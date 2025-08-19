# backend/main.py
import os
import asyncio
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, Path, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import assemblyai as aai
# Updated imports: streaming v3 API (per AssemblyAI docs you provided)
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingParameters,
    StreamingEvents,
    TurnEvent,
    BeginEvent,
    TerminationEvent,
    StreamingError,
)
from murf import Murf
import google.generativeai as genai

# ---------------------------------------------------------
# Env & SDK setup
# ---------------------------------------------------------
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") or ""
MURF_API_KEY       = os.getenv("MURF_API_KEY") or ""
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY") or ""

if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

murf_client = Murf(api_key=MURF_API_KEY) if MURF_API_KEY else None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

VOICE_ID        = "en-US-terrell"   # any valid Murf voice id
LLM_MODEL       = "gemini-1.5-flash"
CONTEXT_TURNS   = 12
MURF_CHAR_LIMIT = 3000
FALLBACK_LINE   = "I'm having trouble connecting right now."

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Voice Agent - Day 11 (API fallback)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    return {
        "ok": True,
        "assemblyai": bool(ASSEMBLYAI_API_KEY),
        "murf": bool(MURF_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
    }

# ---------------------------------------------------------
# In-memory chat history
# ---------------------------------------------------------
CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return CHAT_STORE.setdefault(session_id, [])

def append_turn(session_id: str, role: str, text: str) -> None:
    CHAT_STORE.setdefault(session_id, []).append({"role": role, "text": text})

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def safe_transcribe(audio_bytes: bytes) -> str:
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY missing")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_bytes)
    text = (getattr(transcript, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Empty STT result")
    return text

def safe_llm(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    model = genai.GenerativeModel(LLM_MODEL)
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError("Empty LLM result")
    return text

def try_murf_tts(text: str) -> str | None:
    """Return Murf audio URL if possible, otherwise None."""
    if not text:
        return None
    if not murf_client:
        return None
    try:
        url = murf_client.text_to_speech.generate(
            text=text[:MURF_CHAR_LIMIT],
            voice_id=VOICE_ID
        ).audio_file
        return url or None
    except Exception:
        return None

def build_prompt_from_history(history: List[Dict[str, str]]) -> str:
    context = history[-CONTEXT_TURNS:]
    lines = []
    for m in context:
        prefix = "User:" if m["role"] == "user" else "Assistant:"
        lines.append(f"{prefix} {m['text']}")
    lines.append("Assistant:")
    return "\n".join(lines)

def api_audio_fallback_payload(stage: str, details: str | None = None):
    """
    Prefer API-generated fallback via Murf. If Murf unavailable, return speak_text for client SpeechSynthesis.
    """
    url = try_murf_tts(FALLBACK_LINE)
    if url:
        return {"audio_url": url, "error": stage, "details": details}
    # Client will synthesize audibly with Web Speech API
    return {"speak_text": FALLBACK_LINE, "error": stage, "details": details}

# ---------------------------------------------------------
# Day 10/11 endpoint (with API fallbacks)
# ---------------------------------------------------------
@app.post("/agent/chat/{session_id}")
async def agent_chat(
    session_id: str = Path(..., description="Conversation session id"),
    file: UploadFile = File(...)
):
    try:
        audio_bytes = await file.read()

        # 1) STT
        try:
            user_text = safe_transcribe(audio_bytes)
        except Exception as e:
            return api_audio_fallback_payload("stt_failed", str(e))

        # 2) Append user
        append_turn(session_id, "user", user_text)

        # 3) LLM with history
        prompt = build_prompt_from_history(get_history(session_id))
        try:
            reply_text = safe_llm(prompt)
        except Exception as e:
            # Return API-generated fallback (or speak_text)
            return api_audio_fallback_payload("llm_failed", str(e)) | {"user_text": user_text}

        # 4) Append assistant
        append_turn(session_id, "assistant", reply_text)

        # 5) TTS
        url = try_murf_tts(reply_text)
        if url:
            return {"audio_url": url, "user_text": user_text, "assistant_text": reply_text}

        # Murf failed: API-generated fallback (or speak_text)
        return api_audio_fallback_payload("tts_failed") | {
            "user_text": user_text,
            "assistant_text": reply_text
        }

    except Exception as e:
        # Unexpected crash → still return audible fallback path
        return JSONResponse(
            content=api_audio_fallback_payload("server_error", str(e)),
            status_code=200
        )

# ---------------------------------------------------------
# WebSocket endpoint (Day 12+ real-time upgrade)
# ---------------------------------------------------------
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"WebSocket connected for session: {session_id}")

    if not ASSEMBLYAI_API_KEY:
        await websocket.send_json({"status": "error", "message": "No AssemblyAI API key"})
        await websocket.close()
        return

    # -------------------------
    # Create StreamingClient (AssemblyAI streaming v3)
    # -------------------------
    loop = asyncio.get_running_loop()
    print("[WebSocket] Initializing AssemblyAI StreamingClient…")
    client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )

    # Event handlers
    def _on_begin(self: StreamingClient, event: BeginEvent):
        print(f"[AssemblyAI] Session started: {event.id}")

    def _on_turn(self: StreamingClient, event: TurnEvent):
        # event.transcript may be partial or final; forward to client
        txt = getattr(event, "transcript", None)
        is_final = getattr(event, "end_of_turn", False)
        if txt:
            print(f"[Transcript]{' [FINAL]' if is_final else ''} {txt}")
            # Callback runs in a background thread; marshal to main event loop
            try:
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({
                        "status": "transcript",
                        "text": txt,
                        "final": bool(is_final)
                    }),
                    loop,
                )
                if is_final:
                    # Explicit turn-end signal for Day 18
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({
                            "status": "turn_end",
                            "text": txt
                        }),
                        loop,
                    )
            except RuntimeError as e:
                print(f"[Transcript] failed to dispatch to loop: {e}")

            # If end_of_turn and not formatted, request formatted turns (optional)
            try:
                if event.end_of_turn and not event.turn_is_formatted:
                    from assemblyai.streaming.v3 import StreamingSessionParameters
                    params = StreamingSessionParameters(format_turns=True)
                    self.set_params(params)
            except Exception:
                pass

    def _on_terminated(self: StreamingClient, event: TerminationEvent):
        print(f"[AssemblyAI] Session terminated: {event.audio_duration_seconds} seconds processed")

    def _on_error(self: StreamingClient, error: StreamingError):
        print(f"[AssemblyAI] Error: {error}")

    # Register handlers
    client.on(StreamingEvents.Begin, _on_begin)
    client.on(StreamingEvents.Turn, _on_turn)
    client.on(StreamingEvents.Termination, _on_terminated)
    client.on(StreamingEvents.Error, _on_error)

    # Connect with desired params (16 kHz)
    print("[AssemblyAI] Connecting with sample_rate=16000…")
    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,
        )
    )
    print("[AssemblyAI] Connected.")

    # Buffer in case you still want to save the file locally (kept as before)
    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if isinstance(message, dict):
                mtype = message.get("type")
                if mtype == "websocket.receive":
                    if "bytes" in message:
                        print(f"[WS<-] received {len(message['bytes'])} bytes")
                    elif "text" in message:
                        preview = (message["text"] or "")[:80]
                        print(f"[WS<-] received text: {preview}")
            # message may be dict-form (FastAPI) or bytes; handle both

            # Binary chunk (FastAPI returns a dict with "bytes" key)
            if isinstance(message, dict) and message.get("type") == "websocket.receive" and "bytes" in message:
                audio_chunk = message["bytes"]
                audio_buffer.extend(audio_chunk)
                # Stream the raw bytes to AssemblyAI
                try:
                    # Assumes client.stream accepts bytes — per docs it accepts audio sources;
                    # this call is safe in modern SDKs. Use await if stream is async.
                    maybe_awaitable = client.stream(audio_chunk)
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
                    print(f"[Aai->] streamed {len(audio_chunk)} bytes")
                except Exception as e:
                    print(f"[AssemblyAI] stream error: {e}")

            # Binary directly (some FastAPI versions might give bytes)
            elif isinstance(message, (bytes, bytearray)):
                audio_chunk = bytes(message)
                audio_buffer.extend(audio_chunk)
                try:
                    maybe_awaitable = client.stream(audio_chunk)
                    if asyncio.iscoroutine(maybe_awaitable):
                        await maybe_awaitable
                    print(f"[Aai->] streamed {len(audio_chunk)} bytes (raw)")
                except Exception as e:
                    print(f"[AssemblyAI] stream error: {e}")

            # Text message (end_of_audio event)
            elif isinstance(message, dict) and message.get("type") == "websocket.receive" and "text" in message:
                import json
                try:
                    data = json.loads(message["text"])
                except Exception:
                    data = None

                if isinstance(data, dict) and data.get("event") == "end_of_audio":
                    print("End of audio event received. Closing AssemblyAI stream.")
                    # Give AssemblyAI a moment to flush and then disconnect/terminate
                    try:
                        # disconnect can be awaitable
                        maybe_awaitable = client.disconnect(terminate=True)
                        if asyncio.iscoroutine(maybe_awaitable):
                            await maybe_awaitable
                        print("[AssemblyAI] disconnect requested (terminate=True)")
                    except Exception as e:
                        print(f"[AssemblyAI] disconnect error: {e}")

                    # optionally save audio to disk (preserve existing behavior)
                    try:
                        import uuid
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"session_{session_id}_{timestamp}_{unique_id}.webm"
                        filepath = os.path.join("uploads", filename)
                        os.makedirs("uploads", exist_ok=True)
                        with open(filepath, "wb") as f:
                            f.write(bytes(audio_buffer))
                        print(f"Audio file saved successfully: {filepath}")

                        await websocket.send_json({
                            "status": "success",
                            "message": "Audio file saved successfully",
                            "filename": filename,
                            "filepath": filepath
                        })
                    except Exception as e:
                        print(f"Error saving audio file: {e}")
                        await websocket.send_json({
                            "status": "error",
                            "message": "Failed to save audio file",
                            "error": str(e)
                        })

                    audio_buffer.clear()
                    print("Audio buffer cleared after save")
                    # Break the loop if you want to close socket after one recording
                    # break

    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # Make sure AssemblyAI client is disconnected
        try:
            maybe_awaitable = client.disconnect(terminate=True)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
            print("[AssemblyAI] disconnect on finally")
        except Exception:
            pass
