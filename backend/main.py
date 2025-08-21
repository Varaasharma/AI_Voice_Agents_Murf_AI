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
import json

# Optional: Murf WS
try:
    import websockets  # type: ignore
except Exception:
    websockets = None  # defer import error until used

# ---------------------------------------------------------
# Env & SDK setup
# ---------------------------------------------------------
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY") or ""
MURF_API_KEY       = os.getenv("MURF_API_KEY") or ""
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY") or ""

# Debug: Log API keys (masked for security)
print(f"[DEBUG] ASSEMBLYAI_API_KEY: {'*' * (len(ASSEMBLYAI_API_KEY) - 4) + ASSEMBLYAI_API_KEY[-4:] if ASSEMBLYAI_API_KEY else 'NOT_SET'}")
print(f"[DEBUG] MURF_API_KEY: {'*' * (len(MURF_API_KEY) - 4) + MURF_API_KEY[-4:] if MURF_API_KEY else 'NOT_SET'}")
print(f"[DEBUG] GEMINI_API_KEY: {'*' * (len(GEMINI_API_KEY) - 4) + GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'NOT_SET'}")

# Murf WS configuration
MURF_WS_URL        = os.getenv("MURF_WS_URL") or "wss://api.murf.ai/v1/speech/stream-input"

MURF_WS_CONTEXT_ID = os.getenv("MURF_WS_CONTEXT_ID") or "day20-context"

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

# ---------------- Murf WebSocket helpers (Day 20) ----------------
async def _ws_connect_with_headers(url: str, headers: Dict[str, str]):
    """Try multiple websockets.connect signatures to pass headers across versions."""
    if websockets is None:
        raise RuntimeError("websockets package not available")
    # 1) extra_headers=dict
    try:
        return await websockets.connect(url, extra_headers=headers, max_size=None)  # type: ignore
    except TypeError:
        pass
    # 2) extra_headers=list of tuples
    try:
        return await websockets.connect(url, extra_headers=[(k, v) for k, v in headers.items()], max_size=None)  # type: ignore
    except TypeError:
        pass
    # 3) additional_headers=dict
    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)  # type: ignore
    except TypeError:
        pass
    # 4) additional_headers=list of tuples
    return await websockets.connect(url, additional_headers=[(k, v) for k, v in headers.items()], max_size=None)  # type: ignore

async def start_murf_ws_session(context_id: str, voice_id: str):
    """Open a Murf WS session, send init/config, and start a reader that prints base64 audio."""
    if not MURF_API_KEY:
        print("[MurfWS] Missing MURF_API_KEY; skipping WS TTS")
        return None, None
    if websockets is None:
        print("[MurfWS] 'websockets' package not installed; skipping WS TTS")
        return None, None

    headers = {
        "Authorization": MURF_API_KEY,
        "Accept": "application/json",
        "User-Agent": "voice-agents/1.0",
    }
    ws = await _ws_connect_with_headers(MURF_WS_URL, headers)
    # minimal init/config — adjust fields if Murf docs require different names
    await ws.send(json.dumps({
        "type": "init",
        "voice_id": voice_id,
        "context_id": context_id,
    }))
    print(f"[MurfWS] Connected (context_id={context_id}, voice_id={voice_id})")

    async def reader():
        idx = 0
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                print("[MurfWS][RAW]", str(raw)[:120], "…")
                continue

            if isinstance(msg, dict):
                # Adjust keys/types to Murf's schema if needed
                b64 = msg.get("audio_base64") or msg.get("audio")
                if b64:
                    idx += 1
                    print(f"[MurfWS][AUDIO_BASE64][chunk {idx}] len={len(b64)}")
                    # Print the full base64 (as required by the task)
                    print(b64)
                elif msg.get("type") in {"error", "warning"}:
                    print("[MurfWS][ERROR]", msg)
                else:
                    # Helpful to see acks/config/done events
                    preview = {k: msg[k] for k in list(msg.keys())[:4]}
                    print("[MurfWS][INFO]", preview)

    reader_task = asyncio.create_task(reader())
    return ws, reader_task

async def close_murf_ws_session(ws, reader_task, context_id: str):
    if ws:
        try:
            # optional finalize/end message if Murf expects it
            await ws.send(json.dumps({"type": "finalize", "context_id": context_id}))
        except Exception:
            pass
        await asyncio.sleep(0.1)
        try:
            await ws.close()
        except Exception:
            pass
    if reader_task:
        reader_task.cancel()

# keep your original single-shot helper (not used by streaming, but retained)
async def send_text_to_murf_ws(text: str, voice_id: str, context_id: str) -> None:
    """Send text to Murf TTS over WebSocket and print base64 audio chunks to console."""
    if not MURF_API_KEY:
        print("[MurfWS] Missing MURF_API_KEY; skipping WS TTS")
        return
    if websockets is None:
        print("[MurfWS] 'websockets' package not installed; skipping WS TTS")
        return
    if not text:
        return

    # Murf API key goes in query params, not headers
    ws_url = f"{MURF_WS_URL}?api-key={MURF_API_KEY}&sample_rate=44100&channel_type=MONO&format=WAV"
    
    # Debug: Log the connection details
    print(f"[DEBUG] Murf WS URL: {ws_url}")
    print(f"[DEBUG] Murf API Key (masked): {'*' * (len(MURF_API_KEY) - 4) + MURF_API_KEY[-4:]}")
    
    ws = None
    try:
        print(f"[MurfWS] Connecting to Murf WS (context_id={context_id})")
        # Connect without custom headers - API key is in URL
        ws = await websockets.connect(ws_url, max_size=None)
        
        # Send voice config first (as per Murf example)
        voice_config_msg = {
            "voice_config": {
                "voiceId": voice_id,
                "style": "Conversational",
                "rate": 0,
                "pitch": 0,
                "variation": 1
            }
        }
        print(f"[DEBUG] Murf Voice Config: {voice_config_msg}")
        await ws.send(json.dumps(voice_config_msg))
        
        # Send text message (as per Murf example)
        text_msg = {
            "text": text,
            "end": True  # Close context after this text
        }
        print(f"[DEBUG] Murf Text Message: {text_msg}")
        await ws.send(json.dumps(text_msg))
        
        print("[MurfWS] Text sent; awaiting audio chunks…")
        
        # Receive audio chunks
        first_chunk = True
        chunk_count = 0
        while True:
            try:
                response = await ws.recv()
                data = json.loads(response)
                print(f"[MurfWS] Received: {data}")
                
                if "audio" in data:
                    chunk_count += 1
                    audio_b64 = data["audio"]
                    print(f"[MurfWS] audio_base64 [chunk {chunk_count}]: {audio_b64[:64]}… (len={len(audio_b64)})")
                    # Print the full base64 (as required by the task)
                    print(f"[MurfWS] FULL_BASE64_CHUNK_{chunk_count}:")
                    print(audio_b64)
                
                if data.get("final"):
                    print("[MurfWS] Synthesis completed (final=true)")
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                print("[MurfWS] Connection closed by server")
                break
            except Exception as e:
                print(f"[MurfWS] Error processing response: {e}")
                break
                
    except Exception as e:
        print(f"[MurfWS] Connection error: {e}")
    finally:
        try:
            if ws is not None:
                await ws.close()
        except Exception:
            pass

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
    print("[WebSocket] Initializing AssemblyAI StreamingClient…")
    loop = asyncio.get_running_loop()
    client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )

    # Accumulate final transcripts for LLM processing
    final_transcripts = []

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
                    # Deduplicate near-identical finals
                    if not final_transcripts or not is_similar_transcript(txt, final_transcripts[-1]):
                        final_transcripts.append(txt)
                        print(f"[Transcript] Added to final_transcripts: {txt}")
                    else:
                        print(f"[Transcript] Skipped duplicate: {txt}")
            except RuntimeError as e:
                print(f"[Transcript] failed to dispatch to loop: {e}")

    def _on_terminated(self: StreamingClient, event: TerminationEvent):
        print(f"[AssemblyAI] Session terminated: {event.audio_duration_seconds} seconds processed")

    def _on_error(self: StreamingClient, error: StreamingError):
        print(f"[AssemblyAI] Error: {error}")

    # Helper function to detect similar/duplicate transcripts
    def is_similar_transcript(new_txt: str, last_txt: str) -> bool:
        """Check if two transcripts are essentially the same (ignoring case, punctuation, spacing)"""
        if not new_txt or not last_txt:
            return False
        def normalize(t):
            return ' '.join(t.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split())
        norm_new = normalize(new_txt)
        norm_last = normalize(last_txt)
        if norm_new == norm_last:
            return True
        if norm_new in norm_last or norm_last in norm_new:
            return True
        new_words = set(norm_new.split())
        last_words = set(norm_last.split())
        if len(new_words) > 0 and len(last_words) > 0:
            common_words = new_words.intersection(last_words)
            similarity = len(common_words) / max(len(new_words), len(last_words))
            if similarity > 0.8:
                return True
        return False

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
            # message may be dict-form (FastAPI) or bytes; handle both

            # Binary chunk (FastAPI returns a dict with "bytes" key)
            if isinstance(message, dict) and message.get("type") == "websocket.receive" and "bytes" in message:
                audio_chunk = message["bytes"]
                audio_buffer.extend(audio_chunk)
                try:
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
                try:
                    data = json.loads(message["text"])
                except Exception:
                    data = None

                if isinstance(data, dict) and data.get("event") == "end_of_audio":
                    print("End of audio event received. Closing AssemblyAI stream.")

                    # Day 19: Process accumulated final transcripts with LLM
                    if final_transcripts:
                        print(f"[LLM] Processing {len(final_transcripts)} final transcripts: {final_transcripts}")
                        combined_transcript = " ".join(final_transcripts)
                        asyncio.create_task(stream_llm_response(combined_transcript, websocket, loop, session_id))
                        final_transcripts.clear()

                    # Give AssemblyAI a moment to flush and then disconnect/terminate
                    try:
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
                    # break  # keep socket open for next turns if you want

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

# ---------------- LLM Streaming -> Murf WS (Day 19 + Day 20) ----------------
async def stream_llm_response(user_text: str, websocket: WebSocket, loop: asyncio.AbstractEventLoop, session_id: str):
    """Day 19+20: Stream LLM response; forward chunks to Murf WS; print base64 audio in console."""
    if not GEMINI_API_KEY:
        print("[LLM] No API key available")
        return

    try:
        # ---------- Build prompt from history ----------
        history = get_history(session_id)
        append_turn(session_id, "user", user_text)
        prompt = build_prompt_from_history(get_history(session_id))

        print(f"[LLM] Streaming response...")
        model = genai.GenerativeModel(LLM_MODEL)

        # ---------- Stream Gemini response ----------
        accumulated = ""
        try:
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                part = getattr(chunk, "text", "") or ""
                if not part.strip():
                    continue

                accumulated += part
                print(f"[LLM][chunk] {part}")

                # Optional: echo to client
                try:
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({"status": "llm_chunk", "text": part}),
                        loop,
                    )
                except Exception as e:
                    print(f"[LLM] client send error: {e}")

        except Exception as e:
            print(f"[LLM] Streaming error: {e}")
            return

        # ---------- Send complete response to Murf WS ----------
        if accumulated:
            print(f"[LLM] Final response: {accumulated}")
            append_turn(session_id, "assistant", accumulated)
            
            # Day 20: Send complete response to Murf WS
            await send_text_to_murf_ws(accumulated, VOICE_ID, MURF_WS_CONTEXT_ID)
            
            # Send completion signal to client
            try:
                asyncio.create_task(websocket.send_json({
                    "status": "llm_complete",
                    "text": accumulated
                }))
            except Exception as e:
                print(f"[LLM] completion send error: {e}")

    except Exception as e:
        print(f"[LLM] General error: {e}")
        try:
            asyncio.create_task(websocket.send_json({
                "status": "llm_error",
                "error": str(e)
            }))
        except Exception:
            pass
