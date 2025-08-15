# backend/main.py
import os
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import assemblyai as aai
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