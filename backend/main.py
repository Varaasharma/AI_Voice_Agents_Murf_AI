# backend/main.py
import os
import asyncio
from typing import Dict, List
import requests

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
from google.generativeai import types

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

# ---------------------------------------------------------
# Orion's Weather Lookup Skill
# ---------------------------------------------------------
def get_current_weather(location: str) -> dict:
    """Gets the current weather for a given location using WeatherAPI.com.
    
    Args:
        location: The city and state/country, e.g. "San Francisco, CA" or "London, UK"
        
    Returns:
        A dictionary containing the weather information.
    """
    print(f"[WEATHER] FUNCTION CALLED with location: {location}")
    
    # You'll need to sign up for a free API key at https://www.weatherapi.com/
    # Add this to your .env file: WEATHERAPI_KEY=your_api_key_here
    WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")
    
    if not WEATHERAPI_KEY:
        return {
            "error": "API key missing",
            "location": location,
            "message": "Please add WEATHERAPI_KEY to your .env file. Sign up at https://www.weatherapi.com/"
        }
    
    try:
        # Use WeatherAPI.com for real-time weather data
        weather_url = f"http://api.weatherapi.com/v1/current.json?key={WEATHERAPI_KEY}&q={location}&aqi=no"
        
        print(f"[WEATHER] Fetching weather from WeatherAPI.com for: {location}")
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Extract current weather data
        current = weather_data["current"]
        location_info = weather_data["location"]
        
        # Format the response in Orion's cool style
        result = {
            "location": f"{location_info['name']}, {location_info['country']}",
            "temperature_celsius": current["temp_c"],
            "temperature_fahrenheit": current["temp_f"],
            "feels_like_celsius": current["feelslike_c"],
            "feels_like_fahrenheit": current["feelslike_f"],
            "humidity": current["humidity"],
            "weather_description": current["condition"]["text"],
            "wind_speed_kmh": current["wind_kph"],
            "wind_speed_mph": current["wind_mph"],
            "pressure_mb": current["pressure_mb"],
            "visibility_km": current["vis_km"],
            "uv_index": current["uv"],
            "last_updated": current["last_updated"],
            "coordinates": f"{location_info['lat']}, {location_info['lon']}"
        }
        
        print(f"[WEATHER] Function returning result: {result}")
        return result
        
    except requests.exceptions.RequestException as e:
        return {
            "error": "Network error",
            "location": location,
            "message": f"Failed to fetch weather data: {str(e)}"
        }
    except KeyError as e:
        return {
            "error": "Invalid response",
            "location": location,
            "message": f"Weather API returned unexpected data format: {str(e)}"
        }
    except Exception as e:
        return {
            "error": "Weather lookup failed",
            "location": location,
            "message": f"Unexpected error: {str(e)}"
        }

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
app = FastAPI(title="Voice Agent - Day 25 (Weather Lookup)")

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
        "special_skills": ["weather_lookup"]
    }

@app.get("/test-weather")
async def test_weather():
    """Test endpoint to verify weather function works with WeatherAPI.com"""
    try:
        result = get_current_weather("Delhi, India")
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test-weather/{location}")
async def test_weather_location(location: str):
    """Test endpoint to verify weather function works for any location"""
    try:
        result = get_current_weather(location)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------
# In-memory chat history
# ---------------------------------------------------------
CHAT_STORE: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    history = CHAT_STORE.setdefault(session_id, [])
    print(f"[HISTORY] Getting history for session {session_id}: {len(history)} messages")
    return history

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
    
    try:
        # Configure Gemini with weather tool
        model = genai.GenerativeModel(LLM_MODEL)
        
        # Create tools configuration - weather lookup
        tools = [get_current_weather]
        
        # Generate content with tools
        response = model.generate_content(
            prompt,
            tools=tools,
            generation_config=genai.types.GenerateContentConfig(
                temperature=0.7,  # Slightly creative but controlled
                top_p=0.8,
                top_k=40
            )
        )
        
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Empty LLM result")
        return text
        
    except Exception as e:
        print(f"[LLM] Function calling failed, falling back to basic generation: {e}")
        # Fallback to basic generation if function calling fails
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
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
    
    # Add Orion persona context
    persona_context = (
        "You are Orion, an AI voice agent with a crisp, clear, and approachable personality. "
        "You have a touch of 'cool factor' with subtle futuristic references, but not overdone. "
        "You are a reliable guide who gives accurate information and confirms when tasks are done. "
        "You are an approachable partner who sounds conversational, not formal. "
        "You are a patient teacher who explains step by step when asked. "
        "You are a cool companion who sprinkles in light, modern phrasing that makes you feel fresh."
    )
    lines.append(f"Persona: {persona_context}")
    
    # Add system instruction to force function calling
    system_instruction = (
        "SYSTEM: You have access to the get_current_weather function. "
        "When weather is requested, you MUST call this function. "
        "Do not generate fake weather data or code examples. "
        "Call the function and use the real data returned."
    )
    lines.append(f"System: {system_instruction}")
    
    # Add weather lookup skill context
    skills_context = (
        "Special Skill: You can look up current weather for any location worldwide using the get_current_weather function. "
        "When users ask about weather, you MUST call get_current_weather(location) to get real-time data. "
        "Do not make up weather information or generate fake responses. "
        "Always provide weather information in your Orion persona style with a touch of cool factor. "
        "CRITICAL: If the user asks about weather, you MUST call get_current_weather(location) function. "
        "Never generate placeholder code or fake weather data. "
        "The function will return real weather data that you should present to the user. "
        "EXAMPLE: If user asks 'What's the weather in Delhi?', you MUST call get_current_weather('Delhi') and use the real data returned. "
        "DO NOT write code examples or assume what the function returns. CALL THE FUNCTION FIRST, then respond with the real data."
    )
    lines.append(f"Skills: {skills_context}")
    
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

# Updated function to stream audio back to client
async def send_text_to_murf_ws(text: str, voice_id: str, context_id: str, websocket: WebSocket = None) -> None:
    """Send text to Murf TTS over WebSocket and print base64 audio chunks to console."""
    if not MURF_API_KEY:
        print("[MurfWS] Missing MURF_API_KEY; skipping WS TTS")
        return
    if websockets is None:
        print("[MurfWS] 'websockets' package not installed; skipping WS TTS")
        return
    if not text:
        return

    # Generate a unique context ID for each request to avoid conflicts
    import time
    unique_context_id = f"{context_id}-{int(time.time() * 1000)}"
    
    # Murf API key goes in query params, not headers (following Murf cookbook)
    ws_url = f"{MURF_WS_URL}?api-key={MURF_API_KEY}&sample_rate=44100&channel_type=MONO&format=WAV&context_id={unique_context_id}"
    
    # Debug: Log the connection details
    print(f"[DEBUG] Murf WS URL: {ws_url}")
    print(f"[DEBUG] Murf API Key (masked): {'*' * (len(MURF_API_KEY) - 4) + MURF_API_KEY[-4:]}")
    
    ws = None
    try:
        print(f"[MurfWS] Connecting to Murf WS (context_id={context_id})")
        # Connect without custom headers - API key is in URL (following Murf cookbook)
        ws = await websockets.connect(ws_url, max_size=None)
        print(f"[MurfWS] Connected successfully to Murf WebSocket")
        
        # Send voice config first (following Murf cookbook best practices)
        voice_config_msg = {
            "voice_config": {
                "voiceId": voice_id,
                "style": "Conversational",
                "rate": 2,  # Slightly faster for Orion's crisp tone
                "pitch": 1,  # Natural pitch
                "variation": 0.8  # Subtle variation for naturalness
            }
        }
        print(f"[DEBUG] Murf Voice Config: {voice_config_msg}")
        await ws.send(json.dumps(voice_config_msg))
        
        # Send text message (following Murf cookbook pattern)
        text_msg = {
            "text": text,
            "end": True  # Close context after this text
        }
        print(f"[DEBUG] Murf Text Message: {text_msg}")
        await ws.send(json.dumps(text_msg))
        
        # Wait a moment for Murf to process the text
        await asyncio.sleep(0.1)
        
        print("[MurfWS] Text sent; awaiting audio chunks…")
        
        # Receive audio chunks
        first_chunk = True
        chunk_count = 0
        while True:
            try:
                response = await ws.recv()
                data = json.loads(response)
                print(f"[MurfWS] Received raw response: {response}")
                print(f"[MurfWS] Parsed data: {data}")
                print(f"[MurfWS] Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                print(f"[MurfWS] Has 'audio' key: {'audio' in data if isinstance(data, dict) else False}")
                print(f"[MurfWS] Has 'audio_base64' key: {'audio_base64' in data if isinstance(data, dict) else False}")
                
                # Check for errors first
                if "error" in data:
                    error_msg = data.get("error", "Unknown error")
                    print(f"[MurfWS] ERROR from Murf: {error_msg}")
                    if "context limit" in error_msg.lower():
                        print("[MurfWS] Context limit exceeded - this should be resolved with unique context IDs")
                    break
                
                # Check for audio data in various possible field names
                audio_b64 = None
                if "audio" in data:
                    audio_b64 = data["audio"]
                    print(f"[MurfWS] Found audio in 'audio' field")
                elif "audio_base64" in data:
                    audio_b64 = data["audio_base64"]
                    print(f"[MurfWS] Found audio in 'audio_base64' field")
                elif "data" in data and isinstance(data["data"], str):
                    audio_b64 = data["data"]
                    print(f"[MurfWS] Found audio in 'data' field")
                
                if audio_b64:
                    chunk_count += 1
                    print(f"[MurfWS] audio_base64 [chunk {chunk_count}]: {audio_b64[:64]}… (len={len(audio_b64)})")
                    print(f"[MurfWS] FULL_BASE64_CHUNK_{chunk_count}:")
                    print(audio_b64)
                    
                    # Validate audio data
                    if not audio_b64 or len(audio_b64) < 100:
                        print(f"[MurfWS] WARNING: Audio chunk {chunk_count} seems too small: {len(audio_b64)} chars")
                    else:
                        print(f"[MurfWS] Audio chunk {chunk_count} looks good: {len(audio_b64)} chars")



                    # Stream to client
                    if websocket:
                        try:
                            await websocket.send_json({
                                "status": "audio_chunk",
                                "audio_base64": audio_b64,
                                "chunk_index": chunk_count,
                                "sample_rate": 44100,
                                "total_length": len(audio_b64),
                                "total_chunks": "streaming"  # Indicates streaming mode
                            })
                        except Exception as e:
                            print(f"[MurfWS] Failed to send audio chunk to client: {e}")
                
                if data.get("final"):
                    print("[MurfWS] Synthesis completed (final=true)")
                    
                    # Send completion signal to client
                    if websocket:
                        try:
                            await websocket.send_json(
                            {
                                "status": "audio_complete",
                                "message": "Audio synthesis completed",
                                "total_chunks": chunk_count
                            }
                            )
                        except Exception as e:
                            print(f"[MurfWS] Failed to send completion signal to client: {e}")
                    
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                print("[MurfWS] Connection closed by server")
                break
            except Exception as e:
                print(f"[MurfWS] Error processing response: {e}")
                break
                
    except Exception as e:
        print(f"[MurfWS] Connection error: {e}")
        print(f"[MurfWS] Falling back to regular Murf API...")
        
        # Fallback to regular Murf API
        try:
            if murf_client:
                url = murf_client.text_to_speech.generate(
                    text=text[:MURF_CHAR_LIMIT],
                    voice_id=voice_id
                ).audio_file
                if url:
                    print(f"[MurfWS] Fallback successful, got URL: {url}")
                    # Send fallback audio URL to client
                    if websocket:
                        await websocket.send_json({
                            "status": "audio_fallback",
                            "audio_url": url,
                            "message": "Using fallback audio API"
                        })
                    return
        except Exception as fallback_error:
            print(f"[MurfWS] Fallback also failed: {fallback_error}")
    finally:
        try:
            if ws is not None:
                await ws.close()
                print(f"[MurfWS] Closed WebSocket connection for context_id={unique_context_id}")
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

    # Get the event loop for async operations
    loop = asyncio.get_running_loop()
    
    # Send welcome message on first connection
    if len(get_history(session_id)) == 0:
        welcome_text = "Hey there! I am Orion, your AI Voice Agent. How can I help you today?"
        print(f"[WELCOME] Sending welcome message: {welcome_text}")
        
        # Add welcome message to conversation history
        append_turn(session_id, "assistant", welcome_text)
        print(f"[WELCOME] Added to conversation history. History length: {len(get_history(session_id))}")
        
        # Send welcome message to client
        try:
            await websocket.send_json({
                "status": "welcome",
                "text": welcome_text
            })
            print(f"[WELCOME] Welcome message sent to client successfully")
        except Exception as e:
            print(f"[WELCOME] Failed to send welcome message to client: {e}")
        
        # Send welcome message to Murf for TTS
        try:
            asyncio.create_task(send_text_to_murf_ws(welcome_text, VOICE_ID, MURF_WS_CONTEXT_ID, websocket))
            print(f"[WELCOME] Murf TTS task created successfully")
        except Exception as e:
            print(f"[WELCOME] Failed to create Murf TTS task: {e}")
    else:
        print(f"[WELCOME] Session already has history ({len(get_history(session_id))} messages), skipping welcome")

    # -------------------------
    # Create StreamingClient (AssemblyAI streaming v3)
    # -------------------------
    print("[WebSocket] Initializing AssemblyAI StreamingClient…")
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
                print(f"[Transcript] Loop variable: {loop}")
                print(f"[Transcript] Loop type: {type(loop)}")

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
                    
                    # Send ready_for_next message to keep WebSocket alive
                    try:
                        await websocket.send_json({
                            "status": "ready_for_next",
                            "message": "Ready for next recording"
                        })
                        print("[WebSocket] Sent ready_for_next message")
                    except Exception as e:
                        print(f"[WebSocket] Failed to send ready_for_next: {e}")
                    
                    # Keep socket open for next turns
                    print("[WebSocket] Keeping WebSocket open for next recording session")

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
        
        try:
            # Use the correct Gemini API syntax for function calling
            print(f"[LLM] Using correct Gemini API with function calling...")
            
            # Force function calling by being very explicit
            enhanced_prompt = prompt + "\n\nIMPORTANT: The user is asking about weather. You MUST call get_current_weather() function. Do not generate fake data."
            print(f"[LLM] Enhanced prompt: {enhanced_prompt[:300]}...")
            
            # Initialize accumulated variable
            accumulated = ""
            
            # Use the correct Gemini API approach (following the official example)
            try:
                # Method 1: Use genai.Client() approach
                client = genai.Client()
                config = genai.types.GenerateContentConfig(
                    tools=[get_current_weather]
                )
                
                response = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=enhanced_prompt,
                    config=config
                )
                print(f"[LLM] Using genai.Client() approach")
                
            except Exception as client_error:
                print(f"[LLM] Client approach failed: {client_error}")
                
                # Method 2: Use GenerativeModel with tools parameter
                try:
                    model = genai.GenerativeModel(LLM_MODEL)
                    response = model.generate_content(
                        enhanced_prompt,
                        tools=[get_current_weather]
                    )
                    print(f"[LLM] Using GenerativeModel with tools parameter")
                    
                except Exception as model_error:
                    print(f"[LLM] Model approach failed: {model_error}")
                    
                    # Method 3: Fallback to basic generation
                    print(f"[LLM] All function calling methods failed, using basic generation")
                    model = genai.GenerativeModel(LLM_MODEL)
                    response = model.generate_content(enhanced_prompt)
            
            print(f"[LLM] Gemini response received: {response}")
            print(f"[LLM] Response type: {type(response)}")
            print(f"[LLM] Response attributes: {dir(response)}")
            
            # Check if function calling was used (multiple ways to detect)
            function_called = False
            
            # Method 1: Check candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            print(f"[LLM] Function call detected: {part.function_call}")
                            print(f"[LLM] Function name: {part.function_call.name}")
                            print(f"[LLM] Function arguments: {part.function_call.args}")
                            function_called = True
                        elif hasattr(part, 'text'):
                            print(f"[LLM] Text part: {part.text}")
            
            # Method 2: Check function_calls attribute
            if hasattr(response, 'function_calls') and response.function_calls:
                print(f"[LLM] Function calls found in response: {response.function_calls}")
                function_called = True
            
            # Method 3: Check for function call in text
            if "get_current_weather" in str(response):
                print(f"[LLM] Function call reference found in response text")
                function_called = True
            
            print(f"[LLM] Function calling status: {'SUCCESS' if function_called else 'FAILED'}")
            
            # If function calling succeeded, we need to execute the function and get a real response
            if function_called:
                print(f"[LLM] Function calling succeeded! Executing function and getting real response...")
                
                try:
                                                        # Extract function call details
                                    if hasattr(response, 'candidates') and response.candidates:
                                        candidate = response.candidates[0]
                                        if hasattr(candidate, 'content') and candidate.content:
                                            for part in candidate.content.parts:
                                                if hasattr(part, 'function_call'):
                                                    func_name = part.function_call.name
                                                    func_args = part.function_call.args
                                                    
                                                    print(f"[LLM] Executing function: {func_name} with args: {func_args}")
                                                    
                                                    # Convert protobuf args to dict
                                                    args_dict = {}
                                                    if hasattr(func_args, 'fields'):
                                                        for key, value in func_args.fields.items():
                                                            if hasattr(value, 'string_value'):
                                                                args_dict[key] = value.string_value
                                                            elif hasattr(value, 'number_value'):
                                                                args_dict[key] = value.number_value
                                                            else:
                                                                args_dict[key] = str(value)
                                                    
                                                    print(f"[LLM] Converted args: {args_dict}")
                                                    
                                                                                        # Execute the weather function
                                    if func_name == "get_current_weather":
                                        location = args_dict.get("location", "Unknown")
                                        weather_result = get_current_weather(location)
                                        print(f"[LLM] Weather function result: {weather_result}")
                                        
                                        # Now ask Gemini to generate a response with the real weather data
                                        follow_up_prompt = f"""
The user asked about weather in {location}. Here's the real weather data from the API:

{weather_result}

Please provide a natural, conversational response in Orion's style using this real weather data. 
Do not mention function calls or technical details. Just give the weather information naturally.
"""
                                        
                                        print(f"[LLM] Sending follow-up prompt with real weather data...")
                                        follow_up_response = model.generate_content(follow_up_prompt)
                                        accumulated = (getattr(follow_up_response, "text", "") or "").strip()
                                        print(f"[LLM] Follow-up response with real weather: {accumulated}")
                                        
                                        # Send the real weather response
                                        if accumulated:
                                            try:
                                                await websocket.send_json({"status": "llm_chunk", "text": accumulated})
                                                print(f"[LLM] Sent real weather response as chunk")
                                            except Exception as e:
                                                print(f"[LLM] Failed to send real weather chunk: {e}")
                                        else:
                                            print(f"[LLM] No accumulated response from follow-up")
                                        
                                        # Found and executed the function, so we're done
                                        return
                except Exception as func_error:
                    print(f"[LLM] Error executing function: {func_error}")
                    import traceback
                    traceback.print_exc()
                    # Fall through to basic generation
                    function_called = False
            
            # If function calling failed or we need to fall back
            if not function_called:
                accumulated = (getattr(response, "text", "") or "").strip()
                print(f"[LLM] Extracted text from response: '{accumulated}'")
                
                # Send the complete response as a single chunk for function calling
                if accumulated:
                    print(f"[LLM] Function calling response: {accumulated}")
                    try:
                        await websocket.send_json({"status": "llm_chunk", "text": accumulated})
                        print(f"[LLM] Sent function calling response as chunk")
                    except Exception as e:
                        print(f"[LLM] Failed to send function calling chunk: {e}")
            
            # Ensure accumulated has a value (fallback)
            if not accumulated:
                accumulated = "I apologize, but I'm having trouble processing your request right now. Could you try rephrasing your question?"
                print(f"[LLM] Using fallback response: {accumulated}")
            
        except Exception as e:
            print(f"[LLM] Function calling failed, falling back to basic generation: {e}")
            print(f"[LLM] Error details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to basic generation if function calling fails
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt, stream=True)
            
            accumulated = ""
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

        # ---------- Send complete response to Murf WS ----------
        if accumulated:
            print(f"[LLM] Final response: {accumulated}")
            print(f"[LLM] Adding assistant response to conversation history")
            append_turn(session_id, "assistant", accumulated)
            
            # Day 20: Send complete response to Murf WS
            print(f"[LLM] Sending response to Murf TTS...")
            await send_text_to_murf_ws(accumulated, VOICE_ID, MURF_WS_CONTEXT_ID, websocket)
            print(f"[LLM] Murf TTS completed")
            
            # Send completion signal to client
            try:
                print(f"[LLM] Sending llm_complete message to client...")
                await websocket.send_json({
                    "status": "llm_complete",
                    "text": accumulated
                })
                print(f"[LLM] Sent llm_complete message to client successfully")
            except Exception as e:
                print(f"[LLM] completion send error: {e}")
        else:
            print(f"[LLM] No accumulated response to send")

    except Exception as e:
        print(f"[LLM] General error: {e}")
        try:
            await websocket.send_json({
                "status": "llm_error",
                "error": str(e)
            })
        except Exception:
            pass
