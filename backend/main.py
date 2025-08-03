import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from murf import Murf

# Load environment variables
load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY")

app = FastAPI()

# Path to frontend folder
frontend_path = Path(__file__).parent.parent / "frontend"

# Serve static files (JavaScript, CSS, etc.)
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
def read_index():
    """Serve the main HTML file."""
    return FileResponse(frontend_path / "index.html")


@app.post("/tts")
def generate_tts(text: str = Body(..., embed=True)):
    """
    Accept text, send it to Murf TTS API, return the audio file URL.
    """
    client = Murf(api_key=MURF_API_KEY)

    result = client.text_to_speech.generate(
        text=text,
        voice_id="en-US-terrell"  # You can choose any available voice
    )

    return {"audio_url": result.audio_file}
