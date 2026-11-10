"""
api/main.py
-----------
Production-ready FastAPI REST API for Hindi TTS.

Endpoints:
  GET  /health          → system health check
  GET  /voices          → list all voice profiles
  POST /tts             → synthesize Hindi text → MP3 download
  POST /story           → AI multi-character story narration
  GET  /outputs         → list generated audio files

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Env vars (set in .env):
  GROQ_API_KEY   — required for /story endpoint
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Bootstrap ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from src.tts_engine import HindiTTSEngine, VOICE_PROFILES  # noqa: E402

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hindi TTS API",
    description=(
        "Production-grade Hindi Text-to-Speech REST API powered by "
        "Microsoft Edge-TTS Neural voices and Groq LLM story narration."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared TTS engine (cached) ────────────────────────────────────────────────
_tts_engine: Optional[HindiTTSEngine] = None

def get_engine() -> HindiTTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = HindiTTSEngine(output_dir=OUTPUTS_DIR)
    return _tts_engine


# ── Request / Response models ─────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Hindi text to synthesize (Devanagari preferred)",
        example="नमस्ते! आपका स्वागत है।",
    )
    voice: str = Field(
        default="narrator_male",
        description="Voice profile key (see GET /voices)",
    )
    rate: Optional[str] = Field(
        default=None,
        description="Speech rate override, e.g. '+10%' or '-20%'",
        example="+10%",
    )
    pitch: Optional[str] = Field(
        default=None,
        description="Pitch override in Hz, e.g. '+5Hz' or '-10Hz'",
        example="+5Hz",
    )
    volume: str = Field(default="+0%", description="Volume adjustment")


class StoryRequest(BaseModel):
    story: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Raw Hindi story text with narrator + character dialogue",
        example=(
            "एक जंगल में शेर रहता था। उसने कहा, \"मैं इस जंगल का राजा हूँ!\"\n"
            "खरगोश ने जवाब दिया, \"लेकिन राजा को दयालु भी होना चाहिए।\""
        ),
    )
    project_id: str = Field(
        default="",
        description="Optional project name prefix for output filenames",
    )


class TTSResponse(BaseModel):
    file: str
    voice: str
    duration_ms: int
    download_url: str


class HealthResponse(BaseModel):
    status: str
    version: str
    groq_configured: bool
    output_dir: str
    generated_files: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """System health check — always returns 200 if the API is running."""
    files = list(OUTPUTS_DIR.glob("*.mp3"))
    return HealthResponse(
        status="ok",
        version="1.0.0",
        groq_configured=bool(os.environ.get("GROQ_API_KEY")),
        output_dir=str(OUTPUTS_DIR),
        generated_files=len(files),
    )


@app.get("/voices", tags=["TTS"])
def list_voices():
    """
    List all available Hindi voice profiles.

    Returns each profile's key, label, underlying neural voice,
    and default acoustic settings.
    """
    return {
        "voices": [
            {
                "key": key,
                "label": profile["label"],
                "voice": profile["voice"],
                "rate": profile["rate"],
                "pitch": profile["pitch"],
            }
            for key, profile in VOICE_PROFILES.items()
        ],
        "count": len(VOICE_PROFILES),
    }


@app.post("/tts", tags=["TTS"])
def synthesize(req: TTSRequest):
    """
    Synthesize Hindi text into an MP3 file.

    - **text**: Hindi text (Devanagari). Max 5000 chars.
    - **voice**: One of the keys returned by GET /voices.
    - **rate / pitch**: Optional acoustic overrides.

    Returns the MP3 file as a direct download.
    """
    if req.voice not in VOICE_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown voice '{req.voice}'. "
                   f"Valid options: {list(VOICE_PROFILES.keys())}",
        )

    profile = VOICE_PROFILES[req.voice]
    filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"

    t0 = time.time()
    try:
        engine = get_engine()
        output_path = engine.generate_speech(
            text=req.text,
            filename=filename,
            voice=profile["voice"],
            rate=req.rate or profile["rate"],
            pitch=req.pitch or profile["pitch"],
            volume=req.volume,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")

    elapsed_ms = int((time.time() - t0) * 1000)

    return FileResponse(
        path=str(output_path),
        media_type="audio/mpeg",
        filename=filename,
        headers={
            "X-Voice": req.voice,
            "X-Duration-Ms": str(elapsed_ms),
            "X-Text-Length": str(len(req.text)),
        },
    )


@app.post("/story", tags=["Story Narration"])
def narrate_story(req: StoryRequest):
    """
    AI-directed multi-character Hindi story narration.

    **Requires GROQ_API_KEY in .env.**

    Two LLM agents process the story:
    1. **Script Analyst** — splits into narrator vs character turns
    2. **Acoustic Director** — assigns voice profiles, emotion, pacing

    Returns a JSON screenplay with audio file paths for each turn.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not configured. Set it in .env to use this endpoint.",
        )

    try:
        from src.story_director import StoryDirector
        director = StoryDirector(output_dir=OUTPUTS_DIR)
        project_id = req.project_id or f"story_{uuid.uuid4().hex[:6]}"
        result = director.narrate_story(req.story, project_id=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Story narration failed: {e}")

    return JSONResponse(content={
        "project_id": project_id,
        "turns": len(result["turns"]),
        "screenplay": result["turns"],
        "audio_files": [str(p) for p in result["clips"]],
    })


@app.get("/outputs", tags=["System"])
def list_outputs(
    limit: int = Query(default=20, ge=1, le=200, description="Max files to return"),
):
    """List the most recently generated MP3 files."""
    files = sorted(OUTPUTS_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {
        "files": [
            {
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "created": f.stat().st_mtime,
            }
            for f in files[:limit]
        ],
        "total": len(list(OUTPUTS_DIR.glob("*.mp3"))),
    }


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
