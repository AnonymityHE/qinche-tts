"""FastAPI backend — exposes the Context-Aware Emotional TTS pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from src.context_engine.analyzer import EmotionAnalyzer
from src.context_engine.models import (
    EmotionAnnotation,
    EmotionCategory,
)
from src.emotion_tracker.tracker import EmotionArcTracker
from src.pipeline import load_emotion_buckets, select_ref_audio
from src.rag.retriever import CharacterRetriever

logger = logging.getLogger("emotional_tts")

app = FastAPI(title="Context-Aware Emotional TTS", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SPEAKER = "秦彻"

_retriever: Optional[CharacterRetriever] = None
_analyzer: Optional[EmotionAnalyzer] = None
_buckets: dict[str, list[dict]] = {}
_rag_cache: dict[str, list[str]] = {}


def _get_retriever() -> CharacterRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CharacterRetriever()
    return _retriever


def _get_analyzer() -> EmotionAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EmotionAnalyzer()
    return _analyzer


def _get_buckets() -> dict[str, list[dict]]:
    global _buckets
    if not _buckets:
        _buckets = load_emotion_buckets()
    return _buckets


def _cached_retrieve(scene: str) -> list[str]:
    if scene not in _rag_cache:
        _rag_cache[scene] = _get_retriever().retrieve(scene)
    return _rag_cache[scene]


class AnalyzeRequest(BaseModel):
    scene: str = Field(description="Scene description")
    dialogue: list[dict] = Field(description="List of {speaker, text}")


class LineResult(BaseModel):
    index: int
    speaker: str
    text: str
    emotion: Optional[dict] = None
    rag_context: list[str] = []
    emotion_history: list[dict] = []
    ref_audio: Optional[str] = None
    audio: dict[str, str] = {}
    skipped: bool = False


class AnalyzeResponse(BaseModel):
    session_id: str
    results: list[LineResult]
    emotion_arc: list[dict]


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_script(req: AnalyzeRequest):
    """Analyze a script: RAG + LLM emotion + emotion arc."""
    try:
        analyzer = _get_analyzer()
        tracker = EmotionArcTracker()

        session_id = uuid.uuid4().hex[:12]
        results: list[LineResult] = []
        arc: list[dict] = []

        for i, line in enumerate(req.dialogue):
            speaker = line.get("speaker", "")
            text = line.get("text", "")

            if speaker != TARGET_SPEAKER:
                tracker.update(
                    EmotionAnnotation(
                        emotion="other_speaker",
                        intensity=0.0,
                        pace="normal",
                        style="",
                        ref_emotion_category=EmotionCategory.CALM,
                    ),
                    text,
                )
                results.append(LineResult(index=i, speaker=speaker, text=text, skipped=True))
                arc.append({"index": i, "speaker": speaker, "category": None, "intensity": 0})
                continue

            rag_ctx = _cached_retrieve(req.scene)

            emotion = analyzer.analyze(
                line=text,
                scene_description=req.scene,
                emotion_history=tracker.history,
                character_context=rag_ctx,
            )

            tracker.update(emotion, text)

            ref_audio_path = None
            try:
                ref_audio_path = str(select_ref_audio(
                    emotion.ref_emotion_category, _get_buckets(),
                    intensity=emotion.intensity,
                ))
            except FileNotFoundError:
                pass

            results.append(
                LineResult(
                    index=i,
                    speaker=speaker,
                    text=text,
                    emotion=emotion.model_dump(),
                    rag_context=rag_ctx[:3],
                    emotion_history=[s.model_dump() for s in tracker.history[-5:]],
                    ref_audio=ref_audio_path,
                )
            )
            arc.append({
                "index": i,
                "speaker": speaker,
                "category": emotion.ref_emotion_category.value,
                "intensity": emotion.intensity,
                "emotion": emotion.emotion,
                "style": emotion.style,
            })

        return AnalyzeResponse(session_id=session_id, results=results, emotion_arc=arc)

    except Exception as e:
        logger.exception("analyze_script failed")
        raise HTTPException(500, detail=f"Analysis failed: {type(e).__name__}: {e}")


class PipelineRequest(BaseModel):
    scene: str
    dialogue: list[dict]
    backends: list[str] = ["auto", "clone_xvec", "baseline"]


@app.post("/api/pipeline")
async def run_full_pipeline(req: PipelineRequest):
    """Run full pipeline including TTS generation.

    Runs the synchronous pipeline in a thread to avoid blocking the event loop.
    """
    try:
        session_id = uuid.uuid4().hex[:12]
        output_dir = Path("output") / session_id

        script_data = {"scene": req.scene, "dialogue": req.dialogue}
        tmp_script = output_dir / "script.json"
        tmp_script.parent.mkdir(parents=True, exist_ok=True)
        tmp_script.write_text(json.dumps(script_data, ensure_ascii=False, indent=2))

        from src.pipeline import run_pipeline

        pipeline_results = await asyncio.to_thread(
            run_pipeline,
            script_path=tmp_script,
            backends=req.backends,
            output_dir=output_dir,
        )

        return {"session_id": session_id, "results": pipeline_results}

    except Exception as e:
        logger.exception("run_full_pipeline failed")
        raise HTTPException(500, detail=f"Pipeline failed: {type(e).__name__}: {e}")


@app.get("/api/audio/{session_id}/{backend}/{filename}")
async def serve_audio(session_id: str, backend: str, filename: str):
    """Serve generated audio files."""
    path = Path("output") / session_id / backend / filename
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/emotion-buckets")
async def get_emotion_buckets():
    """Return the emotion bucket summary."""
    buckets = _get_buckets()
    return {cat: len(files) for cat, files in buckets.items()}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": _retriever is not None,
        "analyzer_ready": _analyzer is not None,
        "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
