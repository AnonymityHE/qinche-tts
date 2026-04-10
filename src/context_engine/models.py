"""Pydantic models for emotion annotation and pipeline data structures."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmotionCategory(str, Enum):
    TENDER = "tender"
    CALM = "calm"
    PLAYFUL = "playful"
    INTENSE = "intense"
    COLD = "cold"
    INTIMATE = "intimate"


class EmotionAnnotation(BaseModel):
    """Structured emotion annotation output from the LLM Context Engine."""

    emotion: str = Field(description="Fine-grained emotion label, e.g. 'gentle_reassuring'")
    intensity: float = Field(ge=0.0, le=1.0, description="Emotion intensity from 0 to 1")
    pace: str = Field(description="Speaking pace: 'slow', 'normal', 'fast'")
    style: str = Field(description="Natural language style description for TTS instruction")
    ref_emotion_category: EmotionCategory = Field(
        description="Coarse emotion bucket for reference audio selection"
    )
    fish_audio_tags: Optional[str] = Field(
        default=None,
        description="Inline [tag] formatted text for Fish Audio S2 Pro",
    )


class DialogueLine(BaseModel):
    """A single line in a game script."""

    speaker: str
    text: str


class GameScript(BaseModel):
    """Input game script format."""

    scene: str = Field(description="Scene description, e.g. '深夜书房，雨声淅沥'")
    dialogue: list[DialogueLine]


class EmotionState(BaseModel):
    """Recorded emotion state for arc tracking."""

    text: str
    emotion: str
    intensity: float
    ref_emotion_category: EmotionCategory
