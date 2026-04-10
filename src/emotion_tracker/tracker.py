"""Emotion Arc Tracker — maintain a sliding window of recent emotion states."""

from __future__ import annotations

from collections import deque

from src.context_engine.models import EmotionAnnotation, EmotionCategory, EmotionState

TRANSITION_DISTANCE: dict[tuple[EmotionCategory, EmotionCategory], float] = {
    (EmotionCategory.INTENSE, EmotionCategory.CALM): 0.3,
    (EmotionCategory.INTENSE, EmotionCategory.TENDER): 0.5,
    (EmotionCategory.INTENSE, EmotionCategory.INTIMATE): 0.9,
    (EmotionCategory.COLD, EmotionCategory.INTIMATE): 0.8,
    (EmotionCategory.COLD, EmotionCategory.PLAYFUL): 0.6,
    (EmotionCategory.PLAYFUL, EmotionCategory.INTENSE): 0.5,
    (EmotionCategory.TENDER, EmotionCategory.COLD): 0.7,
}

ABRUPT_THRESHOLD = 0.7


class EmotionArcTracker:
    """Track emotion states across dialogue turns with a sliding window."""

    def __init__(self, window_size: int = 10):
        self._window: deque[EmotionState] = deque(maxlen=window_size)

    @property
    def history(self) -> list[EmotionState]:
        return list(self._window)

    def update(self, annotation: EmotionAnnotation, text: str) -> None:
        """Record a new emotion state from an annotation."""
        state = EmotionState(
            text=text,
            emotion=annotation.emotion,
            intensity=annotation.intensity,
            ref_emotion_category=annotation.ref_emotion_category,
        )
        self._window.append(state)

    def check_transition(self, proposed: EmotionCategory) -> bool:
        """Return True if the proposed emotion is a natural transition from the last state.

        Returns True (safe) when history is empty or transition distance is below threshold.
        """
        if not self._window:
            return True

        last = self._window[-1].ref_emotion_category
        if last == proposed:
            return True

        pair = (last, proposed)
        reverse_pair = (proposed, last)
        distance = TRANSITION_DISTANCE.get(
            pair, TRANSITION_DISTANCE.get(reverse_pair, 0.4)
        )
        return distance < ABRUPT_THRESHOLD

    def reset(self) -> None:
        """Reset the emotion state (e.g. at a story branch point)."""
        self._window.clear()

    def format_history_for_prompt(self) -> str:
        """Serialize recent emotion history as a string for LLM prompt injection."""
        if not self._window:
            return "（无历史情感记录）"
        entries = [
            f"{s.emotion}({s.intensity:.1f})" for s in self._window
        ]
        return ", ".join(entries)
