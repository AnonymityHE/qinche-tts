"""Evaluation framework for Context-Aware Emotional TTS.

Public API:
    - compute_sim / compute_sim_against_mean: speaker similarity
    - compute_wer / transcribe: ASR-based WER
    - compute_emotion_accuracy: emotion2vec classification match
    - run_evaluation: full three-way comparative evaluation
    - analyze_existing_results: summarize prior eval_from_project/ results
"""

from .sim import compute_sim, compute_sim_against_mean, get_embedding
from .wer import compute_wer, transcribe
from .emotion_acc import compute_emotion_accuracy
from .run_eval import run_evaluation
from .analyze_existing import analyze_existing_results

__all__ = [
    "compute_sim",
    "compute_sim_against_mean",
    "get_embedding",
    "compute_wer",
    "transcribe",
    "compute_emotion_accuracy",
    "run_evaluation",
    "analyze_existing_results",
]
