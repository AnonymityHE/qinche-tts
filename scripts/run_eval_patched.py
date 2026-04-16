"""Wrapper to run eval with torch.load patch for pyannote compatibility."""

import functools
import torch

# Patch torch.load to use weights_only=False (pyannote model uses pytorch_lightning checkpoint)
_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Also disable cuDNN (driver compatibility issue on this machine)
torch.backends.cudnn.enabled = False

from dotenv import load_dotenv
load_dotenv()

import sys
from src.eval.run_eval import run_evaluation

if __name__ == "__main__":
    import click

    conditions = sys.argv[1] if len(sys.argv) > 1 else "auto,clone_xvec,baseline"
    cond_list = [c.strip() for c in conditions.split(",")]

    run_evaluation(
        output_dir="output",
        ref_audio_dir="data/ref_audio",
        test_manifest="data/test_manifest.jsonl",
        conditions=cond_list,
        skip_emotion=True,
        report_path="output/eval_report.json",
    )
