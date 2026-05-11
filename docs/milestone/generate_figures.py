"""Generate Figure 4 for the milestone report.

Produces two PDF panels (emotion-bucket distribution + Phase-1 SIM/WER bar)
saved under docs/milestone/figs/.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

EMOTION_ORDER = ["tender", "calm", "playful", "intense", "cold", "intimate"]
EMOTION_COLORS = {
    "tender":   "#f472b6",
    "calm":     "#60a5fa",
    "playful":  "#34d399",
    "intense":  "#f87171",
    "cold":     "#94a3b8",
    "intimate": "#c084fc",
}


def _panel_a(ax):
    buckets = json.loads((ROOT / "data" / "emotion_buckets.json").read_text())
    counts = Counter(e["emotion"] for e in buckets)
    x = np.arange(len(EMOTION_ORDER))
    vals = [counts.get(k, 0) for k in EMOTION_ORDER]
    colors = [EMOTION_COLORS[k] for k in EMOTION_ORDER]
    bars = ax.bar(x, vals, color=colors, edgecolor="#1f2937", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_ORDER, rotation=0)
    ax.set_ylabel("# curated reference clips")
    ax.set_title("(a) Emotion buckets (curated from 664 training clips)")
    ax.set_ylim(0, max(vals) + 2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v),
                ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_b(ax):
    summary = json.loads((ROOT / "docs" / "eval_summary.json").read_text())

    systems = [
        ("Qwen ZS\n(1.7B base)",       summary["qwen_1.7b_zs"]),
        ("Fish S2 Pro\nZero-shot",     {"avg_sim_gt": 0.6627, "avg_wer": 0.0209}),
        ("Qwen SFT v5 ep3\n(ours)",    summary["ft_v5_fast_checkpoint-epoch-3"]),
    ]

    labels = [lbl for lbl, _ in systems]
    sim = [d.get("avg_sim_gt", 0) for _, d in systems]
    wer = [d.get("avg_wer", 0) for _, d in systems]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, sim, w, label="SIM\\textsubscript{gt}", color="#a855f7")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, wer, w, label="WER", color="#f59e0b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("SIM\\textsubscript{gt} (higher is better)")
    ax2.set_ylabel("WER (lower is better)")
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, max(wer) * 1.6 + 1e-3)
    ax.set_title("(b) Phase-1 foundation: timbre solved, WER low")
    for b, v in zip(b1, sim):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, color="#6b21a8")
    for b, v in zip(b2, wer):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7, color="#b45309")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax2.legend(loc="upper right", frameon=False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2))
    _panel_a(axes[0])
    _panel_b(axes[1])
    fig.tight_layout()
    out = FIGS / "emotion_and_phase1.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
