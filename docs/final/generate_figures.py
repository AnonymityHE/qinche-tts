#!/usr/bin/env python3
"""
Final-report figure generator — Morandi color scheme.

Reuses the palette and rcParams from `docs/report/generate_figures.py` so
that NEW Phase-2 figures match the look of the Phase-1 figures we re-use
in `final.tex`.

Inputs (from the eval-a100 branch, pre-pulled to /tmp by the build step):
    /tmp/pipeline_timed.json   - per-component pipeline timing
    /tmp/benchmark.json        - native vs cuda-graph vs clone_xvec
    /tmp/eval_report.json      - three-condition SIM_ref + WER

Outputs (PDF, written under docs/final/figs/):
    emotion_and_phase1.pdf     - 6 emotion buckets + Phase-1 head-to-head
    pipeline_timing.pdf        - per-line latency stack (LLM/RAG/TTS)
    a100_benchmark.pdf         - horizontal bar of inference RTF on A100
    a100_three_condition.pdf   - SIM_ref + WER twin-axis bar
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Morandi palette (identical to docs/report) ────────────────────────────
MORANDI = {
    "dusty_blue":    "#9db4c0",
    "sage_green":    "#a8b5a0",
    "dusty_rose":    "#c4a5a0",
    "muted_yellow":  "#d4c5a9",
    "lavender_gray": "#b5a8c0",
    "soft_coral":    "#c9a99e",
    "olive_green":   "#a5a894",
    "dusty_purple":  "#a89db5",
    "warm_beige":    "#c7b8a8",
    "muted_teal":    "#8fb0a8",
    "soft_pink":     "#c2a8a8",
    "warm_gray":     "#b5a99c",
    "background":    "#f7f6f3",
    "axes_bg":       "#faf9f7",
    "dark_text":     "#5a5550",
    "medium_text":   "#7a7066",
    "light_text":    "#9a8f82",
    "accent_red":    "#b8857a",
    "accent_blue":   "#7a9bb8",
    "grid":          "#e8e6e1",
    "border":        "#d0c8b8",
}

# ── Global rcParams ───────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor":   MORANDI["background"],
    "axes.facecolor":     MORANDI["axes_bg"],
    "font.size":          11,
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     1.0,
    "grid.alpha":         0.3,
    "grid.color":         MORANDI["grid"],
    "text.color":         MORANDI["dark_text"],
    "axes.labelcolor":    MORANDI["medium_text"],
    "xtick.color":        MORANDI["medium_text"],
    "ytick.color":        MORANDI["medium_text"],
    "savefig.dpi":        200,
    "savefig.facecolor":  MORANDI["background"],
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.2,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})


# ── Helpers ───────────────────────────────────────────────────────────────
def _save(fig, name):
    fig.savefig(FIGS / name, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {name}")


def _title(fig, main, sub=None, y_main=0.97, y_sub=None):
    fig.suptitle(main, fontsize=15, fontweight="400", y=y_main,
                 color=MORANDI["dark_text"])
    if sub:
        if y_sub is None:
            y_sub = y_main - 0.05
        fig.text(0.5, y_sub, sub, ha="center", fontsize=10,
                 color=MORANDI["medium_text"], style="italic")


def _legend(ax, **kw):
    defaults = dict(fontsize=9, framealpha=0.95, facecolor="white",
                    edgecolor=MORANDI["light_text"])
    defaults.update(kw)
    ax.legend(**defaults)


# ════════════════════════════════════════════════════════════
# Fig 1 — Emotion buckets (left) + Phase-1 head-to-head (right)
#         (replaces the old milestone "emotion_and_phase1.pdf")
# ════════════════════════════════════════════════════════════
EMOTION_ORDER = ["tender", "calm", "playful", "intense", "cold", "intimate"]
EMOTION_COLORS = {
    "tender":   MORANDI["dusty_rose"],
    "calm":     MORANDI["dusty_blue"],
    "playful":  MORANDI["sage_green"],
    "intense":  MORANDI["accent_red"],
    "cold":     MORANDI["lavender_gray"],
    "intimate": MORANDI["dusty_purple"],
}


def fig_emotion_and_phase1():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))

    # ---- Panel (a): emotion bucket counts ---------------------------------
    ax = axes[0]
    buckets = json.loads((ROOT / "data" / "emotion_buckets.json").read_text())
    counts = Counter(e["emotion"] for e in buckets)
    x = np.arange(len(EMOTION_ORDER))
    vals = [counts.get(k, 0) for k in EMOTION_ORDER]
    cols = [EMOTION_COLORS[k] for k in EMOTION_ORDER]
    bars = ax.bar(x, vals, color=cols, edgecolor="white", linewidth=2,
                  alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.18, str(v),
                ha="center", va="bottom", fontsize=9,
                color=MORANDI["dark_text"], fontweight="500")
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_ORDER, rotation=0, fontsize=9)
    ax.set_ylabel("# curated reference clips")
    ax.set_title("(a) Emotion buckets (curated from 664 training clips)",
                 fontsize=11, color=MORANDI["dark_text"], pad=8)
    ax.set_ylim(0, max(vals) + 2)
    ax.grid(axis="y", alpha=0.3, color=MORANDI["grid"])

    # ---- Panel (b): Phase-1 head-to-head ----------------------------------
    ax = axes[1]
    summary = json.loads((ROOT / "docs" / "eval_summary.json").read_text())
    systems = [
        ("Qwen ZS\n(1.7B base)",       summary["qwen_1.7b_zs"]),
        ("Fish S2 Pro\nzero-shot",     {"avg_sim_gt": 0.6627, "avg_wer": 0.0209}),
        ("Qwen SFT v5 ep3\n(ours)",    summary["ft_v5_fast_checkpoint-epoch-3"]),
    ]
    labels = [lbl for lbl, _ in systems]
    sim = [d["avg_sim_gt"] for _, d in systems]
    wer = [d["avg_wer"]    for _, d in systems]

    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, sim, w, color=MORANDI["dusty_blue"],
                edgecolor="white", lw=2, alpha=0.9, label="SIM_gt")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, wer, w, color=MORANDI["soft_coral"],
                 edgecolor="white", lw=2, alpha=0.9, label="WER")
    ax2.grid(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for b, v in zip(b1, sim):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
                color=MORANDI["dark_text"])
    for b, v in zip(b2, wer):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8,
                 color=MORANDI["accent_red"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("SIM_gt (higher is better)",
                  color=MORANDI["dusty_blue"])
    ax2.set_ylabel("WER (lower is better)",
                   color=MORANDI["accent_red"])
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, max(wer) * 1.7 + 1e-3)
    ax.set_title("(b) Phase-1 foundation: timbre solved, WER low",
                 fontsize=11, color=MORANDI["dark_text"], pad=8)
    ax.tick_params(axis="y", labelcolor=MORANDI["dusty_blue"])
    ax2.tick_params(axis="y", labelcolor=MORANDI["accent_red"])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "emotion_and_phase1.pdf")


# ════════════════════════════════════════════════════════════
# Fig 2 — Per-line latency stack (LLM / RAG / TTS)
# ════════════════════════════════════════════════════════════
def fig_pipeline_timing():
    data = json.loads(Path("/tmp/pipeline_timed.json").read_text())
    s = data["timing_summary"]
    runs = data["pipeline_results"]

    llm = s["emotion_analysis_mean_s"]
    rag = s["rag_retrieval_mean_s"]
    auto_warm = float(np.mean([r["timing"]["auto"] for r in runs[1:]]))
    clone     = s["clone_xvec_mean_s"]
    base_warm = float(np.mean([r["timing"]["baseline"] for r in runs[1:]]))

    backends = ["auto\n(router)", "clone_xvec", "baseline"]
    tts_means = [auto_warm, clone, base_warm]
    llm_arr = [llm] * 3
    rag_arr = [rag] * 3
    totals = [a + b + c for a, b, c in zip(tts_means, llm_arr, rag_arr)]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(backends))
    w = 0.55
    ax.bar(x, llm_arr, w, color=MORANDI["lavender_gray"],
           edgecolor="white", lw=2, alpha=0.9,
           label="LLM Emotion (GPT-4o)")
    ax.bar(x, rag_arr, w, bottom=llm_arr, color=MORANDI["sage_green"],
           edgecolor="white", lw=2, alpha=0.9,
           label="Character RAG (bge + Chroma)")
    ax.bar(x, tts_means, w,
           bottom=[a + b for a, b in zip(llm_arr, rag_arr)],
           color=MORANDI["soft_coral"], edgecolor="white", lw=2, alpha=0.9,
           label="TTS synthesis")

    for xi, t in zip(x, totals):
        ax.text(xi, t + 0.4, f"{t:.1f}s", ha="center", fontsize=9,
                color=MORANDI["dark_text"], fontweight="500")

    # share annotation
    share_txt = (
        f"LLM   {llm:.1f}s  ({llm/totals[0]*100:.0f}%)\n"
        f"RAG   {rag:.1f}s  ({rag/totals[0]*100:.0f}%)\n"
        f"TTS  ~{tts_means[0]:.1f}s  ({tts_means[0]/totals[0]*100:.0f}%)"
    )
    ax.text(0.02, 0.97, share_txt, transform=ax.transAxes, fontsize=8.5,
            color=MORANDI["dark_text"], va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white",
                      edgecolor=MORANDI["light_text"], linewidth=0.8))

    ax.set_xticks(x); ax.set_xticklabels(backends, fontsize=10)
    ax.set_ylabel("Mean per-sentence latency (s)")
    ax.set_ylim(0, max(totals) * 1.28)
    _legend(ax, loc="upper right", ncol=1)
    _title(fig, "End-to-end pipeline: LLM dominates, TTS is a small share",
           y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "pipeline_timing.pdf")


# ════════════════════════════════════════════════════════════
# Fig 3 — A100 inference benchmark (horizontal bar)
# ════════════════════════════════════════════════════════════
def fig_a100_benchmark():
    rows = json.loads(Path("/tmp/benchmark.json").read_text())
    rtf = {r["method"]: r["mean_rtf"] for r in rows}
    times = {r["method"]: float(np.mean(r["times"])) for r in rows}

    cfgs = [
        ("clone_xvec\n(Base + native)",       rtf["clone_xvec"],     MORANDI["warm_gray"]),
        ("Native SFT\n(SDPA)",                rtf["native_sft"],     MORANDI["dusty_purple"]),
        ("CUDA Graph SFT\n(FasterQwen3TTS)",  rtf["cuda_graph_sft"], MORANDI["sage_green"]),
    ]
    names, rtfs, cols = zip(*cfgs)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bars = ax.barh(range(len(names)), rtfs, color=cols, edgecolor="white",
                   height=0.6, linewidth=2, alpha=0.9)

    for b, r, n in zip(bars, rtfs, names):
        method_key = n.split("\n")[0].lower().replace(" ", "_")
        # find time by best match
        if "cuda graph" in n.lower():
            t = times["cuda_graph_sft"]
        elif "native" in n.lower():
            t = times["native_sft"]
        else:
            t = times["clone_xvec"]
        ax.text(b.get_width() + 0.04, b.get_y() + b.get_height() / 2,
                f"RTF={r:.3f}   ({t:.1f}s)",
                va="center", fontsize=9, fontweight="500",
                color=MORANDI["dark_text"])

    ax.axvline(1.0, color=MORANDI["accent_red"], ls="--", lw=1.5,
               label="Real-time (RTF=1.0)")

    speedup = rtf["native_sft"] / rtf["cuda_graph_sft"]
    ax.text(0.99, 0.96, f"x{speedup:.1f} faster\n(CUDA Graph vs Native)",
            transform=ax.transAxes, fontsize=9, color=MORANDI["sage_green"],
            ha="right", va="top", fontweight="500",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=MORANDI["sage_green"], alpha=0.9))

    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("RTF (Real-Time Factor, lower is better)")
    ax.set_xlim(0, max(rtfs) * 1.35)
    _legend(ax, loc="lower right")
    _title(fig, "A100 inference benchmark (4 sentences each)", y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "a100_benchmark.pdf")


# ════════════════════════════════════════════════════════════
# Fig 4 — Three-condition end-to-end SIM_ref + WER
# ════════════════════════════════════════════════════════════
def fig_a100_three_condition():
    data = json.loads(Path("/tmp/eval_report.json").read_text())
    summ = data["summary"]
    order = ["auto", "clone_xvec", "baseline"]
    label_map = {
        "auto":       "auto\n(router → blend)",
        "clone_xvec": "clone_xvec\n(ICL fallback)",
        "baseline":   "baseline\n(SFT, no emotion)",
    }
    sim = [summ[k]["sim_ref_mean"] for k in order]
    wer = [summ[k]["wer_mean"]    for k in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    x = np.arange(len(order))
    w = 0.36
    b1 = ax.bar(x - w / 2, sim, w, color=MORANDI["dusty_blue"],
                edgecolor="white", lw=2, alpha=0.9, label="SIM_ref")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, wer, w, color=MORANDI["soft_coral"],
                 edgecolor="white", lw=2, alpha=0.9, label="WER")
    ax2.grid(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    for b, v in zip(b1, sim):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.013, f"{v:.3f}",
                ha="center", fontsize=8.5, color=MORANDI["dark_text"])
    for b, v in zip(b2, wer):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                 ha="center", fontsize=8.5, color=MORANDI["accent_red"])

    # mark baseline as the timbre upper-bound
    ax.annotate("timbre\nupper bound",
                xy=(2 - w / 2, sim[2]), xytext=(1.55, 0.85),
                fontsize=8, ha="center", color=MORANDI["medium_text"],
                arrowprops=dict(arrowstyle="->", color=MORANDI["light_text"],
                                lw=0.8))
    # mark the cost of injecting emotion
    drop = sim[2] - sim[0]
    ax.annotate(f"emotion costs\n~{drop*100:.1f}% SIM",
                xy=(0 - w / 2, sim[0]), xytext=(0.15, 0.92),
                fontsize=8, ha="center", color=MORANDI["accent_red"],
                arrowprops=dict(arrowstyle="->", color=MORANDI["accent_red"],
                                lw=0.8, alpha=0.6))

    ax.set_xticks(x); ax.set_xticklabels([label_map[k] for k in order], fontsize=9)
    ax.set_ylabel("SIM_ref (higher is better)",
                  color=MORANDI["dusty_blue"])
    ax2.set_ylabel("WER (lower is better; ASR=whisper-base)",
                   color=MORANDI["accent_red"])
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, max(wer) * 1.7)
    ax.tick_params(axis="y", labelcolor=MORANDI["dusty_blue"])
    ax2.tick_params(axis="y", labelcolor=MORANDI["accent_red"])
    _legend(ax, loc="upper left")
    _legend(ax2, loc="upper right")
    _title(fig, "End-to-end three-condition evaluation on A100", y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "a100_three_condition.pdf")


# ════════════════════════════════════════════════════════════
def main():
    print("Generating Morandi-style final-report figures …")
    fig_emotion_and_phase1()
    fig_pipeline_timing()
    fig_a100_benchmark()
    fig_a100_three_condition()
    print(f"\nAll 4 figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
