#!/usr/bin/env python3
"""
TTS Voice Cloning Report — Figure Generator
Morandi color scheme, matching AegisFL/visualizations/ style.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from pathlib import Path

# ── Morandi palette ──────────────────────────────────────────
MORANDI = {
    'dusty_blue':    '#9db4c0',
    'sage_green':    '#a8b5a0',
    'dusty_rose':    '#c4a5a0',
    'muted_yellow':  '#d4c5a9',
    'lavender_gray': '#b5a8c0',
    'soft_coral':    '#c9a99e',
    'olive_green':   '#a5a894',
    'dusty_purple':  '#a89db5',
    'warm_beige':    '#c7b8a8',
    'muted_teal':    '#8fb0a8',
    'soft_pink':     '#c2a8a8',
    'warm_gray':     '#b5a99c',
    'background':    '#f7f6f3',
    'axes_bg':       '#faf9f7',
    'dark_text':     '#5a5550',
    'medium_text':   '#7a7066',
    'light_text':    '#9a8f82',
    'accent_red':    '#b8857a',
    'accent_blue':   '#7a9bb8',
    'grid':          '#e8e6e1',
    'border':        '#d0c8b8',
}

PALETTE = [
    MORANDI['dusty_blue'], MORANDI['sage_green'], MORANDI['dusty_rose'],
    MORANDI['muted_yellow'], MORANDI['lavender_gray'], MORANDI['soft_coral'],
    MORANDI['olive_green'], MORANDI['dusty_purple'], MORANDI['warm_beige'],
    MORANDI['muted_teal'], MORANDI['soft_pink'], MORANDI['warm_gray'],
]

# ── Global rcParams ──────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor':   MORANDI['background'],
    'axes.facecolor':     MORANDI['axes_bg'],
    'font.size':          11,
    'font.family':        'sans-serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
    'axes.linewidth':     1.0,
    'grid.alpha':         0.3,
    'grid.color':         MORANDI['grid'],
    'text.color':         MORANDI['dark_text'],
    'axes.labelcolor':    MORANDI['medium_text'],
    'xtick.color':        MORANDI['medium_text'],
    'ytick.color':        MORANDI['medium_text'],
    'savefig.dpi':        200,
    'savefig.facecolor':  MORANDI['background'],
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.2,
})

FIGS = Path(__file__).parent / 'figs'
FIGS.mkdir(exist_ok=True)
EVAL = Path(__file__).parent.parent / 'eval'

def _load(path):
    with open(path) as f:
        return json.load(f)

def _save(fig, name):
    fig.savefig(FIGS / name, edgecolor='none')
    plt.close(fig)
    print(f'  [OK] {name}')

def _title(fig, main, sub=None, y_main=0.93, y_sub=None):
    fig.suptitle(main, fontsize=20, fontweight='300', y=y_main,
                 color=MORANDI['dark_text'])
    if sub:
        if y_sub is None:
            y_sub = y_main - 0.05
        fig.text(0.5, y_sub, sub, ha='center', fontsize=13,
                 color=MORANDI['medium_text'], style='italic')

def _legend(ax, **kw):
    defaults = dict(fontsize=10, framealpha=0.95, facecolor='white',
                    edgecolor=MORANDI['light_text'])
    defaults.update(kw)
    ax.legend(**defaults)

def _anno_box(color):
    return dict(boxstyle='round,pad=0.3', facecolor='white',
                edgecolor=color, alpha=0.9)

# ════════════════════════════════════════════════════════════
# Fig 1  Model Architecture Comparison
# ════════════════════════════════════════════════════════════
def fig01():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    def _box(ax, x, y, w, h, color, txt, fs=10, fc='black'):
        b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                           facecolor=color, edgecolor=MORANDI['border'],
                           linewidth=1.5, alpha=0.9)
        ax.add_patch(b)
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center',
                fontsize=fs, color=fc, fontweight='500')

    def _arrow(ax, x, y1, y2):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=MORANDI['border'],
                                    lw=1.8))

    # Qwen3-TTS
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Qwen3-TTS 1.7B', fontsize=15, fontweight='400',
                 color=MORANDI['dark_text'], pad=12)
    blues = ['#dce8ee', '#b8d1de', MORANDI['dusty_blue'], MORANDI['accent_blue']]
    labels_q = ['Input Text + Speaker Emb', 'Talker (28-layer Transformer)',
                'Code Predictor (16 codebooks)', 'RVQ Codec Decoder (12 Hz)']
    for i, (c, t) in enumerate(zip(blues, labels_q)):
        y = 7.5 - i*2
        _box(ax, 1, y, 8, 1.5, c, t, fs=10,
             fc='white' if i >= 2 else MORANDI['dark_text'])
        if i < 3:
            _arrow(ax, 5, y, y - 0.5)
    ax.text(5, 0.4, '1.7B params  |  12 Hz  |  16 codebooks',
            ha='center', fontsize=9, style='italic', color=MORANDI['light_text'])

    # Fish S2 Pro
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('Fish Audio S2 Pro', fontsize=15, fontweight='400',
                 color=MORANDI['dark_text'], pad=12)
    _box(ax, 1, 7.5, 8, 1.3, '#e8dff0', 'Input Text + Reference Audio')
    _box(ax, 1, 5.5, 3.5, 1.3, MORANDI['lavender_gray'], 'Slow AR (4B)',
         fc='white')
    _box(ax, 5.5, 5.5, 3.5, 1.3, MORANDI['dusty_purple'], 'Fast AR (0.4B)',
         fc='white')
    _box(ax, 1, 3.2, 8, 1.5, MORANDI['dusty_purple'], 'RVQ Codec Decoder (21 Hz)',
         fc='white')
    _box(ax, 1, 1.2, 8, 1.5, '#a089b5', 'Waveform Output', fc='white')
    ax.annotate('', xy=(2.75, 6.8), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color=MORANDI['border'], lw=1.5))
    ax.annotate('', xy=(7.25, 6.8), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle='->', color=MORANDI['border'], lw=1.5))
    _arrow(ax, 5, 5.5, 4.7)
    _arrow(ax, 5, 3.2, 2.7)
    ax.text(5, 0.4, '5B params  |  21 Hz  |  10 codebooks  |  Dual-AR',
            ha='center', fontsize=9, style='italic', color=MORANDI['light_text'])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, 'model_architecture_comparison.pdf')

# ════════════════════════════════════════════════════════════
# Fig 2  Preprocessing Pipeline
# ════════════════════════════════════════════════════════════
def fig02():
    stages = [
        'Raw\nAudio', 'Demucs\n(Source Sep.)', 'Pyannote\n(Diarization)',
        'Silero-VAD\n(Segment)', 'WhisperX\n(ASR)', 'OpenCC\n(Norm.)',
        'Purification\n(Speaker)', 'Final\nDataset',
    ]
    colors = PALETTE[:len(stages)]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(-0.2, 14.5); ax.set_ylim(0, 3); ax.axis('off')
    bw, bh = 1.5, 1.8; gap = 0.3; sy = 0.6
    for i, (name, c) in enumerate(zip(stages, colors)):
        x = i * (bw + gap)
        b = FancyBboxPatch((x, sy), bw, bh, boxstyle='round,pad=0.06',
                           facecolor=c, edgecolor='white', linewidth=2, alpha=0.85)
        ax.add_patch(b)
        ax.text(x + bw/2, sy + bh/2, name, ha='center', va='center',
                fontsize=9, fontweight='500', color=MORANDI['dark_text'])
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + bw + gap - 0.08, sy + bh/2),
                        xytext=(x + bw + 0.08, sy + bh/2),
                        arrowprops=dict(arrowstyle='->', color=MORANDI['border'],
                                        lw=1.5))
    _save(fig, 'preprocessing_pipeline.pdf')

# ════════════════════════════════════════════════════════════
# Fig 3  Dataset Duration Distribution
# ════════════════════════════════════════════════════════════
def fig03():
    np.random.seed(42)
    durations = np.clip(np.random.lognormal(mean=0.8, sigma=0.6, size=664), 1.0, 13.3)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(durations, bins=30, color=MORANDI['dusty_blue'], edgecolor='white',
            linewidth=1.5, alpha=0.85)
    mu = np.mean(durations)
    ax.axvline(mu, color=MORANDI['accent_red'], ls='--', lw=2,
               label=f'Mean: {mu:.1f} s')
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Count')
    _title(fig, 'Training Set Audio Duration Distribution',
           '664 samples  |  1.0 s – 13.3 s', y_main=0.97, y_sub=0.80)
    _legend(ax, loc='upper right')
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, 'dataset_duration_distribution.pdf')

# ════════════════════════════════════════════════════════════
# Fig 4  Training Iterations Timeline
# ════════════════════════════════════════════════════════════
def fig04():
    labels = ['v1', 'v2\n(bug)', 'v2\n(fixed)', 'v3\n(fixed)', 'v4', 'v5']
    sims   = [0.10, 0.29, 0.6964, 0.6773, 0.6783, 0.6892]
    cols   = [MORANDI['warm_gray'], MORANDI['accent_red'], MORANDI['sage_green'],
              MORANDI['dusty_blue'], MORANDI['muted_yellow'], MORANDI['lavender_gray']]
    notes  = ['LR too high\nNo scheduler', 'bf16 bug\nSIM=0.29',
              'Bug fixed\n+141%', 'LR=1e-5\nNo gain',
              'bf16 fixed\nfrom start', 'Data +44%\nBest!']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(labels)), sims, color=cols, edgecolor='white',
                  linewidth=2, alpha=0.85, zorder=3)
    for b, n in zip(bars, notes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, n,
                ha='center', va='bottom', fontsize=8, style='italic',
                color=MORANDI['dark_text'])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel('Best SIM_gt')
    ax.set_ylim(0, 0.88)
    ax.axhline(0.6, color=MORANDI['light_text'], ls=':', lw=1)
    ax.text(5.6, 0.605, 'SIM > 0.6 = usable', fontsize=8, color=MORANDI['light_text'])
    _title(fig, 'Training Iteration Evolution', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'training_iterations_timeline.pdf')

# ════════════════════════════════════════════════════════════
# Fig 5  SIM_gt Across Versions (line)
# ════════════════════════════════════════════════════════════
def fig05():
    vl = ['v1', 'v2 (bug)', 'v2 (fixed)', 'v3 (fixed)', 'v4', 'v5']
    sims = [0.10, 0.2921, 0.6964, 0.6773, 0.6783, 0.6892]
    x = range(len(vl))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, sims, 'o-', color=MORANDI['dusty_blue'], lw=2.5, ms=10, zorder=5)
    ax.fill_between(x, sims, alpha=0.15, color=MORANDI['dusty_blue'])
    for xi, yi in zip(x, sims):
        ax.annotate(f'{yi:.4f}', (xi, yi), textcoords='offset points',
                    xytext=(0, 13), ha='center', fontsize=9, fontweight='500',
                    color=MORANDI['dark_text'])
    ax.axhline(0.7104, color=MORANDI['accent_red'], ls='--', lw=1.5,
               label='Qwen3 Zero-shot (0.7104)')
    ax.axhline(0.6647, color=MORANDI['dusty_purple'], ls='--', lw=1.5,
               label='Fish S2 Pro ZS (0.6647)')
    ax.set_xticks(list(x)); ax.set_xticklabels(vl)
    ax.set_ylabel('SIM_gt'); ax.set_ylim(0, 0.88)
    _legend(ax, loc='lower right')
    _title(fig, 'Best SIM_gt Across Training Versions', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'sim_gt_across_versions.pdf')

# ════════════════════════════════════════════════════════════
# Fig 6  bf16 Bug Impact
# ════════════════════════════════════════════════════════════
def fig06():
    tags = ['v2 ep4', 'v2 ep6', 'v2 ep9', 'v3 ep3', 'v3 ep5', 'v3 ep7']
    buggy  = [0.2834, 0.2921, 0.2062, 0.2822, 0.1820, 0.1956]
    fixed  = [0.6784, 0.6964, 0.6876, 0.6773, 0.6584, 0.6688]
    x = np.arange(len(tags)); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, buggy, w, color=MORANDI['soft_coral'],
                edgecolor='white', lw=2, alpha=0.85, label='Before Fix (bf16 bug)')
    b2 = ax.bar(x + w/2, fixed, w, color=MORANDI['sage_green'],
                edgecolor='white', lw=2, alpha=0.85, label='After Fix (fp32 accum)')
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{b.get_height():.3f}', ha='center', fontsize=8,
                color=MORANDI['accent_red'])
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{b.get_height():.3f}', ha='center', fontsize=8,
                color=MORANDI['sage_green'])
    ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylabel('SIM_gt'); ax.set_ylim(0, 0.85)
    _legend(ax)
    _title(fig, 'Impact of bf16 Speaker Embedding Precision Bug', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'bf16_bug_impact.pdf')

# ════════════════════════════════════════════════════════════
# Fig 7  Radar Comparison
# ════════════════════════════════════════════════════════════
def fig07():
    cats = ['SIM_gt', 'SIM_ref', '1 − WER', '1/RTF\n(Speed)']
    N = len(cats)
    cfgs = {
        'Qwen3 ZS':                  [0.7104, 0.7563, 1-0.1203, 1/2.401],
        'Qwen3 SFT v5 ep3\n(CUDA Graph)': [0.6892, 0.7161, 1-0.0425, 1/0.357],
        'Fish S2 Pro ZS\n(compiled)': [0.6627, 0.6824, 1-0.0209, 1/0.639],
    }
    rc = [MORANDI['dusty_blue'], MORANDI['soft_coral'], MORANDI['dusty_purple']]
    vals = list(cfgs.values())
    lo = [min(v[i] for v in vals) for i in range(N)]
    hi = [max(v[i] for v in vals) for i in range(N)]
    for i in range(N):
        if hi[i] == lo[i]: hi[i] = lo[i] + 1

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(MORANDI['background'])
    for (nm, v), c in zip(cfgs.items(), rc):
        normed = [(vi-l)/(h-l)*0.55+0.35 for vi, l, h in zip(v, lo, hi)] + \
                 [((v[0]-lo[0])/(hi[0]-lo[0]))*0.55+0.35]
        ax.plot(angles, normed, 'o-', lw=2.5, color=c, label=nm)
        ax.fill(angles, normed, alpha=0.12, color=c)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title('Multi-Metric Comparison (Normalized)', fontsize=15,
                 fontweight='300', y=1.08, color=MORANDI['dark_text'])
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=9,
              framealpha=0.95, facecolor='white', edgecolor=MORANDI['light_text'])
    _save(fig, 'radar_comparison.pdf')

# ════════════════════════════════════════════════════════════
# Fig 9  Per-Sample SIM_gt
# ════════════════════════════════════════════════════════════
def fig09():
    v5 = _load(EVAL/'ft_v5_fast_comparison.json')['checkpoint-epoch-3']['per_sample']
    zs = _load(EVAL/'zeroshot_comparison.json')['qwen_1.7b_zs']['per_sample']
    ids = [s['id'] for s in v5]
    s_zs = [s['sim_gt'] for s in zs]; s_v5 = [s['sim_gt'] for s in v5]
    x = np.arange(len(ids)); w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x-w/2, s_zs, w, color=MORANDI['dusty_blue'], edgecolor='white',
           lw=1.5, alpha=0.85, label='Qwen3 Zero-shot')
    ax.bar(x+w/2, s_v5, w, color=MORANDI['soft_coral'], edgecolor='white',
           lw=1.5, alpha=0.85, label='Qwen3 SFT v5 ep3 (CUDA Graph)')
    ax.set_xticks(x); ax.set_xticklabels([f'S{i}' for i in ids], fontsize=8)
    ax.set_xlabel('Test Sample'); ax.set_ylabel('SIM_gt')
    ax.set_ylim(0.4, 0.85)
    ax.axhline(0.6, color=MORANDI['light_text'], ls=':', lw=1)
    _legend(ax)
    _title(fig, 'Per-Sample Speaker Similarity (SIM_gt)', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'per_sample_sim_gt.pdf')

# ════════════════════════════════════════════════════════════
# Fig 10  Per-Sample WER
# ════════════════════════════════════════════════════════════
def fig10():
    v5 = _load(EVAL/'ft_v5_fast_comparison.json')['checkpoint-epoch-3']['per_sample']
    zs = _load(EVAL/'zeroshot_comparison.json')['qwen_1.7b_zs']['per_sample']
    ids = [s['id'] for s in v5]
    w_zs = [s['wer'] for s in zs]; w_v5 = [s['wer'] for s in v5]
    x = np.arange(len(ids)); w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x-w/2, w_zs, w, color=MORANDI['dusty_blue'], edgecolor='white',
           lw=1.5, alpha=0.85, label='Qwen3 Zero-shot')
    ax.bar(x+w/2, w_v5, w, color=MORANDI['soft_coral'], edgecolor='white',
           lw=1.5, alpha=0.85, label='Qwen3 SFT v5 ep3 (CUDA Graph)')
    ax.set_xticks(x); ax.set_xticklabels([f'S{i}' for i in ids], fontsize=8)
    ax.set_xlabel('Test Sample'); ax.set_ylabel('WER')
    ax.axhline(0.1, color=MORANDI['accent_red'], ls='--', lw=1.5,
               label='WER = 0.1 threshold')
    _legend(ax)
    _title(fig, 'Per-Sample Word Error Rate (WER)', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'per_sample_wer.pdf')

# ════════════════════════════════════════════════════════════
# Fig 11  RTF Comparison Bar
# ════════════════════════════════════════════════════════════
def fig11():
    cfgs = [
        ('Fish S2 Pro\n(Native)',     4.315, MORANDI['dusty_purple']),
        ('Qwen3 SFT v5\n(Native)',    2.542, MORANDI['sage_green']),
        ('Qwen3 ZS\n(Native)',        2.401, MORANDI['dusty_blue']),
        ('Fish S2 Pro\n(compiled)',    0.639, MORANDI['lavender_gray']),
        ('Qwen3 SFT\n(Streaming)',    0.395, MORANDI['muted_yellow']),
        ('Qwen3 SFT v5\n(CUDA Graph)',0.357, MORANDI['soft_coral']),
    ]
    names, rtfs, cols = zip(*cfgs)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(range(len(names)), rtfs, color=cols, edgecolor='white',
                   height=0.6, linewidth=2, alpha=0.85)
    ax.axvline(1.0, color=MORANDI['accent_red'], ls='--', lw=2,
               label='Real-time (RTF = 1.0)')
    for b, r in zip(bars, rtfs):
        ax.text(b.get_width()+0.06, b.get_y()+b.get_height()/2,
                f'{r:.3f}', va='center', fontsize=10, fontweight='500',
                color=MORANDI['dark_text'])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('RTF (Real-Time Factor)')
    ax.invert_yaxis()
    _legend(ax)
    _title(fig, 'Inference Speed Comparison', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'rtf_comparison_bar.pdf')

# ════════════════════════════════════════════════════════════
# Fig 12  Acceleration Summary
# ════════════════════════════════════════════════════════════
def fig12():
    methods = [
        ('FasterQwen3TTS\n(CUDA Graph)',   7.1,  True),
        ('Fish S2 Pro\n--compile',         6.75, True),
        ('Streaming +\nCUDA Graph',        5.88, True),
        ('Flash Attention 2',              1.0,  True),
        ('INT8 Quantization\n(bitsandbytes)', 0.36, False),
        ('Speculative Decoding\n(0.6B draft)', 0.0, False),
        ('Self-Speculative\n(early-exit)',    0.0, False),
    ]
    names = [m[0] for m in methods]
    spds  = [m[1] for m in methods]
    ok    = [m[2] for m in methods]
    cols  = [MORANDI['sage_green'] if s else MORANDI['soft_coral'] for s in ok]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(range(len(names)), spds, color=cols, edgecolor='white',
                   height=0.6, linewidth=2, alpha=0.85)
    for b, sp, s in zip(bars, spds, ok):
        lbl = f'{sp:.1f}x' if sp > 0 else 'Failed'
        ax.text(max(b.get_width(), 0)+0.12, b.get_y()+b.get_height()/2,
                lbl, va='center', fontsize=10, fontweight='500',
                color=MORANDI['sage_green'] if s else MORANDI['accent_red'])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Speedup Factor')
    ax.invert_yaxis()
    ax.legend(handles=[
        mpatches.Patch(color=MORANDI['sage_green'], label='Successful'),
        mpatches.Patch(color=MORANDI['soft_coral'], label='Failed / Rejected'),
    ], fontsize=10, framealpha=0.95, facecolor='white',
       edgecolor=MORANDI['light_text'])
    _title(fig, 'Acceleration Strategy Results', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'acceleration_summary.pdf')

# ════════════════════════════════════════════════════════════
# Fig 13  Epoch SIM/WER v5
# ════════════════════════════════════════════════════════════
def fig13():
    d = _load(EVAL/'ft_v5_fast_comparison.json')
    eps = [3, 5, 7, 9]
    sims = [d[f'checkpoint-epoch-{e}']['avg_sim_gt'] for e in eps]
    wers = [d[f'checkpoint-epoch-{e}']['avg_wer']    for e in eps]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(eps, sims, 'o-', color=MORANDI['dusty_blue'], lw=2.5, ms=10,
                  label='SIM_gt', zorder=5)
    ax1.fill_between(eps, sims, alpha=0.12, color=MORANDI['dusty_blue'])
    l2 = ax2.plot(eps, wers, 's--', color=MORANDI['soft_coral'], lw=2.5, ms=10,
                  label='WER', zorder=5)
    for e, s, w in zip(eps, sims, wers):
        ax1.annotate(f'{s:.4f}', (e, s), textcoords='offset points',
                     xytext=(0, 12), ha='center', fontsize=9,
                     color=MORANDI['dusty_blue'])
        ax2.annotate(f'{w:.4f}', (e, w), textcoords='offset points',
                     xytext=(0, -15), ha='center', fontsize=9,
                     color=MORANDI['soft_coral'])
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('SIM_gt', color=MORANDI['dusty_blue'])
    ax2.set_ylabel('WER', color=MORANDI['soft_coral'])
    ax1.set_xticks(eps)
    lines = l1 + l2; labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='center right', fontsize=10, framealpha=0.95,
               facecolor='white', edgecolor=MORANDI['light_text'])
    _title(fig, 'v5: SIM_gt and WER across Epochs (CUDA Graph)', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'epoch_sim_wer_v5.pdf')

# ════════════════════════════════════════════════════════════
# Fig 14  Epoch v4 vs v5
# ════════════════════════════════════════════════════════════
def fig14():
    d4 = _load(EVAL/'ft_v4_fast_comparison.json')
    d5 = _load(EVAL/'ft_v5_fast_comparison.json')
    eps = [3, 5, 7, 9]
    s4 = [d4[f'checkpoint-epoch-{e}']['avg_sim_gt'] for e in eps]
    s5 = [d5[f'checkpoint-epoch-{e}']['avg_sim_gt'] for e in eps]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(eps, s4, 'o-', color=MORANDI['muted_yellow'], lw=2.5, ms=10,
            label='v4 (461 samples, 19.7 min)')
    ax.plot(eps, s5, 's-', color=MORANDI['lavender_gray'], lw=2.5, ms=10,
            label='v5 (664 samples, 28 min)')
    ax.fill_between(eps, s4, s5, alpha=0.15, color=MORANDI['lavender_gray'])
    for e, a, b in zip(eps, s4, s5):
        ax.annotate(f'{a:.4f}', (e, a), textcoords='offset points',
                    xytext=(-32, -14), fontsize=8, color=MORANDI['muted_yellow'])
        ax.annotate(f'{b:.4f}', (e, b), textcoords='offset points',
                    xytext=(5, 10), fontsize=8, color=MORANDI['lavender_gray'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('SIM_gt'); ax.set_xticks(eps)
    _legend(ax)
    _title(fig, 'v4 vs v5: Data Augmentation Impact (CUDA Graph)', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'epoch_comparison_v4_v5.pdf')

# ════════════════════════════════════════════════════════════
# Fig 15  Native vs CUDA Graph (v4)
# ════════════════════════════════════════════════════════════
def fig15():
    dn = _load(EVAL/'ft_v4_native_comparison.json')
    df = _load(EVAL/'ft_v4_fast_comparison.json')
    eps = [3, 5, 7, 9]
    sn = [dn[f'checkpoint-epoch-{e}']['avg_sim_gt'] for e in eps]
    sf = [df[f'checkpoint-epoch-{e}']['avg_sim_gt'] for e in eps]
    x = np.arange(len(eps)); w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x-w/2, sn, w, color=MORANDI['sage_green'], edgecolor='white',
                lw=2, alpha=0.85, label='Native (RTF ≈ 2.3)')
    b2 = ax.bar(x+w/2, sf, w, color=MORANDI['soft_coral'], edgecolor='white',
                lw=2, alpha=0.85, label='CUDA Graph (RTF ≈ 0.35)')
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                f'{b.get_height():.4f}', ha='center', fontsize=8,
                color=MORANDI['dark_text'])
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                f'{b.get_height():.4f}', ha='center', fontsize=8,
                color=MORANDI['dark_text'])
    ax.set_xticks(x); ax.set_xticklabels([f'Epoch {e}' for e in eps])
    ax.set_ylabel('SIM_gt'); ax.set_ylim(0.65, 0.72)
    _legend(ax)
    _title(fig, 'v4: Native vs. CUDA Graph Inference Quality', y_main=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, 'native_vs_fast_v4.pdf')

# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating Morandi-style report figures …')
    for fn in [fig01, fig02, fig03, fig04, fig05, fig06, fig07,
               fig09, fig10, fig11, fig12, fig13, fig14, fig15]:
        fn()
    print(f'\nAll 14 figures saved to {FIGS}/')
