"""
Paper B: Generate all figures for the paper.
- Fig 1: Bias heatmap (probe × judge matrix, Cliff's delta color scale)
- Fig 2: Recovery 5-dimension delta chart (grouped bar, 5dims × 7judges)
- Fig 3: Recency position gradient (line plot, delta vs position per judge)
- Fig 4: Bidirectional outcome bias (grouped bar, resolved/unresolved × 7judges)
- Fig 5: Debiasing comparison
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

STATS_DIR = Path("results/statistics_v2")
DEBIASING_DIR = Path("results/debiasing")
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Short judge names for display
JUDGE_SHORT = {
    "claude-sonnet-4-5": "Claude 4.5",
    "deepseek-v3.2": "DeepSeek",
    "gemini-3-pro": "Gemini 3",
    "glm-5": "GLM-5",
    "gpt-5.2": "GPT-5.2",
    "qwen3-235b": "Qwen3",
    "kimi-k2.5": "Kimi K2.5",
}

PROBE_LABELS = {
    "recovery": "Error Recovery",
    "recency": "Recency",
    "outcome": "Outcome",
    "peakend": "Position Sensitivity",
    "length": "Length",
}

JUDGES_ORDER = ["claude-sonnet-4-5", "deepseek-v3.2", "gemini-3-pro", "glm-5", "gpt-5.2", "qwen3-235b", "kimi-k2.5"]
PROBES_ORDER = ["recovery", "recency", "outcome", "peakend", "length"]

# NeurIPS-friendly styling
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Fig 1: Bias Heatmap
# ============================================================
def fig1_bias_heatmap():
    """Probe × Judge heatmap of Cliff's delta values."""
    results = load_json(STATS_DIR / "main_results.json")

    # Build matrix
    data = {}
    sig_data = {}
    for r in results:
        data[(r["probe"], r["judge"])] = r["cliffs_delta"]
        sig_data[(r["probe"], r["judge"])] = r.get("bh_significant", False)

    matrix = np.full((len(PROBES_ORDER), len(JUDGES_ORDER)), np.nan)
    sig_matrix = np.full((len(PROBES_ORDER), len(JUDGES_ORDER)), False)

    for i, probe in enumerate(PROBES_ORDER):
        for j, judge in enumerate(JUDGES_ORDER):
            if (probe, judge) in data:
                matrix[i, j] = data[(probe, judge)]
                sig_matrix[i, j] = sig_data.get((probe, judge), False)

    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    # Add text annotations
    for i in range(len(PROBES_ORDER)):
        for j in range(len(JUDGES_ORDER)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha='center', va='center', fontsize=7, color='gray')
            else:
                sig_marker = "*" if sig_matrix[i, j] else ""
                color = 'white' if abs(val) > 0.4 else 'black'
                ax.text(j, i, f"{val:+.2f}{sig_marker}", ha='center', va='center',
                       fontsize=7, color=color, fontweight='bold' if sig_matrix[i, j] else 'normal')

    ax.set_xticks(range(len(JUDGES_ORDER)))
    ax.set_xticklabels([JUDGE_SHORT[j] for j in JUDGES_ORDER], rotation=30, ha='right')
    ax.set_yticks(range(len(PROBES_ORDER)))
    ax.set_yticklabels([PROBE_LABELS[p] for p in PROBES_ORDER])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cliff's $\\delta$", fontsize=8)

    ax.set_title("Bias Probe × Judge: Cliff's Delta (* = BH-significant)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_bias_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig1_bias_heatmap.png")
    plt.close()
    print("Fig 1: Bias heatmap saved")


# ============================================================
# Fig 2: Recovery 5-Dimension Analysis
# ============================================================
def fig2_recovery_5dim():
    """Grouped bar chart: 5 dimensions × judges."""
    recovery = load_json(STATS_DIR / "recovery_5dim.json")

    dims = ["task_progress", "efficiency", "action_correctness", "error_handling", "reasoning_quality"]
    dim_labels = ["Task\nProgress", "Efficiency", "Action\nCorrectness", "Error\nHandling", "Reasoning\nQuality"]

    judges_with_data = [j for j in JUDGES_ORDER if j in recovery and recovery[j]]
    n_judges = len(judges_with_data)
    n_dims = len(dims)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(n_dims)
    width = 0.8 / n_judges
    colors = plt.cm.Set2(np.linspace(0, 1, n_judges))

    for idx, judge in enumerate(judges_with_data):
        deltas = []
        for dim in dims:
            d = recovery[judge].get(dim, {})
            deltas.append(d.get("cliffs_delta", 0))

        offset = (idx - n_judges / 2 + 0.5) * width
        bars = ax.bar(x + offset, deltas, width * 0.9, label=JUDGE_SHORT[judge],
                      color=colors[idx], edgecolor='gray', linewidth=0.5)

        # Add significance markers
        for i, dim in enumerate(dims):
            d = recovery[judge].get(dim, {})
            p = d.get("p_value", 1.0)
            if p < 0.001:
                marker = "***"
            elif p < 0.01:
                marker = "**"
            elif p < 0.05:
                marker = "*"
            else:
                marker = ""
            if marker:
                y_pos = deltas[i] - 0.03 if deltas[i] < 0 else deltas[i] + 0.01
                ax.text(x[i] + offset, y_pos, marker, ha='center', va='top' if deltas[i] < 0 else 'bottom',
                       fontsize=5, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("Cliff's $\\delta$ (perturbed - original)")
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_title("Error Recovery Penalty: Per-Dimension Analysis")
    ax.legend(loc='lower left', ncol=4, fontsize=6, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_recovery_5dim.pdf")
    fig.savefig(FIG_DIR / "fig2_recovery_5dim.png")
    plt.close()
    print("Fig 2: Recovery 5-dim saved")


# ============================================================
# Fig 3: Recency Position Gradient
# ============================================================
def fig3_recency_gradient():
    """Line plot: mean score delta vs position per judge. Full-width, improved readability."""
    gradient = load_json(STATS_DIR / "recency_gradient.json")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    positions = [25, 50, 75]
    pos_keys = ["p25", "p50", "p75"]
    # Use distinct colors + linestyles for colorblind accessibility
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    linestyles = ['-', '-', '--', '-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    for idx, judge in enumerate(JUDGES_ORDER):
        if judge not in gradient:
            continue

        deltas = []
        valid_pos = []
        for pk, pos in zip(pos_keys, positions):
            if pk in gradient[judge]:
                deltas.append(gradient[judge][pk]["mean_delta"])
                valid_pos.append(pos)

        if valid_pos:
            gt = gradient[judge].get("gradient_test", {})
            gt_p = gt.get("p_value", 1.0)
            label_suffix = " *" if gt_p < 0.05 else ""

            ax.plot(valid_pos, deltas, marker=markers[idx], color=colors[idx],
                   linestyle=linestyles[idx],
                   label=f"{JUDGE_SHORT[judge]}{label_suffix}",
                   linewidth=1.8, markersize=6, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel("Error Position (% through trajectory)")
    ax.set_ylabel("Mean $\\bar{\\delta}$ (score difference)")
    ax.set_xticks(positions)
    ax.set_xticklabels(["25%", "50%", "75%"])
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title("Recency Bias: Position Gradient (* = significant gradient test)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_recency_gradient.pdf")
    fig.savefig(FIG_DIR / "fig3_recency_gradient.png")
    plt.close()
    print("Fig 3: Recency gradient saved")


# ============================================================
# Fig 4: Bidirectional Outcome Bias
# ============================================================
def fig4_outcome_bidirectional():
    """Grouped bar chart: resolved/unresolved × judges."""
    bidir = load_json(STATS_DIR / "outcome_bidirectional.json")

    judges_with_data = [j for j in JUDGES_ORDER if j in bidir]
    n = len(judges_with_data)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(n)
    width = 0.35

    resolved_deltas = []
    unresolved_deltas = []
    for judge in judges_with_data:
        resolved_deltas.append(bidir[judge].get("resolved", {}).get("mean_delta", 0))
        unresolved_deltas.append(bidir[judge].get("unresolved", {}).get("mean_delta", 0))

    bars1 = ax.bar(x - width/2, resolved_deltas, width, label='Resolved→"Failed"',
                   color='#d73027', alpha=0.8, edgecolor='gray', linewidth=0.5)
    bars2 = ax.bar(x + width/2, unresolved_deltas, width, label='Unresolved→"Success"',
                   color='#4575b4', alpha=0.8, edgecolor='gray', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([JUDGE_SHORT[j] for j in judges_with_data], rotation=20, ha='right')
    ax.set_ylabel("Mean $\\bar{\\delta}$ (perturbed $-$ original)")
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title("Bidirectional Outcome Bias")
    ax.legend(fontsize=8)

    # Add combined effect annotation
    for i, judge in enumerate(judges_with_data):
        combined = abs(resolved_deltas[i]) + abs(unresolved_deltas[i])
        ax.text(i, max(unresolved_deltas[i], 0) + 0.08, f"|Σ|={combined:.2f}",
               ha='center', va='bottom', fontsize=6, color='gray')

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_outcome_bidirectional.pdf")
    fig.savefig(FIG_DIR / "fig4_outcome_bidirectional.png")
    plt.close()
    print("Fig 4: Outcome bidirectional saved")


# ============================================================
# Fig 5: Debiasing Comparison
# ============================================================
def fig5_debiasing():
    """Bar chart comparing debiasing strategies."""
    debiasing_file = DEBIASING_DIR / "debiasing_results.json"
    if not debiasing_file.exists():
        print("Fig 5: SKIP (no debiasing results found)")
        return

    raw = load_json(debiasing_file)

    # Parse list format: [{strategy, judge, delta, ...}]
    data = {}
    for item in raw:
        judge = item["judge"]
        strategy = item["strategy"]
        if judge not in data:
            data[judge] = {}
        data[judge][strategy] = item["delta"]

    # Add baseline from bidirectional outcome results (avg of |resolved| and |unresolved|)
    # Using directional subgroup means, NOT the aggregate (which the paper shows is misleading)
    bidir_file = STATS_DIR / "outcome_bidirectional.json"
    if bidir_file.exists():
        bidir = load_json(bidir_file)
        for judge in data:
            if judge in bidir:
                res_d = abs(bidir[judge].get("resolved", {}).get("mean_delta", 0))
                unres_d = abs(bidir[judge].get("unresolved", {}).get("mean_delta", 0))
                data[judge]["baseline"] = (res_d + unres_d) / 2  # avg absolute directional effect
    else:
        # Fallback to main results
        main_results = load_json(STATS_DIR / "main_results.json")
        for r in main_results:
            if r["probe"] == "outcome" and r["judge"] in data:
                data[r["judge"]]["baseline"] = r["mean_delta"]

    judges_tested = sorted(data.keys())
    n = len(judges_tested)

    strategies = ["baseline", "outcome_masking", "explicit_instruction"]
    strategy_labels = ["Baseline", "Outcome\nMasking", "Explicit\nInstruction"]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(n)
    width = 0.25
    colors_list = ["#d73027", "#4575b4", "#fdae61"]

    for idx, (strategy, label) in enumerate(zip(strategies, strategy_labels)):
        deltas = [abs(data.get(j, {}).get(strategy, 0)) for j in judges_tested]
        ax.bar(x + (idx - 1) * width, deltas, width * 0.9, label=label,
               color=colors_list[idx], edgecolor='gray', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([JUDGE_SHORT.get(j, j) for j in judges_tested], rotation=20, ha='right')
    ax.set_ylabel("|Mean $\\delta$| (absolute bias)")
    ax.set_title("Debiasing Strategy Comparison")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_debiasing.pdf")
    fig.savefig(FIG_DIR / "fig5_debiasing.png")
    plt.close()
    print("Fig 5: Debiasing comparison saved")


if __name__ == "__main__":
    fig1_bias_heatmap()
    fig2_recovery_5dim()
    fig3_recency_gradient()
    fig4_outcome_bidirectional()
    fig5_debiasing()
    print(f"\nAll figures saved to {FIG_DIR}")
