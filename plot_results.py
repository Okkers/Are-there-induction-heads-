import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BG       = "#0d0d0f"
PANEL    = "#16161a"
GRID     = "#2a2a30"
TEXT     = "#e8e6e0"
MUTED    = "#6b6870"
ACCENT   = "#c8f060" 
ACCENT2  = "#f06060"
ACCENT3  = "#60c8f0"
ACCENT4  = "#f0c060"

FONT_MONO = "monospace"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       FONT_MONO,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  GRID,
    "legend.labelcolor": TEXT,
})

CHANCE = 1 / 10

def _bar(ax, xs, ys, color=ACCENT, width=0.6, label=None):
    bars = ax.bar(xs, ys, width=width, color=color, alpha=0.85,
                  linewidth=0, label=label)
    return bars

def _hline(ax, y, label=None, color=ACCENT2, ls="--"):
    ax.axhline(y, color=color, linewidth=1.2, linestyle=ls,
               alpha=0.7, label=label)

def _style(ax, title, xlabel, ylabel, ylim=(0, 1.05), legend=True):
    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(fontsize=8, framealpha=0.6)

def _split_mean_std(series):
    xs = sorted(series)
    means = []
    stds = []
    for x in xs:
        value = series[x]
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mean, std = value
        else:
            mean, std = value, 0.0
        means.append(float(mean))
        stds.append(float(std))
    return xs, means, stds

def plot_training_curve(losses, save_path="training_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    epochs = list(range(len(losses)))
    ax.plot(epochs, losses, color=ACCENT, linewidth=1.8, label="train loss")
    ax.fill_between(epochs, losses, alpha=0.12, color=ACCENT)
    _style(ax, "Training Loss", "Epoch", "Cross-Entropy Loss",
           ylim=(0, max(losses) * 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")

def plot_accuracy_by_prefix_length(acc_by_length, save_path="eval_by_prefix_length.png"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    lengths, accs, stds = _split_mean_std(acc_by_length)
    _bar(ax, lengths, accs, color=ACCENT, label="mean accuracy")
    ax.errorbar(lengths, accs, yerr=stds, fmt="none", ecolor=TEXT,
                elinewidth=1.0, capsize=3, alpha=0.85)
    _hline(ax, CHANCE, label=f"chance ({CHANCE:.2f})", color=ACCENT2)
    ax.set_xticks(lengths)
    _style(ax, "Accuracy by Prefix Length", "Prefix Length", "Accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")

def plot_gap_length(acc_by_gap, save_path="test_gap_length.png"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    gaps, accs, stds = _split_mean_std(acc_by_gap)
    ax.errorbar(gaps, accs, yerr=stds, color=ACCENT, linewidth=2, marker="o",
            markersize=5, label="accuracy")
    lower = [max(0.0, m - s) for m, s in zip(accs, stds)]
    upper = [min(1.0, m + s) for m, s in zip(accs, stds)]
    ax.fill_between(gaps, lower, upper, alpha=0.10, color=ACCENT)
    _hline(ax, CHANCE, label=f"chance ({CHANCE:.2f})", color=ACCENT2)

    ax.text(gaps[0],  0.97, "flat → induction", color=ACCENT,
            fontsize=7, va="top")
    ax.text(gaps[-1], 0.97, "decay → heuristic", color=ACCENT2,
            fontsize=7, va="top", ha="right")

    _style(ax, "gap_length  ·  Accuracy vs Gap Size",
           "Gap Length (tokens)", "Accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")

def plot_out_of_context(acc_standard, acc_ooc_pos, acc_ooc_no_pos, save_path="test_out_of_context.png"):

    fig, ax = plt.subplots(figsize=(7, 3.5))

    labels = [
        "standard\n(matching prefix)",
        "OOC — pos\n(same position)",
        "OOC — no pos\n(decoy offset)",
    ]
    values = []
    colors = [ACCENT, ACCENT3, ACCENT4]
    stds = []
    for value in [acc_standard, acc_ooc_pos, acc_ooc_no_pos]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mean, std = value
        else:
            mean, std = value, 0.0
        values.append(float(mean))
        stds.append(float(std))

    x_pos = range(len(labels))
    for i, (v, s, c) in enumerate(zip(values, stds, colors)):
        ax.bar(i, v, width=0.55, color=c, alpha=0.85, linewidth=0)
        ax.errorbar(i, v, yerr=s, fmt="none", ecolor=TEXT, elinewidth=1.0, capsize=3)
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=TEXT)

    _hline(ax, CHANCE, label=f"chance ({CHANCE:.2f})", color=ACCENT2)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, fontsize=8)

    def _gap_arrow(ax, x0, x1, y, label, color):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="<->", color=color,
                                   lw=1.2, shrinkA=4, shrinkB=4))
        ax.text((x0 + x1) / 2, y + 0.03, label,
                ha="center", fontsize=7, color=color)

    mid_y = max(values) * 0.55
    _gap_arrow(ax, 0, 1, mid_y,
               f"induction\n−{values[0] - values[1]:.2f}", ACCENT)
    _gap_arrow(ax, 1, 2, mid_y * 0.72,
               f"pos. heuristic\n−{values[1] - values[2]:.2f}", ACCENT3)

    _style(ax, "out_of_context  ·  Dissociating Induction vs Positional Heuristic",
           "Condition", "Accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")

def plot_summary(losses, acc_by_prefix, acc_by_gap, acc_standard, acc_ooc_pos, acc_ooc_no_pos, save_path="summary.png"):
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Induction Head Experiment  ·  Results Summary",
                 fontsize=14, fontweight="bold", y=0.98, color=TEXT)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0])
    epochs = list(range(len(losses)))
    ax0.plot(epochs, losses, color=ACCENT, linewidth=1.8)
    ax0.fill_between(epochs, losses, alpha=0.12, color=ACCENT)
    _style(ax0, "Training Loss", "Epoch", "CE Loss",
           ylim=(0, max(losses) * 1.1), legend=False)

    ax1 = fig.add_subplot(gs[0, 1])
    lengths, accs, stds = _split_mean_std(acc_by_prefix)
    _bar(ax1, lengths, accs, color=ACCENT)
    ax1.errorbar(lengths, accs, yerr=stds, fmt="none", ecolor=TEXT,
                 elinewidth=1.0, capsize=3, alpha=0.85)
    _hline(ax1, CHANCE, color=ACCENT2)
    ax1.set_xticks(lengths)
    _style(ax1, "Eval: Prefix Length", "Prefix Len", "Accuracy", legend=False)

    ax2 = fig.add_subplot(gs[0, 2])
    gaps, gaccs, gstds = _split_mean_std(acc_by_gap)
    ax2.errorbar(gaps, gaccs, yerr=gstds, color=ACCENT, linewidth=2, marker="o", markersize=4)
    lower = [max(0.0, m - s) for m, s in zip(gaccs, gstds)]
    upper = [min(1.0, m + s) for m, s in zip(gaccs, gstds)]
    ax2.fill_between(gaps, lower, upper, alpha=0.10, color=ACCENT)
    _hline(ax2, CHANCE, color=ACCENT2)
    _style(ax2, "Test: Gap Length", "Gap Len", "Accuracy", legend=False)

    ax3 = fig.add_subplot(gs[1, :])
    ooc_labels = ["standard\n(matching prefix)", "OOC — pos\n(same position)", "OOC — no pos\n(decoy offset)"]
    ooc_vals = []
    ooc_stds = []
    ooc_colors = [ACCENT, ACCENT3, ACCENT4]
    for value in [acc_standard, acc_ooc_pos, acc_ooc_no_pos]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mean, std = value
        else:
            mean, std = value, 0.0
        ooc_vals.append(float(mean))
        ooc_stds.append(float(std))
    for i, (v, s, c) in enumerate(zip(ooc_vals, ooc_stds, ooc_colors)):
        ax3.bar(i, v, width=0.35, color=c, alpha=0.85, linewidth=0)
        ax3.errorbar(i, v, yerr=s, fmt="none", ecolor=TEXT, elinewidth=1.0, capsize=3)
        ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color=TEXT)
    _hline(ax3, CHANCE, color=ACCENT2, label=f"chance ({CHANCE:.2f})")
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(ooc_labels, fontsize=9)

    mid = max(ooc_vals) * 0.55
    for x0, x1, y, label, col in [
        (0, 1, mid,        f"induction  -{ooc_vals[0] - ooc_vals[1]:.2f}",        ACCENT),
        (1, 2, mid * 0.72, f"pos. heuristic  -{ooc_vals[1] - ooc_vals[2]:.2f}", ACCENT3),
    ]:
        ax3.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="<->", color=col,
                                     lw=1.2, shrinkA=4, shrinkB=4))
        ax3.text((x0 + x1) / 2, y + 0.03, label,
                 ha="center", fontsize=8, color=col)

    _style(ax3, "Test: Out-of-Context  ·  Induction vs Positional Heuristic",
           "Condition", "Accuracy")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")
