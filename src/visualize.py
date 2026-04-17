# ============================================================
# src/visualize.py — Step 5: Generate All 4 Evaluation Figures
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Fix Devanagari (Hindi) font rendering
matplotlib.rcParams['font.family'] = ['Noto Sans Devanagari', 'Noto Sans', 'DejaVu Sans']

# ── Colour palette ───────────────────────────────────────────
C = {
    "blue"   : "#0D3B6E",
    "teal"   : "#0E7C7B",
    "mid"    : "#1A6B9A",
    "accent" : "#F4A261",
    "green"  : "#27AE60",
    "red"    : "#E74C3C",
    "grey"   : "#64748B",
    "purple" : "#7D3C98",
    "lightbg": "#F4F9FF",
}

WORD_COLORS = [
    C["blue"], C["teal"], C["mid"],
    C["accent"], C["green"], C["grey"],
]


# ── Master function ──────────────────────────────────────────

def generate_all_figures(features, alignments, variants,
                         figures_dir, dpi=150,
                         transcript=None):
    """
    Generate all 4 evaluation figures and save to figures_dir.

    Parameters
    ----------
    features     : dict  — from src/acoustic.py
    alignments   : list  — from src/alignment.py
    variants     : list  — from src/linguistic.py
    figures_dir  : str   — output directory path
    dpi          : int   — image resolution
    transcript   : str   — real Hindi transcript from audio (optional)
    """
    print("\n" + "=" * 60)
    print("  STEP 5 — GENERATING EVALUATION FIGURES")
    print("=" * 60)

    os.makedirs(figures_dir, exist_ok=True)
    paths = {}

    p1 = os.path.join(figures_dir, "fig1_acoustic_features.png")
    _fig1_acoustic(features, alignments, p1, dpi)
    paths["acoustic_features"] = p1

    p2 = os.path.join(figures_dir, "fig2_mfa_alignment.png")
    _fig2_mfa(alignments, p2, dpi)
    paths["mfa_alignment"] = p2

    p3 = os.path.join(figures_dir, "fig3_word_order_scoring.png")
    _fig3_word_order(variants, p3, dpi, transcript=transcript)
    paths["word_order_scoring"] = p3

    p4 = os.path.join(figures_dir, "fig4_pipeline_overview.png")
    _fig4_pipeline(p4, dpi)
    paths["pipeline_overview"] = p4

    print("\n  All figures saved to:", figures_dir)
    for name, path in paths.items():
        size_kb = os.path.getsize(path) // 1024
        print(f"  ✓ {os.path.basename(path):40s} ({size_kb} KB)")

    return paths


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Acoustic Features
# ══════════════════════════════════════════════════════════════

def _fig1_acoustic(features, alignments, save_path, dpi):
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.suptitle(
        "Figure 1 — Acoustic Feature Extraction with MFA Word Boundaries",
        fontsize=14, fontweight="bold", color=C["blue"], y=0.98,
    )
    gs = GridSpec(3, 1, hspace=0.50, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0])
    y      = features["y"]
    dur    = features["duration"]
    t_wave = np.linspace(0, dur, len(y))
    ax1.fill_between(t_wave, y, alpha=0.65, color=C["mid"], linewidth=0.3)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Waveform", fontsize=11, loc="left",
                  color=C["blue"], fontweight="bold")
    ax1.set_xlim(0, dur)
    ax1.tick_params(labelbottom=False)
    ax1.set_facecolor(C["lightbg"])
    _draw_word_boundaries(ax1, alignments, label_words=True, y_frac=0.88)

    ax2 = fig.add_subplot(gs[1])
    f0      = features["f0"].copy().astype(float)
    vf      = features["voiced_flag"]
    f0[~vf] = np.nan
    ax2.plot(features["f0_times"], f0,
             color=C["teal"], linewidth=2.2, label="F0 (Hz)", zorder=3)
    ax2.fill_between(features["f0_times"],
                     np.where(~np.isnan(f0), f0, 0),
                     alpha=0.18, color=C["teal"])
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_title(
        "Pitch Contour F0 — falling intonation pattern in Hindi SOV sentence",
        fontsize=11, loc="left", color=C["blue"], fontweight="bold",
    )
    ax2.set_xlim(0, dur)
    ax2.set_ylim(60, 280)
    ax2.tick_params(labelbottom=False)
    ax2.set_facecolor(C["lightbg"])
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.8)
    _draw_word_boundaries(ax2, alignments, label_words=False)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(features["rms_times"], features["rms"],
             color=C["accent"], linewidth=1.8, label="RMS Energy")
    ax3.fill_between(features["rms_times"], features["rms"],
                     alpha=0.25, color=C["accent"])
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylabel("RMS Energy", fontsize=10)
    ax3.set_title(
        "Energy Envelope — dips at word / phrase boundaries",
        fontsize=11, loc="left", color=C["blue"], fontweight="bold",
    )
    ax3.set_xlim(0, dur)
    ax3.set_facecolor(C["lightbg"])
    ax3.legend(fontsize=9, loc="upper right", framealpha=0.8)
    _draw_word_boundaries(ax3, alignments, label_words=False)

    if features.get("synthetic"):
        fig.text(
            0.5, 0.005,
            "★ Synthetic demo features — replace AUDIO_FILE in config.py "
            "with real Hindi audio for actual analysis",
            ha="center", fontsize=8, color=C["grey"], style="italic",
        )

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 1 saved")


def _draw_word_boundaries(ax, alignments, label_words=False, y_frac=0.88):
    ymin, ymax = ax.get_ylim()
    y_label    = ymin + y_frac * (ymax - ymin)
    for i, a in enumerate(alignments):
        c = WORD_COLORS[i % len(WORD_COLORS)]
        ax.axvline(a["start"], color=c, linestyle="--",
                   alpha=0.55, linewidth=0.9, zorder=2)
        ax.axvspan(a["start"], a["end"], alpha=0.06, color=c, zorder=1)
        if label_words:
            mid = (a["start"] + a["end"]) / 2
            ax.text(mid, y_label, a["word"],
                    ha="center", va="top", fontsize=8.5,
                    color=c, fontweight="bold", clip_on=True)


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — MFA Alignment (TextGrid)
# ══════════════════════════════════════════════════════════════

def _fig2_mfa(alignments, save_path, dpi):
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="white")
    ax.set_facecolor(C["lightbg"])

    # ── FIXED: Updated title — real MFA implemented ──────────
    fig.suptitle(
        "Figure 2 — MFA Acoustic–Linguistic Alignment  (TextGrid)\n"
        "Each word is mapped to its start time, end time, and phoneme sequence",
        fontsize=13, fontweight="bold", color=C["blue"],
    )

    total_dur = max(a["end"] for a in alignments) + 0.15
    ax.set_xlim(0, total_dur)
    ax.set_ylim(-0.35, 3.4)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_yticks([0.5, 1.55, 2.6])
    ax.set_yticklabels(["Phoneme tier", "Word tier", "Role tier"], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.axhline(1.08, color=C["grey"], linewidth=0.5, linestyle=":")
    ax.axhline(2.12, color=C["grey"], linewidth=0.5, linestyle=":")

    for i, a in enumerate(alignments):
        c   = WORD_COLORS[i % len(WORD_COLORS)]
        s   = a["start"]
        e   = a["end"]
        w   = a["word"]
        pad = 0.008

        # Role tier
        ax.text((s + e) / 2, 2.6, "",
                ha="center", va="center", fontsize=9,
                color=c, fontweight="bold")
        ax.plot([s, e], [2.12, 2.12], color=c, linewidth=1.5)

        # Word tier
        rect_w = mpatches.FancyBboxPatch(
            (s + pad, 1.12), e - s - 2 * pad, 0.88,
            boxstyle="round,pad=0.01",
            linewidth=1.8, edgecolor=c,
            facecolor=c, alpha=0.82,
        )
        ax.add_patch(rect_w)
        ax.text((s + e) / 2, 1.56, w,
                ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

        # Phoneme tier
        phones = a["phones"]
        n_ph   = max(len(phones), 1)
        ph_w   = (e - s) / n_ph
        for j, ph in enumerate(phones):
            ps = s + j * ph_w
            pe = ps + ph_w
            rect_p = mpatches.FancyBboxPatch(
                (ps + pad, 0.1), ph_w - 2 * pad, 0.88,
                boxstyle="round,pad=0.005",
                linewidth=0.8, edgecolor=c,
                facecolor=c, alpha=0.28,
            )
            ax.add_patch(rect_p)
            ax.text((ps + pe) / 2, 0.54, ph,
                    ha="center", va="center",
                    fontsize=9, color=C["blue"])

        # Duration arrow
        ax.annotate(
            "", xy=(e - 0.005, -0.12), xytext=(s + 0.005, -0.12),
            arrowprops=dict(arrowstyle="<->", color=c, lw=1.0),
        )
        ax.text((s + e) / 2, -0.28, f"{a['duration']:.2f}s",
                ha="center", fontsize=8, color=c)

    # ── FIXED: Show real MFA source info instead of MTP-2 note
    # Count how many are real MFA aligned vs unk
    real_mfa = sum(1 for a in alignments if a.get("in_dict", True))
    total    = len(alignments)
    method   = alignments[0].get("method", "MFA") if alignments else "MFA"

    ax.text(
        total_dur * 0.98, 3.25,
        f"Method: {method}  |  Aligned: {real_mfa}/{total} words  "
        f"|  Source: AI4Bharat IndicMFA Hindi (255h training data)",
        ha="right", fontsize=8, color=C["teal"],
        style="italic", fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.87])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 2 saved")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — Word Order Variant Scoring
# ══════════════════════════════════════════════════════════════

def _fig3_word_order(variants, save_path, dpi, transcript=None):
    """
    FIXED: Uses real transcript from audio instead of hardcoded
    'Ramesh ne kitaab Neha ko di' example sentence.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), facecolor="white")

    # ── FIXED: Use real transcript from audio ────────────────
    if transcript and len(transcript.strip()) > 0:
        # Truncate long transcripts for display
        display_sentence = transcript.strip()
        if len(display_sentence) > 70:
            display_sentence = display_sentence[:70] + "..."
        title_sentence = f'"{display_sentence}"'
        title_sub = "(Real Hindi audio transcript)"
    else:
        title_sentence = '"Ramesh ne kitaab Neha ko di"'
        title_sub = "(Ramesh gave the book to Neha)"

    fig.suptitle(
        "Figure 3 — Cognitive Constraint Scoring of Hindi Word Order Variants\n"
        f"Sentence: {title_sentence}  {title_sub}",
        fontsize=12, fontweight="bold", color=C["blue"],
    )

    orders  = [v["order"]      for v in variants]
    colors  = [C["green"] if v["canonical"] else C["mid"] for v in variants]
    edgecol = [C["green"] if v["canonical"] else C["blue"] for v in variants]

    _bar_chart(
        axes[0],
        labels=orders,
        values=[v["dep_length"] for v in variants],
        colors=colors, edge_colors=edgecol,
        title="Dependency Length\n(lower = more efficient)",
        ylabel="Dependency Length",
        fmt="{:.0f}",
        source="Theory: Paper 1 (Zafar & Husain 2023)",
    )

    _bar_chart(
        axes[1],
        labels=orders,
        values=[v["surprisal"] for v in variants],
        colors=colors, edge_colors=edgecol,
        title="Discourse Surprisal\n(lower = more predictable)",
        ylabel="Surprisal Score",
        fmt="{:.1f}",
        source="Theory: Paper 2 (Ranjan et al. 2022)",
    )

    _bar_chart(
        axes[2],
        labels=orders,
        values=[v["naturalness"] for v in variants],
        colors=colors, edge_colors=edgecol,
        title="Overall Naturalness Rank\n(higher = more natural)",
        ylabel="Naturalness Score (1–5)",
        fmt=None,
        ylim=(0, 6.2),
        source="Combined: Dep.Length + Surprisal + IS Score",
        star_values=[v["naturalness"] for v in variants],
    )

    p1 = mpatches.Patch(color=C["green"], label="Canonical (predicted best)")
    p2 = mpatches.Patch(color=C["mid"],   label="Non-canonical variant")
    fig.legend(handles=[p1, p2], loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 3 saved")


def _bar_chart(ax, labels, values, colors, edge_colors,
               title, ylabel, fmt, source="",
               ylim=None, star_values=None):
    bars = ax.bar(
        labels, values,
        color=colors, edgecolor=edge_colors,
        linewidth=1.5, zorder=3, width=0.55,
    )
    ax.set_title(title, fontsize=11, color=C["blue"], pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_facecolor(C["lightbg"])
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    if source:
        ax.text(0.98, -0.28, source,
                transform=ax.transAxes,
                ha="right", fontsize=7, color=C["grey"], style="italic")
    for idx, (bar, val) in enumerate(zip(bars, values)):
        if star_values is not None:
            n_star = star_values[idx]
            stars  = "★" * n_star + "☆" * (5 - n_star)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.12,
                    stars, ha="center", fontsize=9, color=C["blue"])
        elif fmt:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(val * 0.015, 0.1),
                    fmt.format(val), ha="center",
                    fontsize=10, fontweight="bold", color=C["blue"])


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Full Pipeline Overview
# ══════════════════════════════════════════════════════════════

def _fig4_pipeline(save_path, dpi):
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="white")
    ax.axis("off")
    fig.suptitle(
        "Figure 4 — Proposed Two-Stream Pipeline: "
        "Acoustic–Linguistic Modelling of Cognitive Constraints",
        fontsize=13, fontweight="bold", color=C["blue"],
    )

    steps = [
        ("Hindi\nSpeech\nInput",   C["blue"],   "User utterance\n(conversational)"),
        ("ASR\n(Whisper)",         C["mid"],    "Speech → Text\nHindi transcript"),
        ("MFA\nAlignment",         C["teal"],   "Audio ↔ Text\nword boundaries"),
        ("Acoustic\nFeatures",     "#7D6608",   "Pitch  Energy\nDuration  MFCC"),
        ("Linguistic\nFeatures",   C["purple"], "Dep. length\nSurprisal  IS"),
        ("Cognitive\nModelling",   "#922B21",   "Efficiency +\nPredictability"),
        ("Response\nSelection",    C["green"],  "Most natural\nHindi response"),
    ]

    bw, bh = 1.55, 1.5
    gap    = 0.28
    total  = len(steps) * bw + (len(steps) - 1) * gap
    x0     = (14 - total) / 2
    y0     = 1.5

    for i, (label, color, sub) in enumerate(steps):
        x = x0 + i * (bw + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y0), bw, bh,
            boxstyle="round,pad=0.07",
            linewidth=2, edgecolor=color,
            facecolor=color, alpha=0.88,
            transform=ax.transData,
        )
        ax.add_patch(rect)
        ax.text(x + bw / 2, y0 + bh / 2 + 0.08, label,
                ha="center", va="center",
                fontsize=9.5, color="white",
                fontweight="bold", transform=ax.transData,
                linespacing=1.4)
        ax.text(x + bw / 2, y0 - 0.22, sub,
                ha="center", va="top",
                fontsize=7.5, color=C["grey"],
                transform=ax.transData, linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + bw + gap, y0 + bh / 2),
                xytext=(x + bw, y0 + bh / 2),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=2.0),
                transform=ax.transData,
            )

    ax.annotate(
        "", xy=(x0 + 3 * (bw + gap), y0 - 0.8),
        xytext=(x0 + 3 * (bw + gap) + bw, y0 - 0.8),
        arrowprops=dict(arrowstyle="<->", color=C["mid"], lw=1.2),
        transform=ax.transData,
    )
    ax.text(x0 + 3 * (bw + gap) + bw / 2, y0 - 1.05,
            "Two-stream feature extraction",
            ha="center", fontsize=8.5, color=C["mid"],
            style="italic", transform=ax.transData)

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 4 saved")