# ============================================================
# config.py — Project Configuration
# MTP-1: Acoustic-Linguistic Modelling of Cognitive Constraints
# ============================================================
# 
# HOW TO USE:
#   1. Place your Hindi .wav audio file in:  data/audio/
#   2. Set AUDIO_FILE to the filename below
#   3. Run:  python run_demo.py
#
# If AUDIO_FILE = None, the demo runs with synthetic data
# and still produces all 4 figures correctly.
# ============================================================

import os

# ── Audio File ───────────────────────────────────────────────
# Set to your Hindi audio filename (must be inside data/audio/)
# Example: AUDIO_FILE = "hindi_sample.wav"
#AUDIO_FILE = None
AUDIO_FILE = "hin_04.wav"

# ── Paths (do not change) ────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
AUDIO_DIR     = os.path.join(DATA_DIR, "audio")
SAMPLES_DIR   = os.path.join(DATA_DIR, "samples")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR   = os.path.join(OUTPUT_DIR, "figures")

# Full path to audio file (auto-constructed)
AUDIO_PATH = os.path.join(AUDIO_DIR, AUDIO_FILE) if AUDIO_FILE else None

# ── Sample Sentence (used when no audio provided) ────────────
SAMPLE_SENTENCE_ROMAN = "Ramesh ne kitaab Neha ko di"
SAMPLE_SENTENCE_HINDI = "रमेश ने किताब नेहा को दी"
SAMPLE_SENTENCE_TRANS = "Ramesh gave the book to Neha"

# ── Audio Settings ───────────────────────────────────────────
SAMPLE_RATE = 16000      # Hz — standard for speech processing

# ── ASR Settings ─────────────────────────────────────────────
WHISPER_MODEL = "small"
#WHISPER_MODEL = "base"   # Options: tiny, base, small, medium, large
                          # 'base' is fastest, good enough for demo
LANGUAGE      = "hi"     # Hindi

# ── Figure Settings ──────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"

# ── Colour Palette ───────────────────────────────────────────
COLORS = {
    "blue"   : "#0D3B6E",
    "teal"   : "#0E7C7B",
    "mid"    : "#1A6B9A",
    "accent" : "#F4A261",
    "green"  : "#27AE60",
    "red"    : "#E74C3C",
    "grey"   : "#64748B",
    "purple" : "#7D3C98",
    "lightbg": "#F4F9FF",
    "white"  : "#FFFFFF",
}

# ── Ensure output directories exist ─────────────────────────
os.makedirs(FIGURES_DIR, exist_ok=True)
