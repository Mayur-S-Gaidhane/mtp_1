#!/usr/bin/env python3
# ============================================================
# run_demo.py  —  MTP-1 EVALUATION DEMO
#
# HOW TO RUN:
#   1. (First time only)  bash setup.sh
#   2. source venv/bin/activate
#   3. python run_demo.py
#
# To use YOUR Hindi audio:
#   - Place .wav file in:  data/audio/
#   - Open config.py and set:  AUDIO_FILE = "your_file.wav"
#   - Run again: python run_demo.py
# ============================================================

import os
import sys
import time

# ── Ensure project root is on path ──────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Project imports ──────────────────────────────────────────
import config

from src.asr        import run_asr, get_transcript_words
from src.acoustic   import extract_features
from src.alignment  import run_alignment
from src.linguistic import run_analysis
from src.visualize  import generate_all_figures


# ── Banner ───────────────────────────────────────────────────
def print_banner():
    print("\n" + "=" * 60)
    print("  MTP-1 PRELIMINARY ANALYSIS DEMO")
    print("  Acoustic–Linguistic Modelling of Cognitive Constraints")
    print("  for Context-Aware Response Selection in Hindi Speech")
    print()
    print("  Student  : Mayur Shriram Gaidhane")
    print("  Supervisor: Prof. Sidharth Ranjan")
    print("=" * 60)


# ── Main pipeline ─────────────────────────────────────────────
def main():
    print_banner()
    start_time = time.time()

    audio_path = config.AUDIO_PATH

    # ── STEP 1: ASR ──────────────────────────────────────────
    transcript, whisper_words, asr_success = run_asr(
        audio_path,
        model_name=config.WHISPER_MODEL,
        language=config.LANGUAGE,
    )

    # Decide which words to use downstream
    if asr_success and transcript:
        words = get_transcript_words(transcript, whisper_words)
        sentence_for_analysis = transcript
    else:
        # Fallback to pre-defined sample sentence
        words = config.SAMPLE_SENTENCE_ROMAN.split()
        sentence_for_analysis = config.SAMPLE_SENTENCE_ROMAN
        print(f"  Using sample: \"{config.SAMPLE_SENTENCE_ROMAN}\"")
        print(f"  Translation : \"{config.SAMPLE_SENTENCE_TRANS}\"")

    # ── STEP 2: Acoustic Features ────────────────────────────
    features = extract_features(audio_path, config.SAMPLE_RATE)

    # ── STEP 3: Real WhisperX Alignment ──────────────────────
    alignments = run_alignment(words, features["duration"], audio_path)

    # ── STEP 4: Linguistic Analysis ──────────────────────────
    parse, variants = run_analysis(sentence_for_analysis)

    # ── STEP 5: Generate Figures ─────────────────────────────
    figure_paths = generate_all_figures(
    features    = features,
    alignments  = alignments,
    variants    = variants,
    figures_dir = config.FIGURES_DIR,
    dpi         = config.FIGURE_DPI,
    transcript  = sentence_for_analysis,
    )

    # ── Final Summary ─────────────────────────────────────────
    elapsed = time.time() - start_time
    best    = min(variants, key=lambda v: v["dep_length"])

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Sentence analysed : {sentence_for_analysis}")
    print(f"  Words aligned     : {len(alignments)}")
    print(f"  Variants scored   : {len(variants)}")
    print(f"  Best word order   : {best['order']}  "
          f"(dep_length={best['dep_length']}, "
          f"surprisal={best['surprisal']})")
    print(f"  Audio mode        : "
          f"{'Real audio' if asr_success else 'Synthetic (demo)'}")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print()
    print("  Output figures:")
    for name, path in figure_paths.items():
        print(f"    → {path}")
    print()
    print("  Open the figures folder to view all 4 evaluation images:")
    print(f"  {config.FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
