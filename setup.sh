#!/bin/bash
# ============================================================
# MTP-1 PROJECT SETUP SCRIPT FOR WSL (Windows 11)
# Run this ONCE before your first demo
# Usage: bash setup.sh
# ============================================================

set -e   # Stop on any error

echo ""
echo "============================================================"
echo "  MTP-1 PROJECT SETUP"
echo "  Acoustic-Linguistic Modelling — Hindi Speech"
echo "============================================================"

# ── STEP 1: System packages (WSL Ubuntu) ────────────────────
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    2>/dev/null
echo "  ✓ System packages installed"

# ── STEP 2: Create virtual environment ──────────────────────
echo ""
echo "[2/5] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created: ./venv"
else
    echo "  ✓ Virtual environment already exists"
fi

# ── STEP 3: Activate venv and upgrade pip ───────────────────
echo ""
echo "[3/5] Activating venv and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip -q
echo "  ✓ pip upgraded"

# ── STEP 4: Install Python packages ─────────────────────────
echo ""
echo "[4/5] Installing Python packages from requirements.txt..."
echo "  (This may take 3-5 minutes the first time)"
pip install -r requirements.txt -q
echo "  ✓ All packages installed"

# ── STEP 5: Verify key imports ──────────────────────────────
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
import numpy as np
print('  ✓ numpy', np.__version__)
import scipy
print('  ✓ scipy', scipy.__version__)
import matplotlib
print('  ✓ matplotlib', matplotlib.__version__)
try:
    import librosa
    print('  ✓ librosa', librosa.__version__)
except Exception as e:
    print('  ✗ librosa:', e)
try:
    import soundfile
    print('  ✓ soundfile')
except Exception as e:
    print('  ✗ soundfile:', e)
try:
    import whisper
    print('  ✓ openai-whisper')
except Exception as e:
    print('  ✗ whisper:', e)
try:
    import stanza
    print('  ✓ stanza', stanza.__version__)
except Exception as e:
    print('  ✗ stanza:', e)
"

echo ""
echo "============================================================"
echo "  SETUP COMPLETE!"
echo ""
echo "  To run the demo:"
echo "  1. source venv/bin/activate"
echo "  2. python run_demo.py"
echo ""
echo "  To add your Hindi audio file:"
echo "  1. Place .wav file in:  data/audio/"
echo "  2. Open config.py and set AUDIO_FILE"
echo "============================================================"
echo ""
