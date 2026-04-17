# ============================================================
# src/acoustic.py — Step 2: Acoustic Feature Extraction
# Extracts pitch (F0), energy, MFCCs from Hindi speech audio
# Falls back to synthetic features if librosa not available
# ============================================================

import numpy as np


# ── Public entry point ───────────────────────────────────────

def extract_features(audio_path, sample_rate=16000):
    """
    Extract acoustic features from audio file.

    Returns a dictionary with keys:
      y, sr, duration, f0, f0_times, voiced_flag,
      rms, rms_times, mfcc, synthetic (bool)
    """
    print("\n" + "=" * 60)
    print("  STEP 2 — ACOUSTIC FEATURE EXTRACTION")
    print("=" * 60)

    if audio_path:
        features = _extract_real(audio_path, sample_rate)
    else:
        print("  [INFO] No audio provided — using synthetic demo features.")
        features = _synthetic_features()

    _print_summary(features)
    return features


# ── Real extraction using librosa ───────────────────────────

def _extract_real(audio_path, sample_rate):
    try:
        import librosa
        import os
        print(f"  Loading: {os.path.basename(audio_path)}")
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  Duration : {duration:.2f}s  |  Sample rate: {sr} Hz")

        # ── Pitch (F0) via PYIN algorithm ───────────────────
        print("  Extracting pitch (F0)...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        f0_times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=256
        )

        # ── RMS Energy ──────────────────────────────────────
        print("  Extracting RMS energy...")
        rms = librosa.feature.rms(
            y=y, frame_length=1024, hop_length=256
        )[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=256
        )

        # ── MFCCs (13 coefficients) ──────────────────────────
        print("  Extracting MFCCs...")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        print("  ✓ Real acoustic features extracted")
        return {
            "y"          : y,
            "sr"         : sr,
            "duration"   : duration,
            "f0"         : f0,
            "f0_times"   : f0_times,
            "voiced_flag": voiced_flag,
            "rms"        : rms,
            "rms_times"  : rms_times,
            "mfcc"       : mfcc,
            "synthetic"  : False,
        }

    except ImportError:
        print("  [WARNING] librosa not installed.")
        print("  Install with: pip install librosa soundfile")
        print("  [FALLBACK] Generating synthetic features.")
        return _synthetic_features()

    except Exception as e:
        print(f"  [WARNING] Audio extraction failed: {e}")
        print("  [FALLBACK] Generating synthetic features.")
        return _synthetic_features()


# ── Synthetic fallback ───────────────────────────────────────

def _synthetic_features(duration=2.5, sr=16000):
    """
    Generate realistic synthetic acoustic features for demo.
    Produces a falling F0 (natural Hindi intonation pattern)
    and energy dips at word boundaries.
    """
    n_samples  = int(sr * duration)
    t_wave     = np.linspace(0, duration, n_samples)

    # Waveform — simple sine with amplitude envelope
    envelope   = np.exp(-0.5 * t_wave / duration)
    y          = 0.05 * envelope * np.sin(2 * np.pi * 150 * t_wave)
    y         += np.random.normal(0, 0.002, n_samples)

    # F0 — falling intonation (160 → 120 Hz) with micro-variation
    n_frames   = 200
    f0_times   = np.linspace(0, duration, n_frames)
    f0         = 160 - 40 * (f0_times / duration)
    f0        += 8 * np.sin(2 * np.pi * 2.5 * f0_times / duration)
    f0        += np.random.normal(0, 2, n_frames)
    voiced_flag = np.ones(n_frames, dtype=bool)
    # Unvoice the last 5% (sentence-final)
    voiced_flag[int(0.95 * n_frames):] = False
    f0[~voiced_flag]                    = np.nan

    # RMS — energy dips at word boundaries (6 words)
    n_rms      = 300
    rms_times  = np.linspace(0, duration, n_rms)
    rms        = 0.055 + 0.025 * np.sin(2 * np.pi * 2.5 * rms_times / duration)
    rms       += np.random.normal(0, 0.003, n_rms)
    rms        = np.abs(rms)
    # Add dips at approximate word boundaries
    for boundary_frac in [0.17, 0.33, 0.50, 0.67, 0.83]:
        idx = int(boundary_frac * n_rms)
        rms[max(0, idx-4): idx+4] *= 0.25

    # MFCC — random coefficients (shape: 13 x 100)
    mfcc = np.random.randn(13, 100) * 10
    # Add some structure to MFCC 1 (energy-correlated)
    mfcc[0, :] = 20 + 10 * np.sin(np.linspace(0, np.pi, 100))

    return {
        "y"          : y,
        "sr"         : sr,
        "duration"   : duration,
        "f0"         : f0,
        "f0_times"   : f0_times,
        "voiced_flag": voiced_flag,
        "rms"        : rms,
        "rms_times"  : rms_times,
        "mfcc"       : mfcc,
        "synthetic"  : True,
    }


# ── Summary printer ──────────────────────────────────────────

def _print_summary(features):
    f0_voiced = features["f0"][features["voiced_flag"]]
    valid_f0  = f0_voiced[~np.isnan(f0_voiced)]
    if len(valid_f0) > 0:
        print(f"  F0 range   : {np.nanmin(valid_f0):.1f} – "
              f"{np.nanmax(valid_f0):.1f} Hz")
    print(f"  RMS mean   : {np.mean(features['rms']):.4f}")
    print(f"  Duration   : {features['duration']:.2f}s")
    if features["synthetic"]:
        print("  [NOTE] Using synthetic features — "
              "add real audio in config.py for actual analysis")
