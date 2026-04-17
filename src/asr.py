# ============================================================
# src/asr.py — Step 1: Automatic Speech Recognition
# Uses faster-whisper (installed with WhisperX) for stable ASR
# faster-whisper is more efficient than openai-whisper and
# works correctly with torch 2.8.0+
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")


def run_asr(audio_path, model_name="small", language="hi"):
    """
    Convert Hindi speech audio to text using faster-whisper.

    Parameters
    ----------
    audio_path  : str  — path to .wav audio file
    model_name  : str  — model size: tiny/base/small/medium/large
    language    : str  — language code, 'hi' for Hindi

    Returns
    -------
    transcript  : str  — transcribed text
    words       : list — word-level tokens (if available)
    success     : bool — True if real ASR ran, False if fallback
    """
    print("\n" + "=" * 60)
    print("  STEP 1 — AUTOMATIC SPEECH RECOGNITION (ASR)")
    print("=" * 60)

    # ── Validate audio file ──────────────────────────────────
    if not audio_path or not os.path.exists(audio_path):
        print(f"  [INFO] No audio file found at: {audio_path}")
        print("  [INFO] Using placeholder sentence for demo.")
        return None, [], False

    # ── Try faster-whisper (comes with WhisperX) ─────────────
    try:
        from faster_whisper import WhisperModel

        print(f"  Loading faster-whisper model: '{model_name}'")
        print("  (CPU mode — stable with torch 2.8.0)")

        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="float32"
        )

        print(f"  Transcribing: {os.path.basename(audio_path)}")

        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
        )

        # Collect transcript and word tokens
        transcript_parts = []
        words = []

        for seg in segments:
            transcript_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word" : w.word.strip(),
                        "start": round(w.start, 3),
                        "end"  : round(w.end,   3),
                    })

        transcript = " ".join(transcript_parts).strip()

        print(f"\n  Transcript     : {transcript}")
        print(f"  Language       : {info.language} "
              f"(confidence: {info.language_probability:.2f})")
        print(f"  Word tokens    : {len(words)} words detected")
        print("  ✓ ASR complete")

        return transcript, words, True

    except ImportError:
        print("  [INFO] faster-whisper not found — trying openai-whisper")
        return _try_openai_whisper(audio_path, model_name, language)

    except Exception as e:
        print(f"  [WARNING] faster-whisper failed: {type(e).__name__}: {e}")
        print("  Trying openai-whisper fallback...")
        return _try_openai_whisper(audio_path, model_name, language)


def _try_openai_whisper(audio_path, model_name, language):
    """Fallback to openai-whisper if faster-whisper fails."""
    try:
        import whisper
        print(f"  Loading openai-whisper model: '{model_name}'")
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, language=language)
        transcript = result["text"].strip()

        words = []
        if "segments" in result:
            for seg in result["segments"]:
                if "words" in seg:
                    words.extend(seg["words"])

        print(f"  Transcript : {transcript}")
        print(f"  Words      : {len(transcript.split())} detected")
        print("  ✓ ASR complete (openai-whisper)")
        return transcript, words, True

    except Exception as e:
        print(f"  [WARNING] openai-whisper also failed: {e}")
        print("  [FALLBACK] Using placeholder sentence.")
        return None, [], False


def get_transcript_words(transcript, whisper_words):
    """
    Return clean list of word strings from transcript.
    Uses word tokens if available, otherwise splits on spaces.
    """
    if whisper_words:
        words = []
        for w in whisper_words:
            # Handle both dict and object formats
            if isinstance(w, dict):
                word = w.get("word", "").strip()
            else:
                word = getattr(w, "word", "").strip()
            if word:
                words.append(word)
        return words
    elif transcript:
        return [w for w in transcript.split() if w.strip()]
    return []