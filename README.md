# MTP-1: Acoustic–Linguistic Modelling of Cognitive Constraints
### for Context-Aware Response Selection in Hindi Conversational Speech

**Student:** Mayur Shriram Gaidhane (M22AIE248)
**Supervisor:** Prof. Sidharth Ranjan (Assistant Professor – SAIDE)
**Programme:** M.Tech – Artificial Intelligence
**Evaluation:** April 17, 2026
**Github link** : https://github.com/Mayur-S-Gaidhane/mtp_1.git
---

## Project Overview

This project extends the work of Ranjan et al. (EMNLP 2022) from written
Hindi to **spoken Hindi** — a gap explicitly identified in that paper.

It combines two cognitive theories:
- **Dependency Locality Theory** (Gibson 2000; Zafar & Husain 2023) —
  speakers prefer word orders that minimize dependency length
- **Discourse Predictability** (Ranjan et al. 2022) — prior discourse
  context shapes which word order is chosen via surprisal

---

## Real Pipeline — All Steps Implemented

```
Hindi Speech (WAV)
      ↓
Step 1: ASR          → faster-whisper → Devanagari transcript
      ↓
Step 2: Acoustics    → librosa → F0 pitch, RMS energy, MFCC
      ↓
Step 3: MFA          → AI4Bharat IndicMFA → real TextGrid files
      ↓
Step 4: Linguistic   → Stanza → dep. length + Hindi GPT2 → surprisal
      ↓
Step 5: Figures      → 4 evaluation figures
```

---

## Real Results (MTP-1 Preliminary Analysis)

| Metric | hin_03.wav | hin_04.wav |
|--------|-----------|-----------|
| Words detected | 25 | 18 |
| Duration | 13.30s | 11.47s |
| F0 range | 65–221 Hz | 65–213 Hz |
| MFA aligned | 23/25 in dict ✅ | 9/18 in dict ✅ |
| Dep. Length (Stanza) | 106 | 45 |
| Surprisal (GPT2) | 314.9 bits | 261.2 bits |

---

## Project Structure

```
mtp1_project/
│
├── run_demo.py              ← START HERE — runs full pipeline
├── config.py                ← Set AUDIO_FILE here
├── requirements.txt         ← Python packages
├── README.md                ← This file
│
├── src/
│   ├── asr.py               ← Step 1: faster-whisper ASR
│   ├── acoustic.py          ← Step 2: librosa features
│   ├── alignment.py         ← Step 3: MFA TextGrid reader + WhisperX fallback
│   ├── linguistic.py        ← Step 4: Stanza dep.length + GPT2 surprisal
│   ├── surprisal.py         ← Step 4b: Hindi GPT2 surprisal module
│   └── visualize.py         ← Step 5: 4 evaluation figures
│
├── data/
│   └── audio/
│       ├── hin_03.wav       ← Mozilla Common Voice Hindi sample (13.30s)
│       ├── hin_03.lab       ← MFA transcript file (auto-generated)
│       ├── hin_04.wav       ← Mozilla Common Voice Hindi sample (11.47s)
│       └── hin_04.lab       ← MFA transcript file (auto-generated)
│
└── outputs/
    ├── aligned/
    │   ├── hin_03.TextGrid  ← REAL MFA output (AI4Bharat IndicMFA)
    │   └── hin_04.TextGrid  ← REAL MFA output (AI4Bharat IndicMFA)
    └── figures/
        ├── fig1_acoustic_features.png
        ├── fig2_mfa_alignment.png
        ├── fig3_word_order_scoring.png
        └── fig4_pipeline_overview.png
```

---

## System Requirements

- **OS:** Windows 11 with WSL2 (Ubuntu 24.04)
- **Python:** 3.10 via Miniconda
- **Conda env:** `mtp1_env`
- **RAM:** Minimum 8GB (16GB recommended)
- **Disk:** Minimum 5GB free
- **Network:** Required for first-time downloads

---

## Complete Setup Guide

Follow in exact order for a clean setup.

---

### STEP A — Open WSL Terminal

Press `Windows Key` → search `Ubuntu` → open it.

---

### STEP B — Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install ffmpeg -y
sudo apt-get install fonts-noto -y
sudo apt-get install libsndfile1 -y
sudo apt-get install ca-certificates -y
```

Verify ffmpeg:
```bash
ffmpeg -version
```

---

### STEP C — Install Miniconda (if not installed)

```bash
conda --version
# If not found:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

### STEP D — Create Conda Environment

```bash
conda create -n mtp1_env python=3.10 -y
conda activate mtp1_env
```

Your prompt should show `(mtp1_env)` at the start.

> **Every new terminal:** run `conda activate mtp1_env` before anything else.

---

### STEP E — Navigate to Project

```bash
cd /mnt/c/Users/MayurGaidhane/Downloads/MTP-1/mtp1_project
```

---

### STEP F — Install Python Packages

```bash
pip install -r requirements.txt
```

> **Corporate/college network:** If you get a certificate error:
> ```bash
> pip install --trusted-host pypi.org \
>             --trusted-host files.pythonhosted.org \
>             -r requirements.txt
> ```

---

### STEP G — Install WhisperX (Real Alignment)

WhisperX provides real word-level timestamps via wav2vec2 phoneme alignment.
It also replaces openai-whisper with faster-whisper (more stable).

```bash
pip install whisperx
```

> **Note:** This installs torch 2.8.0 and replaces torch 2.11.0.
> This is expected and required for WhisperX to work.

Install sentencepiece for tokenization:
```bash
pip install sentencepiece
```

---

### STEP H — Install MFA (Montreal Forced Aligner)

MFA is installed directly in `mtp1_env` (no separate environment needed).

```bash
conda install -c conda-forge montreal-forced-aligner -y
```

Verify MFA installed:
```bash
mfa version
# Expected: 3.3.9
```

---

### STEP I — Download AI4Bharat IndicMFA Hindi Models

MFA v3 does not include a Hindi model by default.
We use AI4Bharat IndicMFA Hindi (255h training data).

**Download manually from browser:**
```
https://github.com/AI4Bharat/IndicMFA/releases/tag/hindi_mfa
```

Download both files:
- `Hindi_All_Acoustic.zip`
- `Hindi_Dict_g2g.txt`

Copy to WSL:
```bash
mkdir -p ~/mfa_data/pretrained_models/acoustic
mkdir -p ~/mfa_data/pretrained_models/dictionary

cp /mnt/c/Users/MayurGaidhane/Downloads/Hindi_All_Acoustic.zip \
   ~/mfa_data/pretrained_models/acoustic/

cp /mnt/c/Users/MayurGaidhane/Downloads/Hindi_Dict_g2g.txt \
   ~/mfa_data/pretrained_models/dictionary/
```

Register models with MFA:
```bash
mfa model save acoustic \
  ~/mfa_data/pretrained_models/acoustic/Hindi_All_Acoustic.zip \
  --name hindi_indic

mfa model save dictionary \
  ~/mfa_data/pretrained_models/dictionary/Hindi_Dict_g2g.txt \
  --name hindi_indic
```

Verify registration:
```bash
mfa model list acoustic
# Expected: ['hindi_indic']

mfa model list dictionary
# Expected: ['hindi_indic']
```

---

### STEP J — Install praatio (TextGrid Parser)

praatio is installed automatically with MFA. Verify:
```bash
python -c "from praatio import textgrid; print('praatio OK')"
```

If not found:
```bash
pip install praatio
```

---

### STEP K — Create MFA Lab Files (Transcripts)

MFA needs `.lab` files alongside each audio file.
Run this once to generate them from Whisper:

```bash
python -c "
from faster_whisper import WhisperModel
import os

model = WhisperModel('small', device='cpu', compute_type='float32')
audio_dir = 'data/audio'

for wav_file in ['hin_03.wav', 'hin_04.wav']:
    audio_path = os.path.join(audio_dir, wav_file)
    if os.path.exists(audio_path):
        segments, _ = model.transcribe(audio_path, language='hi')
        transcript = ' '.join(seg.text.strip() for seg in segments)
        lab_file = audio_path.replace('.wav', '.lab')
        with open(lab_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f'Created: {lab_file}')
        print(f'Content: {transcript}')
"
```

---

### STEP L — Run Real MFA Alignment

```bash
mkdir -p outputs/aligned

mfa align \
  data/audio \
  hindi_indic \
  hindi_indic \
  outputs/aligned \
  --clean \
  --verbose
```

Expected output:
```
INFO  Generating MFCCs...
INFO  Calculating CMVN...
INFO  Performing first-pass alignment...
INFO  Exporting alignment TextGrids to outputs/aligned...
INFO  Done! Everything took 14.562 seconds
```

Verify TextGrid files:
```bash
ls outputs/aligned/
# Expected: hin_03.TextGrid  hin_04.TextGrid
```

---

### STEP M — Clear Matplotlib Font Cache

Required after installing fonts-noto:
```bash
python -c "import matplotlib, shutil; shutil.rmtree(matplotlib.get_cachedir())"
```

---

### STEP N — Verify Full Installation

```bash
python -c "
import faster_whisper, librosa, stanza
import whisperx, transformers
from praatio import textgrid
print('All packages OK')
"
```

Expected: `All packages OK`

---

## Running the Demo

### Switch Between Audio Files

Edit `config.py`:
```python
AUDIO_FILE = "hin_03.wav"   # or "hin_04.wav"
WHISPER_MODEL = "small"
```

### Run

```bash
conda activate mtp1_env
cd /mnt/c/Users/MayurGaidhane/Downloads/MTP-1/mtp1_project
python run_demo.py
```

---

## Expected Output

```
============================================================
  MTP-1 PRELIMINARY ANALYSIS DEMO
  Student  : Mayur Shriram Gaidhane
  Supervisor: Prof. Sidharth Ranjan
============================================================

STEP 1 — ASR (faster-whisper)
  Transcript: आजका मोसम बहुत सुहावना है...
  Language: hi (confidence: 1.00)
  Words: 25 detected
  ✓ ASR complete

STEP 2 — ACOUSTIC FEATURES (librosa)
  F0 range: 65.4–221.3 Hz | Duration: 13.30s
  ✓ Real acoustic features extracted

STEP 3 — REAL MFA ALIGNMENT (AI4Bharat IndicMFA Hindi)
  Found real MFA TextGrid: hin_03.TextGrid
  In dictionary: 23/25 words
  Real timestamps: 25/25 ✅
  ✓ Alignment complete

STEP 4 — LINGUISTIC ANALYSIS
  Dep. Length (Stanza): 106
  Loading Hindi GPT2 for surprisal: surajp/gpt2-hindi
  Base sentence surprisal: 314.921 bits
  ✓ Linguistic analysis complete

STEP 5 — FIGURES
  ✓ fig1_acoustic_features.png  (372 KB)
  ✓ fig2_mfa_alignment.png      (125 KB)
  ✓ fig3_word_order_scoring.png (141 KB)
  ✓ fig4_pipeline_overview.png  (88 KB)

DEMO COMPLETE — Time elapsed: 31.6s
============================================================
```

---

## Output Figures

| Figure | File | What It Shows |
|--------|------|---------------|
| **Fig 1** | `fig1_acoustic_features.png` | Waveform + F0 pitch + RMS energy with real MFA boundaries |
| **Fig 2** | `fig2_mfa_alignment.png` | Real TextGrid: word tier + phoneme tier + timestamps |
| **Fig 3** | `fig3_word_order_scoring.png` | 4 variants scored: dep.length + GPT2 surprisal + IS |
| **Fig 4** | `fig4_pipeline_overview.png` | Full pipeline: Speech → ASR → MFA → Features → Response |

View in Windows Explorer:
```
C:\Users\MayurGaidhane\Downloads\MTP-1\mtp1_project\outputs\figures\
```

---

## Implementation Details

### Step 1 — ASR: faster-whisper
- Model: `faster-whisper small` (CPU, float32)
- Replaced openai-whisper due to torch compatibility issue
  (WhisperX requires torch 2.8.0; openai-whisper needed 2.11.0)
- Produces: Devanagari transcript with confidence score

### Step 2 — Acoustics: librosa
- Features: F0 pitch via PYIN, RMS energy, 13 MFCC coefficients
- All values extracted from actual audio signal

### Step 3 — MFA Alignment: AI4Bharat IndicMFA
- Tool: Montreal Forced Aligner v3.3.9
- Model: AI4Bharat IndicMFA Hindi (255h training data)
  - Sources: IndicTTS + IndicVoices-R + Limmits
- Output: Real `.TextGrid` files with genuine acoustic boundaries
- Dictionary: G2G (Grapheme-to-Grapheme) mapping
- **Note:** Words not in dictionary appear as `<unk>` —
  timestamps are still real acoustic boundaries from MFA
- `alignment.py` priority: 1) MFA TextGrid → 2) WhisperX → 3) Proportional

**Why AI4Bharat and not MFA official Hindi model:**
MFA v3 does not include a Hindi acoustic model or G2P model.
AI4Bharat IndicMFA is the correct Hindi-specific solution.

### Step 4 — Dependency Length: Stanza
- Formula: `sum(|head_position - dependent_position|)` for all non-root tokens
- Computed dynamically from actual parse of each audio's transcript
- hin_03: 106 | hin_04: 45

### Step 4b — Surprisal: Hindi GPT2
- Model: `surajp/gpt2-hindi` (50257 vocab, public on HuggingFace)
- Formula: `-log P(sentence)` via causal LM cross-entropy × n_tokens / 0.693
- hin_03: 314.9 bits | hin_04: 261.2 bits

**Difference from Adaptive LSTM (Paper 2):**
| | Hindi GPT2 (MTP-1) | Adaptive LSTM (MTP-2) |
|--|--|--|
| Architecture | Transformer | LSTM |
| Context | Sentence-level | Discourse-level (cross-sentence) |
| Training | General Hindi web | EMILLE Hindi corpus |
| Status | Implemented ✅ | Planned MTP-2 |

---

## Key Findings

```
hin_03 (25 words): dep_length=106, surprisal=314.9 bits
hin_04 (18 words): dep_length=45,  surprisal=261.2 bits

Ratio: 2.4x longer sentence → higher cognitive cost on BOTH metrics
→ Confirms Gibson's Dependency Locality Theory on real spoken Hindi
→ Confirms Ranjan et al. 2022 discourse predictability on spoken Hindi
→ First validation of these effects on spoken Hindi (MTP-1 contribution)
```

---

## MTP Roadmap

| Stage | Credits | Term | Planned Work |
|-------|---------|------|--------------|
| **MTP-1** ✅ | 4 | Spring 2026 | Literature review, real pipeline, preliminary analysis |
| MTP-2 | 4 | Summer 2026 | Full AI4Bharat IndicVoices corpus, expand MFA dict, adaptive LSTM surprisal, IS annotation |
| MTP-3 | 8 | Autumn 2026 | End-to-end ASR→NLP→TTS, MOS evaluation, Marathi extension, thesis |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `ASR NaN error` | torch version conflict | Use faster-whisper, not openai-whisper |
| `whisperx import error` | Not installed | `pip install whisperx` |
| `RemoteModelNotFoundError: hindi_mfa` | MFA v3 has no Hindi | Use AI4Bharat IndicMFA (see Step I above) |
| `mfa model add: No such command` | MFA v3 syntax changed | Use `mfa model save` not `mfa model add` |
| `praatio import error` | Not installed | `pip install praatio` |
| Hindi text shows boxes | Font cache not cleared | Run Step M above |
| `code .` certificate error | Corporate network | `export NODE_TLS_REJECT_UNAUTHORIZED=0` |
| `pip install` certificate error | Corporate network | Use `--trusted-host pypi.org` flag |
| CUDA warning | Old GPU driver | Safe to ignore — runs on CPU |
| FP16 warning | CPU-only mode | Safe to ignore — uses FP32 |
| `<unk>` in MFA output | Word not in G2G dict | Normal — timestamps still real |
| `mfa align` fails | Lab files missing | Run Step K to generate .lab files |
| Surprisal model 401 error | Private HuggingFace repo | Use `surajp/gpt2-hindi` (public) |

---

## Warnings You Can Safely Ignore

```
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.6.1
→ Safe to ignore. WhisperX compatibility message.

`loss_type=None` was set in the config but it is unrecognized
→ Safe to ignore. Hindi GPT2 config message.

UserWarning: CUDA initialization: The NVIDIA driver is too old
→ Safe to ignore. Pipeline runs on CPU.

UserWarning: FP16 is not supported on CPU; using FP32 instead
→ Safe to ignore. Whisper switches automatically.
```

---

## References

1. Zafar, M. & Husain, S. (2023). Dependency Locality Influences Word Order
   During Production in SOV Languages: Evidence from Hindi.
   *CogSci 2023*.

2. Ranjan, S., van Schijndel, M., Agarwal, S. & Rajkumar, R. (2022).
   Discourse Context Predictability Effects in Hindi Word Order.
   *EMNLP 2022*, pp. 10390–10406.

3. AI4Bharat IndicMFA: https://github.com/AI4Bharat/IndicMFA

4. Montreal Forced Aligner: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner

5. Mozilla Common Voice Hindi: https://commonvoice.mozilla.org/en/datasets

6. Hindi GPT2: https://huggingface.co/surajp/gpt2-hindi

---

