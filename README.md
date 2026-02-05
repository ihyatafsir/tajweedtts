# TajweedSST - Quranic Precision Alignment & Tajweed Analysis Tool

A Python-based pipeline that generates letter-level precise timing data for Quran recitations, prevents timing drift, and uses signal processing to validate Tajweed rules.

## Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 1. Tajweed      │───▶│ 2. Hierarchical  │───▶│ 3. Physics &    │
│    Parser       │    │    Alignment     │    │    Validator    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                      │                       │
   Visual Stream          WhisperX              Signal Analysis
   Phonetic Stream           MFA               Tajweed Scoring
   Rule Tags             Normalization              JSON
```

## Tech Stack

- **Text Processing:** Camel-Tools, Regex
- **Alignment:** WhisperX (word-level), MFA (phoneme-level)
- **Acoustic Model:** Wav2Vec2-Large-XLSR-53-Arabic
- **Signal Processing:** Librosa, Parselmouth (Praat)

## Tajweed Rules Covered

| Rule | Arabic | Physics Check |
|------|--------|---------------|
| Qalqalah | قلقلة | RMS Energy bounce (dip→spike) |
| Madd | مد | Duration vs. ROS ratio |
| Ghunnah | غنة | Formant analysis + nasalization |
| Idgham | إدغام | Duration compression |
| Iqlab | إقلاب | Ba→Mim substitution |
| Ikhfa | إخفاء | Partial nasalization |
| Tafkheem | تفخيم | F2 formant depression |
| Tarqeeq | ترقيق | F2 formant elevation |

## Project Structure

```
tajweedsst/
├── src/
│   ├── tajweed_parser.py     # Step 1: Text preprocessing & rule tagging
│   ├── alignment_engine.py   # Step 2: WhisperX + MFA + Normalization
│   ├── physics_validator.py  # Step 3: Signal processing & validation
│   └── pipeline.py           # Main orchestrator
├── data/
│   ├── quran_uthmani.json    # Source text
│   └── audio/                # Recitation files
├── models/                   # Acoustic models
├── output/                   # Generated JSON
└── tests/
```

## Installation

```bash
pip install -r requirements.txt

# Additional setup
pip install camel-tools
pip install whisperx
pip install montreal-forced-aligner
pip install librosa parselmouth-praat
```

## Usage

```python
from src.pipeline import TajweedPipeline

pipeline = TajweedPipeline()
result = pipeline.process(
    audio_path="data/audio/surah_112.mp3",
    surah=112,
    ayah=1
)
```

## Output Format

```json
{
  "surah": 112,
  "ayah": 1,
  "words": [{
    "word_text": "أَحَدٌ",
    "whisper_anchor": { "start": 3.50, "end": 4.10 },
    "phonemes": [{
      "char_visual": "د",
      "char_phonetic": "d",
      "start": 3.85, "end": 4.10,
      "tajweed_type": "Qalqalah_Kubra",
      "physics_analysis": {
        "rms_profile": "dip_then_spike",
        "intensity": 0.95,
        "status": "PASS"
      }
    }]
  }]
}
```
