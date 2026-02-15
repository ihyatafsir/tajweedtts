#!/usr/bin/env python3
"""
Single Surah Processing Script
Runs CTC forced alignment + TajweedSST physics validation for a given surah.
Called by batch_render_subs.py

Usage:
    python process_single_surah.py --surah 36 --audio path/to/audio.wav --output path/to/timing.json
"""
import argparse
import sys
import json
import torch
import librosa
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions, 
    preprocess_text, get_alignments, get_spans, postprocess_results
)
from src.tajweed_parser import TajweedParser, TajweedType, PhysicsCheck
from src.physics_validator import PhysicsValidator, ValidationStatus
from src.duration_model import DurationModel

# Config
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/tajweedtts")
VERSES_PATH = PROJECT_ROOT / "dependencies/verses_v4.json" 
# Note: Path above might need adjustment based on where verses file actually is.
# user's prev scripts pointed to PROJECT_ROOT / "public/data/verses_v4.json" (inside MahQuranApp?)
# I will try to locate verses_v4.json dynamically or use the one from MahQuranApp path

MAH_QURAN_APP_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

def load_quran_text(surah_num):
    if not MAH_QURAN_APP_PATH.exists():
        raise FileNotFoundError(f"Verses file not found at {MAH_QURAN_APP_PATH}")
    
    with open(MAH_QURAN_APP_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    
    verses = all_verses.get(str(surah_num), [])
    # Return both the full joined text (for CTC) and per-verse info (for ayah tracking)
    verse_info = []
    for v in verses:
        text = v.get('text', '')
        ayah = v.get('verse', v.get('ayah', len(verse_info) + 1))
        verse_info.append((ayah, text))
    full_text = ' '.join(text for _, text in verse_info)
    return full_text, verse_info

def run_ctc_alignment(text, audio_path):
    print(f"  [CTC] Loading model on {DEVICE}...", flush=True)
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    
    print(f"  [CTC] Loading audio from {audio_path}...", flush=True)
    # alignment_model.dtype is likely float16 or float32
    audio_waveform = load_audio(str(audio_path), alignment_model.dtype, alignment_model.device)
    
    print("  [CTC] Generating emissions...", flush=True)
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=BATCH_SIZE
    )
    
    print("  [CTC] Preprocessing text...", flush=True)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language="ara")
    
    print("  [CTC] Aligning...", flush=True)
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer,
    )
    
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    # Cleanup
    del alignment_model
    torch.cuda.empty_cache()
    
    return word_timestamps

def convert_to_char_timings(word_timestamps, verse_info):
    """Convert word timestamps to char-level timings with ayah tracking.
    
    verse_info: list of (ayah_num, verse_text) tuples from load_quran_text.
    We build a map of character positions -> ayah numbers from the original text.
    """
    # Build char->ayah map from the original verse texts
    # The full text is verses joined by spaces
    char_ayah_map = []
    for ayah_num, verse_text in verse_info:
        for ch in verse_text:
            char_ayah_map.append(ayah_num)
        char_ayah_map.append(0)  # space between verses
    
    char_timings = []
    word_idx = 0
    source_pos = 0  # tracks position in original full text
    
    for wt in word_timestamps:
        word = wt['text']
        start = wt['start']
        end = wt['end']
        duration = end - start
        char_dur = duration / len(word) if word else 0
        
        has_chars = False
        for i, char in enumerate(word):
            if not char.isspace():
                has_chars = True
                # Find ayah from map
                ayah = 1
                if source_pos + i < len(char_ayah_map):
                    ayah = char_ayah_map[source_pos + i]
                    if ayah == 0:  # inter-verse space
                        ayah = char_ayah_map[min(source_pos + i + 1, len(char_ayah_map) - 1)]
                
                char_timings.append({
                    "char": char,
                    "start": round(start + i * char_dur, 3),
                    "end": round(start + (i + 1) * char_dur, 3),
                    "idx": len(char_timings),
                    "wordIdx": word_idx,
                    "ayah": ayah
                })
        source_pos += len(word) + 1  # +1 for space
        if has_chars: word_idx += 1
    return char_timings

def apply_physics(char_timings, text, audio_path, surah_num):
    print("  [Physics] Parsing Tajweed rules...", flush=True)
    parser = TajweedParser()
    
    # Load verse text again to parse tajweed per word
    with open(MAH_QURAN_APP_PATH, 'r', encoding='utf-8') as f:
        verses = json.load(f).get(str(surah_num), [])
    
    all_tags = []
    for verse in verses:
        word_tags = parser.parse_text(verse['text'])
        for word_tag in word_tags:
            for letter in word_tag.letters:
                all_tags.append({
                    'char': letter.char_visual,
                    'tajweed_type': letter.tajweed_type,
                    'physics_check': letter.physics_check,
                    'madd_count': letter.madd_count
                })

    print(f"  [Physics] Validating {len(all_tags)} letters...", flush=True)
    audio, sr = librosa.load(str(audio_path), sr=22050)
    physics = PhysicsValidator(sample_rate=sr)
    
    for i, entry in enumerate(char_timings):
        if i < len(all_tags):
            tag = all_tags[i]
            entry['tajweed'] = tag['tajweed_type'].value
            
            if tag['physics_check'] != PhysicsCheck.NONE:
                start, end = entry['start'], entry['end']
                try:
                    check = tag['physics_check']
                    val = None
                    if check == PhysicsCheck.CHECK_RMS_BOUNCE:
                        val = physics.validate_qalqalah(audio, start, end)
                    elif check == PhysicsCheck.CHECK_DURATION:
                        val = physics.validate_madd(audio, start, end, tag['madd_count'] or 2)
                    elif check == PhysicsCheck.CHECK_GHUNNAH:
                        val = physics.validate_ghunnah(audio, start, end)
                    elif check == PhysicsCheck.CHECK_FORMANT_F2:
                        val = physics.validate_tafkheem(audio, start, end)
                    
                    if val:
                        entry['physics'] = val.status.value
                        entry['score'] = float(round(val.score, 2))
                except Exception:
                    pass
    return char_timings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Processing Surah {args.surah}...", flush=True)
    
    full_text, verse_info = load_quran_text(args.surah)
    if not full_text:
        print("Error: No text found for surah", flush=True)
        sys.exit(1)

    # 1. CTC
    word_timestamps = run_ctc_alignment(full_text, args.audio)
    
    # 2. Convert (with ayah tracking)
    char_timings = convert_to_char_timings(word_timestamps, verse_info)
    
    # 3. Physics
    char_timings = apply_physics(char_timings, full_text, args.audio, args.surah)
    
    # 4. Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(char_timings, f, ensure_ascii=False, indent=2)
        
    print(f"Saved alignment to {args.output}", flush=True)

if __name__ == "__main__":
    main()
