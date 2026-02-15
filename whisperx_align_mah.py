#!/usr/bin/env python3
"""
WhisperX Character-Level Alignment for MAH Recitation

Uses WhisperX (wav2vec2-based) to get TRUE character-level timing from audio,
not linear interpolation. WhisperX's align() with return_char_alignments=True
gives acoustic boundaries per character.

Then maps those character timings to Quranic graphemes with ayah/wordIdx.

Usage:
    python whisperx_align_mah.py --surah 80 \
        --audio output/surah_080.wav \
        --output output/timing_080.json
"""
import json
import sys
import argparse
from pathlib import Path

VERSES_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json")
DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def is_diacritic(ch):
    cp = ord(ch)
    return ch in DIACRITICS or (0x064B <= cp <= 0x0652) or (0x0610 <= cp <= 0x061A)


def split_into_graphemes(text):
    """Exact same logic as App.tsx splitIntoGraphemes"""
    graphemes = []
    current = ''
    for ch in text:
        if ch == ' ':
            if current:
                graphemes.append(current)
                current = ''
        elif is_diacritic(ch) and current:
            current += ch
        else:
            if current:
                graphemes.append(current)
            current = ch
    if current:
        graphemes.append(current)
    return graphemes


def get_grapheme_list(all_verses, surah_num):
    """Get graphemes with ayah and wordIdx info."""
    verses = all_verses.get(str(surah_num), [])
    grapheme_list = []
    word_idx = 0
    for v in verses:
        ayah = v.get('ayah', v.get('verse', 0))
        words = v['text'].split()
        for word in words:
            for g in split_into_graphemes(word):
                grapheme_list.append({
                    'char': g,
                    'ayah': ayah,
                    'wordIdx': word_idx,
                })
            word_idx += 1
    return grapheme_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import torch
    # Fix PyTorch 2.6+ weights_only default breaking speechbrain
    import omegaconf
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ])
    import whisperx

    device = args.device
    compute_type = "float16" if device == "cuda" else "float32"
    
    print(f"[1] Loading WhisperX model (device={device})...", flush=True)
    model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)
    
    print(f"[2] Transcribing {args.audio}...", flush=True)
    audio = whisperx.load_audio(args.audio)
    result = model.transcribe(audio, batch_size=4, language="ar")
    
    print(f"[3] Loading alignment model for Arabic...", flush=True)
    align_model, align_metadata = whisperx.load_align_model(
        language_code="ar", device=device
    )
    
    print(f"[4] Aligning with character-level precision...", flush=True)
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=True  # KEY: get per-character acoustic timing
    )
    
    # Extract character-level timings from WhisperX
    char_timings = []
    for segment in aligned["segments"]:
        for char_data in segment.get("chars", []):
            if char_data.get("char", "").strip():
                char_timings.append({
                    'char': char_data['char'],
                    'start': char_data.get('start', 0),
                    'end': char_data.get('end', 0),
                })
        # Also collect word-level for fallback
    
    # Also get word-level for reference
    word_timings = []
    for segment in aligned["segments"]:
        for word_data in segment.get("words", []):
            word_timings.append({
                'word': word_data.get('word', ''),
                'start': word_data.get('start', 0),
                'end': word_data.get('end', 0),
            })
    
    print(f"    Got {len(char_timings)} char timings, {len(word_timings)} word timings", flush=True)
    
    # Load verse data for ayah/wordIdx mapping
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    grapheme_list = get_grapheme_list(all_verses, args.surah)
    print(f"[5] Mapping {len(char_timings)} chars to {len(grapheme_list)} graphemes...", flush=True)
    
    # Map character timings to graphemes
    # Strategy: accumulate char timings into graphemes (base + diacritics)
    enriched = []
    ci = 0  # char timing index
    for gi, ginfo in enumerate(grapheme_list):
        g = ginfo['char']
        s, e = None, None
        # Each grapheme consumes one or more raw characters
        base_count = sum(1 for ch in g if not is_diacritic(ch))
        consume = max(1, base_count)
        
        for _ in range(consume):
            if ci < len(char_timings):
                if s is None:
                    s = char_timings[ci]['start']
                e = char_timings[ci]['end']
                ci += 1
        
        if s is None:
            s = enriched[-1]['end'] if enriched else 0
            e = s + 0.05
        
        enriched.append({
            'idx': gi,
            'char': g,
            'ayah': ginfo['ayah'],
            'wordIdx': ginfo['wordIdx'],
            'start': round(s, 4),
            'end': round(e, 4),
            'duration': round(e - s, 4),
            'weight': 1.0,
        })
    
    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    
    ayahs = set(e['ayah'] for e in enriched)
    print(f"Saved {len(enriched)} entries ({len(ayahs)} ayahs) to {args.output}", flush=True)
    print(f"Time range: {enriched[0]['start']:.2f}s - {enriched[-1]['end']:.2f}s", flush=True)
    for e in enriched[:3]:
        print(f"  {e}")

    # Cleanup
    del model, align_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
