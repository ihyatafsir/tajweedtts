#!/usr/bin/env python3
"""
Viterbi Forced Alignment for Quran Recitation

Uses ctc_forced_aligner's FULL pipeline which internally does:
  1. generate_emissions → wav2vec2 logits (batched, windowed)
  2. preprocess_text → romanized tokens with <star> separators
  3. get_alignments → C++ Viterbi forced alignment
  4. get_spans → per-character span extraction
  5. postprocess_results → word-level grouping

We use steps 1-4 for CHARACTER-level timing, then map to graphemes.

Usage:
    python viterbi_align.py --surah 80 \
        --audio output/surah_080.wav \
        --output output/timing_080.json
"""
import json
import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

VERSES_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json")
DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def is_diacritic(ch):
    cp = ord(ch)
    return ch in DIACRITICS or (0x064B <= cp <= 0x0652) or (0x0610 <= cp <= 0x061A)


def split_into_graphemes(text):
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
    verses = all_verses.get(str(surah_num), [])
    grapheme_list = []
    word_idx = 0
    for v in verses:
        ayah = v.get('ayah', v.get('verse', 0))
        words = v['text'].split()
        for word in words:
            for g in split_into_graphemes(word):
                grapheme_list.append({'char': g, 'ayah': ayah, 'wordIdx': word_idx})
            word_idx += 1
    return grapheme_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = "cpu"

    # Load verses
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    text = ' '.join(v.get('text', '') for v in all_verses.get(str(args.surah), []))
    grapheme_list = get_grapheme_list(all_verses, args.surah)
    print(f"Surah {args.surah}: {len(grapheme_list)} graphemes", flush=True)

    # ── Step 1: Load wav2vec2 model ──
    print("[1] Loading wav2vec2 alignment model...", flush=True)
    model, tokenizer = load_alignment_model(device, dtype=torch.float32)
    print("    Model loaded.", flush=True)

    # ── Step 2: Load audio (using ctc_forced_aligner's load_audio = 1D tensor) ──
    print("[2] Loading audio...", flush=True)
    audio_waveform = load_audio(args.audio, model.dtype, model.device)
    # load_audio returns 1D tensor [N] at 16kHz mono — exactly what generate_emissions wants
    duration_s = audio_waveform.shape[0] / 16000
    print(f"    Shape: {audio_waveform.shape}, Duration: {duration_s:.1f}s", flush=True)

    # ── Step 3: Generate emissions (batched, windowed) ──
    print("[3] Generating emissions...", flush=True)
    emissions, stride = generate_emissions(model, audio_waveform, batch_size=1)
    num_frames = emissions.shape[0]
    print(f"    Emissions: {emissions.shape}, stride={stride}ms", flush=True)

    # ── Step 4: Preprocess text ──
    print("[4] Preprocessing text...", flush=True)
    tokens_starred, text_starred = preprocess_text(text, romanize=True, language="ara")
    print(f"    {len(tokens_starred)} starred tokens", flush=True)
    print(f"    Sample: {tokens_starred[:6]}", flush=True)

    # ── Step 5: Viterbi alignment (C++ forced_align under the hood) ──
    print("[5] Running Viterbi alignment...", flush=True)
    segments, scores, blank_id = get_alignments(
        emissions, tokens_starred, tokenizer  # Pass the tokenizer object!
    )
    print(f"    {len(segments)} segments, blank='{blank_id}'", flush=True)

    # ── Step 6: Get per-word spans ──
    print("[6] Extracting spans...", flush=True)
    spans = get_spans(tokens_starred, segments, blank_id)
    print(f"    {len(spans)} spans", flush=True)

    # ── Step 7: Extract per-character timings from spans ──
    # Each span corresponds to a token in tokens_starred
    # For non-<star> tokens, the span contains sub-segments for each romanized character
    # We use word-level: each non-<star> token = one Arabic word
    # Then split by characters within each word using the span's sub-segments
    print("[7] Extracting character timings...", flush=True)

    # First get word-level timings via postprocess_results
    word_results = postprocess_results(text_starred, spans, stride, scores)
    print(f"    {len(word_results)} words with Viterbi boundaries", flush=True)

    # Collect ALL non-blank segment centers (frame positions) in order
    # These are the Viterbi-detected positions for each romanized character
    all_char_frames = []
    for i, (tok, span_list) in enumerate(zip(tokens_starred, spans)):
        if tok == '<star>':
            continue
        non_blank_segs = [s for s in span_list if s.label != blank_id]
        for seg in non_blank_segs:
            all_char_frames.append(seg.start)  # frame index of detection

    # Convert to time boundaries: each char runs from its detection to the next char's detection
    char_times = []
    for j in range(len(all_char_frames)):
        start_f = all_char_frames[j]
        if j + 1 < len(all_char_frames):
            end_f = all_char_frames[j + 1]
        else:
            # Last char — extend to end of audio
            end_f = start_f + int(0.5 * 1000 / stride)  # ~0.5s
        start_s = start_f * stride / 1000
        end_s = end_f * stride / 1000
        char_times.append({
            'start': round(start_s, 4),
            'end': round(end_s, 4),
        })

    print(f"    {len(char_times)} character-level timings", flush=True)

    # ── Step 8: Map romanized characters to graphemes ──
    print("[8] Mapping to graphemes...", flush=True)
    enriched = []
    ci = 0

    for gi, ginfo in enumerate(grapheme_list):
        g = ginfo['char']
        base_count = sum(1 for ch in g if not is_diacritic(ch))
        if base_count == 0:
            base_count = 1

        s, e = None, None
        for _ in range(base_count):
            if ci < len(char_times):
                ct = char_times[ci]
                if s is None:
                    s = ct['start']
                e = ct['end']
                ci += 1

        if s is None:
            s = enriched[-1]['end'] if enriched else 0.0
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
    durations = [e['duration'] for e in enriched]
    print(f"\n{'='*50}", flush=True)
    print(f"Saved {len(enriched)} entries ({len(ayahs)} ayahs)", flush=True)
    print(f"Time: {enriched[0]['start']:.2f}s - {enriched[-1]['end']:.2f}s", flush=True)
    print(f"Durations: min={min(durations):.4f}s max={max(durations):.4f}s", flush=True)
    print(f"Unique durations: {len(set(round(d, 3) for d in durations))}", flush=True)
    print(f"\nFirst 5:")
    for e in enriched[:5]:
        print(f"  {e}")
    print(f"Last 3:")
    for e in enriched[-3:]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
