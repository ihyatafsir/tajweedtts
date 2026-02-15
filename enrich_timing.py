#!/usr/bin/env python3
"""
Enrich MAH timing data with ayah/wordIdx fields from verses_v4.json.

Takes the existing letter_timing_80.json from MahQuranApp (which has char/start/end/idx)
and adds ayah, wordIdx, duration, weight fields needed by render_video_overlay.py.
"""
import json
import sys
from pathlib import Path

VERSES_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json")
DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def is_diacritic(ch):
    cp = ord(ch)
    return ch in DIACRITICS or (0x064B <= cp <= 0x0652) or (0x0610 <= cp <= 0x061A)


def split_into_graphemes(text):
    """Exact same logic as App.tsx splitIntoGraphemes and batch_align_all.py"""
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--surah", type=int, required=True)
    parser.add_argument("--input", type=str, required=True, help="MAH timing JSON from MahQuranApp")
    parser.add_argument("--output", type=str, required=True, help="Enriched output JSON")
    args = parser.parse_args()

    # Load timing
    with open(args.input, 'r', encoding='utf-8') as f:
        timing = json.load(f)
    print(f"Loaded {len(timing)} timing entries from {args.input}")

    # Load verses
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)

    # Build grapheme list with ayah/wordIdx
    grapheme_list = get_grapheme_list(all_verses, args.surah)
    print(f"Built {len(grapheme_list)} graphemes from verses_v4.json")

    if len(timing) != len(grapheme_list):
        print(f"WARNING: timing entries ({len(timing)}) != graphemes ({len(grapheme_list)})")
        print(f"  Will map min({len(timing)}, {len(grapheme_list)}) entries")

    # Merge: keep timing's start/end, add grapheme's ayah/wordIdx
    n = min(len(timing), len(grapheme_list))
    enriched = []
    for i in range(n):
        t = timing[i]
        g = grapheme_list[i]
        start = t['start']
        end = t['end']
        enriched.append({
            'idx': i,
            'char': g['char'],  # Use grapheme char (has proper diacritics grouping)
            'ayah': g['ayah'],
            'wordIdx': g['wordIdx'],
            'start': start,
            'end': end,
            'duration': round(end - start, 4),
            'weight': 1.0,
        })

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # Summary
    ayahs = set(e['ayah'] for e in enriched)
    print(f"Saved {len(enriched)} enriched entries ({len(ayahs)} ayahs) to {args.output}")
    print(f"Time range: {enriched[0]['start']:.2f}s - {enriched[-1]['end']:.2f}s")
    # Show first few
    for e in enriched[:3]:
        print(f"  {e}")


if __name__ == "__main__":
    main()
