#!/usr/bin/env python3
"""
Batch Grapheme Alignment for All Abdul Basit Surahs

Processes surahs 1-90 and 92-114 (skipping 91 which is already done)
Maps original timing to graphemes from verse text.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
ORIG_DIR = PROJECT_ROOT / "public/data/abdul_basit_original"
OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"

# Arabic diacritics
DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def split_graphemes(text: str) -> list[str]:
    """Split Arabic text into graphemes (base letter + following diacritics)"""
    graphemes = []
    current = ''
    
    for ch in text:
        is_diacritic = (ch in DIACRITICS or 
                        (0x064B <= ord(ch) <= 0x0652) or 
                        (0x0610 <= ord(ch) <= 0x061A))
        
        if ch == ' ':
            if current:
                graphemes.append(current)
                current = ''
        elif is_diacritic and current:
            current += ch
        else:
            if current:
                graphemes.append(current)
            current = ch
    
    if current:
        graphemes.append(current)
    
    return graphemes


def get_all_graphemes(verses: list) -> list[dict]:
    """Extract all graphemes from verse text"""
    all_graphemes = []
    word_idx = 0
    
    for verse in verses:
        ayah = verse.get('ayah', 0)
        words = verse.get('words', [])
        
        for word in words:
            arabic = word.get('arabic', '')
            graphemes = split_graphemes(arabic)
            
            for g in graphemes:
                all_graphemes.append({
                    'char': g,
                    'ayah': ayah,
                    'wordIdx': word_idx
                })
            
            word_idx += 1
    
    return all_graphemes


def strip_diacritics(text: str) -> str:
    """Remove diacritics from Arabic text"""
    return ''.join(ch for ch in text if ch not in DIACRITICS and not (0x064B <= ord(ch) <= 0x0652))


def is_standalone_diacritic(char: str) -> bool:
    """Check if char is a standalone diacritic"""
    if len(char) != 1:
        return False
    return char in DIACRITICS or (0x064B <= ord(char) <= 0x0652)


def distribute_timing(graphemes: list[dict], original_timing: list[dict]) -> list[dict]:
    """Map original timing to graphemes by matching base letters"""
    if not original_timing:
        return []
    
    # Filter out standalone diacritics and merge their duration
    filtered_timing = []
    for entry in original_timing:
        char = entry['char']
        if is_standalone_diacritic(char):
            if filtered_timing:
                filtered_timing[-1]['end'] = entry['end']
        else:
            filtered_timing.append(dict(entry))
    
    aligned_timing = []
    orig_idx = 0
    
    for i, g in enumerate(graphemes):
        grapheme_char = g['char']
        base_letter = strip_diacritics(grapheme_char)
        
        # Try to find matching original timing entry
        # STRICTLY FORWARD: only search ahead from current position
        matched = None
        search_end = min(len(filtered_timing), orig_idx + 10)
        
        for j in range(orig_idx, search_end):
            orig_char = filtered_timing[j]['char']
            orig_base = strip_diacritics(orig_char)
            if orig_base == base_letter or orig_char in grapheme_char or base_letter in orig_char:
                matched = filtered_timing[j]
                orig_idx = j + 1
                break
        
        if not matched and orig_idx < len(filtered_timing):
            matched = filtered_timing[orig_idx]
            orig_idx += 1
        
        if matched:
            aligned_timing.append({
                'idx': i,
                'char': grapheme_char,
                'ayah': g['ayah'],
                'start': matched['start'],
                'end': matched['end'],
                'duration': matched.get('duration', matched['end'] - matched['start']),
                'wordIdx': g['wordIdx'],
                'weight': matched.get('weight', 1.0)
            })
        elif aligned_timing:
            # Silent letter: share timing with previous (don't advance time)
            prev = aligned_timing[-1]
            aligned_timing.append({
                'idx': i,
                'char': grapheme_char,
                'ayah': g['ayah'],
                'start': prev['start'],  # Same start as previous
                'end': prev['end'],      # Same end as previous
                'duration': prev['duration'],
                'wordIdx': g['wordIdx'],
                'weight': 0.0,  # Mark as silent
                'silent': True
            })
    
    return aligned_timing


def process_surah(surah_num: int, all_verses: dict) -> dict:
    """Process a single surah"""
    timing_path = ORIG_DIR / f"letter_timing_{surah_num}.json"
    output_path = OUTPUT_DIR / f"letter_timing_{surah_num}.json"
    
    if not timing_path.exists():
        return {'status': 'skip', 'reason': 'no timing file'}
    
    verses = all_verses.get(str(surah_num), [])
    if not verses:
        return {'status': 'skip', 'reason': 'no verses'}
    
    # Get graphemes
    graphemes = get_all_graphemes(verses)
    
    # Load original timing
    with open(timing_path, 'r', encoding='utf-8') as f:
        original_timing = json.load(f)
    
    # Align
    aligned = distribute_timing(graphemes, original_timing)
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)
    
    return {
        'status': 'ok',
        'graphemes': len(graphemes),
        'original': len(original_timing),
        'aligned': len(aligned)
    }


def main():
    print("=" * 60)
    print("Batch Grapheme Alignment: Abdul Basit")
    print("=" * 60)
    
    # Load all verses
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    
    # Surahs to process (1-90, 92-114, skip 91)
    surahs = list(range(1, 91)) + list(range(92, 115))
    
    stats = {'ok': 0, 'skip': 0, 'errors': []}
    
    for surah in surahs:
        try:
            result = process_surah(surah, all_verses)
            if result['status'] == 'ok':
                stats['ok'] += 1
                print(f"  {surah:3d}: ✓ {result['graphemes']} graphemes <- {result['original']} original")
            else:
                stats['skip'] += 1
                print(f"  {surah:3d}: - {result['reason']}")
        except Exception as e:
            stats['errors'].append(surah)
            print(f"  {surah:3d}: ✗ {e}")
    
    print()
    print("=" * 60)
    print(f"✓ Processed: {stats['ok']}")
    print(f"- Skipped: {stats['skip']}")
    print(f"✗ Errors: {len(stats['errors'])} {stats['errors'] if stats['errors'] else ''}")
    print("=" * 60)


if __name__ == "__main__":
    main()
