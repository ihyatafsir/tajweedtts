#!/usr/bin/env python3
"""
TajweedSST - Physics Refinement Pipeline for Surah 91

Uses EXISTING timing from MahQuranApp + applies physics refinement.
No WhisperX needed - just physics validation and boundary refinement.

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python3 surah_91_full_pipeline.py
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.tajweed_parser import TajweedParser, TajweedType, PhysicsCheck
from src.physics_validator import PhysicsValidator, ValidationStatus
from src.duration_model import DurationModel, MaddType

import librosa

# Paths
MAHQURAN_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = MAHQURAN_PATH / "public/data/verses_v4.json"
AUDIO_PATH = MAHQURAN_PATH / "public/audio/abdul_basit/surah_091.mp3"
EXISTING_TIMING = MAHQURAN_PATH / "public/data/abdul_basit/letter_timing_91.json"
OUTPUT_TIMING = MAHQURAN_PATH / "public/data/abdul_basit/letter_timing_91_physics.json"


def load_verses():
    """Load Surah 91 verses"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('91', [])


def load_existing_timing():
    """Load existing letter timing"""
    with open(EXISTING_TIMING, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_tajweed_tags(verses):
    """Parse all verses for Tajweed tags"""
    parser = TajweedParser()
    all_tags = []
    
    for verse in verses:
        word_tags = parser.parse_text(verse['text'])
        for word_tag in word_tags:
            for letter in word_tag.letters:
                all_tags.append({
                    'char': letter.char_visual,
                    'phonetic': letter.char_phonetic,
                    'tajweed_type': letter.tajweed_type,
                    'physics_check': letter.physics_check,
                    'madd_count': letter.madd_count,
                    'is_silent': letter.is_silent
                })
    
    return all_tags


def refine_with_physics(timing_data, tags, audio, sr, physics, duration_model):
    """Apply physics refinement to existing timing"""
    refined = []
    stats = {'total': 0, 'validated': 0, 'passed': 0, 'marginal': 0, 'failed': 0}
    
    for i, entry in enumerate(timing_data):
        stats['total'] += 1
        
        # Copy existing data
        result = entry.copy()
        # CRITICAL PRECISION FIX: Times are stored in milliseconds, convert to seconds
        start = entry['start'] / 1000.0
        end = entry['end'] / 1000.0
        
        # Get corresponding Tajweed tag
        if i < len(tags):
            tag = tags[i]
            result['tajweed'] = tag['tajweed_type'].value
            result['phonetic'] = tag['phonetic']
            
            # Run physics validation if needed
            if tag['physics_check'] != PhysicsCheck.NONE:
                stats['validated'] += 1
                
                try:
                    check = tag['physics_check']
                    
                    if check == PhysicsCheck.CHECK_RMS_BOUNCE:
                        val = physics.validate_qalqalah(audio, start, end)
                    elif check == PhysicsCheck.CHECK_DURATION:
                        val = physics.validate_madd(audio, start, end, tag['madd_count'] or 2)
                    elif check == PhysicsCheck.CHECK_GHUNNAH:
                        if tag['tajweed_type'] == TajweedType.IKHFA:
                            val = physics.validate_ikhfa(audio, start, end)
                        elif tag['tajweed_type'] == TajweedType.IQLAB:
                            val = physics.validate_iqlab(audio, start, end)
                        else:
                            val = physics.validate_ghunnah(audio, start, end)
                    elif check == PhysicsCheck.CHECK_FORMANT_F2:
                        val = physics.validate_tafkheem(audio, start, end)
                    else:
                        val = None
                    
                    if val:
                        result['physics'] = val.status.value
                        result['score'] = round(val.score, 2)
                        
                        if val.status == ValidationStatus.PASS:
                            stats['passed'] += 1
                        elif val.status == ValidationStatus.MARGINAL:
                            stats['marginal'] += 1
                        else:
                            stats['failed'] += 1
                        
                except Exception as e:
                    result['error'] = str(e)
            
            # Duration validation for Madd
            if tag['tajweed_type'] in [TajweedType.MADD_ASLI, TajweedType.MADD_WAJIB, TajweedType.MADD_LAZIM]:
                duration = end - start
                madd_map = {
                    TajweedType.MADD_ASLI: MaddType.ASLI,
                    TajweedType.MADD_WAJIB: MaddType.WAJIB,
                    TajweedType.MADD_LAZIM: MaddType.LAZIM
                }
                dur_result = duration_model.validate_duration(
                    duration, 
                    madd_map.get(tag['tajweed_type'], MaddType.ASLI),
                    tag['madd_count'] or 2
                )
                result['harakat'] = round(dur_result.harakat_count, 1)
        
        refined.append(result)
    
    return refined, stats


def main():
    print("=" * 60)
    print("TajweedSST - Physics Refinement: Surah 91")
    print("=" * 60)
    
    # Load existing timing
    print("\n[1] Loading existing timing...")
    timing_data = load_existing_timing()
    print(f"    Entries: {len(timing_data)}")
    
    # Load verses and parse Tajweed
    print("\n[2] Parsing Tajweed rules...")
    verses = load_verses()
    tags = get_tajweed_tags(verses)
    print(f"    Tajweed tags: {len(tags)}")
    
    # Load audio
    print("\n[3] Loading audio...")
    audio, sr = librosa.load(str(AUDIO_PATH), sr=22050)
    print(f"    Duration: {len(audio)/sr:.1f}s")
    
    # Initialize validators
    physics = PhysicsValidator(sample_rate=sr)
    duration_model = DurationModel()
    
    # Calibrate
    vowels = [e['end'] - e['start'] for e in timing_data if 0.05 <= (e['end'] - e['start']) <= 0.15]
    if vowels:
        duration_model.calibrate_from_samples("Abdul_Basit", vowels)
        print(f"    Harakat: {duration_model.calibration.harakat_base_ms:.1f}ms")
    
    # Refine
    print("\n[4] Applying physics refinement...")
    refined, stats = refine_with_physics(timing_data, tags, audio, sr, physics, duration_model)
    
    print(f"\n[5] Statistics:")
    print(f"    Total: {stats['total']}")
    print(f"    Validated: {stats['validated']}")
    print(f"    ✓ Passed: {stats['passed']}")
    print(f"    ~ Marginal: {stats['marginal']}")
    print(f"    ✗ Failed: {stats['failed']}")
    
    if stats['validated'] > 0:
        rate = (stats['passed'] + stats['marginal']) / stats['validated'] * 100
        print(f"    Pass Rate: {rate:.1f}%")
    
    # Save
    print(f"\n[6] Saving to MahQuranApp...")
    with open(OUTPUT_TIMING, 'w', encoding='utf-8') as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {OUTPUT_TIMING}")
    
    # Show sample
    print("\n[7] Sample refined entries:")
    for entry in refined[:5]:
        tj = entry.get('tajweed', 'None')
        ph = entry.get('physics', '-')
        sc = entry.get('score', '-')
        print(f"    {entry['char']}: {tj} | physics={ph} score={sc}")
    
    print("\n" + "=" * 60)
    print("✓ Done! Test in MahQuranApp with:")
    print(f"  letter_timing_91_physics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
