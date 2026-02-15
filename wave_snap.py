#!/usr/bin/env python3
"""
Wave Analysis Snap — Sub-millisecond onset refinement for Viterbi timing.

Takes 20ms-quantized Viterbi timing and snaps each character boundary to the
exact acoustic event using librosa onset detection, ZCR analysis, and spectral
centroid tracking.

Usage:
    python wave_snap.py \
        --timing output/timing_080_viterbi.json \
        --audio output/surah_080.wav \
        --output output/timing_080_snapped.json
"""
import json
import argparse
import numpy as np
import librosa
import soundfile as sf


# ─── Arabic letter classification for onset detection ───

# Plosives / Qalqalah: sudden energy burst
PLOSIVES = set('قطبدكتجضظء')

# Fricatives: high zero-crossing rate (breath/static)
FRICATIVES = set('سشفصزثذهخغعح')

# Nasals / Liquids: smooth spectral transition
NASALS_LIQUIDS = set('منلرو')

# Long vowels / Madd carriers: stable formant energy
MADD_CARRIERS = set('اوي')

# Diacritics (not base letters)
DIACRITICS_SET = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def classify_letter(char):
    """Classify the base letter of a grapheme for onset detection method."""
    base = ''.join(c for c in char if c not in DIACRITICS_SET)
    if not base:
        return 'default'
    first = base[0]
    if first in PLOSIVES:
        return 'plosive'
    elif first in FRICATIVES:
        return 'fricative'
    elif first in NASALS_LIQUIDS:
        return 'nasal'
    elif first in MADD_CARRIERS:
        return 'madd'
    return 'default'


def snap_onset_energy(y, sr, viterbi_time, window_ms=40, hop_length=64):
    """Snap to nearest energy onset within ±window_ms/2 of viterbi_time.
    
    Uses librosa.onset.onset_detect with a fine hop_length for sub-ms resolution.
    Returns the snapped time, or the original if no onset found.
    """
    half_win = window_ms / 2000.0  # seconds
    start_s = max(0, viterbi_time - half_win)
    end_s = min(len(y) / sr, viterbi_time + half_win)
    
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    
    if end_sample - start_sample < hop_length * 4:
        return viterbi_time
    
    segment = y[start_sample:end_sample]
    
    try:
        onsets = librosa.onset.onset_detect(
            y=segment, sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units='samples'
        )
    except Exception:
        return viterbi_time
    
    if len(onsets) == 0:
        return viterbi_time
    
    # Find closest onset to the center (viterbi position)
    center_sample = int(half_win * sr)
    closest = min(onsets, key=lambda o: abs(o - center_sample))
    snapped_time = start_s + closest / sr
    
    return round(snapped_time, 6)


def snap_zcr(y, sr, viterbi_time, window_ms=40, hop_length=64):
    """Snap to ZCR spike for fricatives.
    
    Finds where the zero-crossing rate peaks within the window.
    """
    half_win = window_ms / 2000.0
    start_s = max(0, viterbi_time - half_win)
    end_s = min(len(y) / sr, viterbi_time + half_win)
    
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    
    if end_sample - start_sample < hop_length * 4:
        return viterbi_time
    
    segment = y[start_sample:end_sample]
    
    zcr = librosa.feature.zero_crossing_rate(
        y=segment, frame_length=hop_length * 2, hop_length=hop_length
    )[0]
    
    if len(zcr) == 0:
        return viterbi_time
    
    # Find steepest increase in ZCR (onset of fricative)
    diff = np.diff(zcr)
    if len(diff) == 0:
        return viterbi_time
    
    peak_frame = np.argmax(diff)
    onset_sample = peak_frame * hop_length
    snapped_time = start_s + onset_sample / sr
    
    return round(snapped_time, 6)


def snap_spectral(y, sr, viterbi_time, window_ms=40, hop_length=64):
    """Snap to spectral centroid change for nasals/liquids.
    
    Finds where the spectral centroid shifts most dramatically.
    """
    half_win = window_ms / 2000.0
    start_s = max(0, viterbi_time - half_win)
    end_s = min(len(y) / sr, viterbi_time + half_win)
    
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    
    if end_sample - start_sample < hop_length * 4:
        return viterbi_time
    
    segment = y[start_sample:end_sample]
    
    centroid = librosa.feature.spectral_centroid(
        y=segment, sr=sr, hop_length=hop_length
    )[0]
    
    if len(centroid) < 2:
        return viterbi_time
    
    diff = np.abs(np.diff(centroid))
    peak_frame = np.argmax(diff)
    onset_sample = peak_frame * hop_length
    snapped_time = start_s + onset_sample / sr
    
    return round(snapped_time, 6)


def snap_madd_end(y, sr, viterbi_end, window_ms=80, hop_length=128):
    """Extend Madd end to where formant energy actually drops.
    
    For long vowels, the Viterbi end might cut short. We look for the
    actual energy drop-off point after the Viterbi boundary.
    """
    start_s = viterbi_end
    end_s = min(len(y) / sr, viterbi_end + window_ms / 1000.0)
    
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    
    if end_sample - start_sample < hop_length * 4:
        return viterbi_end
    
    segment = y[start_sample:end_sample]
    
    rms = librosa.feature.rms(y=segment, frame_length=hop_length * 2, hop_length=hop_length)[0]
    
    if len(rms) == 0:
        return viterbi_end
    
    # Find where RMS drops below 30% of its initial value
    threshold = rms[0] * 0.3
    drop_frames = np.where(rms < threshold)[0]
    
    if len(drop_frames) > 0:
        drop_sample = drop_frames[0] * hop_length
        return round(start_s + drop_sample / sr, 6)
    
    return viterbi_end


def main():
    parser = argparse.ArgumentParser(description="Wave Analysis Snap for Viterbi timing")
    parser.add_argument("--timing", required=True, help="Input Viterbi timing JSON")
    parser.add_argument("--audio", required=True, help="Audio WAV file")
    parser.add_argument("--output", required=True, help="Output snapped timing JSON")
    parser.add_argument("--window", type=int, default=40, help="Search window in ms (default: 40)")
    args = parser.parse_args()

    # Load audio
    print("[1] Loading audio...", flush=True)
    y, sr = librosa.load(args.audio, sr=16000, mono=True)
    duration = len(y) / sr
    print(f"    {duration:.1f}s @ {sr}Hz, {len(y)} samples", flush=True)

    # Load timing
    print("[2] Loading Viterbi timing...", flush=True)
    with open(args.timing, 'r', encoding='utf-8') as f:
        timing = json.load(f)
    print(f"    {len(timing)} graphemes", flush=True)

    # Snap each boundary
    print("[3] Snapping boundaries...", flush=True)
    snap_counts = {'plosive': 0, 'fricative': 0, 'nasal': 0, 'madd': 0, 'default': 0, 'unchanged': 0}
    
    hop = 64  # ~4ms at 16kHz → then sample-level = 0.0625ms
    
    snapped = []
    for i, entry in enumerate(timing):
        char = entry['char']
        letter_type = classify_letter(char)
        
        orig_start = entry['start']
        orig_end = entry['end']
        
        # Snap START boundary
        if letter_type == 'plosive':
            new_start = snap_onset_energy(y, sr, orig_start, args.window, hop)
            snap_counts['plosive'] += 1
        elif letter_type == 'fricative':
            new_start = snap_zcr(y, sr, orig_start, args.window, hop)
            snap_counts['fricative'] += 1
        elif letter_type == 'nasal':
            new_start = snap_spectral(y, sr, orig_start, args.window, hop)
            snap_counts['nasal'] += 1
        elif letter_type == 'madd':
            new_start = snap_onset_energy(y, sr, orig_start, args.window, hop)
            snap_counts['madd'] += 1
        else:
            new_start = snap_onset_energy(y, sr, orig_start, args.window, hop)
            snap_counts['default'] += 1
        
        # Snap END for Madd carriers (extend if energy sustains)
        if letter_type == 'madd':
            new_end = snap_madd_end(y, sr, orig_end, window_ms=80, hop_length=128)
        else:
            new_end = orig_end  # End will be set by next char's start
        
        new_entry = dict(entry)
        new_entry['start'] = round(new_start, 6)
        new_entry['end'] = round(new_end, 6)
        new_entry['snap_type'] = letter_type
        snapped.append(new_entry)
    
    # Post-processing: ensure end[i] = start[i+1] for continuity
    # (except across ayah boundaries or where Madd extension applies)
    print("[4] Post-processing continuity...", flush=True)
    for i in range(len(snapped) - 1):
        if snapped[i]['ayah'] == snapped[i + 1]['ayah']:
            # Within same ayah: next char's start becomes this char's end
            snapped[i]['end'] = snapped[i + 1]['start']
    
    # Recompute durations
    for entry in snapped:
        entry['duration'] = round(entry['end'] - entry['start'], 6)
    
    # Fix any negative durations (rare edge case from aggressive snapping)
    neg_count = 0
    for i, entry in enumerate(snapped):
        if entry['duration'] <= 0:
            # Fall back: use midpoint between neighbors
            if i > 0 and i < len(snapped) - 1:
                entry['start'] = snapped[i - 1]['end']
                entry['end'] = snapped[i + 1]['start']
                entry['duration'] = round(entry['end'] - entry['start'], 6)
            if entry['duration'] <= 0:
                entry['duration'] = 0.02  # minimum 20ms
                entry['end'] = entry['start'] + 0.02
            neg_count += 1
    
    # Save
    print("[5] Saving...", flush=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(snapped, f, ensure_ascii=False, indent=2)
    
    durations = [e['duration'] for e in snapped]
    deltas = [abs(snapped[i]['start'] - timing[i]['start']) for i in range(len(timing))]
    
    print(f"\n{'='*60}", flush=True)
    print(f"Saved {len(snapped)} entries to {args.output}", flush=True)
    print(f"Snap counts: {snap_counts}", flush=True)
    print(f"Avg shift: {np.mean(deltas)*1000:.2f}ms, Max: {np.max(deltas)*1000:.2f}ms", flush=True)
    print(f"Durations: min={min(durations)*1000:.2f}ms max={max(durations)*1000:.2f}ms", flush=True)
    print(f"Unique durations: {len(set(round(d, 4) for d in durations))}", flush=True)
    if neg_count:
        print(f"Fixed {neg_count} negative durations", flush=True)
    
    print(f"\nFirst 5:")
    for e in snapped[:5]:
        print(f"  {e['char']:>6s}  {e['snap_type']:>10s}  {e['start']:.4f}-{e['end']:.4f} ({e['duration']*1000:.1f}ms)")
    print(f"Last 3:")
    for e in snapped[-3:]:
        print(f"  {e['char']:>6s}  {e['snap_type']:>10s}  {e['start']:.4f}-{e['end']:.4f} ({e['duration']*1000:.1f}ms)")


if __name__ == "__main__":
    main()
