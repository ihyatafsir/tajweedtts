#!/usr/bin/env python3
"""
Batch Processing Pipeline for Mohammad Ahmad Hassan Recitations

Runs the full TajweedSST pipeline for each MAH recitation:
  1. Viterbi forced alignment (CTC)
  2. Wave-snap onset refinement
  3. Tilawah overlay video rendering

Surah identification is manual (filename → surah number mapping) because
  the downloaded filenames use English transliterations.

Usage:
    python batch_process_mah.py                    # Process all
    python batch_process_mah.py --surah abasa      # Process one
    python batch_process_mah.py --skip-render       # Alignment only
"""
import json
import subprocess
import sys
import argparse
from pathlib import Path

# ── Surah name → number mapping for MAH downloads ──
SURAH_MAP = {
    'abasa':    80,   # عبس
    'fajr':     89,   # الفجر
    'yasin':    36,   # يس
    'waqiah':   56,   # الواقعة
    'ibrahim':  14,   # إبراهيم
    'baqarah':  2,    # البقرة
}

OUTPUT_DIR = Path(__file__).parent / "output"
DOWNLOADS_DIR = Path(__file__).parent / "downloads"

def find_audio(surah_name):
    """Find WAV audio for a surah (extracted by extract_audio.sh)."""
    wav = OUTPUT_DIR / f"mah_{surah_name}.wav"
    if wav.exists():
        return wav
    # Try extracting from video directly
    video = DOWNLOADS_DIR / f"surah_{surah_name}_mohammad_ahmad_hassan.mp4"
    if video.exists():
        print(f"  Extracting audio from {video.name}...")
        subprocess.run([
            'ffmpeg', '-i', str(video),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            str(wav), '-y', '-loglevel', 'warning'
        ], check=True)
        return wav
    return None


def run_viterbi(surah_num, audio_path, output_path):
    """Run Viterbi forced alignment."""
    script = Path(__file__).parent / "viterbi_align.py"
    cmd = [
        sys.executable, str(script),
        '--surah', str(surah_num),
        '--audio', str(audio_path),
        '--output', str(output_path),
    ]
    print(f"  [CTC] Running Viterbi alignment...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Viterbi failed:\n{result.stderr}")
        return False
    # Print last few lines of output
    lines = result.stdout.strip().split('\n')
    for line in lines[-5:]:
        print(f"    {line}")
    return True


def run_wave_snap(surah_num, audio_path, timing_path, output_path):
    """Run wave-snap onset refinement."""
    script = Path(__file__).parent / "wave_snap.py"
    cmd = [
        sys.executable, str(script),
        '--audio', str(audio_path),
        '--timing', str(timing_path),
        '--output', str(output_path),
    ]
    print(f"  [SNAP] Running wave-snap refinement...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Wave-snap failed:\n{result.stderr}")
        return False
    lines = result.stdout.strip().split('\n')
    for line in lines[-3:]:
        print(f"    {line}")
    return True


def run_render(surah_num, audio_path, timing_path, video_bg_path, output_path):
    """Run tilawah overlay renderer."""
    script = Path(__file__).parent / "render_video_overlay.py"
    cmd = [
        sys.executable, str(script),
        '--surah', str(surah_num),
        '--audio', str(audio_path),
        '--timing', str(timing_path),
        '--video', str(video_bg_path),
        '--output', str(output_path),
    ]
    print(f"  [RENDER] Rendering tilawah overlay video...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Render failed:\n{result.stderr}")
        return False
    lines = result.stdout.strip().split('\n')
    for line in lines[-3:]:
        print(f"    {line}")
    return True


def process_surah(surah_name, surah_num, skip_render=False):
    """Full pipeline for one surah."""
    print(f"\n{'='*60}")
    print(f"  Surah {surah_name.title()} (#{surah_num})")
    print(f"{'='*60}")

    # Step 1: Find audio
    audio = find_audio(surah_name)
    if not audio:
        print(f"  SKIP: No audio found for {surah_name}")
        return False

    # Step 2: Viterbi alignment
    timing_ctc = OUTPUT_DIR / f"mah_{surah_name}_ctc.json"
    if timing_ctc.exists():
        print(f"  SKIP: CTC timing already exists: {timing_ctc.name}")
    else:
        if not run_viterbi(surah_num, audio, timing_ctc):
            return False

    # Step 3: Wave-snap refinement
    timing_snapped = OUTPUT_DIR / f"mah_{surah_name}_snapped.json"
    if timing_snapped.exists():
        print(f"  SKIP: Snapped timing already exists: {timing_snapped.name}")
    else:
        if not run_wave_snap(surah_num, audio, timing_ctc, timing_snapped):
            # Fall back to CTC timing if snap fails
            print(f"  WARN: Using CTC timing (snap failed)")
            timing_snapped = timing_ctc

    # Step 4: Render tilawah video
    if skip_render:
        print(f"  SKIP: Render skipped (--skip-render)")
    else:
        video_bg = DOWNLOADS_DIR / f"surah_{surah_name}_mohammad_ahmad_hassan.mp4"
        video_out = OUTPUT_DIR / f"mah_{surah_name}_tilawah.mp4"
        if video_out.exists():
            print(f"  SKIP: Tilawah video already exists: {video_out.name}")
        elif video_bg.exists():
            final_timing = timing_snapped if timing_snapped.exists() else timing_ctc
            run_render(surah_num, audio, final_timing, video_bg, video_out)
        else:
            print(f"  SKIP: No background video for rendering")

    # Summary
    entries = 0
    final = timing_snapped if timing_snapped.exists() else timing_ctc
    if final.exists():
        with open(final, 'r') as f:
            data = json.load(f)
            entries = len(data)
    print(f"\n  ✓ {surah_name.title()}: {entries} graphemes aligned")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch MAH recitation pipeline")
    parser.add_argument('--surah', type=str, help="Process single surah by name")
    parser.add_argument('--skip-render', action='store_true', help="Skip video rendering")
    parser.add_argument('--list', action='store_true', help="List available surahs")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.list:
        print("Available surahs:")
        for name, num in sorted(SURAH_MAP.items(), key=lambda x: x[1]):
            video = DOWNLOADS_DIR / f"surah_{name}_mohammad_ahmad_hassan.mp4"
            status = "✓ video" if video.exists() else "✗ no video"
            wav = OUTPUT_DIR / f"mah_{name}.wav"
            wav_status = "✓ audio" if wav.exists() else "✗ no audio"
            print(f"  #{num:3d}  {name:12s}  [{status}]  [{wav_status}]")
        return

    if args.surah:
        name = args.surah.lower()
        if name not in SURAH_MAP:
            print(f"Unknown surah: {name}")
            print(f"Available: {', '.join(sorted(SURAH_MAP.keys()))}")
            sys.exit(1)
        process_surah(name, SURAH_MAP[name], skip_render=args.skip_render)
    else:
        # Process all available
        results = {}
        for name in sorted(SURAH_MAP.keys(), key=lambda n: SURAH_MAP[n]):
            ok = process_surah(name, SURAH_MAP[name], skip_render=args.skip_render)
            results[name] = ok

        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE")
        print(f"{'='*60}")
        for name, ok in results.items():
            status = "✓" if ok else "✗"
            print(f"  {status} {name.title()} (#{SURAH_MAP[name]})")


if __name__ == "__main__":
    main()
