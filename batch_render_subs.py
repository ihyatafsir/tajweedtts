#!/usr/bin/env python3
"""
Batch Render Subtitles for Downloaded Quran Videos
Iterates through downloads/*.mp4, extracts audio, runs CTC alignment + physics,
and renders the overlay video.

Features:
- Robust timeout handling (prevents stalling)
- Auto-surah detection from filename
- Skips already processed files
"""

import os
import sys
import subprocess
import shutil
import json
import time
import re
from pathlib import Path

# Config
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/tajweedtts")
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
OUTPUT_DIR = PROJECT_ROOT / "output"
VENV_PYTHON = PROJECT_ROOT / "venv/bin/python"

# Timeout settings (seconds)
TIMEOUT_AUDIO_EXTRACT = 60
TIMEOUT_ALIGNMENT = 600  # 10 mins
TIMEOUT_RENDER = 1200    # 20 mins

# Surah Map (Filename -> Surah Number)
SURAH_MAP = {
    "surah_yasin": 36,
    "surah_ibrahim": 14,
    "surah_baqarah": 2,
    "surah_abasa": 80,
    "surah_waqiah": 56,
    "surah_fajr": 89,
    # Add more as needed or use regex
}

def get_surah_num(filename):
    """Refined surah detection logic"""
    lower_name = filename.lower()
    for key, num in SURAH_MAP.items():
        if key in lower_name:
            return num
    
    # Regex fallback for "surah_X"
    match = re.search(r'surah_(\d+)', lower_name)
    if match:
        return int(match.group(1))
    
    if "kahf" in lower_name: return 18
    if "rahman" in lower_name: return 55
    if "mulk" in lower_name: return 67
    
    return None

def run_command(cmd, timeout_sec, desc):
    """Run command with timeout and clear logging"""
    print(f"  > {desc}...", flush=True)
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"    ✓ Done in {elapsed:.1f}s", flush=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"    ✗ TIMEOUT after {timeout_sec}s!", flush=True)
        return False
    except subprocess.CalledProcessError as e:
        print(f"    ✗ FAILED with exit code {e.returncode}", flush=True)
        print(f"    Error output:\n{e.stderr}", flush=True)
        return False

def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    print("=" * 60)
    print("TajweedTTS Batch Video Processing")
    print("=" * 60)

    video_files = sorted(DOWNLOADS_DIR.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {DOWNLOADS_DIR}")

    for video_path in video_files:
        print("\n" + "-" * 60)
        print(f"Processing: {video_path.name}")
        
        surah_num = get_surah_num(video_path.name)
        if not surah_num:
            print(f"  ⚠ Could not identify surah number from filename. Skipping.")
            continue
            
        print(f"  Detected Surah: {surah_num}")
        
        # 1. Extract Audio
        audio_path = OUTPUT_DIR / f"surah_{surah_num:03d}.wav"
        if not audio_path.exists():
            cmd = [
                "timeout", str(TIMEOUT_AUDIO_EXTRACT),
                "ffmpeg", "-y", "-i", str(video_path),
                "-ac", "1", "-ar", "22050", 
                str(audio_path)
            ]
            if not run_command(cmd, TIMEOUT_AUDIO_EXTRACT + 5, "Extracting audio"):
                continue
        else:
            print("  ✓ Audio already extracted")

        # 2. Run Alignment
        timing_file = OUTPUT_DIR / f"timing_{surah_num:03d}.json"
        
        if not timing_file.exists():
            cmd = [
                "timeout", str(TIMEOUT_ALIGNMENT),
                str(VENV_PYTHON), 
                "process_single_surah.py",
                "--surah", str(surah_num),
                "--audio", str(audio_path),
                "--output", str(timing_file)
            ]
            if not run_command(cmd, TIMEOUT_ALIGNMENT + 5, "Running CTC Alignment + Physics"):
                continue
        else:
             print("  ✓ Alignment already exists")

        # 3. Render Video
        output_video = OUTPUT_DIR / f"render_surah_{surah_num:03d}.mp4"
        if not output_video.exists():
            cmd = [
                "timeout", str(TIMEOUT_RENDER),
                str(VENV_PYTHON),
                "render_video_overlay.py",
                "--video", str(video_path),
                "--audio", str(audio_path),
                "--timing", str(timing_file),
                "--output", str(output_video),
                "--surah", str(surah_num)
            ]
            if not run_command(cmd, TIMEOUT_RENDER + 5, "Rendering Video Overlay"):
                continue
        else:
             print("  ✓ Video already rendered")
             
        print(f"  ✓ SURAH {surah_num} COMPLETE: {output_video}")

    print("\n" + "=" * 60)
    print("Batch Processing Finished")
    print("=" * 60)

if __name__ == "__main__":
    main()
