#!/bin/bash
# Extract 16kHz mono WAV audio from downloaded recitation videos
# Usage: ./extract_audio.sh

DOWNLOADS="$(dirname "$0")/downloads"
OUTPUT="$(dirname "$0")/output"

mkdir -p "$OUTPUT"

echo "=== Extracting audio from downloaded recitation videos ==="

for video in "$DOWNLOADS"/surah_*_mohammad_ahmad_hassan.mp4; do
    [ -f "$video" ] || continue
    
    # Extract surah name from filename: surah_abasa_mohammad_ahmad_hassan.mp4 -> abasa
    basename=$(basename "$video" .mp4)
    surah_name=$(echo "$basename" | sed 's/surah_\(.*\)_mohammad_ahmad_hassan/\1/')
    
    wav_out="$OUTPUT/mah_${surah_name}.wav"
    
    if [ -f "$wav_out" ]; then
        echo "  SKIP: $wav_out already exists"
        continue
    fi
    
    echo "  Extracting: $surah_name -> $wav_out"
    ffmpeg -i "$video" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$wav_out" -y -loglevel warning
    
    if [ $? -eq 0 ]; then
        duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$wav_out" 2>/dev/null)
        echo "    Done: ${duration}s"
    else
        echo "    ERROR: Failed to extract $surah_name"
    fi
done

echo ""
echo "=== Audio extraction complete ==="
ls -lh "$OUTPUT"/mah_*.wav 2>/dev/null || echo "No WAV files produced."
