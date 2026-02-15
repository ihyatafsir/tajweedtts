#!/usr/bin/env python3
"""
Smooth Karaoke Overlay Renderer for Quran Recitation

Renders per-character gradient color sweep proportional to duration:
- Active char: smooth green fill sweeping right→left (RTL)
- Madd letters: fill slows visibly as the reciter holds
- Past chars: dimmed green
- Future chars: white/light gray
- Non-current ayahs: dark gray, fixed position

Usage:
    python render_video_overlay.py --video in.mp4 --audio in.wav \
        --timing timing.json --output out.mp4 --surah 80
"""
import argparse
import json
import subprocess
import wave
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ─── CONFIG ────────────────────────────────────────────────────
WIDTH = 1024
HEIGHT = 576
FPS = 30
OVERLAY_HEIGHT = 380
WAVEFORM_HEIGHT = 26

# Colors
COLOR_FUTURE    = (230, 230, 230, 255)    # White-ish for unrecited
COLOR_ACTIVE    = (51, 255, 51, 255)      # Bright green
COLOR_ACTIVE_GLOW = (51, 255, 51, 80)    # Glow around active
COLOR_PAST      = (80, 180, 100, 255)     # Dimmed green for recited
COLOR_DIM       = (100, 100, 100, 160)    # Non-current ayahs
AYAH_COLOR      = (255, 215, 0, 200)       # Gold for ayah numbers
BG_COLOR        = (0, 0, 0, 210)          # Semi-transparent overlay bg

FONT_SIZE = 42
FONT_SIZE_SM = 28
AYAHS_VISIBLE = 5

DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')
QURAN_STOP_MARKS = set('\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DD\u06DE\u06DF\u06E0\u06E1\u06E2\u06E3\u06E4\u06E5\u06E6\u06E7\u06E8\u06E9\u06EA\u06EB\u06EC\u06ED')


def is_diacritic(ch):
    cp = ord(ch)
    return ch in DIACRITICS or (0x064B <= cp <= 0x0652) or (0x0610 <= cp <= 0x061A)


def find_font():
    for p in ["/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
              "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
              "/home/absolut7/Documents/26apps/MahQuranApp/public/fonts/AmiriQuran.ttf"]:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No Arabic font found!")


def load_uthmani_text(surah_num):
    path = Path("/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json")
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get(str(surah_num), [])
    return []


def clean_timing_data(timing):
    for g in timing:
        g['char'] = ''.join(c for c in g['char'] if c not in QURAN_STOP_MARKS)
    return timing


def load_waveform(audio_path, duration_s, rate=80):
    with wave.open(str(audio_path), 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total = int(duration_s * rate)
    if total == 0:
        return np.array([])
    chunk = max(1, len(samples) // total)
    return np.array([np.max(np.abs(samples[i:i+chunk])) for i in range(0, len(samples), chunk)])[:total]


def precompute_ayahs(timing):
    ayahs = {}
    for t in timing:
        ayahs.setdefault(t['ayah'], []).append(t)

    result = []
    for ayah_num in sorted(ayahs.keys()):
        graphemes = ayahs[ayah_num]
        words = []
        current_chars = []
        current_widx = None

        for g in graphemes:
            if current_widx is not None and g.get('wordIdx') != current_widx:
                raw = ''.join(c['char'] for c in current_chars)
                words.append({'raw': raw, 'display': raw, 'graphemes': current_chars})
                current_chars = []
            current_chars.append(g)
            current_widx = g.get('wordIdx')

        if current_chars:
            raw = ''.join(c['char'] for c in current_chars)
            words.append({'raw': raw, 'display': raw, 'graphemes': current_chars})

        result.append({
            'ayah': ayah_num, 'words': words,
            'start_s': graphemes[0]['start'],
            'end_s': graphemes[-1]['end'],
        })
    return result


def lerp_color(c1, c2, t):
    """Linear interpolation between two RGBA colors. t=0→c1, t=1→c2."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(len(c1)))


def get_char_color_and_progress(g, current_time):
    """Return (color, fill_progress) for a grapheme at current_time.
    
    fill_progress: 0.0 = not started, 1.0 = fully filled (past)
    For the ACTIVE char: 0.0 < progress < 1.0
    """
    start = g['start']
    end = g['end']
    
    if current_time < start:
        return COLOR_FUTURE, 0.0
    elif current_time >= end:
        return COLOR_PAST, 1.0
    else:
        # ACTIVE — compute fill progress
        duration = end - start
        if duration <= 0:
            return COLOR_ACTIVE, 1.0
        progress = (current_time - start) / duration
        return COLOR_ACTIVE, progress


def measure_grapheme_width(char_text, font):
    """Measure the actual pixel width of a single grapheme using font metrics."""
    try:
        bbox = font.getbbox(char_text, direction='rtl')
    except Exception:
        bbox = font.getbbox(char_text)
    return max(1, bbox[2] - bbox[0])


def render_colored_word_smooth(draw, word, current_time, x, y, font, is_current):
    """Render a word with precise per-character coloring using real font metrics.
    
    Each grapheme is measured and rendered individually, then composited
    in correct RTL order. Gradient fills sweep right-to-left for active chars.
    """
    graphemes = word['graphemes']
    display_text = word['display']
    n = len(graphemes)

    try:
        bbox = font.getbbox(display_text, direction='rtl')
    except Exception:
        bbox = font.getbbox(display_text)
    word_w = bbox[2] - bbox[0]

    if not is_current:
        draw.text((x, y), display_text, fill=COLOR_DIM[:3], font=font, direction='rtl')
        return word_w

    # Determine each grapheme's color and progress
    colors = []
    progresses = []
    has_gradient = False
    for g in graphemes:
        color, progress = get_char_color_and_progress(g, current_time)
        colors.append(color)
        progresses.append(progress)
        if 0 < progress < 1:
            has_gradient = True

    # Fast path: all same color, no gradient needed
    if not has_gradient and all(c == colors[0] for c in colors):
        if colors[0] == COLOR_ACTIVE:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), display_text, fill=COLOR_ACTIVE_GLOW,
                              font=font, direction='rtl')
        draw.text((x, y), display_text, fill=colors[0][:3], font=font, direction='rtl')
        return word_w

    # ── Per-character rendering with real font metrics ──
    # Measure each grapheme's actual pixel width
    char_widths = []
    for g in graphemes:
        w = measure_grapheme_width(g['char'], font)
        char_widths.append(w)
    
    # Scale measured widths to fit the actual word width (font shaping may differ)
    total_measured = sum(char_widths)
    if total_measured > 0:
        scale = word_w / total_measured
        char_widths = [max(1, int(w * scale)) for w in char_widths]
        # Fix rounding: adjust last char to absorb remainder
        remainder = word_w - sum(char_widths)
        if char_widths:
            char_widths[-1] = max(1, char_widths[-1] + remainder)

    # Create per-grapheme images and composite in RTL order
    # In RTL display: grapheme[0] is rightmost, grapheme[n-1] is leftmost
    pad = 10
    tmp_h = FONT_SIZE + 40
    tmp_w = word_w + pad * 2
    
    # Render the full word as a white mask to get proper glyph shapes
    word_img = Image.new('RGBA', (tmp_w, tmp_h), (0, 0, 0, 0))
    ImageDraw.Draw(word_img).text((pad, 10), display_text,
                                  fill=(255, 255, 255, 255), font=font, direction='rtl')
    word_arr = np.array(word_img)
    alpha_mask = word_arr[:, :, 3]
    result_arr = np.zeros_like(word_arr)

    # Map graphemes to pixel columns using measured widths
    # RTL: first grapheme (index 0) starts from the RIGHT side
    # Pixel layout: [pad ... leftmost_char ... rightmost_char ... pad]
    # grapheme[n-1] is leftmost in display, grapheme[0] is rightmost
    cum_x = pad  # Start from left pixel edge
    for display_i in range(n):
        # display_i=0 is leftmost on screen = last grapheme in logical order (n-1)
        logical_i = n - 1 - display_i
        g_w = char_widths[logical_i]
        color = colors[logical_i]
        progress = progresses[logical_i]

        x_start = cum_x
        x_end = min(cum_x + g_w, tmp_w)

        if 0 < progress < 1:
            # ── GRADIENT FILL (RTL sweep: right to left) ──
            # progress increases as recitation proceeds through the char
            # In RTL, the char starts from the right side
            # fill_boundary moves from right (x_end) toward left (x_start)
            filled_pixels = int((x_end - x_start) * progress)
            fill_boundary = x_end - filled_pixels  # Sweep from right to left

            for col_x in range(x_start, x_end):
                col_mask = alpha_mask[:, col_x] > 0
                if col_x >= fill_boundary:
                    # Already recited (right side in RTL)
                    px_color = COLOR_ACTIVE
                else:
                    # Not yet recited — interpolate near boundary edge
                    dist = fill_boundary - col_x
                    edge_width = max(2, int(g_w * 0.15))
                    if dist < edge_width:
                        t = 1.0 - (dist / edge_width)
                        px_color = lerp_color(COLOR_FUTURE, COLOR_ACTIVE, t)
                    else:
                        px_color = COLOR_FUTURE

                for c_idx in range(3):
                    result_arr[:, col_x, c_idx] = np.where(
                        col_mask, px_color[c_idx], result_arr[:, col_x, c_idx])
                result_arr[:, col_x, 3] = np.where(
                    col_mask, alpha_mask[:, col_x], result_arr[:, col_x, 3])
        else:
            # Solid color (fully future or fully past/active)
            col_mask = alpha_mask[:, x_start:x_end] > 0
            for c_idx in range(3):
                result_arr[:, x_start:x_end, c_idx] = np.where(
                    col_mask, color[c_idx], result_arr[:, x_start:x_end, c_idx])
            result_arr[:, x_start:x_end, 3] = np.where(
                col_mask, alpha_mask[:, x_start:x_end], result_arr[:, x_start:x_end, 3])

        cum_x += g_w

    result_img = Image.fromarray(result_arr)
    draw._image.paste(result_img, (x - pad, y - 10), result_img)
    return word_w


def find_active_idx(timing, t):
    if not timing or t < timing[0]['start']:
        return -1
    if t >= timing[-1]['start']:
        return timing[-1]['idx']

    lo, hi = 0, len(timing) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if timing[mid]['start'] <= t:
            lo = mid
        else:
            hi = mid - 1
    return timing[lo]['idx']


def wrap_words_to_lines(words, font, max_width, space_w):
    lines = []
    current_words = []
    current_widths = []
    current_w = 0

    for w in words:
        try:
            bbox = font.getbbox(w['display'], direction='rtl')
        except:
            bbox = font.getbbox(w['display'])
        ww = bbox[2] - bbox[0]
        needed = ww + (space_w if current_words else 0)

        if current_words and current_w + needed > max_width:
            lines.append((current_words, current_widths))
            current_words = []
            current_widths = []
            current_w = 0

        current_words.append(w)
        current_widths.append(ww)
        current_w += needed

    if current_words:
        lines.append((current_words, current_widths))
    return lines


def render_frame_overlay(frame_img, ayah_groups, timing, current_time, waveform, wf_rate, font, font_sm):
    draw = ImageDraw.Draw(frame_img, 'RGBA')

    overlay_y = HEIGHT - OVERLAY_HEIGHT
    draw.rectangle([(0, overlay_y), (WIDTH, HEIGHT)], fill=BG_COLOR)

    # ── Waveform ──
    wf_y = overlay_y + 2
    wf_mid = wf_y + WAVEFORM_HEIGHT // 2
    playback_pos = int(current_time * wf_rate)
    playhead_x = WIDTH - 80

    for x in range(10, WIDTH - 10):
        si = playback_pos - (playhead_x - x)
        if 0 <= si < len(waveform):
            amp = waveform[si]
            h = max(1, int(amp * WAVEFORM_HEIGHT * 0.85))
            c = (51, 255, 51, 150)
            draw.line([(x, wf_mid - h), (x, wf_mid + h)], fill=c, width=1)

    draw.line([(playhead_x, wf_y), (playhead_x, wf_y + WAVEFORM_HEIGHT)],
              fill=(255, 255, 255, 200), width=2)

    # ── Find current ayah group ──
    cur_group_i = 0
    for i, ag in enumerate(ayah_groups):
        if ag['start_s'] <= current_time <= ag['end_s']:
            cur_group_i = i
            break
        elif ag['start_s'] > current_time:
            cur_group_i = max(0, i - 1)
            break
        else:
            cur_group_i = i

    # ── Visible ayah range ──
    half = AYAHS_VISIBLE // 2
    start_i = max(0, cur_group_i - half)
    end_i = min(len(ayah_groups), start_i + AYAHS_VISIBLE)
    start_i = max(0, end_i - AYAHS_VISIBLE)
    visible_groups = ayah_groups[start_i:end_i]

    # ── Layout lines ──
    line_height = FONT_SIZE + 10
    space_w = 8
    text_area_top = overlay_y + WAVEFORM_HEIGHT + 8
    text_area_h = OVERLAY_HEIGHT - WAVEFORM_HEIGHT - 16

    all_lines = []
    for gi, ag in enumerate(visible_groups):
        real_gi = start_i + gi
        is_current = (real_gi == cur_group_i)
        lines = wrap_words_to_lines(ag['words'], font, WIDTH - 80, space_w)
        for lw, lww in lines:
            all_lines.append((lw, lww, ag, is_current))

    total_text_h = len(all_lines) * line_height
    y_offset = text_area_top + max(0, (text_area_h - total_text_h) // 2)

    # ── Render text ──
    for line_words, line_widths, ag, is_current in all_lines:
        if y_offset + line_height > HEIGHT - 4:
            break

        total_line_w = sum(line_widths) + space_w * (len(line_widths) - 1)
        x = (WIDTH + total_line_w) // 2
        line_y = y_offset

        for w, ww in zip(line_words, line_widths):
            x -= ww
            render_colored_word_smooth(draw, w, current_time, x, line_y, font, is_current)
            x -= space_w

        # Ayah number
        if is_current:
            ayah_str = f"\ufd3f{ag['ayah']}\ufd3e"
            draw.text((WIDTH - 50, line_y + 2), ayah_str, fill=AYAH_COLOR, font=font_sm)

        y_offset += line_height

    return frame_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--timing", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--surah", type=int, required=True)
    args = parser.parse_args()

    print(f"Rendering overlay for Surah {args.surah}...", flush=True)

    with open(args.timing, 'r', encoding='utf-8') as f:
        timing = json.load(f)
    timing = clean_timing_data(timing)
    duration_s = timing[-1]['end']

    font_path = find_font()
    font = ImageFont.truetype(font_path, FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    font_sm = ImageFont.truetype(font_path, FONT_SIZE_SM, layout_engine=ImageFont.Layout.RAQM)

    ayah_groups = precompute_ayahs(timing)
    wf_rate = 80
    waveform = load_waveform(args.audio, duration_s, wf_rate)

    FRAMES_DIR = Path("temp_frames_" + str(args.surah))
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir()

    # Extract frames
    subprocess.run([
        'ffmpeg', '-y', '-i', args.video,
        '-vf', f'fps={FPS}', '-q:v', '3',
        str(FRAMES_DIR / 'frame_%05d.jpg')
    ], check=True, capture_output=True)

    frame_files = sorted(FRAMES_DIR.glob('frame_*.jpg'))
    out_dir = FRAMES_DIR / 'out'
    out_dir.mkdir()

    # Render
    total_frames = int(duration_s * FPS)
    for i, frame_path in enumerate(frame_files[:total_frames]):
        frame_img = Image.open(frame_path).convert('RGBA')
        if frame_img.size != (WIDTH, HEIGHT):
            frame_img = frame_img.resize((WIDTH, HEIGHT), Image.LANCZOS)

        t = i / FPS
        frame_img = render_frame_overlay(frame_img, ayah_groups, timing, t, waveform, wf_rate, font, font_sm)
        frame_img.convert('RGB').save(out_dir / f'frame_{i+1:05d}.jpg', quality=90)

        if i % 100 == 0:
            print(f"  Frame {i}/{total_frames}", flush=True)

    # Assemble
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(FPS),
        '-i', str(out_dir / 'frame_%05d.jpg'),
        '-i', args.audio,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-c:a', 'aac', '-b:a', '128k', '-pix_fmt', 'yuv420p',
        '-shortest', args.output
    ], check=True, capture_output=True)

    shutil.rmtree(FRAMES_DIR)
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
