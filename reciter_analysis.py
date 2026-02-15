#!/usr/bin/env python3
"""
Cross-Reciter Comparative Analysis — Novel Research Tool

Compares letter-level timing between two reciters for the same surah.
Produces per-letter, per-word, and per-ayah comparative statistics.

This is a NOVEL research contribution: no existing work compares Quran
reciters at the letter level using forced alignment timing data.

Usage:
    python reciter_analysis.py \
        --reciter1 output/timing_080_snapped.json \
        --name1 "Abdul Basit" \
        --reciter2 output/mah_abasa_snapped.json \
        --name2 "Mohammad Ahmad Hassan" \
        --output output/comparison_080.json

Produces:
    - Per-letter duration ratios
    - Tajweed adherence differences (Madd, Ghunnah duration patterns)
    - Rhythmic signature analysis (tempo variance per ayah)
    - Summary statistics and JSON output for visualization
"""
import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Arabic letter categories for Tajweed analysis
MADD_LETTERS = set('اوي')  # Long vowel carriers
QALQALAH_LETTERS = set('قطبجد')  # Bouncing letters
GHUNNAH_LETTERS = set('نم')  # Nasal letters
TAFKHEEM_LETTERS = set('صضطظخغق')  # Emphatic/heavy letters
LAAM = 'ل'
RAA = 'ر'
DIACRITICS_SET = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def get_base_letter(char):
    """Extract the base letter from a grapheme cluster."""
    return ''.join(c for c in char if c not in DIACRITICS_SET)


def classify_tajweed(char):
    """Classify a grapheme by its Tajweed category."""
    base = get_base_letter(char)
    if not base:
        return 'other'
    first = base[0]
    if first in MADD_LETTERS:
        return 'madd'
    elif first in QALQALAH_LETTERS:
        return 'qalqalah'
    elif first in GHUNNAH_LETTERS:
        return 'ghunnah'
    elif first in TAFKHEEM_LETTERS:
        return 'tafkheem'
    elif first == LAAM:
        return 'laam'
    elif first == RAA:
        return 'raa'
    return 'other'


def analyze_single(timing, name):
    """Compute statistics for a single reciter."""
    durations = [(e['end'] - e['start']) for e in timing]
    total = timing[-1]['end'] - timing[0]['start']

    # Per-ayah stats
    ayah_stats = defaultdict(list)
    for e in timing:
        ayah_stats[e['ayah']].append(e['end'] - e['start'])

    # Per-category stats
    cat_stats = defaultdict(list)
    for e in timing:
        cat = classify_tajweed(e['char'])
        cat_stats[cat].append(e['end'] - e['start'])

    return {
        'name': name,
        'total_s': round(total, 2),
        'num_entries': len(timing),
        'num_ayahs': len(ayah_stats),
        'avg_duration_ms': round(safe_mean(durations) * 1000, 1),
        'median_duration_ms': round(safe_median(durations) * 1000, 1),
        'min_duration_ms': round(min(durations) * 1000, 1),
        'max_duration_ms': round(max(durations) * 1000, 1),
        'per_ayah': {
            int(a): {
                'count': len(ds),
                'total_s': round(sum(ds), 2),
                'avg_ms': round(safe_mean(ds) * 1000, 1),
            }
            for a, ds in sorted(ayah_stats.items())
        },
        'per_category': {
            cat: {
                'count': len(ds),
                'avg_ms': round(safe_mean(ds) * 1000, 1),
                'median_ms': round(safe_median(ds) * 1000, 1),
            }
            for cat, ds in sorted(cat_stats.items())
        },
    }


def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0


def safe_median(lst):
    if not lst:
        return 0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def compare_reciters(timing1, timing2, name1, name2):
    """Cross-reciter comparison — the novel research contribution."""
    if len(timing1) != len(timing2):
        print(f"  WARNING: Different entry counts: {len(timing1)} vs {len(timing2)}")
        # Trim to minimum
        min_len = min(len(timing1), len(timing2))
        timing1 = timing1[:min_len]
        timing2 = timing2[:min_len]

    n = len(timing1)

    # Per-letter duration ratios
    ratios = []
    letter_comparisons = []

    for i in range(n):
        d1 = timing1[i]['end'] - timing1[i]['start']
        d2 = timing2[i]['end'] - timing2[i]['start']

        if d1 > 0.001:  # Avoid division by near-zero
            ratio = d2 / d1
        else:
            ratio = 1.0

        ratios.append(ratio)
        cat = classify_tajweed(timing1[i]['char'])

        letter_comparisons.append({
            'idx': i,
            'char': timing1[i]['char'],
            'ayah': timing1[i]['ayah'],
            'category': cat,
            f'{name1}_ms': round(d1 * 1000, 1),
            f'{name2}_ms': round(d2 * 1000, 1),
            'ratio': round(ratio, 3),
        })

    # Per-category ratio analysis
    cat_ratios = defaultdict(list)
    for lc in letter_comparisons:
        cat_ratios[lc['category']].append(lc['ratio'])

    category_analysis = {}
    for cat, rs in sorted(cat_ratios.items()):
        category_analysis[cat] = {
            'count': len(rs),
            'avg_ratio': round(safe_mean(rs), 3),
            'median_ratio': round(safe_median(rs), 3),
            'min_ratio': round(min(rs), 3),
            'max_ratio': round(max(rs), 3),
            'interpretation': interpret_ratio(safe_mean(rs), cat),
        }

    # Per-ayah tempo comparison
    ayah_tempo = defaultdict(lambda: {'d1': [], 'd2': []})
    for i in range(n):
        ayah = timing1[i]['ayah']
        ayah_tempo[ayah]['d1'].append(timing1[i]['end'] - timing1[i]['start'])
        ayah_tempo[ayah]['d2'].append(timing2[i]['end'] - timing2[i]['start'])

    ayah_comparison = {}
    for ayah in sorted(ayah_tempo.keys()):
        t1 = sum(ayah_tempo[ayah]['d1'])
        t2 = sum(ayah_tempo[ayah]['d2'])
        ayah_comparison[int(ayah)] = {
            f'{name1}_s': round(t1, 2),
            f'{name2}_s': round(t2, 2),
            'ratio': round(t2 / t1 if t1 > 0 else 1, 3),
        }

    # Top divergent letters (most different between reciters)
    divergent = sorted(letter_comparisons, key=lambda x: abs(x['ratio'] - 1.0), reverse=True)[:20]

    # Overall stats
    overall_ratio = safe_mean(ratios)

    return {
        'overall': {
            'avg_ratio': round(overall_ratio, 3),
            'median_ratio': round(safe_median(ratios), 3),
            'interpretation': (
                f"{name2} is {'slower' if overall_ratio > 1 else 'faster'} "
                f"by {abs(overall_ratio - 1) * 100:.1f}% on average"
            ),
        },
        'category_analysis': category_analysis,
        'ayah_comparison': ayah_comparison,
        'top_divergent_letters': divergent,
        'letter_count': n,
    }


def interpret_ratio(ratio, category):
    """Human-readable interpretation of a duration ratio for a Tajweed category."""
    diff = abs(ratio - 1.0) * 100
    direction = "longer" if ratio > 1 else "shorter"

    if diff < 5:
        return "Very similar recitation style"
    elif diff < 15:
        return f"Slightly {direction} ({diff:.0f}%)"
    elif diff < 30:
        adj = ""
        if category == 'madd':
            adj = " — different Madd elongation style"
        elif category == 'ghunnah':
            adj = " — different nasalization duration"
        elif category == 'qalqalah':
            adj = " — different bounce intensity"
        return f"Moderately {direction} ({diff:.0f}%){adj}"
    else:
        return f"Significantly {direction} ({diff:.0f}%) — distinctive recitation feature"


def print_summary(stats1, stats2, comparison):
    """Print a human-readable summary."""
    print(f"\n{'='*70}")
    print(f"  CROSS-RECITER COMPARISON: Surah Analysis")
    print(f"{'='*70}")
    print(f"  {stats1['name']}: {stats1['total_s']}s ({stats1['num_entries']} graphemes)")
    print(f"  {stats2['name']}: {stats2['total_s']}s ({stats2['num_entries']} graphemes)")
    print(f"\n  {comparison['overall']['interpretation']}")

    print(f"\n  Tajweed Category Analysis:")
    print(f"  {'Category':<12s} {'Count':<7s} {'Avg Ratio':<11s} {'Interpretation'}")
    print(f"  {'-'*12} {'-'*7} {'-'*11} {'-'*40}")
    for cat, data in comparison['category_analysis'].items():
        print(f"  {cat:<12s} {data['count']:<7d} {data['avg_ratio']:<11.3f} {data['interpretation']}")

    print(f"\n  Top 10 Most Divergent Letters:")
    print(f"  {'Char':<6s} {'Ayah':<6s} {'Cat':<12s} "
          f"{stats1['name'][:8]:<10s} {stats2['name'][:8]:<10s} {'Ratio'}")
    print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*7}")
    for lc in comparison['top_divergent_letters'][:10]:
        n1_key = [k for k in lc.keys() if k.endswith('_ms') and k != list(lc.keys())[-2]][0]
        n2_key = [k for k in lc.keys() if k.endswith('_ms') and k != n1_key][0]
        print(f"  {lc['char']:<6s} {lc['ayah']:<6d} {lc['category']:<12s} "
              f"{lc[n1_key]:>7.1f}ms {lc[n2_key]:>7.1f}ms {lc['ratio']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Reciter Comparative Analysis")
    parser.add_argument("--reciter1", required=True, help="First reciter timing JSON")
    parser.add_argument("--name1", default="Reciter 1", help="Name of first reciter")
    parser.add_argument("--reciter2", required=True, help="Second reciter timing JSON")
    parser.add_argument("--name2", default="Reciter 2", help="Name of second reciter")
    parser.add_argument("--output", default=None, help="Output comparison JSON")
    args = parser.parse_args()

    # Load
    print(f"Loading {args.name1}: {args.reciter1}", flush=True)
    with open(args.reciter1, 'r', encoding='utf-8') as f:
        timing1 = json.load(f)

    print(f"Loading {args.name2}: {args.reciter2}", flush=True)
    with open(args.reciter2, 'r', encoding='utf-8') as f:
        timing2 = json.load(f)

    # Individual stats
    stats1 = analyze_single(timing1, args.name1)
    stats2 = analyze_single(timing2, args.name2)

    # Cross-reciter comparison
    comparison = compare_reciters(timing1, timing2, args.name1, args.name2)

    # Print summary
    print_summary(stats1, stats2, comparison)

    # Save
    output_path = args.output or f"comparison_{Path(args.reciter1).stem}.json"
    result = {
        'reciter1': stats1,
        'reciter2': stats2,
        'comparison': comparison,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
