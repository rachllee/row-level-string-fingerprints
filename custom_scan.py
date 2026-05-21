"""
Custom row-group-aware scanner comparing:
  1. Fingerprint-guided scan: load fp16_chars first, build boolean mask with
     Arrow compute, filter title column at Arrow level (no Python string objects),
     evaluate match_substring only on passing rows.
  2. Baseline scan: load title column for every row group, evaluate all rows.

Uses pyarrow.compute throughout to avoid Python-object overhead.
"""

import json
import time
import argparse
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


PARQUET = "title_strs_substr_fp16.parquet"
FEATURES_JSON = "substr_features.json"


def load_features():
    with open(FEATURES_JSON) as f:
        data = json.load(f)
    if "selected_features" in data:
        return data["selected_features"], data.get("ngram", 1)
    else:
        return data["selected_chars"], 1


def extract_ngrams(s: str, n: int) -> set:
    s = (s or "").lower()
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def pattern_mask(pattern: str, selected_features: list, ngram: int) -> int:
    mask  = 0
    grams = extract_ngrams(pattern, ngram)
    for i, feat in enumerate(selected_features):
        if feat in grams:
            mask |= (1 << i)
    return mask


def fp_scan(pf: pq.ParquetFile, mask: int, pattern: str):
    """
    Fingerprint-guided scan using Arrow compute throughout.
    Returns (min_match, match_count, rgs_skipped, rgs_loaded, total_rgs, elapsed_ms)
    """
    n_rg = pf.metadata.num_row_groups
    mask_scalar = pa.scalar(mask, type=pa.uint16())

    min_match   = None
    match_count = 0
    rgs_skipped = 0
    rgs_loaded  = 0

    t0 = time.perf_counter()

    for rg in range(n_rg):
        # Step 1: load fp16_chars, compute boolean selection mask at Arrow level
        fp_col = pf.read_row_group(rg, columns=["fp16_chars"])["fp16_chars"]

        if mask == 0:
            bool_mask = None
        else:
            bool_mask = pc.equal(pc.bit_wise_and(fp_col, mask_scalar), mask_scalar)
            if not pc.any(bool_mask).as_py():
                rgs_skipped += 1
                continue

        rgs_loaded += 1

        # Step 2: load title column, filter at Arrow level (no Python string objects)
        titles = pf.read_row_group(rg, columns=["title"])["title"]
        if bool_mask is not None:
            titles = pc.filter(titles, bool_mask)

        # Step 3: substring match via Arrow compute
        hits = pc.match_substring(titles, pattern, ignore_case=True)
        count = pc.sum(hits).as_py() or 0
        match_count += count

        if count > 0:
            matched = pc.filter(titles, hits)
            local_min = pc.min(matched).as_py()
            if min_match is None or local_min < min_match:
                min_match = local_min

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return min_match, match_count, rgs_skipped, rgs_loaded, n_rg, elapsed_ms


def baseline_scan(pf: pq.ParquetFile, pattern: str):
    """
    Baseline: load title for every row group, evaluate all rows via Arrow compute.
    Returns (min_match, match_count, total_rgs, elapsed_ms)
    """
    n_rg = pf.metadata.num_row_groups

    min_match   = None
    match_count = 0

    t0 = time.perf_counter()

    for rg in range(n_rg):
        titles = pf.read_row_group(rg, columns=["title"])["title"]
        hits   = pc.match_substring(titles, pattern, ignore_case=True)
        count  = pc.sum(hits).as_py() or 0
        match_count += count

        if count > 0:
            matched   = pc.filter(titles, hits)
            local_min = pc.min(matched).as_py()
            if min_match is None or local_min < min_match:
                min_match = local_min

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return min_match, match_count, n_rg, elapsed_ms


def run(patterns, reps=3):
    selected_features, ngram = load_features()
    pf = pq.ParquetFile(PARQUET)

    print(f"Parquet: {PARQUET}")
    print(f"Row groups: {pf.metadata.num_row_groups}, "
          f"Total rows: {pf.metadata.num_rows:,}")
    print(f"N-gram size: {ngram}")
    print(f"Features ({len(selected_features)}): {selected_features}")
    print()

    for pattern in patterns:
        mask = pattern_mask(pattern, selected_features, ngram)
        mask_bits = bin(mask).count("1")

        print(f"Pattern: '%{pattern}%'  mask=0x{mask:04X}  ({mask_bits} bits set)")
        print("-" * 70)

        base_times = [baseline_scan(pf, pattern)[3] for _ in range(reps)]
        t_base = np.median(base_times)

        fp_results = [fp_scan(pf, mask, pattern) for _ in range(reps)]
        t_fp = np.median([r[5] for r in fp_results])
        min_f, cnt_f, rgs_skip, rgs_load, n_rg, _ = fp_results[0]

        min_b, cnt_b, _, _ = baseline_scan(pf, pattern)

        print(f"  Baseline scan:  {t_base:7.1f}ms  |  matches={cnt_b:,}  |  min='{min_b}'")
        print(f"  FP-guided scan: {t_fp:7.1f}ms  |  matches={cnt_f:,}  |  min='{min_f}'")
        print(f"  Row groups: {rgs_load} loaded + {rgs_skip} skipped / {n_rg} total "
              f"({100*rgs_skip/n_rg:.1f}% skipped)")
        print(f"  Speedup: {t_base/t_fp:.2f}x")
        print(f"  Results match: {cnt_b == cnt_f and min_b == min_f}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+",
                        default=["anda smi", "ing", "the", "python", "2024",
                                 "ist claudi", "enshurst p"])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    run(args.patterns, args.reps)


if __name__ == "__main__":
    main()
