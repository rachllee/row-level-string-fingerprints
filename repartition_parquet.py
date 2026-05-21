"""
Repartition title_strs_substr_fp16.parquet into smaller row groups
and re-run the custom scanner to test row-group-level skipping.
"""

import json
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from custom_scan import fp_scan, baseline_scan, pattern_mask

SOURCE   = "title_strs_substr_fp16.parquet"
FEATURES_JSON = "substr_features.json"

ROW_GROUP_SIZES = [10_000, 50_000, 122_880, 500_000]

PATTERNS = ["anda smi", "ing", "the", "python", "2024", "ist claudi"]


def repartition(source: str, row_group_size: int) -> str:
    out = f"title_strs_substr_rg{row_group_size}.parquet"
    print(f"  Writing {out} (row_group_size={row_group_size:,})...")
    table = pq.read_table(source)
    pq.write_table(table, out, row_group_size=row_group_size)
    pf = pq.ParquetFile(out)
    print(f"  → {pf.metadata.num_row_groups} row groups, "
          f"{pf.metadata.num_rows:,} rows")
    return out


def run_comparison(parquet_path: str, patterns: list, selected_chars: list, reps=3):
    pf = pq.ParquetFile(parquet_path)
    n_rg = pf.metadata.num_row_groups

    print(f"\n{'Pattern':<16} {'bits':>5} {'rg_skip%':>9} "
          f"{'baseline':>10} {'fp_scan':>10} {'speedup':>9}")
    print("-" * 65)

    for pat in patterns:
        mask = pattern_mask(pat, selected_chars)
        bits = bin(mask).count('1')

        # baseline
        bt = np.median([baseline_scan(pf, pat)[3] for _ in range(reps)])

        # fp scan
        fp_results = [fp_scan(pf, mask, pat) for _ in range(reps)]
        ft = np.median([r[5] for r in fp_results])
        _, _, rgs_skip, rgs_load, _, _ = fp_results[0]
        skip_pct = 100 * rgs_skip / n_rg

        print(f"'%{pat}%'  {bits:>5}  {skip_pct:>8.1f}%  "
              f"{bt:>8.1f}ms  {ft:>8.1f}ms  {bt/ft:>8.2f}x")


def main():
    selected_chars = json.load(open(FEATURES_JSON))["selected_chars"]

    print("Repartitioning parquet with varying row group sizes...")
    for rgs in ROW_GROUP_SIZES:
        print(f"\n{'='*65}")
        print(f"Row group size: {rgs:,}")
        print('='*65)
        path = repartition(SOURCE, rgs)
        run_comparison(path, PATTERNS, selected_chars, reps=3)


if __name__ == "__main__":
    main()
