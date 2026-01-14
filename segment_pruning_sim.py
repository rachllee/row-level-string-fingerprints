import argparse
import csv
import os
import re

import duckdb
import numpy as np

PREFIX_BYTES = 8


def norm(s):
    return (s or "").lower()


def normalize_query(s: str, suffix: bool) -> str:
    s = norm(s)
    return s[::-1] if suffix else s


def key_u64_from_normed(s_norm, nbytes=PREFIX_BYTES):
    b = s_norm.encode("utf-8", errors="ignore")[:nbytes]
    b = b + b"\x00" * (nbytes - len(b))
    return int.from_bytes(b, "big", signed=False)


def next_prefix_normed(s_norm: str) -> str:
    b = bytearray(s_norm.encode("utf-8", errors="ignore"))
    if not b:
        return "\uffff"
    b[-1] = min(255, b[-1] + 1)
    return bytes(b).decode("utf-8", errors="ignore")


def bucket_range(query: str, boundaries: np.ndarray, bits: int, suffix: bool):
    s = normalize_query(query, suffix)
    lo = key_u64_from_normed(s)
    hi = key_u64_from_normed(next_prefix_normed(s))
    if boundaries is None:
        shift = 64 - bits
        jlo = int(np.right_shift(lo, shift))
        jhi = int(np.right_shift(hi, shift))
        return min(jlo, jhi), max(jlo, jhi)
    jlo = np.searchsorted(boundaries, lo, side="right") - 1
    jhi = np.searchsorted(boundaries, hi, side="right") - 1
    jlo = int(np.clip(jlo, 0, len(boundaries) - 1))
    jhi = int(np.clip(jhi, 0, len(boundaries) - 1))
    return min(jlo, jhi), max(jlo, jhi)


def parse_min_max(stats):
    if not isinstance(stats, str):
        return None, None
    m = re.search(r"Min:\s*([0-9]+), Max:\s*([0-9]+)", stats)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_segments(con, table, code_col):
    df = con.execute(f"PRAGMA storage_info('{table}')").fetchdf()
    df = df[df["column_name"] == code_col]
    df = df[df["segment_type"] != "VALIDITY"].copy()
    mins = []
    maxs = []
    for s in df["stats"]:
        mn, mx = parse_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    df["min"] = mins
    df["max"] = maxs
    return df


def summarize_pruning(df, lo, hi):
    total_rows = float(df["count"].sum())
    with_stats = df[df["min"].notna() & df["max"].notna()].copy()
    no_stats = df[df["min"].isna() | df["max"].isna()].copy()

    overlap = with_stats[(with_stats["min"] <= hi) & (with_stats["max"] >= lo)]
    kept_rows = float(overlap["count"].sum() + no_stats["count"].sum())
    kept_pct = (kept_rows / total_rows * 100.0) if total_rows else 0.0

    return {
        "segments_total": int(len(df)),
        "segments_with_stats": int(len(with_stats)),
        "segments_no_stats": int(len(no_stats)),
        "segments_overlap": int(len(overlap)),
        "rows_total": int(total_rows),
        "rows_kept_est": int(kept_rows),
        "rows_skipped_est": int(total_rows - kept_rows),
        "keep_pct_est": kept_pct,
        "skip_pct_est": 100.0 - kept_pct,
    }


def summarize_row_groups(df, lo, hi):
    groups = []
    for rg_id, g in df.groupby("row_group_id"):
        mins = g["min"].dropna()
        maxs = g["max"].dropna()
        if len(mins) == 0 or len(maxs) == 0:
            groups.append((rg_id, None, None, float(g["count"].sum())))
        else:
            groups.append((rg_id, int(mins.min()), int(maxs.max()), float(g["count"].sum())))

    total_rows = sum(x[3] for x in groups)
    overlap = [x for x in groups if x[1] is not None and x[1] <= hi and x[2] >= lo]
    no_stats = [x for x in groups if x[1] is None or x[2] is None]
    kept_rows = sum(x[3] for x in overlap) + sum(x[3] for x in no_stats)
    kept_pct = (kept_rows / total_rows * 100.0) if total_rows else 0.0

    return {
        "row_groups_total": int(len(groups)),
        "row_groups_with_stats": int(len(groups) - len(no_stats)),
        "row_groups_no_stats": int(len(no_stats)),
        "row_groups_overlap": int(len(overlap)),
        "rg_rows_total": int(total_rows),
        "rg_rows_kept_est": int(kept_rows),
        "rg_rows_skipped_est": int(total_rows - kept_rows),
        "rg_keep_pct_est": kept_pct,
        "rg_skip_pct_est": 100.0 - kept_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..28). Default: 8")
    parser.add_argument("--suffix", action="store_true", help="Use suffix fingerprints.")
    parser.add_argument("--queries", type=str, default="", help="Comma-separated queries (without %).")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path.")
    args = parser.parse_args()

    K = args.bits
    if K < 1 or K > 28:
        raise ValueError("--bits must be between 1 and 28")

    mode = "suffix" if args.suffix else "prefix"
    parquet = f"title_strs_{mode}_b{K}.parquet"
    boundaries_npy = f"q{K}_{mode}_boundaries.npy"
    code_col = f"q{K}_{mode}"

    boundaries = np.load(boundaries_npy) if os.path.exists(boundaries_npy) else None

    if args.queries:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    else:
        queries = ["jos", "2012", "the", "a", "(", "#"] if not args.suffix else ["son", "2012", "the", "a", ")", "#"]

    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")
    df = load_segments(con, "t", code_col)

    rows = []
    for q in queries:
        lo, hi = bucket_range(q, boundaries, K, suffix=args.suffix)
        seg = summarize_pruning(df, lo, hi)
        rg = summarize_row_groups(df, lo, hi)
        rows.append(
            {
                "bits": K,
                "mode": mode,
                "query": q,
                "lo": lo,
                "hi": hi,
                **seg,
                **rg,
            }
        )

    for r in rows:
        print(
            f"{r['query']}% [{r['lo']},{r['hi']}] "
            f"segments keep {r['keep_pct_est']:.2f}% (skip {r['skip_pct_est']:.2f}%) "
            f"row_groups keep {r['rg_keep_pct_est']:.2f}% (skip {r['rg_skip_pct_est']:.2f}%)"
        )

    if args.csv:
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
