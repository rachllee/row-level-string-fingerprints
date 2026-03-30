"""
6-way benchmark: prefix / suffix / infix  x  two-column / one-column (bit-shift).

For each sampled (prefix, suffix) pair, runs:
  full_prefix   : WHERE title ILIKE 'p%'
  full_suffix   : WHERE title ILIKE '%s'
  full_infix    : WHERE title ILIKE 'p%' AND title ILIKE '%s'
  fp_prefix_2col: WHERE q8_prefix BETWEEN p_lo AND p_hi AND title ILIKE 'p%'
  fp_suffix_2col: WHERE q8_suffix BETWEEN s_lo AND s_hi AND title ILIKE '%s'
  fp_infix_2col : WHERE q8_prefix BETWEEN ... AND q8_suffix BETWEEN ... AND title ILIKE ...
  fp_prefix_1col: WHERE (q16_infix >> 8) BETWEEN p_lo AND p_hi AND title ILIKE 'p%'
  fp_suffix_1col: WHERE (q16_infix & 255) BETWEEN s_lo AND s_hi AND title ILIKE '%s'
  fp_infix_1col : WHERE (q16_infix >> 8) BETWEEN ... AND (q16_infix & 255) BETWEEN ... AND title ILIKE ...

Plots 6 speedup lines vs selectivity.
"""

import argparse
import csv
import os
import random
import time

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFIX_BYTES = 8


def norm(s):
    return (s or "").lower()


def key_u64_from_normed(s_norm, nbytes=PREFIX_BYTES):
    b = s_norm.encode("utf-8", errors="ignore")[:nbytes]
    b = b + b"\x00" * (nbytes - len(b))
    return int.from_bytes(b, "big", signed=False)


def next_prefix_normed(s_norm):
    b = bytearray(s_norm.encode("utf-8", errors="ignore"))
    if not b:
        return "\uffff"
    b[-1] = min(255, b[-1] + 1)
    return bytes(b).decode("utf-8", errors="ignore")


def bucket_range(query, boundaries, bits, suffix=False):
    s = norm(query)
    if suffix:
        s = s[::-1]
    lo = key_u64_from_normed(s)
    hi = key_u64_from_normed(next_prefix_normed(s))
    jlo = int(np.clip(np.searchsorted(boundaries, lo, side="right") - 1, 0, len(boundaries) - 1))
    jhi = int(np.clip(np.searchsorted(boundaries, hi, side="right") - 1, 0, len(boundaries) - 1))
    return min(jlo, jhi), max(jlo, jhi)


def time_query(con, sql, warmup=1, reps=5):
    for _ in range(warmup):
        con.execute(sql).fetchall()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000  # ms


def extract_pairs(con, prefix_boundaries, suffix_boundaries, pb, sb, n=3):
    print(f"Extracting real {n}-char (prefix, suffix) pairs...")
    sql = f"""
        SELECT DISTINCT
            LOWER(SUBSTRING(title, 1, {n})) as prefix,
            LOWER(SUBSTRING(title, LENGTH(title) - {n-1}, {n})) as suffix
        FROM two_col
        WHERE LENGTH(title) >= {n}
    """
    result = con.execute(sql).fetchall()
    pairs = []
    for prefix, suffix in result:
        if (not prefix or not suffix or
                len(prefix) != n or len(suffix) != n or
                "'" in prefix or "'" in suffix or
                "\\" in prefix or "\\" in suffix):
            continue
        p_lo, p_hi = bucket_range(prefix, prefix_boundaries, pb, suffix=False)
        s_lo, s_hi = bucket_range(suffix, suffix_boundaries, sb, suffix=True)
        p_span = p_hi - p_lo + 1
        s_span = s_hi - s_lo + 1
        c_span = p_span * s_span
        pairs.append((prefix, suffix, p_lo, p_hi, s_lo, s_hi, p_span, s_span, c_span))
    print(f"Found {len(pairs):,} unique pairs")
    return pairs


def stratified_sample(pairs, total=4000, seed=42):
    bins = [
        (1, 1,           "span=1"),
        (2, 5,           "span=2-5"),
        (6, 10,          "span=6-10"),
        (11, 20,         "span=11-20"),
        (21, 50,         "span=21-50"),
        (51, float("inf"), "span>50"),
    ]
    binned = {label: [] for _, _, label in bins}
    for p in pairs:
        c_span = p[8]
        for lo, hi, label in bins:
            if lo <= c_span <= hi:
                binned[label].append(p)
                break

    print("\nDistribution across bucket span bins:")
    for label in binned:
        print(f"  {label}: {len(binned[label]):,} pairs")

    min_per_bin = 50
    allocs = {}
    remaining = total
    total_avail = sum(len(v) for v in binned.values())
    for label, ps in binned.items():
        if ps:
            alloc = min(min_per_bin, len(ps), remaining)
            allocs[label] = alloc
            remaining -= alloc
        else:
            allocs[label] = 0
    if remaining > 0:
        for label, ps in binned.items():
            if ps:
                extra = int(remaining * len(ps) / total_avail)
                allocs[label] += min(extra, len(ps) - allocs[label])

    rng = random.Random(seed)
    sampled = []
    print("\nStratified sampling:")
    for label, ps in binned.items():
        n_take = min(allocs[label], len(ps))
        if n_take > 0:
            sampled.extend(rng.sample(ps, n_take))
            print(f"  {label}: {n_take} / {len(ps)}")
    print(f"\nTotal sampled: {len(sampled)}")
    return sampled


def plot_results(results, out_dir, sb):
    os.makedirs(out_dir, exist_ok=True)

    variants = [
        ("speedup_prefix_2col", "Prefix  / two-col",  "blue",   "-"),
        ("speedup_prefix_1col", "Prefix  / one-col",  "blue",   "--"),
        ("speedup_suffix_2col", "Suffix  / two-col",  "green",  "-"),
        ("speedup_suffix_1col", "Suffix  / one-col",  "green",  "--"),
        ("speedup_infix_2col",  "Infix   / two-col",  "red",    "-"),
        ("speedup_infix_1col",  "Infix   / one-col",  "red",    "--"),
    ]

    sel_keys = {
        "speedup_prefix_2col": "sel_prefix",
        "speedup_prefix_1col": "sel_prefix",
        "speedup_suffix_2col": "sel_suffix",
        "speedup_suffix_1col": "sel_suffix",
        "speedup_infix_2col":  "sel_infix",
        "speedup_infix_1col":  "sel_infix",
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for key, label, color, ls in variants:
        sels = np.array([r[sel_keys[key]] * 100 for r in results if r[key] > 0])
        sus  = np.array([r[key]              for r in results if r[key] > 0])
        if len(sels) == 0:
            continue

        # Binned average on log scale
        lo, hi = sels.min(), sels.max()
        if lo <= 0 or hi <= lo:
            continue
        bins = np.logspace(np.log10(lo), np.log10(hi), 25)
        centers, means = [], []
        for i in range(len(bins) - 1):
            mask = (sels >= bins[i]) & (sels < bins[i + 1])
            if mask.sum() >= 3:
                centers.append(np.sqrt(bins[i] * bins[i + 1]))
                means.append(np.mean(sus[mask]))

        ax.scatter(sels, sus, color=color, alpha=0.08, s=8)
        if centers:
            ax.plot(centers, means, color=color, linestyle=ls, linewidth=2, label=label)

    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Selectivity (% of rows matching)")
    ax.set_ylabel("Speedup vs full scan")
    ax.set_title("Fingerprint Speedup: Prefix / Suffix / Infix  ×  Two-column / One-column")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(out_dir, "speedup_6way.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")
    plt.close(fig)

    # Also plot by combined bucket span
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    span_keys = {
        "speedup_prefix_2col": "prefix_bucket_span",
        "speedup_prefix_1col": "prefix_bucket_span",
        "speedup_suffix_2col": "suffix_bucket_span",
        "speedup_suffix_1col": "suffix_bucket_span",
        "speedup_infix_2col":  "combined_bucket_span",
        "speedup_infix_1col":  "combined_bucket_span",
    }
    for key, label, color, ls in variants:
        spans = np.array([r[span_keys[key]] for r in results if r[key] > 0])
        sus   = np.array([r[key]            for r in results if r[key] > 0])
        if len(spans) == 0:
            continue
        unique_spans = sorted(set(spans))
        centers = [s for s in unique_spans if (spans == s).sum() >= 3]
        means   = [np.mean(sus[spans == s]) for s in centers]
        ax2.scatter(spans, sus, color=color, alpha=0.08, s=8)
        if centers:
            ax2.plot(centers, means, color=color, linestyle=ls, linewidth=2, label=label)

    ax2.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Bucket span (prefix span, suffix span, or prefix×suffix span)")
    ax2.set_ylabel("Speedup vs full scan")
    ax2.set_title("Fingerprint Speedup by Bucket Span: Prefix / Suffix / Infix  ×  Two-column / One-column")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    out2 = os.path.join(out_dir, "speedup_6way_by_span.png")
    fig2.savefig(out2, dpi=150)
    print(f"Wrote {out2}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="6-way fingerprint benchmark")
    parser.add_argument("--prefix-bits", type=int, default=8)
    parser.add_argument("--suffix-bits", type=int, default=8)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="sixway_plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    pb, sb = args.prefix_bits, args.suffix_bits
    two_col_parquet  = f"title_strs_infix_p{pb}_s{sb}.parquet"
    combined_parquet = f"title_strs_infix16_p{pb}_s{sb}.parquet"

    for path in [two_col_parquet, combined_parquet]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run build_infix.py first.")

    prefix_boundaries = np.load(f"q{pb}_prefix_boundaries.npy")
    suffix_boundaries = np.load(f"q{sb}_suffix_boundaries.npy")
    suffix_mask = (1 << sb) - 1

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    con.execute("PRAGMA enable_object_cache=true")

    print("Loading tables...")
    con.execute(f"CREATE TABLE two_col  AS SELECT * FROM read_parquet('{two_col_parquet}')")
    con.execute(f"CREATE TABLE combined AS SELECT * FROM read_parquet('{combined_parquet}')")

    print("Warming cache...")
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM two_col").fetchall()
    con.execute("SELECT COUNT(*), MIN(title), MAX(title) FROM combined").fetchall()

    total_rows = con.execute("SELECT COUNT(*) FROM two_col").fetchone()[0]
    print(f"Total rows: {total_rows:,}")

    pairs = extract_pairs(con, prefix_boundaries, suffix_boundaries, pb, sb, args.n)
    sampled = stratified_sample(pairs, args.samples, args.seed)

    prefix_col = f"q{pb}_prefix"
    suffix_col  = f"q{sb}_suffix"

    results = []
    print(f"\nBenchmarking {len(sampled)} queries (warmup={args.warmup}, reps={args.reps})...")

    for i, (pq, sq, p_lo, p_hi, s_lo, s_hi, p_span, s_span, c_span) in enumerate(sampled):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {i+1}/{len(sampled)} ({100*(i+1)/len(sampled):.0f}%)")

        pe = pq.replace("'", "''")
        se = sq.replace("'", "''")

        # Full scans
        q_fp = f"SELECT COUNT(*) FROM two_col WHERE title ILIKE '{pe}%'"
        q_fs = f"SELECT COUNT(*) FROM two_col WHERE title ILIKE '%{se}'"
        q_fi = f"SELECT COUNT(*) FROM two_col WHERE title ILIKE '{pe}%' AND title ILIKE '%{se}'"

        mc_p = con.execute(q_fp).fetchone()[0]
        mc_s = con.execute(q_fs).fetchone()[0]
        mc_i = con.execute(q_fi).fetchone()[0]
        if mc_i == 0:
            continue

        # Fingerprint queries — two-column
        q_p2 = (f"SELECT COUNT(*) FROM two_col "
                f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
                f"AND title ILIKE '{pe}%'")
        q_s2 = (f"SELECT COUNT(*) FROM two_col "
                f"WHERE {suffix_col} BETWEEN {s_lo} AND {s_hi} "
                f"AND title ILIKE '%{se}'")
        q_i2 = (f"SELECT COUNT(*) FROM two_col "
                f"WHERE {prefix_col} BETWEEN {p_lo} AND {p_hi} "
                f"AND {suffix_col} BETWEEN {s_lo} AND {s_hi} "
                f"AND title ILIKE '{pe}%' AND title ILIKE '%{se}'")

        # Fingerprint queries — one-column raw BETWEEN
        # Prefix codes live in MSBs: [p_lo<<sb, (p_hi<<sb)|suffix_mask] is contiguous
        # Suffix codes live in LSBs: not contiguous, must keep mask expression
        q_p1 = (f"SELECT COUNT(*) FROM combined "
                f"WHERE q16_infix BETWEEN {p_lo << sb} AND {(p_hi << sb) | suffix_mask} "
                f"AND title ILIKE '{pe}%'")
        q_s1 = (f"SELECT COUNT(*) FROM combined "
                f"WHERE (q16_infix & {suffix_mask}) BETWEEN {s_lo} AND {s_hi} "
                f"AND title ILIKE '%{se}'")
        q_i1 = (f"SELECT COUNT(*) FROM combined "
                f"WHERE q16_infix BETWEEN {p_lo << sb} AND {(p_hi << sb) | suffix_mask} "
                f"AND (q16_infix & {suffix_mask}) BETWEEN {s_lo} AND {s_hi} "
                f"AND title ILIKE '{pe}%' AND title ILIKE '%{se}'")

        t_fp = time_query(con, q_fp, args.warmup, args.reps)
        t_fs = time_query(con, q_fs, args.warmup, args.reps)
        t_fi = time_query(con, q_fi, args.warmup, args.reps)
        t_p2 = time_query(con, q_p2, args.warmup, args.reps)
        t_s2 = time_query(con, q_s2, args.warmup, args.reps)
        t_i2 = time_query(con, q_i2, args.warmup, args.reps)
        t_p1 = time_query(con, q_p1, args.warmup, args.reps)
        t_s1 = time_query(con, q_s1, args.warmup, args.reps)
        t_i1 = time_query(con, q_i1, args.warmup, args.reps)

        results.append({
            "prefix": pq, "suffix": sq,
            "match_count_prefix": mc_p,
            "match_count_suffix": mc_s,
            "match_count_infix":  mc_i,
            "sel_prefix": mc_p / total_rows,
            "sel_suffix": mc_s / total_rows,
            "sel_infix":  mc_i / total_rows,
            "prefix_bucket_span":   p_span,
            "suffix_bucket_span":   s_span,
            "combined_bucket_span": c_span,
            "time_full_prefix_ms": t_fp,
            "time_full_suffix_ms": t_fs,
            "time_full_infix_ms":  t_fi,
            "time_prefix_2col_ms": t_p2,
            "time_suffix_2col_ms": t_s2,
            "time_infix_2col_ms":  t_i2,
            "time_prefix_1col_ms": t_p1,
            "time_suffix_1col_ms": t_s1,
            "time_infix_1col_ms":  t_i1,
            "speedup_prefix_2col": t_fp / t_p2 if t_p2 > 0 else 0,
            "speedup_suffix_2col": t_fs / t_s2 if t_s2 > 0 else 0,
            "speedup_infix_2col":  t_fi / t_i2 if t_i2 > 0 else 0,
            "speedup_prefix_1col": t_fp / t_p1 if t_p1 > 0 else 0,
            "speedup_suffix_1col": t_fs / t_s1 if t_s1 > 0 else 0,
            "speedup_infix_1col":  t_fi / t_i1 if t_i1 > 0 else 0,
        })

    print(f"\nCompleted {len(results)} queries with matches")

    if results:
        keys = ["speedup_prefix_2col", "speedup_suffix_2col", "speedup_infix_2col",
                "speedup_prefix_1col", "speedup_suffix_1col", "speedup_infix_1col"]
        labels = ["Prefix/2col", "Suffix/2col", "Infix/2col",
                  "Prefix/1col", "Suffix/1col", "Infix/1col"]

        print("\n" + "=" * 65)
        print("SUMMARY (geometric mean speedup)")
        print("=" * 65)
        for k, lbl in zip(keys, labels):
            vals = [r[k] for r in results if r[k] > 0]
            gm = np.exp(np.mean(np.log(vals))) if vals else 0
            print(f"  {lbl:<18}: {gm:.2f}x")
        print("=" * 65)

        out_csv = args.csv or f"sixway_p{pb}_s{sb}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {out_csv}")

        plot_results(results, args.out_dir, sb)

    con.close()


if __name__ == "__main__":
    main()
