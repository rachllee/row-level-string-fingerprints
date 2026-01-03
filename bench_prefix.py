import argparse
import csv
import duckdb
import numpy as np
import time

PREFIX_BYTES = 8

def norm(s): return (s or "").lower()

def prefix_key_u64(s, nbytes=PREFIX_BYTES):
    b = norm(s).encode("utf-8", errors="ignore")[:nbytes]
    b = b + b"\x00" * (nbytes - len(b))
    return int.from_bytes(b, "big", signed=False)

def next_prefix(p: str) -> str:
    b = bytearray(norm(p).encode("utf-8", errors="ignore"))
    if not b:
        return "\uffff"
    b[-1] = min(255, b[-1] + 1)
    return bytes(b).decode("utf-8", errors="ignore")

def bucket_range(prefix: str, boundaries: np.ndarray):
    lo = prefix_key_u64(prefix)
    hi = prefix_key_u64(next_prefix(prefix))
    jlo = np.searchsorted(boundaries, lo, side="right") - 1
    jhi = np.searchsorted(boundaries, hi, side="right") - 1
    jlo = int(np.clip(jlo, 0, len(boundaries)-1))
    jhi = int(np.clip(jhi, 0, len(boundaries)-1))
    return min(jlo, jhi), max(jlo, jhi)

def time_query(con, sql, warmup=1, reps=10):
    for _ in range(warmup):
        con.execute(sql).fetchone()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        con.execute(sql).fetchone()
        times.append(time.perf_counter() - t0)
    return times


def summarize_times(times):
    p10 = float(np.percentile(times, 10))
    p90 = float(np.percentile(times, 90))
    med = float(np.median(times))
    iqr = float(np.percentile(times, 75) - np.percentile(times, 25))
    return med, p10, p90, iqr


def geomean(values):
    vals = np.asarray(values, dtype=float)
    if np.any(vals <= 0):
        raise ValueError("geomean requires all values > 0")
    return float(np.exp(np.mean(np.log(vals))))


def run_benchmark(con, parquet, boundaries, code_col, prefixes, warmup, reps, csv_rows, source_label, bits, explain):
    con.execute("DROP VIEW IF EXISTS t")
    con.execute("DROP TABLE IF EXISTS t")
    if source_label == "table":
        con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")
    else:
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{parquet}')")

    all_full = []
    all_fp = []
    speedups = []

    for p in prefixes:
        lo, hi = bucket_range(p, boundaries)
        q_full = f"SELECT COUNT(*) FROM t WHERE title LIKE '{p}%'"
        q_fp = (
            f"SELECT COUNT(*) FROM t WHERE {code_col} BETWEEN {lo} AND {hi} "
            f"AND title LIKE '{p}%'"
        )

        if explain:
            print(f"\n[{source_label}] EXPLAIN ANALYZE {p}% full scan:")
            plan_full = con.execute("EXPLAIN ANALYZE " + q_full).fetchall()
            print(plan_full)
            print(f"\n[{source_label}] EXPLAIN ANALYZE {p}% fp+exact:")
            plan_fp = con.execute("EXPLAIN ANALYZE " + q_fp).fetchall()
            print(plan_fp)

        full_times = time_query(con, q_full, warmup=warmup, reps=reps)
        fp_times = time_query(con, q_fp, warmup=warmup, reps=reps)

        #again to clear the cache
        full_times = time_query(con, q_full, warmup=warmup, reps=reps)
        fp_times = time_query(con, q_fp, warmup=warmup, reps=reps)

        all_full.extend(full_times)
        all_fp.extend(fp_times)

        if csv_rows is not None:
            for i, t in enumerate(full_times):
                csv_rows.append((bits, source_label, p, "full", i, t))
            for i, t in enumerate(fp_times):
                csv_rows.append((bits, source_label, p, "fp_exact", i, t))

        full_med, full_p10, full_p90, full_iqr = summarize_times(full_times)
        fp_med, fp_p10, fp_p90, fp_iqr = summarize_times(fp_times)
        speedup = full_med / fp_med
        speedups.append(speedup)

        print(f"\n[{source_label}] {p}%  buckets[{lo},{hi}]")
        print(
            f"  full scan: median {full_med*1000:.2f} ms "
            f"(P10 {full_p10*1000:.2f}, P90 {full_p90*1000:.2f}, IQR {full_iqr*1000:.2f})"
        )
        print(
            f"  fp+exact:  median {fp_med*1000:.2f} ms "
            f"(P10 {fp_p10*1000:.2f}, P90 {fp_p90*1000:.2f}, IQR {fp_iqr*1000:.2f})"
        )
        print(f"  speedup (med/med): {speedup:.2f}x")

    if all_full and all_fp:
        full_mean = float(np.mean(all_full))
        fp_mean = float(np.mean(all_fp))
        speedup_geo = geomean(speedups)
        print(f"\n[{source_label}] Aggregate across prefixes:")
        print(f"  mean full scan: {full_mean*1000:.2f} ms")
        print(f"  mean fp+exact:  {fp_mean*1000:.2f} ms")
        print(f"  geomean speedup (med/med): {speedup_geo:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..16). Default: 8")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs to discard per query.")
    parser.add_argument("--reps", type=int, default=10, help="Timed runs per query after warmup.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV path for per-run timings.")
    parser.add_argument("--explain", action="store_true", help="Print EXPLAIN ANALYZE for each query.")
    args = parser.parse_args()

    K = args.bits
    if K < 1 or K > 16:
        raise ValueError("--bits must be between 1 and 16")

    parquet = f"title_strs_prefix_b{K}.parquet"
    boundaries_npy = f"q{K}_prefix_boundaries.npy"
    code_col = f"q{K}_prefix"

    boundaries = np.load(boundaries_npy)

    con = duckdb.connect()

    # Helpful to reduce variability:
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA enable_object_cache=true")

    prefixes = ["jos", "2012", "the", "a", "(", "#"]

    csv_rows = [] if args.csv else None

    run_benchmark(
        con,
        parquet,
        boundaries,
        code_col,
        prefixes,
        warmup=args.warmup,
        reps=args.reps,
        csv_rows=csv_rows,
        source_label="view",
        bits=K,
        explain=args.explain,
    )
    run_benchmark(
        con,
        parquet,
        boundaries,
        code_col,
        prefixes,
        warmup=args.warmup,
        reps=args.reps,
        csv_rows=csv_rows,
        source_label="table",
        bits=K,
        explain=args.explain,
    )

    if args.csv and csv_rows is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bits", "source", "prefix", "query", "run", "time_s"])
            w.writerows(csv_rows)
        print(f"\nWrote per-run timings to {args.csv}")


if __name__ == "__main__":
    main()
