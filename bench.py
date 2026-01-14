import argparse
import csv
import json
import os
import shutil
import subprocess
import duckdb
import numpy as np
import time

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


def run_benchmark(
    con,
    parquet,
    boundaries,
    code_col,
    queries,
    warmup,
    reps,
    csv_rows,
    source_label,
    bits,
    explain,
    suffix,
    mode,
    profile_dir,
    profile_rows,
    profile_shell,
    duckdb_bin,
    force_uncompressed_table,
):
    con.execute("DROP VIEW IF EXISTS t")
    con.execute("DROP TABLE IF EXISTS t")
    if source_label == "table":
        if force_uncompressed_table:
            con.execute("PRAGMA force_compression='uncompressed'")
        con.execute(f"CREATE TABLE t AS SELECT * FROM read_parquet('{parquet}')")
    else:
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{parquet}')")

    all_full = []
    all_fp = []
    speedups = []

    like_left = "%" if suffix else ""
    like_right = "" if suffix else "%"

    for q in queries:
        lo, hi = bucket_range(q, boundaries, bits, suffix=suffix)
        q_full = f"SELECT COUNT(*) FROM t WHERE title LIKE '{like_left}{q}{like_right}'"
        q_fp = (
            f"SELECT COUNT(*) FROM t WHERE {code_col} BETWEEN {lo} AND {hi} "
            f"AND title LIKE '{like_left}{q}{like_right}'"
        )

        if explain:
            label = f"{like_left}{q}{like_right}"
            print(f"\n[{source_label}] EXPLAIN ANALYZE {label} full scan:")
            plan_full = con.execute("EXPLAIN ANALYZE " + q_full).fetchall()
            print(plan_full)
            print(f"\n[{source_label}] EXPLAIN ANALYZE {label} fp+exact:")
            plan_fp = con.execute("EXPLAIN ANALYZE " + q_fp).fetchall()
            print(plan_fp)

        full_times = time_query(con, q_full, warmup=warmup, reps=reps)
        fp_times = time_query(con, q_fp, warmup=warmup, reps=reps)

        full_times = time_query(con, q_full, warmup=warmup, reps=reps)
        fp_times = time_query(con, q_fp, warmup=warmup, reps=reps)

        all_full.extend(full_times)
        all_fp.extend(fp_times)

        if csv_rows is not None:
            for i, t in enumerate(full_times):
                csv_rows.append((bits, source_label, q, "full", i, t))
            for i, t in enumerate(fp_times):
                csv_rows.append((bits, source_label, q, "fp_exact", i, t))

        full_med, full_p10, full_p90, full_iqr = summarize_times(full_times)
        fp_med, fp_p10, fp_p90, fp_iqr = summarize_times(fp_times)
        speedup = full_med / fp_med
        speedups.append(speedup)

        label = f"{like_left}{q}{like_right}"
        print(f"\n[{source_label}] {label}  buckets[{lo},{hi}]")
        print(
            f"  full scan: median {full_med*1000:.2f} ms "
            f"(P10 {full_p10*1000:.2f}, P90 {full_p90*1000:.2f}, IQR {full_iqr*1000:.2f})"
        )
        print(
            f"  fp+exact:  median {fp_med*1000:.2f} ms "
            f"(P10 {fp_p10*1000:.2f}, P90 {fp_p90*1000:.2f}, IQR {fp_iqr*1000:.2f})"
        )
        print(f"  speedup (med/med): {speedup:.2f}x")

        if profile_dir and profile_rows is not None:
            label = f"{like_left}{q}{like_right}".replace("%", "pct")
            base = f"{mode}_b{bits}_{source_label}_{label}"
            full_path = os.path.join(profile_dir, f"{base}_full.json")
            fp_path = os.path.join(profile_dir, f"{base}_fp.json")

            if profile_shell:
                rows_full = run_profile_shell(
                    duckdb_bin,
                    parquet,
                    source_label,
                    q_full,
                    full_path,
                    force_uncompressed_table,
                )
                rows_fp = run_profile_shell(
                    duckdb_bin,
                    parquet,
                    source_label,
                    q_fp,
                    fp_path,
                    force_uncompressed_table,
                )
            else:
                rows_full = None
                rows_fp = None

            profile_rows.append((bits, mode, source_label, q, "full", rows_full, full_path))
            profile_rows.append((bits, mode, source_label, q, "fp_exact", rows_fp, fp_path))

    if all_full and all_fp:
        full_mean = float(np.mean(all_full))
        fp_mean = float(np.mean(all_fp))
        speedup_geo = geomean(speedups)
        print(f"\n[{source_label}] Aggregate across queries:")
        print(f"  mean full scan: {full_mean*1000:.2f} ms")
        print(f"  mean fp+exact:  {fp_mean*1000:.2f} ms")
        print(f"  geomean speedup (med/med): {speedup_geo:.2f}x")


def main(argv=None, default_suffix=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..28). Default: 8")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs to discard per query.")
    parser.add_argument("--reps", type=int, default=10, help="Timed runs per query after warmup.")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV path for per-run timings.")
    parser.add_argument("--explain", action="store_true", help="Print EXPLAIN ANALYZE for each query.")
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="Write JSON query profiles (and a summary CSV) to this directory.",
    )
    parser.add_argument(
        "--profile-shell",
        action="store_true",
        help="Use the duckdb shell to generate per-query JSON profiles.",
    )
    parser.add_argument(
        "--duckdb-bin",
        type=str,
        default="duckdb",
        help="Path to duckdb CLI (used with --profile-shell).",
    )
    parser.add_argument(
        "--suffix",
        action="store_true",
        default=default_suffix,
        help="Benchmark suffix queries instead of prefix queries.",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="",
        help="Override parquet path. Supports {bits} and {mode} format tokens.",
    )
    parser.add_argument(
        "--force-uncompressed-table",
        action="store_true",
        help="Force uncompressed storage for CTAS tables (disables FSST-style compression).",
    )
    args = parser.parse_args(argv)

    K = args.bits
    if K < 1 or K > 28:
        raise ValueError("--bits must be between 1 and 28")

    mode = "suffix" if args.suffix else "prefix"
    parquet = f"title_strs_{mode}_b{K}.parquet"
    if args.parquet_path:
        parquet = args.parquet_path.format(bits=K, mode=mode)
    boundaries_npy = f"q{K}_{mode}_boundaries.npy"
    code_col = f"q{K}_{mode}"

    boundaries = np.load(boundaries_npy) if os.path.exists(boundaries_npy) else None

    con = duckdb.connect()

    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA enable_object_cache=true")

    queries = ["jos", "2012", "the", "a", "(", "#", "interest"]

    csv_rows = [] if args.csv else None
    profile_rows = [] if args.profile_dir else None
    if args.profile_dir:
        os.makedirs(args.profile_dir, exist_ok=True)

    run_benchmark(
        con,
        parquet,
        boundaries,
        code_col,
        queries,
        warmup=args.warmup,
        reps=args.reps,
        csv_rows=csv_rows,
        source_label="view",
        bits=K,
        explain=args.explain,
        suffix=args.suffix,
        mode=mode,
        profile_dir=args.profile_dir,
        profile_rows=profile_rows,
        profile_shell=args.profile_shell,
        duckdb_bin=args.duckdb_bin,
        force_uncompressed_table=args.force_uncompressed_table,
    )
    run_benchmark(
        con,
        parquet,
        boundaries,
        code_col,
        queries,
        warmup=args.warmup,
        reps=args.reps,
        csv_rows=csv_rows,
        source_label="table",
        bits=K,
        explain=args.explain,
        suffix=args.suffix,
        mode=mode,
        profile_dir=args.profile_dir,
        profile_rows=profile_rows,
        profile_shell=args.profile_shell,
        duckdb_bin=args.duckdb_bin,
        force_uncompressed_table=args.force_uncompressed_table,
    )

    if args.csv and csv_rows is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bits", "source", "prefix", "query", "run", "time_s"])
            w.writerows(csv_rows)
        print(f"\nWrote per-run timings to {args.csv}")

    if args.profile_dir and profile_rows is not None:
        summary_path = os.path.join(args.profile_dir, "profile_rows.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bits", "mode", "source", "query", "variant", "rows_scanned", "profile_path"])
            w.writerows(profile_rows)
        print(f"Wrote profile summary to {summary_path}")


def sql_quote(s: str) -> str:
    return s.replace("'", "''")


def run_profile_shell(
    duckdb_bin, parquet, source_label, sql, out_path, force_uncompressed_table
):
    if not duckdb_bin:
        return None
    if not shutil.which(duckdb_bin):
        return None

    quoted_path = sql_quote(out_path)
    quoted_parquet = sql_quote(parquet)
    if source_label == "table":
        if force_uncompressed_table:
            create_stmt = (
                "PRAGMA force_compression='uncompressed';\n"
                f"CREATE TABLE t AS SELECT * FROM read_parquet('{quoted_parquet}')"
            )
        else:
            create_stmt = f"CREATE TABLE t AS SELECT * FROM read_parquet('{quoted_parquet}')"
    else:
        create_stmt = f"CREATE VIEW t AS SELECT * FROM read_parquet('{quoted_parquet}')"

    sql_script = "\n".join(
        [
            "PRAGMA enable_profiling='json';",
            f"PRAGMA profiling_output='{quoted_path}';",
            "DROP VIEW IF EXISTS t;",
            "DROP TABLE IF EXISTS t;",
            f"{create_stmt};",
            f"{sql};",
        ]
    )

    try:
        subprocess.run(
            [duckdb_bin, "-c", sql_script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None

    if not os.path.exists(out_path):
        return None

    try:
        with open(out_path, "r") as f:
            data = json.load(f)
        return data.get("cumulative_rows_scanned")
    except Exception:
        return None


if __name__ == "__main__":
    main()
