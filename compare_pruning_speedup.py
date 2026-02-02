import argparse
import os
import re

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sanitize(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def render_table_png(df, out_path, max_rows=80):
    df_show = df.copy()
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    fig, ax = plt.subplots(figsize=(14, 0.35 * (len(df_show) + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def norm(s):
    return (s or "").lower()


def normalize_query(s: str, suffix: bool) -> str:
    s = norm(s)
    return s[::-1] if suffix else s


def key_u64_from_normed(s_norm, nbytes=8):
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


def load_boundaries(bits: int, mode: str):
    path = f"q{bits}_{mode}_boundaries.npy"
    if os.path.exists(path):
        return np.load(path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-summary",
        default="bench_summary.csv",
        help="CSV produced by summarize_bench.py",
    )
    parser.add_argument(
        "--profile-rows",
        required=True,
        help="profile_rows.csv produced by bench.py --profile-dir",
    )
    parser.add_argument("--mode", default="", help="Filter mode (prefix or suffix).")
    parser.add_argument("--source", default="", help="Filter source (view or table).")
    parser.add_argument(
        "--parquet-template",
        default="title_strs_{mode}_b{bits}.parquet",
        help="Parquet path template for counting rows (uses {mode} and {bits}).",
    )
    parser.add_argument("--out-csv", default="pruning_speedup.csv", help="Output CSV.")
    parser.add_argument("--out-dir", default="pruning_speedup_plots", help="Output directory.")
    parser.add_argument("--table-png", default="pruning_speedup_table.png", help="PNG table output.")
    args = parser.parse_args()

    bench_df = pd.read_csv(args.bench_summary)
    prof_df = pd.read_csv(args.profile_rows)

    mode = args.mode or "prefix"
    if args.mode:
        prof_df = prof_df[prof_df["mode"] == args.mode]
    if args.source:
        prof_df = prof_df[prof_df["source"] == args.source]

    prof_df["rows_scanned"] = pd.to_numeric(prof_df["rows_scanned"], errors="coerce")
    pivot = (
        prof_df.pivot_table(
            index=["bits", "mode", "source", "query"],
            columns="variant",
            values="rows_scanned",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"full": "rows_full", "fp_exact": "rows_fp"})
    )

    pivot["rows_skipped"] = pivot["rows_full"] - pivot["rows_fp"]
    pivot["skip_pct"] = 1.0 - (pivot["rows_fp"] / pivot["rows_full"])

    merged = bench_df.merge(
        pivot,
        left_on=["bits", "source", "prefix"],
        right_on=["bits", "source", "query"],
        how="left",
    )

    # Compute prunable rows directly from prefix buckets to avoid relying on
    # cumulative scan counts (which are often identical for full/fp scans).
    suffix = mode == "suffix"
    prefixes = sorted(merged["prefix"].dropna().unique())
    bits_list = sorted(merged["bits"].dropna().unique())
    prunable = {}
    for bits in bits_list:
        parquet_path = args.parquet_template.format(bits=bits, mode=mode)
        if not os.path.exists(parquet_path):
            continue
        boundaries = load_boundaries(bits, mode)
        code_col = f"q{bits}_{mode}"
        con = duckdb.connect()
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{parquet_path}')")
        total = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        for prefix in prefixes:
            lo, hi = bucket_range(prefix, boundaries, bits, suffix=suffix)
            rows_fp = con.execute(
                f"SELECT COUNT(*) FROM t WHERE {code_col} BETWEEN {lo} AND {hi}"
            ).fetchone()[0]
            prunable[(bits, prefix)] = (total, rows_fp)
        con.close()

    if prunable:
        rows_full = []
        rows_fp = []
        for _, row in merged.iterrows():
            key = (row["bits"], row["prefix"])
            if key in prunable:
                total, fp = prunable[key]
                rows_full.append(total)
                rows_fp.append(fp)
            else:
                rows_full.append(row.get("rows_full"))
                rows_fp.append(row.get("rows_fp"))
        merged["rows_full"] = rows_full
        merged["rows_fp"] = rows_fp
        merged["rows_skipped"] = merged["rows_full"] - merged["rows_fp"]
        merged["skip_pct"] = 1.0 - (merged["rows_fp"] / merged["rows_full"])

    if "rows_full" in merged.columns and "rows_fp" in merged.columns:
        merged["pruning_rate_pct"] = (merged["rows_fp"] / merged["rows_full"]) * 100.0

    cols = [
        "bits",
        "mode",
        "source",
        "prefix",
        "rows_full",
        "rows_fp",
        "rows_skipped",
        "skip_pct",
        "pruning_rate_pct",
        "speedup_median",
        "speedup_iqr",
        "time_full_ms",
        "time_fp_ms",
    ]
    cols = [c for c in cols if c in merged.columns]
    merged = merged[cols].sort_values(["prefix", "source", "bits"])

    merged.to_csv(args.out_csv, index=False)
    os.makedirs(args.out_dir, exist_ok=True)
    render_table_png(merged, os.path.join(args.out_dir, args.table_png))

    for prefix, g_all in merged.groupby("prefix"):
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        g_all = g_all.sort_values("bits")
        bits = np.sort(g_all["bits"].unique())
        sources = sorted(g_all["source"].dropna().unique())
        colors = {"table": "#1f77b4", "view": "#2ca02c"}
        y_skip = g_all["pruning_rate_pct"].to_numpy()
        ax1.plot(
            g_all["bits"].to_numpy(),
            y_skip,
            marker="o",
            linestyle="--",
            color="#999999",
            label="Pruning rate (% rows checked)",
        )
        ax1.set_xlabel("Bits")
        ax1.set_ylabel("Pruning rate (% rows checked)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_xticks(bits)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = ax1.twinx()
        for source in sources:
            sub = g_all[g_all["source"] == source].sort_values("bits")
            ax2.plot(
                sub["bits"].to_numpy(),
                sub["speedup_median"].to_numpy(),
                marker="o",
                linestyle="-",
                color=colors.get(source, "#ff7f0e"),
                label=f"Speedup ({source})",
            )
        ax2.set_ylabel("Speedup (median full / median fp)", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        fig.tight_layout()
        out_path = os.path.join(
            args.out_dir,
            f"{sanitize(prefix)}_pruning_vs_speedup.png",
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Wrote comparison CSV: {args.out_csv}")
    print(f"Wrote table PNG: {os.path.join(args.out_dir, args.table_png)}")
    print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
