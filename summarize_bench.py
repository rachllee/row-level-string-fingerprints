import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_group(df_group):
    bits = int(df_group["bits"].iloc[0])
    source = df_group["source"].iloc[0]
    prefix = df_group["prefix"].iloc[0]

    full = df_group[df_group["query"] == "full"]
    fp = df_group[df_group["query"] == "fp_exact"]

    full_times = full["time_s"].to_numpy()
    fp_times = fp["time_s"].to_numpy()

    median_full = float(np.median(full_times))
    median_fp = float(np.median(fp_times))
    speedup_med = median_full / median_fp if median_fp > 0 else float("nan")

    merged = pd.merge(
        full[["run", "time_s"]],
        fp[["run", "time_s"]],
        on="run",
        suffixes=("_full", "_fp"),
    )
    ratios = (merged["time_s_full"] / merged["time_s_fp"]).to_numpy()
    if len(ratios) > 0:
        p25 = float(np.percentile(ratios, 25))
        p75 = float(np.percentile(ratios, 75))
        iqr = p75 - p25
    else:
        p25 = float("nan")
        p75 = float("nan")
        iqr = float("nan")

    return {
        "bits": bits,
        "source": source,
        "prefix": prefix,
        "query": f"{prefix}%",
        "sketch_bytes": bits / 8.0,
        "time_full_ms": median_full * 1000.0,
        "time_fp_ms": median_fp * 1000.0,
        "speedup_median": speedup_med,
        "speedup_iqr": iqr,
        "speedup_p25": p25,
        "speedup_p75": p75,
    }


def plot_times(df_summary, out_dir):
    for source, sdf in df_summary.groupby("source"):
        prefixes = sdf["prefix"].tolist()
        full = sdf["time_full_ms"].to_numpy()
        fp = sdf["time_fp_ms"].to_numpy()

        x = np.arange(len(prefixes))
        width = 0.38

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, full, width, label="full scan")
        ax.bar(x + width / 2, fp, width, label="fp+exact")
        ax.set_xticks(x)
        ax.set_xticklabels(prefixes)
        ax.set_ylabel("Median time (ms)")
        ax.set_title(f"Median Query Time by Prefix ({source})")
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"bench_times_{source}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_speedup(df_summary, out_dir):
    for source, sdf in df_summary.groupby("source"):
        prefixes = sdf["prefix"].tolist()
        speedup = sdf["speedup_median"].to_numpy()
        p25 = sdf["speedup_p25"].to_numpy()
        p75 = sdf["speedup_p75"].to_numpy()
        lower = np.maximum(0.0, speedup - p25)
        upper = np.maximum(0.0, p75 - speedup)
        yerr = np.vstack([lower, upper])
        yerr = np.nan_to_num(yerr, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.arange(len(prefixes))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(x, speedup, yerr=yerr, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(prefixes)
        ax.set_ylabel("Speedup (median full / median fp)")
        ax.set_title(f"Speedup by Prefix (IQR error bars) ({source})")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"bench_speedup_{source}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_speedup_box(df_runs, out_dir):
    for source, sdf in df_runs.groupby("source"):
        ratios_by_prefix = []
        labels = []
        for prefix, g in sdf.groupby("prefix"):
            full = g[g["query"] == "full"][["run", "time_s"]]
            fp = g[g["query"] == "fp_exact"][["run", "time_s"]]
            merged = pd.merge(full, fp, on="run", suffixes=("_full", "_fp"))
            ratios = (merged["time_s_full"] / merged["time_s_fp"]).to_numpy()
            if len(ratios) > 0:
                ratios_by_prefix.append(ratios)
                labels.append(prefix)

        if not ratios_by_prefix:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(ratios_by_prefix, labels=labels, showfliers=False)
        ax.set_ylabel("Speedup (full / fp) per run")
        ax.set_title(f"Speedup Distribution by Prefix ({source})")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"bench_speedup_box_{source}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV from bench_prefix.py")
    parser.add_argument("--out-table", default="bench_summary.csv", help="Output summary table CSV")
    parser.add_argument("--out-dir", default="bench_plots", help="Output directory for plots")
    parser.add_argument("--plots", action="store_true", help="Generate plots in out-dir.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"bits", "source", "prefix", "query", "run", "time_s"}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in CSV: {missing_list}")

    df["bits"] = df["bits"].astype(int)
    df["run"] = df["run"].astype(int)
    df["time_s"] = df["time_s"].astype(float)

    summaries = []
    for _, g in df.groupby(["bits", "source", "prefix"]):
        summaries.append(summarize_group(g))

    summary_df = pd.DataFrame(summaries).sort_values(["bits", "source", "prefix"])

    write_header = not os.path.exists(args.out_table)
    summary_df.to_csv(
        args.out_table,
        mode="a",
        index=False,
        header=write_header,
        quoting=csv.QUOTE_MINIMAL,
    )

    if args.plots:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_times(summary_df, args.out_dir)
        plot_speedup(summary_df, args.out_dir)
        plot_speedup_box(df, args.out_dir)

    print(f"Wrote table: {args.out_table}")
    if args.plots:
        print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
