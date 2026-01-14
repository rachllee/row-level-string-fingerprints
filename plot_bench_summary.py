import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sanitize(name):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def render_table_png(df, out_path, max_rows=60):
    df_show = df.copy()
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)
    fig, ax = plt.subplots(figsize=(12, 0.35 * (len(df_show) + 2)))
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


def plot_by_query(df, out_dir, y_col, y_label, y_err_col=None):
    bits = np.sort(df["bits"].unique())
    sources = sorted(df["source"].unique())
    prefixes = sorted(df["prefix"].unique())

    for prefix in prefixes:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for source in sources:
            sub = df[(df["prefix"] == prefix) & (df["source"] == source)]
            if sub.empty:
                continue
            sub = sub.sort_values("bits")
            x = sub["bits"].to_numpy()
            y = sub[y_col].to_numpy()
            yerr = None
            if y_err_col and y_err_col in sub.columns:
                yerr = (sub[y_err_col].to_numpy()) / 2.0
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, label=source, capsize=4)

        ax.set_xticks(bits)
        ax.set_xlabel("Bits")
        ax.set_ylabel(y_label)
        ax.set_title(f"{prefix}%")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{sanitize(prefix)}_{sanitize(y_col)}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_time_by_query(df, out_dir):
    bits = np.sort(df["bits"].unique())
    sources = sorted(df["source"].unique())
    prefixes = sorted(df["prefix"].unique())

    for prefix in prefixes:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for source in sources:
            sub = df[(df["prefix"] == prefix) & (df["source"] == source)]
            if sub.empty:
                continue
            sub = sub.sort_values("bits")
            x = sub["bits"].to_numpy()
            full = sub["time_full_ms"].to_numpy()
            fp = sub["time_fp_ms"].to_numpy()
            ax.plot(x, full, marker="o", linewidth=1.5, label=f"{source} full")
            ax.plot(x, fp, marker="o", linewidth=1.5, label=f"{source} fp+exact")

        ax.set_xticks(bits)
        ax.set_xlabel("Bits")
        ax.set_ylabel("Median time (ms)")
        ax.set_title(f"{prefix}%")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{sanitize(prefix)}_time_ms.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="bench_summary.csv", help="Input summary CSV")
    parser.add_argument("--out-dir", default="bench_summary_plots_16", help="Output directory")
    parser.add_argument("--table-png", default="bench_summary_table.png", help="PNG table output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {
        "bits",
        "source",
        "prefix",
        "time_full_ms",
        "time_fp_ms",
        "speedup_median",
        "speedup_iqr",
    }
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns in summary CSV: {missing_list}")

    df["bits"] = df["bits"].astype(int)
    df = df.sort_values(["prefix", "source", "bits"])

    os.makedirs(args.out_dir, exist_ok=True)
    render_table_png(df, os.path.join(args.out_dir, args.table_png))

    plot_by_query(
        df,
        args.out_dir,
        y_col="speedup_median",
        y_label="Speedup (median full / median fp)",
        y_err_col="speedup_iqr",
    )
    plot_time_by_query(df, args.out_dir)

    print(f"Wrote table PNG: {os.path.join(args.out_dir, args.table_png)}")
    print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
