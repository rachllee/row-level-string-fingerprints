"""
Plot speedup vs. selectivity from bench_all_prefixes.py output.

This shows how fingerprint speedup correlates with query selectivity,
providing insight into when fingerprints help most.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_speedup_vs_selectivity(df, out_path, title_suffix=""):
    """Scatter plot of speedup vs selectivity with trend line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for selectivity since it spans many orders of magnitude
    selectivity_pct = df["selectivity"] * 100
    speedup = df["speedup"]

    # Scatter plot
    scatter = ax.scatter(
        selectivity_pct,
        speedup,
        alpha=0.5,
        s=20,
        c=df["prune_rate"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Selectivity (% of rows matching query)")
    ax.set_ylabel("Speedup (full / fingerprint)")
    ax.set_title(f"Speedup vs. Query Selectivity{title_suffix}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No speedup")
    ax.grid(True, alpha=0.3)

    # Colorbar for prune rate
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Prune Rate (fraction of rows skipped)")

    # Add trend line using binned averages
    bins = np.logspace(np.log10(selectivity_pct.min()), np.log10(selectivity_pct.max()), 20)
    df_temp = df.copy()
    df_temp["sel_pct"] = selectivity_pct
    df_temp["bin"] = pd.cut(df_temp["sel_pct"], bins=bins)
    binned = df_temp.groupby("bin", observed=True)["speedup"].mean()

    bin_centers = [(b.left + b.right) / 2 for b in binned.index]
    ax.plot(bin_centers, binned.values, "r-", linewidth=2, label="Binned average")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_speedup_vs_prune_rate(df, out_path, title_suffix=""):
    """Scatter plot of speedup vs prune rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    prune_pct = df["prune_rate"] * 100
    speedup = df["speedup"]

    ax.scatter(prune_pct, speedup, alpha=0.5, s=20)
    ax.set_xlabel("Prune Rate (% of rows skipped by fingerprint)")
    ax.set_ylabel("Speedup (full / fingerprint)")
    ax.set_title(f"Speedup vs. Prune Rate{title_suffix}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)

    # Trend line
    bins = np.linspace(prune_pct.min(), prune_pct.max(), 20)
    df_temp = df.copy()
    df_temp["prune_pct"] = prune_pct
    df_temp["bin"] = pd.cut(df_temp["prune_pct"], bins=bins)
    binned = df_temp.groupby("bin", observed=True)["speedup"].mean()

    bin_centers = [(b.left + b.right) / 2 for b in binned.index]
    ax.plot(bin_centers, binned.values, "r-", linewidth=2, label="Binned average")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_speedup_histogram(df, out_path, title_suffix=""):
    """Histogram of speedup values."""
    fig, ax = plt.subplots(figsize=(10, 5))

    speedup = df["speedup"]
    ax.hist(speedup, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(speedup.median(), color="red", linestyle="--", linewidth=2,
               label=f"Median: {speedup.median():.2f}x")
    ax.axvline(speedup.mean(), color="orange", linestyle="--", linewidth=2,
               label=f"Mean: {speedup.mean():.2f}x")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=2, label="No speedup (1.0x)")

    ax.set_xlabel("Speedup")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Speedup Across All Queries{title_suffix}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_selectivity_histogram(df, out_path, title_suffix=""):
    """Histogram of selectivity values (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    selectivity_pct = df["selectivity"] * 100

    # Use log-spaced bins
    bins = np.logspace(
        np.log10(max(selectivity_pct.min(), 1e-6)),
        np.log10(selectivity_pct.max()),
        50
    )
    ax.hist(selectivity_pct, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Selectivity (% of rows matching)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Query Selectivity{title_suffix}")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_bucket_span_vs_speedup(df, out_path, title_suffix=""):
    """Plot bucket span (number of buckets query covers) vs speedup."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["bucket_span"], df["speedup"], alpha=0.5, s=20)
    ax.set_xlabel("Bucket Span (number of buckets query covers)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup vs. Bucket Span{title_suffix}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)

    # Log scale if range is large
    if df["bucket_span"].max() > 100:
        ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_summary_by_selectivity_bin(df):
    """Print summary statistics grouped by selectivity bins."""
    df = df.copy()
    df["sel_pct"] = df["selectivity"] * 100

    # Create bins: <0.01%, 0.01-0.1%, 0.1-1%, 1-10%, >10%
    bins = [0, 0.01, 0.1, 1, 10, 100]
    labels = ["<0.01%", "0.01-0.1%", "0.1-1%", "1-10%", ">10%"]
    df["sel_bin"] = pd.cut(df["sel_pct"], bins=bins, labels=labels)

    print("\nSpeedup by Selectivity Bin:")
    print("=" * 70)
    print(f"{'Selectivity':<15} {'Count':>8} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    for label in labels:
        sub = df[df["sel_bin"] == label]
        if len(sub) > 0:
            print(f"{label:<15} {len(sub):>8} {sub['speedup'].mean():>8.2f} "
                  f"{sub['speedup'].median():>8.2f} {sub['speedup'].min():>8.2f} "
                  f"{sub['speedup'].max():>8.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Plot speedup vs. selectivity")
    parser.add_argument("--csv", required=True, help="CSV from bench_all_prefixes.py")
    parser.add_argument("--out-dir", default="selectivity_plots", help="Output directory")
    parser.add_argument("--title", default="", help="Title suffix for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    title_suffix = f" ({args.title})" if args.title else ""

    # Generate all plots
    plot_speedup_vs_selectivity(df, os.path.join(args.out_dir, "speedup_vs_selectivity.png"), title_suffix)
    plot_speedup_vs_prune_rate(df, os.path.join(args.out_dir, "speedup_vs_prune_rate.png"), title_suffix)
    plot_speedup_histogram(df, os.path.join(args.out_dir, "speedup_histogram.png"), title_suffix)
    plot_selectivity_histogram(df, os.path.join(args.out_dir, "selectivity_histogram.png"), title_suffix)
    plot_bucket_span_vs_speedup(df, os.path.join(args.out_dir, "bucket_span_vs_speedup.png"), title_suffix)

    # Print summary
    print_summary_by_selectivity_bin(df)

    print(f"\nAll plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
