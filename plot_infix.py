"""
Plot infix benchmark results with informative x-axis labels showing prefix/suffix combinations.
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_and_merge_csvs(csv_dir):
    """Load all infix CSV files and add prefix_bits/suffix_bits columns."""
    import glob

    pattern = os.path.join(csv_dir, "infix_p*_s*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f"No infix CSV files found in {csv_dir}")

    dfs = []
    for f in files:
        # Extract prefix/suffix bits from filename: infix_p8_s8.csv
        match = re.search(r"infix_p(\d+)_s(\d+)\.csv", os.path.basename(f))
        if not match:
            continue
        pb, sb = int(match.group(1)), int(match.group(2))

        df = pd.read_csv(f)
        df["prefix_bits"] = pb
        df["suffix_bits"] = sb
        df["combo"] = f"p{pb}_s{sb}"
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarize_df(df):
    """Summarize per-run timings into median speedups."""
    rows = []
    for (combo, pb, sb, source, prefix), g in df.groupby(
        ["combo", "prefix_bits", "suffix_bits", "source", "prefix"]
    ):
        full = g[g["query"] == "full"]["time_s"]
        fp = g[g["query"] == "fp_exact"]["time_s"]

        if len(full) == 0 or len(fp) == 0:
            continue

        med_full = float(np.median(full))
        med_fp = float(np.median(fp))
        speedup = med_full / med_fp if med_fp > 0 else 0

        rows.append({
            "combo": combo,
            "prefix_bits": pb,
            "suffix_bits": sb,
            "total_bits": pb + sb,
            "source": source,
            "query": prefix,
            "time_full_ms": med_full * 1000,
            "time_fp_ms": med_fp * 1000,
            "speedup": speedup,
        })

    return pd.DataFrame(rows)


def plot_speedup_by_combo(summary_df, out_dir):
    """Plot speedup for each query, x-axis = combo (p8_s8, etc.)."""
    sources = sorted(summary_df["source"].unique())
    queries = sorted(summary_df["query"].unique())

    # Sort combos by total bits, then by prefix bits
    combos = summary_df.groupby("combo").first().reset_index()
    combos = combos.sort_values(["total_bits", "prefix_bits"])["combo"].tolist()

    for source in sources:
        fig, ax = plt.subplots(figsize=(10, 5))

        for query in queries:
            sub = summary_df[(summary_df["source"] == source) & (summary_df["query"] == query)]
            # Reorder by combo order
            sub = sub.set_index("combo").reindex(combos).reset_index()
            sub = sub.dropna(subset=["speedup"])

            if sub.empty:
                continue

            x = np.arange(len(sub))
            ax.plot(x, sub["speedup"], marker="o", label=query, linewidth=1.5)

        ax.set_xticks(np.arange(len(combos)))
        ax.set_xticklabels(combos, rotation=45, ha="right")
        ax.set_xlabel("Fingerprint Combination (prefix_suffix bits)")
        ax.set_ylabel("Speedup (full / fp+exact)")
        ax.set_title(f"Infix Query Speedup by Fingerprint Combination ({source})")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"infix_speedup_{source}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_speedup_heatmap(summary_df, out_dir):
    """Plot heatmap of average speedup by prefix_bits x suffix_bits."""
    for source in sorted(summary_df["source"].unique()):
        sub = summary_df[summary_df["source"] == source]

        # Pivot to get prefix_bits x suffix_bits matrix
        pivot = sub.groupby(["prefix_bits", "suffix_bits"])["speedup"].mean().reset_index()
        pivot_table = pivot.pivot(index="suffix_bits", columns="prefix_bits", values="speedup")

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot_table.values, cmap="RdYlGn", aspect="auto", vmin=0.8, vmax=2.0)

        # Labels
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticklabels(pivot_table.index)
        ax.set_xlabel("Prefix Bits")
        ax.set_ylabel("Suffix Bits")
        ax.set_title(f"Average Speedup by Prefix/Suffix Bits ({source})")

        # Annotate cells
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = pivot_table.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=10)

        fig.colorbar(im, ax=ax, label="Speedup")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"infix_heatmap_{source}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_times_by_combo(summary_df, out_dir):
    """Plot full vs fp times for each combo."""
    for source in sorted(summary_df["source"].unique()):
        sub = summary_df[summary_df["source"] == source]

        # Get unique combos sorted
        combos = sub.groupby("combo").first().reset_index()
        combos = combos.sort_values(["total_bits", "prefix_bits"])["combo"].tolist()

        # Average across queries for each combo
        avg = sub.groupby("combo")[["time_full_ms", "time_fp_ms"]].mean()
        avg = avg.reindex(combos)

        x = np.arange(len(combos))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, avg["time_full_ms"], width, label="Full scan")
        ax.bar(x + width/2, avg["time_fp_ms"], width, label="FP + exact")

        ax.set_xticks(x)
        ax.set_xticklabels(combos, rotation=45, ha="right")
        ax.set_xlabel("Fingerprint Combination (prefix_suffix bits)")
        ax.set_ylabel("Median Time (ms)")
        ax.set_title(f"Query Time by Fingerprint Combination ({source})")
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"infix_times_{source}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def plot_summary_table(summary_df, out_dir):
    """Render summary as a PNG table."""
    # Aggregate by combo and source
    agg = summary_df.groupby(["combo", "source"]).agg({
        "prefix_bits": "first",
        "suffix_bits": "first",
        "total_bits": "first",
        "speedup": "mean",
        "time_full_ms": "mean",
        "time_fp_ms": "mean",
    }).reset_index()

    agg = agg.sort_values(["total_bits", "prefix_bits", "source"])
    agg["speedup"] = agg["speedup"].apply(lambda x: f"{x:.2f}x")
    agg["time_full_ms"] = agg["time_full_ms"].apply(lambda x: f"{x:.1f}")
    agg["time_fp_ms"] = agg["time_fp_ms"].apply(lambda x: f"{x:.1f}")

    display_cols = ["combo", "source", "prefix_bits", "suffix_bits", "time_full_ms", "time_fp_ms", "speedup"]
    df_show = agg[display_cols].copy()
    df_show.columns = ["Combo", "Source", "Prefix Bits", "Suffix Bits", "Full (ms)", "FP (ms)", "Speedup"]

    fig, ax = plt.subplots(figsize=(12, 0.4 * (len(df_show) + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "infix_summary_table.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot infix benchmark results")
    parser.add_argument("--csv-dir", default="csvs_infix", help="Directory containing infix CSV files")
    parser.add_argument("--out-dir", default="infix_plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading CSVs from {args.csv_dir}...")
    df = load_and_merge_csvs(args.csv_dir)
    print(f"Loaded {len(df)} rows")

    print("Summarizing...")
    summary = summarize_df(df)
    print(f"Summary has {len(summary)} rows")

    # Save summary CSV
    summary_path = os.path.join(args.out_dir, "infix_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    # Generate plots
    plot_speedup_by_combo(summary, args.out_dir)
    plot_speedup_heatmap(summary, args.out_dir)
    plot_times_by_combo(summary, args.out_dir)
    plot_summary_table(summary, args.out_dir)

    print(f"\nAll plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
