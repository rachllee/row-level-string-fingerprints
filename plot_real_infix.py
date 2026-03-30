"""
Plot speedup vs selectivity for real infix (prefix+suffix) benchmark results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Plot real infix benchmark results")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out-dir", default="", help="Output directory for plots")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Create output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = args.csv.replace(".csv", "_plots")

    os.makedirs(out_dir, exist_ok=True)

    # Convert selectivity to percentage
    df['selectivity_pct'] = df['selectivity'] * 100

    # 1. Speedup vs Selectivity (log scale)
    plt.figure(figsize=(10, 6))

    # Color by prune rate
    scatter = plt.scatter(df['selectivity_pct'], df['speedup'],
                         c=df['prune_rate'], cmap='RdYlGn',
                         alpha=0.6, s=20)
    plt.colorbar(scatter, label='Prune Rate (%)')

    # Binned average
    bins = np.logspace(np.log10(df['selectivity_pct'].min()),
                       np.log10(df['selectivity_pct'].max()), 20)
    bin_means = []
    bin_centers = []
    for i in range(len(bins)-1):
        mask = (df['selectivity_pct'] >= bins[i]) & (df['selectivity_pct'] < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(df[mask]['speedup'].mean())
            bin_centers.append((bins[i] + bins[i+1]) / 2)

    if bin_means:
        plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned average')

    plt.xscale('log')
    plt.xlabel('Selectivity (% of rows matching query)')
    plt.ylabel('Speedup (full / fingerprint)')
    plt.title('Speedup vs. Query Selectivity (Real Infix)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'speedup_vs_selectivity.png')
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    plt.close()

    # 2. Speedup vs Prune Rate
    plt.figure(figsize=(10, 6))
    plt.scatter(df['prune_rate'], df['speedup'], alpha=0.5, s=20)

    # Binned average
    prune_bins = np.linspace(df['prune_rate'].min(), df['prune_rate'].max(), 20)
    bin_means = []
    bin_centers = []
    for i in range(len(prune_bins)-1):
        mask = (df['prune_rate'] >= prune_bins[i]) & (df['prune_rate'] < prune_bins[i+1])
        if mask.sum() > 0:
            bin_means.append(df[mask]['speedup'].mean())
            bin_centers.append((prune_bins[i] + prune_bins[i+1]) / 2)

    if bin_means:
        plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned average')

    plt.axhline(y=1, color='gray', linestyle='--', label='No speedup (1.0x)')
    plt.xlabel('Prune Rate (% of rows skipped by fingerprint)')
    plt.ylabel('Speedup (full / fingerprint)')
    plt.title('Speedup vs. Prune Rate (Real Infix)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'speedup_vs_prune_rate.png')
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    plt.close()

    # 3. Speedup histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['speedup'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(df['speedup'].median(), color='r', linestyle='--',
                linewidth=2, label=f"Median: {df['speedup'].median():.2f}x")
    plt.axvline(df['speedup'].mean(), color='orange', linestyle='--',
                linewidth=2, label=f"Mean: {df['speedup'].mean():.2f}x")
    plt.axvline(1.0, color='gray', linestyle=':', linewidth=1,
                label='No speedup (1.0x)')
    plt.xlabel('Speedup')
    plt.ylabel('Count')
    plt.title('Distribution of Speedup Across All Queries')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'speedup_histogram.png')
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    plt.close()

    # 4. Selectivity histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['selectivity_pct'], bins=50, edgecolor='black', alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Selectivity (% of rows matching)')
    plt.ylabel('Count')
    plt.title('Distribution of Query Selectivity')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    out_path = os.path.join(out_dir, 'selectivity_histogram.png')
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    plt.close()

    # 5. Bucket span analysis
    if 'prefix_bucket_span' in df.columns and 'suffix_bucket_span' in df.columns:
        df['combined_bucket_span'] = df['prefix_bucket_span'] * df['suffix_bucket_span']

        plt.figure(figsize=(10, 6))
        plt.scatter(df['combined_bucket_span'], df['speedup'], alpha=0.5, s=20)
        plt.axhline(y=1, color='gray', linestyle='--', label='No speedup (1.0x)')
        plt.xlabel('Combined Bucket Span (prefix × suffix)')
        plt.ylabel('Speedup')
        plt.title('Speedup vs. Combined Bucket Span (Real Infix)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(out_dir, 'bucket_span_vs_speedup.png')
        plt.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close()

    # Print summary statistics
    print(f"\nSpeedup by Selectivity Bin:")
    print("=" * 70)
    print(f"{'Selectivity':<20} {'Count':<10} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-" * 70)

    bins = [(0, 0.0001, "<0.0001%"),
            (0.0001, 0.001, "0.0001-0.001%"),
            (0.001, 0.01, "0.001-0.01%"),
            (0.01, 0.1, "0.01-0.1%"),
            (0.1, 1.0, "0.1-1%")]

    for lo, hi, label in bins:
        mask = (df['selectivity_pct'] >= lo) & (df['selectivity_pct'] < hi)
        subset = df[mask]
        if len(subset) > 0:
            print(f"{label:<20} {len(subset):<10} {subset['speedup'].mean():<8.2f} "
                  f"{subset['speedup'].median():<8.2f} {subset['speedup'].min():<8.2f} "
                  f"{subset['speedup'].max():<8.2f}")

    print("=" * 70)
    print(f"\nAll plots written to {out_dir}/")


if __name__ == "__main__":
    main()
