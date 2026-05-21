"""
Side-by-side comparison of unigram vs bigram fingerprint speedup.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIGRAM_CSV = "custom_scan_bench.csv"
BIGRAM_CSV  = "custom_scan_bench_bigram.csv"
OUT         = "custom_scan_plots/unigram_vs_bigram.png"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summary_by_length(rows):
    lengths = sorted(set(int(r["pattern_length"]) for r in rows))
    medians, fpr_means = [], []
    for l in lengths:
        subset = [r for r in rows if int(r["pattern_length"]) == l]
        sus = [float(r["speedup"]) for r in subset]
        fpr = [float(r["fp_rate"]) * 100 for r in subset]
        medians.append(np.median(sus))
        fpr_means.append(np.mean(fpr))
    return lengths, medians, fpr_means


uni   = load(UNIGRAM_CSV)
bi    = load(BIGRAM_CSV)

ul, u_med, u_fpr = summary_by_length(uni)
bl, b_med, b_fpr = summary_by_length(bi)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(ul))
w = 0.35

# -- Speedup comparison --
ax1.bar(x - w/2, u_med, w, label="Unigram (1-gram)", color="steelblue")
ax1.bar(x + w/2, b_med, w, label="Bigram (2-gram)",  color="coral")
ax1.axhline(1, color="gray", linestyle=":", linewidth=1)
ax1.set_xticks(x)
ax1.set_xticklabels(ul)
ax1.set_xlabel("Pattern length (chars)")
ax1.set_ylabel("Median speedup vs baseline")
ax1.set_title("Speedup: unigram vs bigram features")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# -- FP rate comparison --
ax2.bar(x - w/2, u_fpr, w, label="Unigram (1-gram)", color="steelblue")
ax2.bar(x + w/2, b_fpr, w, label="Bigram (2-gram)",  color="coral")
ax2.set_xticks(x)
ax2.set_xticklabels(ul)
ax2.set_xlabel("Pattern length (chars)")
ax2.set_ylabel("Mean false positive rate (%)")
ax2.set_title("FP rate: unigram vs bigram features")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle("16-bit fingerprint: unigram vs bigram feature selection", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Wrote {OUT}")
