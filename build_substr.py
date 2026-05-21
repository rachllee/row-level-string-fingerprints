"""
Build n-gram presence fingerprint for substring queries using
decision tree induction for feature (n-gram) selection.

For bigrams (--ngram 2), rare bigrams (below --max-freq threshold) are
pre-filtered as candidates. This ensures selected features are absent from
many row groups, enabling actual row-group-level skipping.

Algorithm (from "Feature identification for string fingerprints"):
  1. Pre-filter n-gram candidates by frequency (keep those below max_freq)
  2. Compute per-n-gram binary entropy across all strings
  3. Greedily pick the highest-entropy n-gram as first feature
  4. Split strings into two groups: contains / does not contain that n-gram
  5. Recursively pick the next best n-gram from each group
  6. At each step, expand the partition with the highest weighted entropy
  7. Repeat until K=16 features are selected

Each bit i of the fingerprint = 1 if the string contains selected_features[i].

Usage:
    python build_substr.py [--bits 16] [--sample 200000] [--ngram 2] [--max-freq 0.01]
"""

import argparse
import json
import numpy as np
import pandas as pd

IN_PARQUET  = "title_strs.parquet"
OUT_PARQUET = "title_strs_substr_fp16.parquet"
FEATURES_JSON = "substr_features.json"
COL = "title"


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def extract_ngrams(s: str, n: int) -> set:
    s = (s or "").lower()
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def decision_tree_feature_selection(feature_presence: dict, K: int) -> list:
    """
    Select K features via decision tree induction.

    feature_presence: {feature: np.ndarray(bool, shape=(n,))}
    Returns list of K selected features.
    """
    n = len(next(iter(feature_presence.values())))
    candidates = list(feature_presence.keys())

    selected = []
    # Each partition is a boolean mask over the sample rows
    partitions = [np.ones(n, dtype=bool)]

    for step in range(K):
        best_feat       = None
        best_weighted_h = -1.0
        best_p_idx      = -1

        for p_idx, mask in enumerate(partitions):
            size = int(mask.sum())
            if size == 0:
                continue
            for feat in candidates:
                p = feature_presence[feat][mask].mean()
                # Weight entropy by partition size so larger partitions matter more
                wh = binary_entropy(p) * size
                if wh > best_weighted_h:
                    best_weighted_h = wh
                    best_feat       = feat
                    best_p_idx      = p_idx

        if best_feat is None:
            break

        raw_p = feature_presence[best_feat][partitions[best_p_idx]].mean()
        print(f"  Feature {step+1:2d}: '{best_feat}'  "
              f"entropy={binary_entropy(raw_p):.3f}  "
              f"freq={raw_p*100:.2f}%")

        selected.append(best_feat)
        candidates.remove(best_feat)

        # Split the chosen partition on best_feat
        mask = partitions.pop(best_p_idx)
        feat_arr = feature_presence[best_feat]
        partitions.append(mask & ~feat_arr)   # strings that do NOT contain feat
        partitions.append(mask &  feat_arr)   # strings that DO contain feat

    return selected


def compute_fingerprint(s: str, selected_features: list, ngram: int) -> int:
    fp     = 0
    grams  = extract_ngrams(s, ngram)
    for i, feat in enumerate(selected_features):
        if feat in grams:
            fp |= (1 << i)
    return fp


def pattern_mask(pattern: str, selected_features: list, ngram: int) -> int:
    mask  = 0
    grams = extract_ngrams(pattern, ngram)
    for i, feat in enumerate(selected_features):
        if feat in grams:
            mask |= (1 << i)
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits",     type=int,   default=16,
                        help="Number of features / fingerprint bits (default: 16)")
    parser.add_argument("--sample",   type=int,   default=200_000,
                        help="Rows to sample for feature selection (default: 200000)")
    parser.add_argument("--ngram",    type=int,   default=2,
                        help="N-gram size: 1=chars, 2=bigrams (default: 2)")
    parser.add_argument("--max-freq", type=float, default=0.01,
                        help="Max frequency to include as candidate feature (default: 0.01 = 1%)")
    args = parser.parse_args()

    print(f"Reading {IN_PARQUET}...")
    df = pd.read_parquet(IN_PARQUET)
    n  = len(df)
    print(f"Rows: {n:,}")

    # ── Feature selection on a sample ──────────────────────────────────────
    sample_size = min(args.sample, n)
    sample = df[COL].sample(n=sample_size, random_state=42).astype("string")
    print(f"\nBuilding {args.ngram}-gram presence matrix on {sample_size:,} sample rows...")

    # Collect all n-gram candidates
    print(f"Extracting {args.ngram}-grams...")
    all_grams: set = set()
    for s in sample:
        all_grams.update(extract_ngrams(s, args.ngram))
    print(f"Total unique {args.ngram}-grams: {len(all_grams):,}")

    # Compute frequency and filter by max_freq
    print(f"Computing frequencies and filtering (max_freq={args.max_freq*100:.1f}%)...")
    sample_list = list(sample)
    gram_freq = {}
    for gram in all_grams:
        count = sum(1 for s in sample_list if gram in extract_ngrams(s, args.ngram))
        gram_freq[gram] = count / sample_size

    candidates = {g: f for g, f in gram_freq.items() if f <= args.max_freq and f > 0}
    print(f"Candidates after frequency filter: {len(candidates):,}")

    if len(candidates) < args.bits:
        raise ValueError(
            f"Only {len(candidates)} candidates below {args.max_freq*100:.1f}% frequency — "
            f"need at least {args.bits}. Try raising --max-freq."
        )

    # Build boolean presence arrays for candidates only
    print(f"Building presence arrays for {len(candidates):,} candidates...")
    feature_presence = {}
    for gram in candidates:
        feature_presence[gram] = np.array(
            [gram in extract_ngrams(s, args.ngram) for s in sample_list], dtype=bool
        )

    print(f"\nRunning decision tree feature selection (K={args.bits})...")
    selected_features = decision_tree_feature_selection(feature_presence, args.bits)

    print(f"\nSelected {len(selected_features)} features: {selected_features}")

    # Save features for use by benchmark/scan scripts
    with open(FEATURES_JSON, "w") as f:
        json.dump({"selected_features": selected_features, "ngram": args.ngram}, f, indent=2)
    print(f"Saved feature list to {FEATURES_JSON}")

    # ── Build fingerprints for all rows ────────────────────────────────────
    print(f"\nComputing fingerprints for all {n:,} rows...")
    fp_values = np.fromiter(
        (compute_fingerprint(s, selected_features, args.ngram) for s in df[COL].astype("string")),
        dtype=np.uint16,
        count=n,
    )

    df["fp16_chars"] = fp_values
    df = df[[COL, "fp16_chars"]]
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}")

    # ── Stats ───────────────────────────────────────────────────────────────
    avg_bits = np.mean([bin(v).count('1') for v in fp_values[:10_000]])
    print(f"\nAvg bits set per string (sample of 10k): {avg_bits:.1f} / {args.bits}")

    print("\nExample pattern masks with selected features:")
    for pat in ["the", "c++", "war", "python", "2024", "ing"]:
        m    = pattern_mask(pat, selected_features, args.ngram)
        bits = bin(m).count('1')
        sample_fps = fp_values[:50_000]
        fpr  = np.mean((sample_fps & m) == m) * 100
        print(f"  '{pat}': {bits} bits set, estimated FP rate: {fpr:.1f}%")


if __name__ == "__main__":
    main()
