"""
Build combined prefix and suffix fingerprints for infix-style queries.

This creates a parquet file with both q8_prefix and q8_suffix columns,
allowing queries like: titles that start with "the" AND end with "a".
"""

import argparse
import numpy as np
import pandas as pd

IN_PARQUET = "title_strs.parquet"
COL = "title"
PREFIX_BYTES = 8
SAMPLE_SIZE = 500_000
SEED = 42


def normalize(s: str) -> str:
    return (s or "").lower()


def key_bytes(s_norm: str, nbytes: int) -> bytes:
    b = s_norm.encode("utf-8", errors="ignore")[:nbytes]
    if len(b) < nbytes:
        b = b + b"\x00" * (nbytes - len(b))
    return b


def bytes_to_u64(b: bytes) -> np.uint64:
    return np.uint64(int.from_bytes(b, byteorder="big", signed=False))


def make_keys(series: pd.Series, nbytes: int, suffix: bool) -> np.ndarray:
    def process(x):
        s = normalize(x)
        if suffix:
            s = s[::-1]
        return bytes_to_u64(key_bytes(s, nbytes))

    return np.fromiter(
        (process(x) for x in series.astype("string")),
        dtype=np.uint64,
        count=len(series),
    )


def build_boundaries(sample_keys: np.ndarray, B: int) -> np.ndarray:
    ps = np.linspace(0, 100, num=B, endpoint=False)
    boundaries = np.percentile(sample_keys, ps, method="linear").astype(np.uint64)
    boundaries = np.maximum.accumulate(boundaries)
    return boundaries


def assign_codes(all_keys: np.ndarray, boundaries: np.ndarray, bits: int) -> np.ndarray:
    idx = np.searchsorted(boundaries, all_keys, side="right") - 1
    idx = np.clip(idx, 0, len(boundaries) - 1)

    if bits <= 8:
        return idx.astype(np.uint8)
    if bits <= 16:
        return idx.astype(np.uint16)
    if bits <= 32:
        return idx.astype(np.uint32)
    return idx.astype(np.uint64)


def dump_boundaries_txt(boundaries, path):
    with open(path, "w") as f:
        f.write("bucket_id,boundary_u64\n")
        for i, b in enumerate(boundaries):
            f.write(f"{i},{int(b)}\n")


def main():
    parser = argparse.ArgumentParser(description="Build combined prefix+suffix fingerprints")
    parser.add_argument("--prefix-bits", type=int, default=8, help="Bit width for prefix (default: 8)")
    parser.add_argument("--suffix-bits", type=int, default=8, help="Bit width for suffix (default: 8)")
    args = parser.parse_args()

    prefix_bits = args.prefix_bits
    suffix_bits = args.suffix_bits

    if not (1 <= prefix_bits <= 28) or not (1 <= suffix_bits <= 28):
        raise ValueError("Bits must be between 1 and 28")

    prefix_buckets = 1 << prefix_bits
    suffix_buckets = 1 << suffix_bits

    prefix_col = f"q{prefix_bits}_prefix"
    suffix_col = f"q{suffix_bits}_suffix"

    out_parquet = f"title_strs_infix_p{prefix_bits}_s{suffix_bits}.parquet"
    prefix_boundaries_npy = f"q{prefix_bits}_prefix_boundaries.npy"
    suffix_boundaries_npy = f"q{suffix_bits}_suffix_boundaries.npy"

    print(f"Reading {IN_PARQUET} ...")
    df = pd.read_parquet(IN_PARQUET)

    if COL not in df.columns:
        raise ValueError(f"Expected column '{COL}'. Found: {list(df.columns)}")

    n = len(df)
    print(f"Rows: {n:,}")
    print(f"Prefix: {prefix_bits} bits -> {prefix_buckets} buckets -> column {prefix_col}")
    print(f"Suffix: {suffix_bits} bits -> {suffix_buckets} buckets -> column {suffix_col}")

    # Sample for boundary estimation
    if n > SAMPLE_SIZE:
        sample = df[COL].sample(n=SAMPLE_SIZE, random_state=SEED)
        print(f"Sampling {SAMPLE_SIZE:,} rows for boundary estimation...")
    else:
        sample = df[COL]

    # Build prefix fingerprints
    print("\nBuilding prefix fingerprints...")
    prefix_sample_keys = make_keys(sample, PREFIX_BYTES, suffix=False)
    prefix_boundaries = build_boundaries(prefix_sample_keys, prefix_buckets)
    prefix_all_keys = make_keys(df[COL], PREFIX_BYTES, suffix=False)
    df[prefix_col] = assign_codes(prefix_all_keys, prefix_boundaries, prefix_bits)

    np.save(prefix_boundaries_npy, prefix_boundaries)
    dump_boundaries_txt(prefix_boundaries, f"q{prefix_bits}_prefix_boundaries.txt")
    print(f"  Saved: {prefix_boundaries_npy}")

    # Build suffix fingerprints
    print("\nBuilding suffix fingerprints...")
    suffix_sample_keys = make_keys(sample, PREFIX_BYTES, suffix=True)
    suffix_boundaries = build_boundaries(suffix_sample_keys, suffix_buckets)
    suffix_all_keys = make_keys(df[COL], PREFIX_BYTES, suffix=True)
    df[suffix_col] = assign_codes(suffix_all_keys, suffix_boundaries, suffix_bits)

    np.save(suffix_boundaries_npy, suffix_boundaries)
    dump_boundaries_txt(suffix_boundaries, f"q{suffix_bits}_suffix_boundaries.txt")
    print(f"  Saved: {suffix_boundaries_npy}")

    # Save combined parquet
    df.to_parquet(out_parquet, index=False)
    print(f"\nWrote: {out_parquet}")

    # Stats
    print(f"\n{prefix_col} distribution:")
    vc_p = df[prefix_col].value_counts()
    print(f"  non-empty buckets: {(vc_p > 0).sum()} / {prefix_buckets}")
    print(f"  max bucket size:   {vc_p.max():,}")
    print(f"  min bucket size:   {vc_p.min():,}")

    print(f"\n{suffix_col} distribution:")
    vc_s = df[suffix_col].value_counts()
    print(f"  non-empty buckets: {(vc_s > 0).sum()} / {suffix_buckets}")
    print(f"  max bucket size:   {vc_s.max():,}")
    print(f"  min bucket size:   {vc_s.min():,}")

    # Show some samples
    print("\nSample rows:")
    sample_df = df.sample(n=min(10, len(df)), random_state=42)
    print(sample_df[[COL, prefix_col, suffix_col]].to_string())


if __name__ == "__main__":
    main()
