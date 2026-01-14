import argparse
import numpy as np
import pandas as pd

IN_PARQUET = "title_strs.parquet"
OUT_PARQUET = "title_strs_prefix.parquet"  # will be overridden to include bits

COL = "title"

PREFIX_BYTES = 8
SAMPLE_SIZE = 500_000
SEED = 42


def dump_boundaries_txt(boundaries, path="prefix_boundaries.txt"):
    with open(path, "w") as f:
        f.write("bucket_id,boundary_u64\n")
        for i, b in enumerate(boundaries):
            f.write(f"{i},{int(b)}\n")


def dump_bucket_stats(df, col, path="prefix_bucket_stats.csv"):
    stats = df[col].value_counts().sort_index().reset_index()
    stats.columns = ["bucket_id", "row_count"]
    stats.to_csv(path, index=False)


def dump_samples(df, col, path="title_prefix_samples.csv", n=200):
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample[[COL, col]].to_csv(path, index=False)


def normalize(s: str) -> str:
    return (s or "").lower()


def normalize_query(s: str, suffix: bool) -> str:
    s = normalize(s)
    return s[::-1] if suffix else s


def key_bytes(s_norm: str, nbytes: int) -> bytes:
    b = s_norm.encode("utf-8", errors="ignore")[:nbytes]
    if len(b) < nbytes:
        b = b + b"\x00" * (nbytes - len(b))
    return b


def bytes_to_u64(b: bytes) -> np.uint64:
    return np.uint64(int.from_bytes(b, byteorder="big", signed=False))


def make_keys(series: pd.Series, nbytes: int, suffix: bool) -> np.ndarray:
    return np.fromiter(
        (bytes_to_u64(key_bytes(normalize_query(x, suffix), nbytes)) for x in series.astype("string")),
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

    # if bits <= 8:
    #     return idx.astype(np.uint8)
    # else:
    return idx.astype(np.uint16)


def main(argv=None, default_suffix=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..16). Default: 8")
    parser.add_argument(
        "--suffix",
        action="store_true",
        default=default_suffix,
        help="Build suffix fingerprints instead of prefix fingerprints.",
    )
    args = parser.parse_args(argv)

    K = args.bits
    if K < 1 or K > 16:
        raise ValueError("--bits must be between 1 and 16")

    B = 1 << K
    mode = "suffix" if args.suffix else "prefix"
    code_col = f"q{K}_{mode}"

    out_parquet = f"title_strs_{mode}_b{K}.parquet"
    boundaries_npy = f"{code_col}_boundaries.npy"
    boundaries_txt = f"{code_col}_boundaries.txt"
    bucket_stats_csv = f"{code_col}_bucket_stats.csv"
    samples_csv = f"title_{mode}_samples_b{K}.csv"

    print(f"Reading {IN_PARQUET} ...")
    df = pd.read_parquet(IN_PARQUET)

    if COL not in df.columns:
        raise ValueError(f"Expected column '{COL}'. Found: {list(df.columns)}")

    n = len(df)
    print(f"Rows: {n:,}")

    print(f"mode={mode} bits={K} -> buckets={B} -> column={code_col}")

    if n > SAMPLE_SIZE:
        sample = df[COL].sample(n=SAMPLE_SIZE, random_state=SEED)
        print(f"Sampling {SAMPLE_SIZE:,} rows to estimate {B} boundaries...")
    else:
        sample = df[COL]
        print(f"Using all rows to estimate {B} boundaries...")

    sample_keys = make_keys(sample, PREFIX_BYTES, suffix=args.suffix)
    boundaries = build_boundaries(sample_keys, B)

    print(f"Assigning {code_col} codes for all rows...")
    all_keys = make_keys(df[COL], PREFIX_BYTES, suffix=args.suffix)
    df[code_col] = assign_codes(all_keys, boundaries, K)

    np.save(boundaries_npy, boundaries)
    print(f"Saved boundaries: {boundaries_npy}")

    dump_boundaries_txt(boundaries, path=boundaries_txt)
    print(f"Wrote {boundaries_txt}")

    df.to_parquet(out_parquet, index=False)
    print(f"Wrote: {out_parquet}")

    vc = df[code_col].value_counts()
    print(f"\n{code_col} distribution:")
    print(f"  non-empty buckets: {(vc > 0).sum()} / {B}")
    print(f"  max bucket size:   {vc.max():,}")
    print(f"  min bucket size:   {vc.min():,}")
    print("\nTop 10 buckets:")
    print(vc.head(10))

    dump_bucket_stats(df, col=code_col, path=bucket_stats_csv)
    print(f"Wrote {bucket_stats_csv}")

    dump_samples(df, col=code_col, path=samples_csv)
    print(f"Wrote {samples_csv}")


if __name__ == "__main__":
    main()
