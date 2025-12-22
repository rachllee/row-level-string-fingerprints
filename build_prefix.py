import numpy as np
import pandas as pd

IN_PARQUET  = "title_strs.parquet"
OUT_PARQUET = "title_strs_prefix.parquet"

COL = "title"

K = 8
B = 1 << K

PREFIX_BYTES = 8

SAMPLE_SIZE = 500_000
SEED = 42
def dump_boundaries_txt(boundaries, path="prefix_boundaries.txt"):
    with open(path, "w") as f:
        f.write("bucket_id,boundary_u64\n")
        for i, b in enumerate(boundaries):
            f.write(f"{i},{int(b)}\n")

def dump_bucket_stats(df, col="q8_prefix", path="prefix_bucket_stats.csv"):
    stats = (
        df[col]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    stats.columns = ["bucket_id", "row_count"]
    stats.to_csv(path, index=False)

def dump_samples(df, path="title_prefix_samples.csv", n=200):
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample[["title", "q8_prefix"]].to_csv(path, index=False)


def normalize(s: str) -> str:
    return (s or "").lower()

def prefix_key_bytes(s: str, nbytes: int) -> bytes:
    b = normalize(s).encode("utf-8", errors="ignore")[:nbytes]
    if len(b) < nbytes:
        b = b + b"\x00" * (nbytes - len(b))
    return b

def bytes_to_u64(b: bytes) -> np.uint64:
    return np.uint64(int.from_bytes(b, byteorder="big", signed=False))

def make_keys(series: pd.Series, nbytes: int) -> np.ndarray:
    return np.fromiter(
        (bytes_to_u64(prefix_key_bytes(x, nbytes)) for x in series.astype("string")),
        dtype=np.uint64,
        count=len(series),
    )

def build_boundaries(sample_keys: np.ndarray, B: int) -> np.ndarray:
    ps = np.linspace(0, 100, num=B, endpoint=False)
    boundaries = np.percentile(sample_keys, ps, method="linear").astype(np.uint64)
    boundaries = np.maximum.accumulate(boundaries)
    return boundaries

def assign_codes(all_keys: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(boundaries, all_keys, side="right") - 1
    idx = np.clip(idx, 0, len(boundaries) - 1)
    return idx.astype(np.uint8)
def main():
    print(f"Reading {IN_PARQUET} ...")
    df = pd.read_parquet(IN_PARQUET)

    if COL not in df.columns:
        raise ValueError(f"Expected column '{COL}'. Found: {list(df.columns)}")

    n = len(df)
    print(f"Rows: {n:,}")

    if n > SAMPLE_SIZE:
        sample = df[COL].sample(n=SAMPLE_SIZE, random_state=SEED)
        print(f"Sampling {SAMPLE_SIZE:,} rows to estimate {B} boundaries...")
    else:
        sample = df[COL]
        print(f"Using all rows to estimate {B} boundaries...")

    sample_keys = make_keys(sample, PREFIX_BYTES)
    boundaries = build_boundaries(sample_keys, B)

    print("Assigning q8_prefix codes for all rows...")
    all_keys = make_keys(df[COL], PREFIX_BYTES)
    df["q8_prefix"] = assign_codes(all_keys, boundaries)

    np.save("q8_prefix_boundaries.npy", boundaries)
    print("Saved boundaries: q8_prefix_boundaries.npy")

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote: {OUT_PARQUET}")

    vc = df["q8_prefix"].value_counts()
    print("\nq8_prefix distribution:")
    print(f"  non-empty buckets: {(vc > 0).sum()} / {B}")
    print(f"  max bucket size:   {vc.max():,}")
    print(f"  min bucket size:   {vc.min():,}")
    print("\nTop 10 buckets:")
    print(vc.head(10))

    # ---- Human-readable dumps ----

    dump_bucket_stats(df, col="q8_prefix")
    print("Wrote q8_prefix_bucket_stats.csv")

    dump_samples(df)
    print("Wrote title_prefix_samples.csv")

if __name__ == "__main__":
    main()
