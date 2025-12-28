import argparse
import numpy as np
import pandas as pd

PREFIX_BYTES = 8

def norm(s): return (s or "").lower()

def prefix_key_u64(s, nbytes=PREFIX_BYTES):
    b = norm(s).encode("utf-8", errors="ignore")[:nbytes]
    b = b + b"\x00" * (nbytes - len(b))
    return int.from_bytes(b, "big", signed=False)

def next_prefix(p: str) -> str:
    b = bytearray(norm(p).encode("utf-8", errors="ignore"))
    if not b:
        return "\uffff"
    b[-1] = min(255, b[-1] + 1)
    return bytes(b).decode("utf-8", errors="ignore")

def bucket_range(prefix: str, boundaries: np.ndarray):
    lo = prefix_key_u64(prefix)
    hi = prefix_key_u64(next_prefix(prefix))
    jlo = np.searchsorted(boundaries, lo, side="right") - 1
    jhi = np.searchsorted(boundaries, hi, side="right") - 1
    jlo = int(np.clip(jlo, 0, len(boundaries)-1))
    jhi = int(np.clip(jhi, 0, len(boundaries)-1))
    return min(jlo, jhi), max(jlo, jhi)

def kept_fraction(df, code_col, lo, hi):
    return df[code_col].between(lo, hi).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..16). Default: 8")
    args = parser.parse_args()

    K = args.bits
    if K < 1 or K > 16:
        raise ValueError("--bits must be between 1 and 16")

    parquet = f"title_strs_prefix_b{K}.parquet"
    boundaries_npy = f"q{K}_prefix_boundaries.npy"
    code_col = f"q{K}_prefix"

    df = pd.read_parquet(parquet, columns=["title", code_col])
    boundaries = np.load(boundaries_npy)
    n = len(df)

    prefixes = ["jos", "2012", "the", "a", "(", "#"]

    print(f"N={n:,}")
    for p in prefixes:
        lo, hi = bucket_range(p, boundaries)
        kept = kept_fraction(df, code_col, lo, hi)
        print(
            f"{p}% -> buckets [{lo},{hi}] ({hi-lo+1} buckets), "
            f"keep {kept*100:.2f}%, skip {(1-kept)*100:.2f}%"
        )


if __name__ == "__main__":
    main()
