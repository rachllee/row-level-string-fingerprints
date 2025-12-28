import numpy as np
import pandas as pd
import time

PARQUET = "title_strs_prefix.parquet"
BOUNDARIES_NPY = "q8_prefix_boundaries.npy"
PREFIX_BYTES = 8

df = pd.read_parquet(PARQUET, columns=["title", "q8_prefix"])
boundaries = np.load(BOUNDARIES_NPY)

N = len(df)

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

def bucket_range(prefix: str):
    lo = prefix_key_u64(prefix)
    hi = prefix_key_u64(next_prefix(prefix))
    jlo = np.searchsorted(boundaries, lo, side="right") - 1
    jhi = np.searchsorted(boundaries, hi, side="right") - 1
    jlo = int(np.clip(jlo, 0, len(boundaries)-1))
    jhi = int(np.clip(jhi, 0, len(boundaries)-1))
    return min(jlo, jhi), max(jlo, jhi)

def kept_fraction(lo, hi):
    return df["q8_prefix"].between(lo, hi).mean()

prefixes = ["jos", "2012", "the", "a", "(", "#"]

print(f"N={N:,}")
for p in prefixes:
    lo, hi = bucket_range(p)
    kept = kept_fraction(lo, hi)
    print(f"{p}% -> buckets [{lo},{hi}] ({hi-lo+1} buckets), keep {kept*100:.2f}%, skip {(1-kept)*100:.2f}%")
