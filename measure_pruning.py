import argparse
import numpy as np
import pandas as pd

PREFIX_BYTES = 8


def norm(s):
    return (s or "").lower()


def normalize_query(s: str, suffix: bool) -> str:
    s = norm(s)
    return s[::-1] if suffix else s


def key_u64_from_normed(s_norm, nbytes=PREFIX_BYTES):
    b = s_norm.encode("utf-8", errors="ignore")[:nbytes]
    b = b + b"\x00" * (nbytes - len(b))
    return int.from_bytes(b, "big", signed=False)


def next_prefix_normed(s_norm: str) -> str:
    b = bytearray(s_norm.encode("utf-8", errors="ignore"))
    if not b:
        return "\uffff"
    b[-1] = min(255, b[-1] + 1)
    return bytes(b).decode("utf-8", errors="ignore")


def bucket_range(query: str, boundaries: np.ndarray, suffix: bool):
    s = normalize_query(query, suffix)
    lo = key_u64_from_normed(s)
    hi = key_u64_from_normed(next_prefix_normed(s))
    jlo = np.searchsorted(boundaries, lo, side="right") - 1
    jhi = np.searchsorted(boundaries, hi, side="right") - 1
    jlo = int(np.clip(jlo, 0, len(boundaries) - 1))
    jhi = int(np.clip(jhi, 0, len(boundaries) - 1))
    return min(jlo, jhi), max(jlo, jhi)


def kept_fraction(df, code_col, lo, hi):
    return df[code_col].between(lo, hi).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..16). Default: 8")
    parser.add_argument("--suffix", action="store_true", help="Estimate pruning for suffix queries.")
    args = parser.parse_args()

    K = args.bits
    if K < 1 or K > 16:
        raise ValueError("--bits must be between 1 and 16")

    mode = "suffix" if args.suffix else "prefix"
    parquet = f"title_strs_{mode}_b{K}.parquet"
    boundaries_npy = f"q{K}_{mode}_boundaries.npy"
    code_col = f"q{K}_{mode}"

    df = pd.read_parquet(parquet, columns=["title", code_col])
    boundaries = np.load(boundaries_npy)
    n = len(df)

    if args.suffix:
        queries = ["son", "2012", "the", "a", ")", "#"]
    else:
        queries = ["jos", "2012", "the", "a", "(", "#"]

    print(f"N={n:,}")
    for q in queries:
        lo, hi = bucket_range(q, boundaries, suffix=args.suffix)
        kept = kept_fraction(df, code_col, lo, hi)
        label = f"%{q}" if args.suffix else f"{q}%"
        print(
            f"{label} -> buckets [{lo},{hi}] ({hi-lo+1} buckets), "
            f"keep {kept*100:.2f}%, skip {(1-kept)*100:.2f}%"
        )


if __name__ == "__main__":
    main()
