import argparse
import numpy as np

PREFIX_BYTES = 8


def u64_to_prefix(u, nbytes=PREFIX_BYTES):
    b = int(u).to_bytes(nbytes, byteorder="big", signed=False)
    return b.rstrip(b"\x00").decode("utf-8", errors="replace")


def u64_to_suffix(u, nbytes=PREFIX_BYTES):
    b = int(u).to_bytes(nbytes, byteorder="big", signed=False)
    rev = b.rstrip(b"\x00").decode("utf-8", errors="replace")
    return rev[::-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..16). Default: 8")
    parser.add_argument("--suffix", action="store_true", help="Interpret boundaries as suffix fingerprints.")
    args = parser.parse_args()

    K = args.bits
    if K < 1 or K > 16:
        raise ValueError("--bits must be between 1 and 16")

    mode = "suffix" if args.suffix else "prefix"
    boundaries = np.load(f"q{K}_{mode}_boundaries.npy")
    out_path = f"q{K}_{mode}_boundaries_readable.txt"

    with open(out_path, "w") as f:
        label = "boundary_suffix" if args.suffix else "boundary_prefix"
        f.write(f"bucket_id,boundary_u64,{label}\n")
        for i, u in enumerate(boundaries):
            if args.suffix:
                token = u64_to_suffix(u)
            else:
                token = u64_to_prefix(u)
            f.write(f"{i},{int(u)},{token}\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
