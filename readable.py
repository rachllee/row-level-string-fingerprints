import argparse
import numpy as np

PREFIX_BYTES = 8

def u64_to_prefix(u, nbytes=PREFIX_BYTES):
    b = int(u).to_bytes(nbytes, byteorder="big", signed=False)
    # Replace null bytes for readability
    return b.rstrip(b"\x00").decode("utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, help="Bit width b (1..16). Default: 8")
    args = parser.parse_args()

    K = args.bits
    if K < 1 or K > 16:
        raise ValueError("--bits must be between 1 and 16")

    boundaries = np.load(f"q{K}_prefix_boundaries.npy")
    out_path = f"q{K}_prefix_boundaries_readable.txt"

    with open(out_path, "w") as f:
        f.write("bucket_id,boundary_u64,boundary_prefix\n")
        for i, u in enumerate(boundaries):
            prefix = u64_to_prefix(u)
            f.write(f"{i},{int(u)},{prefix}\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
