import numpy as np

boundaries = np.load("q8_prefix_boundaries.npy")

PREFIX_BYTES = 8

def u64_to_prefix(u, nbytes=PREFIX_BYTES):
    b = int(u).to_bytes(nbytes, byteorder="big", signed=False)
    # Replace null bytes for readability
    return b.rstrip(b"\x00").decode("utf-8", errors="replace")

with open("q8_prefix_boundaries_readable.txt", "w") as f:
    f.write("bucket_id,boundary_u64,boundary_prefix\n")
    for i, u in enumerate(boundaries):
        prefix = u64_to_prefix(u)
        f.write(f"{i},{int(u)},{prefix}\n")

print("Wrote q8_prefix_boundaries_readable.txt")
