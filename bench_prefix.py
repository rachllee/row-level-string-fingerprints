import duckdb
import numpy as np
import time

PARQUET = "title_strs_prefix.parquet"
BOUNDARIES_NPY = "q8_prefix_boundaries.npy"
PREFIX_BYTES = 8

boundaries = np.load(BOUNDARIES_NPY)

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

con = duckdb.connect()
con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{PARQUET}')")

# Helpful to reduce variability:
con.execute("PRAGMA threads=4")
con.execute("PRAGMA enable_object_cache=true")

def time_query(sql, reps=5):
    # warmup
    con.execute(sql).fetchone()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        con.execute(sql).fetchone()
        times.append(time.perf_counter() - t0)
    return min(times), times

prefixes = ["jos", "2012", "the", "a", "(", "#"]

for p in prefixes:
    lo, hi = bucket_range(p)
    q_full = f"SELECT COUNT(*) FROM t WHERE title LIKE '{p}%'"
    q_fp   = f"SELECT COUNT(*) FROM t WHERE q8_prefix BETWEEN {lo} AND {hi} AND title LIKE '{p}%'"

    t_full, _ = time_query(q_full)
    t_fp, _   = time_query(q_fp)

    print(f"\n{p}%  buckets[{lo},{hi}]")
    print(f"  full scan: {t_full*1000:.2f} ms")
    print(f"  fp+exact:  {t_fp*1000:.2f} ms   speedup {t_full/t_fp:.2f}x")
