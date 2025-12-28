# row-level-string-fingerprints

Prototype for building a prefix-based "fingerprint" column on string data, then using it to prune prefix queries (e.g., `LIKE 'foo%'`).

## Files and outputs
- Input: `title_strs.parquet` (expects a `title` column)
- Build script: `build_prefix.py` (creates `q{b}_prefix` and writes artifacts)
- Benchmark: `bench_prefix.py` (DuckDB timing)
- Pruning estimate: `measure_pruning.py`
- Boundary readability: `readable.py`
- Cleanup: `cleanup.sh`

Outputs follow a consistent naming scheme by bit width `b`:
- `title_strs_prefix_b{b}.parquet`
- `q{b}_prefix_boundaries.npy`
- `q{b}_prefix_boundaries.txt`
- `q{b}_prefix_bucket_stats.csv`
- `title_prefix_samples_b{b}.csv`
- `q{b}_prefix_boundaries_readable.txt` (optional)

## Quick start
```bash
python build_prefix.py --bits 8
python readable.py --bits 8
python measure_pruning.py --bits 8
python bench_prefix.py --bits 8
```
