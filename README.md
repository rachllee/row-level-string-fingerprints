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

## Script arguments

### build_prefix.py
- `--bits` (int, default 8): bit width `b` (1..16). Controls bucket count `2^b` and output names.

### readable.py
- `--bits` (int, default 8): reads `q{b}_prefix_boundaries.npy` and writes `q{b}_prefix_boundaries_readable.txt`.

### measure_pruning.py
- `--bits` (int, default 8): reads `title_strs_prefix_b{b}.parquet` and `q{b}_prefix_boundaries.npy`.

### bench_prefix.py
- `--bits` (int, default 8): input files and column names for bit width `b`.
- `--warmup` (int, default 1): warmup runs per query (discarded).
- `--reps` (int, default 10): timed runs per query after warmup.
- `--csv` (string, default empty): write per-run timings to a CSV file.
- `--explain`: print `EXPLAIN ANALYZE` for each query.

### summarize_bench.py
- `--csv` (string, required): input CSV from `bench_prefix.py`.
- `--out-table` (string, default `bench_summary.csv`): output summary table (appends if file exists).
- `--out-dir` (string, default `bench_plots`): output directory for plots.
- `--plots`: enable plot generation (off by default).

### plot_bench_summary.py
- `--csv` (string, default `bench_summary.csv`): input summary table.
- `--out-dir` (string, default `bench_summary_plots`): output directory for plots.
- `--table-png` (string, default `bench_summary_table.png`): output PNG table name.
