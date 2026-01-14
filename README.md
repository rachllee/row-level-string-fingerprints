# row-level-string-fingerprints

Prototype for building a prefix- or suffix-based "fingerprint" column on string data, then using it to prune prefix queries (e.g., `LIKE 'foo%'`) or suffix queries (e.g., `LIKE '%foo'`).

## Files and outputs
- Input: `title_strs.parquet` (expects a `title` column)
- Build script: `build.py` (creates `q{b}_prefix` or `q{b}_suffix` and writes artifacts)
- Benchmark: `bench.py` (DuckDB timing)
- Pruning estimate: `measure_pruning.py`
- Boundary readability: `readable.py`
- Cleanup: `cleanup.sh`

Outputs follow a consistent naming scheme by bit width `b`:
- `title_strs_prefix_b{b}.parquet`
- `title_strs_suffix_b{b}.parquet`
- `q{b}_prefix_boundaries.npy`
- `q{b}_suffix_boundaries.npy`
- `q{b}_prefix_boundaries.txt`
- `q{b}_suffix_boundaries.txt`
- `q{b}_prefix_bucket_stats.csv`
- `q{b}_suffix_bucket_stats.csv`
- `title_prefix_samples_b{b}.csv`
- `title_suffix_samples_b{b}.csv`
- `q{b}_prefix_boundaries_readable.txt` (optional)
- `q{b}_suffix_boundaries_readable.txt` (optional)

## Quick start
```bash
python build.py --bits 8
python readable.py --bits 8
python measure_pruning.py --bits 8
python bench.py --bits 8

python build.py --bits 8 --suffix
python readable.py --bits 8 --suffix
python measure_pruning.py --bits 8 --suffix
python bench.py --bits 8 --suffix
```

## Shell scripts
Simple wrappers for running the full pipeline:

- `bash all.sh`: prefix fingerprints for bits 1..28 (build, bench, summarize, plot).
- `bash all_suffix.sh`: suffix fingerprints for bits 1..28 (build, bench, summarize, plot).
- `bash all_uncompressed.sh`: prefix fingerprints using uncompressed Parquet + uncompressed tables.

## Script arguments

### build.py
- `--bits` (int, default 8): bit width `b` (1..28). Controls bucket count `2^b` and output names.
- `--suffix`: build suffix fingerprints instead of prefix fingerprints.

### readable.py
- `--bits` (int, default 8): reads `q{b}_{prefix|suffix}_boundaries.npy` and writes `q{b}_{prefix|suffix}_boundaries_readable.txt`.
- `--suffix`: interpret boundaries as suffix fingerprints.

### measure_pruning.py
- `--bits` (int, default 8): reads `title_strs_{prefix|suffix}_b{b}.parquet` and `q{b}_{prefix|suffix}_boundaries.npy` when `b <= 16`.
- `--suffix`: estimate pruning for suffix queries.

### bench.py
- `--bits` (int, default 8): input files and column names for bit width `b` (1..28).
- `--warmup` (int, default 1): warmup runs per query (discarded).
- `--reps` (int, default 10): timed runs per query after warmup.
- `--csv` (string, default empty): write per-run timings to a CSV file.
- `--explain`: print `EXPLAIN ANALYZE` for each query.
- `--suffix`: benchmark suffix queries instead of prefix queries.
- `--profile-dir` (string, default empty): write JSON query profiles (and a summary CSV) to this directory.
- `--profile-shell`: use the duckdb shell to generate per-query JSON profiles.
- `--duckdb-bin` (string, default `duckdb`): path to duckdb CLI (used with `--profile-shell`).
- `--parquet-path` (string, default empty): override parquet path (supports `{bits}` and `{mode}`).
- `--force-uncompressed-table`: force uncompressed storage for CTAS tables (disables FSST-style compression).

### summarize_bench.py
- `--csv` (string, required): input CSV from `bench.py`.
- `--out-table` (string, default `bench_summary.csv`): output summary table (appends if file exists).
- `--out-dir` (string, default `bench_plots`): output directory for plots.
- `--plots`: enable plot generation (off by default).

### plot_bench_summary.py
- `--csv` (string, default `bench_summary.csv`): input summary table.
- `--out-dir` (string, default `bench_summary_plots`): output directory for plots.
- `--table-png` (string, default `bench_summary_table.png`): output PNG table name.

### segment_pruning_sim.py
- `--bits` (int, default 8): bit width `b` (1..16).
- `--suffix`: analyze suffix fingerprints.
- `--queries` (string, default empty): comma-separated queries (without `%`).
- `--csv` (string, default empty): append results to a CSV file.
