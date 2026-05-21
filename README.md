# row-level-string-fingerprints

Prototype for building fingerprint columns on string data to accelerate string predicate queries. Covers three query types:
- **Prefix** queries (`LIKE 'foo%'`) and **suffix** queries (`LIKE '%foo'`) via quantile bucket codes with zone map pruning
- **Substring** queries (`LIKE '%foo%'`) via 16-bit character presence bitmask with Arrow-compute row-level filtering

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
- `--force-compression` (string, default empty): force a specific compression for CTAS tables (e.g., `dict_fsst`, `fsst`).

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

---

## Substring fingerprint (16-bit character bitmask)

Bit `i` of `fp16_chars` = 1 if the string contains `selected_features[i]`. Features are chosen via decision tree induction to maximize entropy. At query time, a pattern mask is computed from the query string and used to filter rows before substring evaluation.

### Quick start

```bash
# 1. Build fingerprint parquet (unigrams, recommended)
python build_substr.py --ngram 1 --bits 16

# 2. Run custom Arrow-compute scanner on a few patterns
python custom_scan.py --patterns "ing" "python" "the" "2024"

# 3. Benchmark across 600 sampled patterns and generate plots
python bench_custom_scan.py --samples 100 --lengths 3 4 5 6 8 10 --reps 3
# → writes custom_scan_bench.csv and custom_scan_plots/
```

### Unigram vs bigram comparison

```bash
# Build and benchmark with unigrams
python build_substr.py --ngram 1
python bench_custom_scan.py --csv custom_scan_bench.csv --out-dir custom_scan_plots

# Build and benchmark with bigrams
python build_substr.py --ngram 2
python bench_custom_scan.py --csv custom_scan_bench_bigram.csv --out-dir custom_scan_plots_bigram

# Side-by-side comparison plot → custom_scan_plots/unigram_vs_bigram.png
python plot_comparison.py
```

### DuckDB bitmask benchmark (for reference)

```bash
python bench_substr.py --samples 300 --lengths 3 4 5 6 8 10
```
Note: DuckDB does not evaluate the bitmask filter before ILIKE, so speedup is ~1.0x regardless of selectivity.

### Row-group skipping test

```bash
python repartition_parquet.py
```
Repartitions the parquet into row groups of 10k / 50k / 122k / 500k rows and measures what fraction of row groups can be skipped. With character-level features at ~50% frequency, skipping is 0% at all sizes.

### Script arguments

#### build_substr.py
- `--bits` (int, default 16): number of fingerprint bits / features selected.
- `--sample` (int, default 200000): rows sampled for feature selection.
- `--ngram` (int, default 2): n-gram size — 1 for characters, 2 for bigrams.
- `--max-freq` (float, default 0.01): maximum feature frequency; use 1.0 to allow all frequencies and let entropy selection decide.

#### custom_scan.py
- `--patterns` (list, default several examples): substring patterns to test.
- `--reps` (int, default 3): timed repetitions per pattern.

#### bench_custom_scan.py
- `--samples` (int, default 100): patterns sampled per length.
- `--lengths` (list, default 3 4 5 6 8 10): pattern lengths to test.
- `--reps` (int, default 3): timed repetitions per pattern.
- `--csv` (string, default `custom_scan_bench.csv`): output CSV path.
- `--out-dir` (string, default `custom_scan_plots`): output directory for plots.
- `--seed` (int, default 42): random seed for pattern sampling.
