#!/bin/bash
set -euo pipefail

# Benchmark all distinct 3-char prefixes/suffixes and plot speedup vs selectivity
# This provides unbiased measurements across the actual data distribution

BITS=${1:-8}  # Default to 8 bits, can override with first argument
N=3           # 3-character n-grams
WARMUP=2
REPS=10
OUT_DIR="selectivity_plots_b${BITS}"

echo "=========================================="
echo "Benchmarking all ${N}-char prefixes (${BITS} bits)"
echo "=========================================="

# Ensure the parquet files exist
if [[ ! -f "title_strs_prefix_b${BITS}.parquet" ]]; then
  echo "Building prefix fingerprints with ${BITS} bits..."
  python build.py --bits "${BITS}"
fi

# Benchmark all prefixes
PREFIX_CSV="all_prefixes_b${BITS}.csv"
echo ""
echo "=== Benchmarking all distinct ${N}-char prefixes ==="
python bench_all_prefixes.py \
  --bits "${BITS}" \
  --n "${N}" \
  --warmup "${WARMUP}" \
  --reps "${REPS}" \
  --csv "${PREFIX_CSV}"

# Generate plots
echo ""
echo "=== Generating prefix plots ==="
python plot_selectivity.py \
  --csv "${PREFIX_CSV}" \
  --out-dir "${OUT_DIR}/prefix" \
  --title "Prefix ${BITS}-bit"

# Optionally do suffix if the files exist
if [[ -f "title_strs_suffix_b${BITS}.parquet" ]]; then
  SUFFIX_CSV="all_suffixes_b${BITS}.csv"
  echo ""
  echo "=== Benchmarking all distinct ${N}-char suffixes ==="
  python bench_all_prefixes.py \
    --bits "${BITS}" \
    --suffix \
    --n "${N}" \
    --warmup "${WARMUP}" \
    --reps "${REPS}" \
    --csv "${SUFFIX_CSV}"

  echo ""
  echo "=== Generating suffix plots ==="
  python plot_selectivity.py \
    --csv "${SUFFIX_CSV}" \
    --out-dir "${OUT_DIR}/suffix" \
    --title "Suffix ${BITS}-bit"
fi

echo ""
echo "=========================================="
echo "Done! Results in ${OUT_DIR}/"
echo "=========================================="
