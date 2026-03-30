#!/bin/bash
set -euo pipefail

# Test various prefix/suffix bit combinations
# Format: "prefix_bits,suffix_bits"
BIT_COMBOS=(
  "4,4"
  "8,8"
  "4,8"
  "8,4"
  "8,12"
  "12,8"
  "12,12"
  "16,16"
)

SUMMARY_CSV="infix_bench_summary.csv"
OUT_DIR="infix_plots"

mkdir -p csvs_infix

# Remove old summary if exists (we append to it)
rm -f "${SUMMARY_CSV}"

for combo in "${BIT_COMBOS[@]}"; do
  IFS=',' read -r pb sb <<< "${combo}"

  echo ""
  echo "========================================"
  echo "=== Prefix bits: ${pb}, Suffix bits: ${sb} ==="
  echo "========================================"

  echo "=== build_infix.py --prefix-bits ${pb} --suffix-bits ${sb} ==="
  python build_infix.py --prefix-bits "${pb}" --suffix-bits "${sb}"

  csv_path="csvs_infix/infix_p${pb}_s${sb}.csv"
  echo "=== bench_infix.py --prefix-bits ${pb} --suffix-bits ${sb} ==="
  python bench_infix.py \
    --prefix-bits "${pb}" \
    --suffix-bits "${sb}" \
    --warmup 2 \
    --reps 10 \
    --csv "${csv_path}"

  if [[ -f "${csv_path}" ]]; then
    echo "=== summarize_bench.py --csv ${csv_path} ==="
    python summarize_bench.py --csv "${csv_path}" --out-table "${SUMMARY_CSV}"
  else
    echo "Missing ${csv_path}; skipping summarize for p${pb}_s${sb}"
  fi
done

echo ""
echo "========================================"
echo "=== Generating plots ==="
echo "========================================"

# Use the dedicated infix plotting script
python plot_infix.py --csv-dir csvs_infix --out-dir "${OUT_DIR}"

echo ""
echo "Summary CSV: ${OUT_DIR}/infix_summary.csv"
echo "Plots: ${OUT_DIR}/"
echo ""
echo "=== Done ==="
