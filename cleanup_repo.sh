#!/bin/bash
# Safe repository cleanup for replication
# Removes regenerable and unused files

set -euo pipefail

echo "=========================================="
echo "Repository Cleanup for Replication"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - Old/intermediate results"
echo "  - Unused scripts"
echo "  - Regenerable data files"
echo "  - System files (__pycache__, .DS_Store)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Removing files..."

# Old/intermediate results
rm -f all_1char_prefixes_b8.csv
rm -f bench_summary.csv
rm -f pruning_speedup.csv
echo "  ✓ Removed old result files"

# Regenerable data (title_strs_imdb.parquet can be regenerated from .tsv.gz)
rm -f title_strs_imdb.parquet
echo "  ✓ Removed regenerable parquet files"

# Old/unused scripts
rm -f all.sh
rm -f all_pruning.sh
rm -f all_suffix.sh
rm -f all_uncompressed.sh
rm -f bench.py
rm -f summarize_bench.py
rm -f plot_bench_summary.py
rm -f compare_pruning_speedup.py
rm -f measure_pruning.py
rm -f segment_pruning_sim.py
rm -f readable.py
rm -f data.py
echo "  ✓ Removed old/unused scripts"

# Optionally remove infix-related files (uncomment if not needed)
# rm -f all_infix.sh bench_infix.py bench_sampled_infix.py build_infix.py plot_infix.py
# echo "  ✓ Removed infix files"

# System files
rm -rf __pycache__/
rm -f .DS_Store
echo "  ✓ Removed system files"

# Clean up any generated intermediate files from cleanup.sh
./cleanup.sh

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
echo ""
echo "Kept essential files:"
echo "  - Core scripts: build.py, bench_all_prefixes.py, plot_selectivity.py, etc."
echo "  - Source data: title.basics.tsv.gz, title_strs.parquet"
echo "  - Key results: all_prefixes_imdb_b8.csv"
echo ""
echo "To regenerate IMDB parquet:"
echo "  python prepare_imdb.py"
echo ""
echo "To run full benchmark:"
echo "  ./all_selectivity.sh"
