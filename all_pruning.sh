#!/bin/bash
set -euo pipefail

BITS=(1 2 4 8 12 16 20 24 28)
PROFILE_BASE="profiles_bits"
SUMMARY_CSV="bench_summary.csv"
OUT_DIR="pruning_speedup_plots"
OUT_CSV="pruning_speedup.csv"

mkdir -p csvs
mkdir -p "${PROFILE_BASE}"

for b in "${BITS[@]}"; do
  profile_dir="${PROFILE_BASE}/b${b}"
  echo "=== build.py --bits ${b} ==="
  python build.py --bits "${b}"

  echo "=== bench.py --bits ${b} --profile-dir ${profile_dir} --profile-shell ==="
  csv_path="csvs/${b}bit-result.csv"
  python bench.py --bits "${b}" --csv "${csv_path}" --profile-dir "${profile_dir}" --profile-shell

  if [[ -f "${csv_path}" ]]; then
    echo "=== summarize_bench.py --csv ${csv_path} ==="
    python summarize_bench.py --csv "${csv_path}" --out-table "${SUMMARY_CSV}"
  else
    echo "Missing ${csv_path}; skipping summarize for bits ${b}"
  fi
done

echo "=== merge profile_rows.csv ==="
python - <<'PY'
import csv
import glob

paths = sorted(glob.glob("profiles_bits/b*/profile_rows.csv"))
if not paths:
    raise SystemExit("No profile_rows.csv files found under profiles_bits/")

rows = []
header = None
for p in paths:
    with open(p, newline="") as f:
        r = csv.reader(f)
        h = next(r, None)
        if header is None:
            header = h
        for row in r:
            rows.append(row)

with open("profiles_bits/profile_rows_all.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print("Wrote profiles_bits/profile_rows_all.csv")
PY

echo "=== compare_pruning_speedup.py ==="
python compare_pruning_speedup.py \
  --bench-summary "${SUMMARY_CSV}" \
  --profile-rows "profiles_bits/profile_rows_all.csv" \
  --out-csv "${OUT_CSV}" \
  --out-dir "${OUT_DIR}"
