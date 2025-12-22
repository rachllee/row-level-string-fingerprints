import pandas as pd

PARQUET_FILE = "title_strs.parquet"

df = pd.read_parquet(PARQUET_FILE)

print("=== Parquet File Summary ===")
print(f"Row count: {len(df)}")
print("\nColumns:")
for col in df.columns:
    print(f" - {col}")

print("\nDtypes:")
print(df.dtypes)

print("\nPreview (first 5 rows):")
print(df.head())
