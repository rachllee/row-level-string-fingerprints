"""
Convert IMDB title.basics.tsv.gz to title_strs.parquet format.

This extracts the primaryTitle column and saves it as a single-column parquet file
compatible with the fingerprint benchmarking tools.
"""

import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert IMDB TSV to parquet")
    parser.add_argument("--input", default="title.basics.tsv.gz", help="Input TSV file (can be gzipped)")
    parser.add_argument("--output", default="title_strs_imdb.parquet", help="Output parquet file")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    # DuckDB can read gzipped TSV directly
    import duckdb

    con = duckdb.connect()

    # Read the TSV and extract just the primaryTitle column
    query = f"""
        SELECT primaryTitle as title
        FROM read_csv_auto('{args.input}', delim='\t', header=true, ignore_errors=true)
        WHERE primaryTitle IS NOT NULL
    """

    df = con.execute(query).df()

    print(f"Loaded {len(df):,} titles")
    print(f"Sample titles:")
    print(df.head(10))

    print(f"\nWriting to {args.output}...")
    df.to_parquet(args.output, index=False)
    print(f"Done! Wrote {len(df):,} rows to {args.output}")

    con.close()

if __name__ == "__main__":
    main()
