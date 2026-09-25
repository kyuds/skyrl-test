"""Keep the first N rows of a parquet file, schema untouched: a smoke-sized eval set.

    uv run skyrl-test/anyscale-pr2198/slice_parquet.py SRC.parquet DST.parquet --rows 32
"""

import argparse

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--rows", type=int, default=32)
a = ap.parse_args()

table = pq.read_table(a.src)
n = min(a.rows, table.num_rows)
pq.write_table(table.slice(0, n), a.dst)
print(f"{a.dst}: {n} of {table.num_rows} rows")
