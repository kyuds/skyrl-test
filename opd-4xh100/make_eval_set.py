"""Build the kit's eval set, AIME24 plus a GSM8K slice, in a form SkyRL can concatenate.

SkyRL's PromptDataset loads each data.val_data file on its own and then calls
datasets.concatenate_datasets, which refuses a column whose Arrow type differs between files, even
when the difference is only the offset width (string vs large_string). The two sources come from
different writers: SkyRL's DAPO prep ends with a pandas to_parquet, and pandas 3 stores string columns
as large_string, while the GSM8K script writes through datasets, which uses string. Seen 2026-09-27:
the smoke run died in get_eval_dataset on data_source, before any GPU work.

_common.sh runs this at the start of every run, and 01_prepare_data.sh once as a check. It:
  1. rewrites both sources into one type convention (string, list; no large_* types, no schema
     metadata), so whatever the upstream writers do, the outputs are the same;
  2. puts the AIME rows in a fixed order (the DAPO prep deduplicates with polars, whose row order
     varies between preparations) and keeps the first N GSM8K validation rows;
  3. writes each output atomically (temp file, then rename), so an interrupted or concurrent build
     never leaves a half-written file behind;
  4. loads the outputs and concatenates them exactly as PromptDataset does, and checks that every row
     carries the ground truth its env reads (aime: reward_model, gsm8k: reward_spec), so a mismatch
     fails here, in seconds.

    python make_eval_set.py --aime-src SRC --aime-out DST --gsm8k-src SRC --gsm8k-out DST [--gsm8k-rows 256]
"""

import argparse
import json
import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def narrow(t: pa.DataType) -> pa.DataType:
    """The same type with every large_* width replaced by the regular one, at any depth."""
    if pa.types.is_large_string(t):
        return pa.string()
    if pa.types.is_large_binary(t):
        return pa.binary()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        f = t.value_field
        return pa.list_(pa.field(f.name, narrow(f.type), f.nullable))
    if pa.types.is_struct(t):
        return pa.struct([pa.field(f.name, narrow(f.type), f.nullable) for f in t])
    return t


def normalize(table: pa.Table) -> pa.Table:
    """Regular widths everywhere; pandas' index column and all schema metadata dropped."""
    table = table.drop_columns([c for c in table.column_names if c.startswith("__index_level_")])
    schema = pa.schema([pa.field(f.name, narrow(f.type), f.nullable) for f in table.schema])
    return table.cast(schema).replace_schema_metadata(None)


def write_atomic(table: pa.Table, dst: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(dst))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".", suffix=".parquet.tmp")
    os.close(fd)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def build_aime(src: str) -> pa.Table:
    """All AIME rows, ordered by prompt text so every preparation yields the same file order."""
    table = normalize(pq.read_table(src))
    rows = table.to_pylist()
    rows.sort(key=lambda row: json.dumps(row.get("prompt"), sort_keys=True, ensure_ascii=False))
    return pa.Table.from_pylist(rows, schema=table.schema)


def build_gsm8k(src: str, rows: int) -> pa.Table:
    """The first `rows` rows of the GSM8K validation split (its order is the dataset's own, fixed)."""
    table = normalize(pq.read_table(src))
    return table.slice(0, min(rows, table.num_rows))


def check(aime_out: str, gsm8k_out: str) -> None:
    """Load and concatenate the outputs with the calls PromptDataset makes, then check each row's ground truth."""
    import datasets

    datasets.disable_progress_bars()
    loaded = [datasets.load_dataset("parquet", data_files=p, keep_in_memory=True)["train"] for p in (aime_out, gsm8k_out)]
    combined = datasets.concatenate_datasets(loaded)  # the call that failed on 2026-09-27
    aime, gsm8k = loaded

    problems = []
    if "reward_model" not in aime.column_names:
        problems.append("the AIME rows have no reward_model column (the aime env reads reward_model.ground_truth)")
    else:
        missing = [i for i, rm in enumerate(aime["reward_model"]) if not (rm or {}).get("ground_truth")]
        if missing:
            problems.append(f"AIME rows {missing[:5]} have no reward_model.ground_truth")
    if "env_class" in aime.column_names and any(e not in (None, "aime") for e in aime["env_class"]):
        problems.append("AIME rows name an env_class other than aime")
    if "env_class" not in gsm8k.column_names or any(e != "gsm8k" for e in gsm8k["env_class"]):
        problems.append("GSM8K rows must carry env_class=gsm8k (the runs default every other row to aime)")
    if "reward_spec" not in gsm8k.column_names:
        problems.append("the GSM8K rows have no reward_spec column (the gsm8k env reads reward_spec.ground_truth)")
    else:
        missing = [i for i, rs in enumerate(gsm8k["reward_spec"]) if not (rs or {}).get("ground_truth")]
        if missing:
            problems.append(f"GSM8K rows {missing[:5]} have no reward_spec.ground_truth")
    if problems:
        sys.exit("eval set check failed:\n  " + "\n  ".join(problems))

    for name, ds, path in (("aime24", aime, aime_out), ("gsm8k", gsm8k, gsm8k_out)):
        sources = sorted({str(s) for s in ds["data_source"]}) if "data_source" in ds.column_names else ["?"]
        print(f"eval set: {name:6s} {len(ds):4d} rows, data_source {', '.join(sources)} -> {path}")
    print(f"eval set: concatenated {len(combined)} rows the way SkyRL loads data.val_data: OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aime-src", required=True)
    ap.add_argument("--aime-out", required=True)
    ap.add_argument("--gsm8k-src", required=True)
    ap.add_argument("--gsm8k-out", required=True)
    ap.add_argument("--gsm8k-rows", type=int, default=256)
    a = ap.parse_args()
    if a.gsm8k_rows <= 0:
        ap.error("--gsm8k-rows must be positive")

    write_atomic(build_aime(a.aime_src), a.aime_out)
    write_atomic(build_gsm8k(a.gsm8k_src, a.gsm8k_rows), a.gsm8k_out)
    check(a.aime_out, a.gsm8k_out)


if __name__ == "__main__":
    main()
