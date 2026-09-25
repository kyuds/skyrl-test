#!/usr/bin/env bash
# Data for both smoke runs. Run from the SkyRL checkout on the workspace, after 00_env.sh.
#   bash skyrl-test/anyscale-pr2198/01_prepare_data.sh              # gsm8k + SQL prompts + sliced eval sets
#   WITH_SQL_DB=1 bash skyrl-test/anyscale-pr2198/01_prepare_data.sh # + the OmniSQL databases (22 GB zip, ~50 GB unzipped)
set -euo pipefail
DATA="${PR2198_DATA:-$HOME/data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SQL_EVAL_ROWS="${SQL_EVAL_ROWS:-32}"
GSM8K_EVAL_ROWS="${GSM8K_EVAL_ROWS:-256}"

hf_dl() { if command -v hf >/dev/null 2>&1; then hf "$@"; else uv run hf "$@"; fi; }

# gsm8k: 7473 train / 1319 validation rows, exactly as ci/gpu_e2e_test_run.sh prepares it.
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA/gsm8k"

# SkyRL-SQL-653 prompts: 4.2 MB train, 0.6 MB validation (1034 Spider rows). Sizes fetched 2026-09-14.
hf_dl download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir "$DATA/sql" --repo-type dataset

# Smoke-sized eval sets. On the full sets the four evals would be the long pole of each run.
uv run "$HERE/slice_parquet.py" "$DATA/gsm8k/validation.parquet" "$DATA/gsm8k/validation_smoke.parquet" --rows "$GSM8K_EVAL_ROWS"
uv run "$HERE/slice_parquet.py" "$DATA/sql/validation.parquet" "$DATA/sql/validation_smoke.parquet" --rows "$SQL_EVAL_ROWS"

if [[ "${WITH_SQL_DB:-0}" == "1" ]]; then
  # The text2sql env executes generated SQL against these; every SQL eval/train row needs them.
  # Charlie's script expects DB_PATH=$HOME/data/sql/db_files/data, so unzip into db_files/ (the
  # zip's root being data/ is ASSUMED from that default -- the ls below is the check).
  mkdir -p "$DATA/sql/db_files"
  hf_dl download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir "$DATA/sql/db_files"
  (cd "$DATA/sql/db_files" && unzip -q data.zip && rm -f data.zip)
  echo "DB root should be $DATA/sql/db_files/data:"; ls "$DATA/sql/db_files/data" | head
fi
echo "done: $DATA"
