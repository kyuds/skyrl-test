"""Writes opd-blog-v2's training and eval data in SkyRL's format: M2RL's RL data (arXiv 2602.12566, appendix A), which
is NVIDIA's Nemotron-3-Nano-RL-Training-Blend split by domain, and AIME 2024 and 2025 as M2RL prompts them.

    cd ~/SkyRL && uv run --isolated --extra fsdp python skyrl-test/opd-blog-v2/data/prepare_data.py --out ~/opd-store/data/v2

data/prepare.sh runs it; see data/README.md. Every source is pinned to a revision, so every node writes the same rows.

Outputs (one row per prompt; ``prompt`` is a chat, every other column reaches the environment as an extra):
  math_train.parquet  M2RL's math set: the blend's DAPO-17k and Skywork-OR1 (no OmniMath) rows, 22,056 in the paper.
  mopd_train.parquet  the math set plus the blend's competitive-coding (19,169) and instruction-following (16,575)
                      rows, each labelled with its teacher in ``skyrl_mopd_domain``: math, code or if.
  deepmath_train.parquet  DeepMath-103K, the prompts of Thinking Machines' OPD (section 1): the question alone, as
                      their cookbook's distillation/datasets.py sends it, graded against final_answer.
  aime24.parquet, aime25.parquet   30 problems each.

The prompts follow M2RL's data/process_rl_data.py and data/download_aime.py. Math gets M2RL's DAPO-style instruction
in front of the blend's question, which already ends by asking for \\boxed{}; code and IF prompts are the blend's,
unchanged. The DAPO and Skywork rows of the blend are placeholders, filled from their source datasets the way the
blend's own create_nanov3_jsonl.py fills them (restore_record below follows it line for line).
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

BLEND = ("nvidia/Nemotron-3-Nano-RL-Training-Blend", "74d1844434885be8237369edaaf534f84fd53d87", "train.jsonl")
DAPO = ("BytedTsinghua-SIA/DAPO-Math-17k", "65877096c24ffa7abc4e4fa5edb95cf3413a5674", "data/dapo-math-17k.parquet")
SKYWORK = ("Skywork/Skywork-OR1-RL-Data", "1cdedc52e0e2db85fdf252f9be682e63a5a38c33", "data/math-00000-of-00001.parquet")
DEEPMATH = ("zwhe99/DeepMath-103K", "5cf055d1fe3d7a2eb19719ac020211469736ae44", 10)  # repo, revision, shards
AIME24 = ("math-ai/aime24", "83a7f387baaa524a8bda0022eac0541582297103", "test-00000-of-00001.parquet")
AIME25 = ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "test.jsonl")

# The blend's dataset names, and the teacher each domain routes to under MOPD.
DOMAINS = {
    "nano_v3_sft_profiled_dapo17k": "math",
    "nano_v3_sft_profiled_skywork_no_omni": "math",
    "nano_v3_sft_profiled_comp_coding_50tests": "code",
    "nano_v3_sft_profiled_instruction_following": "if",
}
PAPER_COUNTS = {"math": 22056, "code": 19169, "if": 16575}  # M2RL, appendix A
MATH_ENV, NO_REWARD_ENV = "opd_v2_math", "opd_v2_noreward"   # registered by common/kit_main.py

# M2RL's process_rl_data.py, for the math rows.
MATH_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
# M2RL's download_aime.py.
AIME_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: \\boxed{$Answer} (without quotes) where $Answer is the answer to the problem.\n\n"
)


# ---- the blend's placeholders, as its create_nanov3_jsonl.py fills them ----
def strip_dapo_prompt(text: str) -> str:
    prefix = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form "
        "Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    )
    suffix = 'Remember to put your answer on its own line after "Answer:".'
    start = text.index(prefix) + len(prefix)
    end = text.rfind(suffix)
    return text[start:end]


def get_answer(raw: Any) -> Any:
    if isinstance(raw, str):
        s = raw.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            return json.loads(s)[0]
        return s
    return raw


def restore_question(dataset: str, placeholder: Dict, source_row: Dict) -> str:
    question = source_row["prompt"][0]["content"]
    template = placeholder["question_template"]
    if dataset == "nano_v3_sft_profiled_dapo17k":
        stripped = strip_dapo_prompt(question)
        if template["prefix"]:
            return f"{template['prefix']}{stripped}".removesuffix("\n\n")
        if template["suffix"]:
            return f"{stripped}{template['suffix']}"
        raise ValueError(f"unknown DAPO template: {template}")
    return template["template"].replace("{question}", question)


class SourceRows:
    """Random access to one source parquet's ``prompt`` and ``reward_model`` columns."""

    def __init__(self, spec):
        path = hf_hub_download(spec[0], spec[2], repo_type="dataset", revision=spec[1])
        self.table = pq.read_table(path, columns=["prompt", "reward_model"])

    def __getitem__(self, i: int) -> Dict:
        return {"prompt": self.table.column("prompt")[i].as_py(), "reward_model": self.table.column("reward_model")[i].as_py()}


# ---- rows in SkyRL's format ----
def messages(raw: List[Dict]) -> List[Dict[str, str]]:
    """Role and content only: the blend's tool fields are empty for these domains."""
    return [{"role": m["role"], "content": m["content"]} for m in raw]


def row(prompt: List[Dict[str, str]], domain: str, answer: str, source: str, extra: Dict) -> Dict:
    return {
        "prompt": prompt,
        "env_class": MATH_ENV if domain == "math" else NO_REWARD_ENV,
        "reward_model": {"ground_truth": answer},
        "data_source": source,
        "skyrl_mopd_domain": domain,
        "extra_info": extra,
    }


def write(rows: List[Dict], path: str) -> None:
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    os.replace(tmp, path)
    print(f"wrote {len(rows):>6} rows to {path}")


def blend_rows(blend_path: str, limit: Optional[int]) -> Dict[str, List[Dict]]:
    dapo = skywork = None
    by_domain: Dict[str, List[Dict]] = {"math": [], "code": [], "if": []}
    with open(blend_path) as f:
        for n, line in enumerate(f):
            if limit is not None and n >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if limit is not None:
                    break  # a test slice cut mid-line
                raise
            name = record.get("dataset")
            domain = DOMAINS.get(name)
            if domain is None:
                continue  # science, agent and structured outputs: not in these experiments
            extra = {"blend_index": n, "pass_rate": float(record.get("pass_rate") or 0.0)}
            if domain == "math":
                placeholder = record["_hf_placeholder"]
                if name == "nano_v3_sft_profiled_dapo17k":
                    dapo = dapo or SourceRows(DAPO)
                    source = dapo[int(placeholder["row"])]
                else:
                    skywork = skywork or SourceRows(SKYWORK)
                    source = skywork[int(placeholder["row"])]
                question = restore_question(name, placeholder, source)
                answer = str(get_answer(source["reward_model"]["ground_truth"]))
                prompt = [{"role": "user", "content": MATH_PREFIX + question}]
                by_domain["math"].append(row(prompt, "math", answer, name, extra))
            else:
                prompt = messages(record["responses_create_params"]["input"])
                by_domain[domain].append(row(prompt, domain, "", name, extra))
    return by_domain


def deepmath_rows() -> List[Dict]:
    repo, revision, shards = DEEPMATH
    rows = []
    for i in range(shards):
        path = hf_hub_download(repo, f"data/train-{i:05d}-of-{shards:05d}.parquet", repo_type="dataset", revision=revision)
        table = pq.read_table(path, columns=["question", "final_answer", "difficulty"])
        for r in table.to_pylist():
            prompt = [{"role": "user", "content": r["question"]}]
            extra = {"difficulty": float(r["difficulty"] or 0.0)}
            rows.append(row(prompt, "math", str(r["final_answer"]), "deepmath-103k", extra))
    return rows


def aime_rows(spec, year: int) -> List[Dict]:
    path = hf_hub_download(spec[0], spec[2], repo_type="dataset", revision=spec[1])
    if path.endswith(".jsonl"):
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = pq.read_table(path).to_pylist()
    rows = []
    for r in records:
        # aime24 keeps the answer as \boxed{...} in `solution`; M2RL takes it as solution[7:-1].
        answer = r["solution"][7:-1] if "solution" in r else str(r["answer"])
        prompt = [{"role": "user", "content": AIME_PREFIX + r["problem"]}]
        rows.append(row(prompt, "math", answer, f"aime-{year}", {"id": str(r.get("id", ""))}))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory, e.g. ~/opd-store/data/v2")
    parser.add_argument("--blend-file", help="a local copy of the blend's train.jsonl instead of the pinned download")
    parser.add_argument("--limit", type=int, help="read only this many blend lines (testing); skips the count check")
    args = parser.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)

    blend = args.blend_file or hf_hub_download(BLEND[0], BLEND[2], repo_type="dataset", revision=BLEND[1])
    by_domain = blend_rows(blend, args.limit)
    counts = {d: len(rows) for d, rows in by_domain.items()}
    print("rows per domain:", counts, "| paper:", PAPER_COUNTS)
    if args.limit is None and counts != PAPER_COUNTS:
        raise SystemExit(f"the blend gave {counts}, not the paper's {PAPER_COUNTS}: check the pinned revision")
    sources = Counter(r["data_source"] for r in by_domain["math"])
    print("math rows by source:", dict(sources))

    write(by_domain["math"], os.path.join(out, "math_train.parquet"))
    write(by_domain["math"] + by_domain["code"] + by_domain["if"], os.path.join(out, "mopd_train.parquet"))
    if args.limit is None:
        write(deepmath_rows(), os.path.join(out, "deepmath_train.parquet"))
    write(aime_rows(AIME24, 2024), os.path.join(out, "aime24.parquet"))
    write(aime_rows(AIME25, 2025), os.path.join(out, "aime25.parquet"))


if __name__ == "__main__":
    main()
