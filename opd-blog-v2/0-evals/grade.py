"""Grades an eval folder: responses.jsonl (generate.py) against grading.jsonl (benchmarks.py). Writes graded.jsonl,
one line per sample, and scores.json, one entry per benchmark, and prints the scores.

    cd ~/SkyRL && uv run --isolated --extra fsdp --with math-verify==0.9.0 <IF grader deps> \\
        python skyrl-test/opd-blog-v2/0-evals/grade.py --dir <eval folder> --tools ~/opd-store/tools

0-evals/run.sh has the full command. Graders:
  math (aime24, aime25, and the traces' prompts)
          common/kit_main.py's is_correct, the one the OPD runs log: the answer after the last </think>, its last
          \\boxed{} or else its last "Answer:" line, compared as a string and then by math-verify. A response that
          never closes its think block was cut off and is wrong. (M2RL grades with QwQ-32B's evaluator, so small
          differences from their numbers are expected.)
  lcb_*   LiveCodeBench's own grader (lcb_runner.evaluation.codegen_metrics, the clone in --tools): the last ``` block
          after </think>, run on the public and private tests; a sample passes when every test passes.
  ifeval, ifbench
          IFBench's strict checker (evaluation_lib.test_instruction_following_strict), which carries the classic IFEval
          checks too; prompt-level strict accuracy on the text after </think>.
A benchmark's score is the mean over all its samples, Avg@K as M2RL reports it, in percent.
"""

import argparse
import importlib.util
import json
import os
import sys
import types
from collections import defaultdict
from typing import Dict, List

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_BENCHES = ("aime24", "aime25", "math")


def answer_text(response: str) -> str:
    """What follows the last </think>; empty when the think block never closes (the answer was cut off)."""
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()
    return "" if "<think>" in response else response.strip()


def load_is_correct():
    """common/kit_main.py's is_correct. kit_main imports ray and skyrl_gym for its environments; where they are missing
    (a laptop), stand-ins let the grader load without them."""
    try:
        import ray  # noqa: F401
        import skyrl_gym.envs.base_text_env  # noqa: F401
    except ImportError:
        ray_stub = types.ModuleType("ray")
        ray_stub.remote = lambda *a, **k: (lambda f: f)
        env_stub = types.ModuleType("skyrl_gym.envs.base_text_env")
        env_stub.BaseTextEnv = object
        env_stub.BaseTextEnvStepOutput = dict
        for name, mod in (("ray", ray_stub), ("skyrl_gym", types.ModuleType("skyrl_gym")),
                          ("skyrl_gym.envs", types.ModuleType("skyrl_gym.envs")),
                          ("skyrl_gym.envs.base_text_env", env_stub)):
            sys.modules.setdefault(name, mod)
    spec = importlib.util.spec_from_file_location("kit_main", os.path.join(KIT, "common", "kit_main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import logging

    logging.getLogger("math_verify").setLevel(logging.ERROR)
    return module.is_correct


def extract_code(text: str) -> str:
    """LiveCodeBench's extract_code for chat models: the last fenced block."""
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def grade_math(rows: List[Dict], meta: Dict[str, Dict]) -> List[bool]:
    is_correct = load_is_correct()
    return [is_correct(r["response"], str(meta[r["id"]]["answer"])) for r in rows]


def grade_lcb(rows: List[Dict], meta: Dict[str, Dict], tools: str, processes: int) -> List[bool]:
    sys.path.insert(0, os.path.join(tools, "LiveCodeBench"))
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    by_problem: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_problem[r["id"]].append(i)
    ids = sorted(by_problem)
    width = max(len(v) for v in by_problem.values())
    # codegen_metrics wants the same number of generations per problem; a missing one is graded as empty and ignored.
    generations = []
    for pid in ids:
        codes = [extract_code(answer_text(rows[i]["response"])) for i in by_problem[pid]]
        generations.append(codes + [""] * (width - len(codes)))
    _, results, _ = codegen_metrics(
        [meta[pid]["sample"] for pid in ids], generations, k_list=[1], num_process_evaluate=processes, timeout=6
    )
    passed = [False] * len(rows)
    for p, pid in enumerate(ids):
        for j, i in enumerate(by_problem[pid]):
            tests = results[p][j]
            passed[i] = bool(tests) and all(t is True or (not isinstance(t, bool) and t > 0) for t in tests)
    return passed


def grade_if(rows: List[Dict], meta: Dict[str, Dict], tools: str) -> List[bool]:
    sys.path.insert(0, os.path.join(tools, "IFBench"))
    import evaluation_lib  # IFBench's; downloads its nltk data on first import
    import langdetect

    langdetect.DetectorFactory.seed = 0  # the language checks are otherwise random from run to run

    out = []
    for r in rows:
        m = meta[r["id"]]
        kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in m["kwargs"]]
        inp = evaluation_lib.InputExample(key=m["key"], instruction_id_list=m["instruction_id_list"], prompt=m["prompt"], kwargs=kwargs)
        result = evaluation_lib.test_instruction_following_strict(inp, {m["prompt"]: answer_text(r["response"])})
        out.append(bool(result.follow_all_instructions))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="eval folder with grading.jsonl and responses.jsonl")
    parser.add_argument("--tools", required=True, help="folder holding the LiveCodeBench and IFBench clones")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    folder, tools = os.path.expanduser(args.dir), os.path.expanduser(args.tools)

    with open(os.path.join(folder, "grading.jsonl")) as f:
        meta = {m["id"]: m for m in map(json.loads, f)}
    with open(os.path.join(folder, "responses.jsonl")) as f:
        responses = [json.loads(line) for line in f]
    by_bench: Dict[str, List[Dict]] = defaultdict(list)
    for r in responses:
        by_bench[r["bench"]].append(r)

    scores, graded = {}, []
    for bench in sorted(by_bench):
        rows = by_bench[bench]
        if bench in MATH_BENCHES:
            ok = grade_math(rows, meta)
        elif bench.startswith("lcb_"):
            ok = grade_lcb(rows, meta, tools, args.processes)
        elif bench in ("ifeval", "ifbench"):
            ok = grade_if(rows, meta, tools)
        else:
            raise SystemExit(f"no grader for {bench}")
        for r, c in zip(rows, ok):
            graded.append({"id": r["id"], "bench": bench, "sample": r["sample"], "correct": bool(c),
                           "finish_reason": r["finish_reason"], "tokens": r["tokens"]})
        scores[bench] = {
            "score": round(100.0 * sum(ok) / len(ok), 2),
            "problems": len({r["id"] for r in rows}),
            "samples": len(rows),
            "cut_rate": round(100.0 * sum(r["finish_reason"] == "length" for r in rows) / len(rows), 2),
            "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows)),
        }
        s = scores[bench]
        print(f"{bench:8s} {s['score']:6.2f}  ({s['problems']} problems, {s['samples']} samples, "
              f"{s['cut_rate']}% cut at the length limit, {s['mean_tokens']} tokens on average)")

    with open(os.path.join(folder, "graded.jsonl"), "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    with open(os.path.join(folder, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
