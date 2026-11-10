"""Task sets for the eval, and the grader.

A task is a stdin/stdout programming problem: Codex writes ``solution.py``, and the grader runs it on the task's
tests. Two sets:

- ``codecontests``: DeepMind's CodeContests (the initiative's target set), read from the dataset's parquet
  conversion on Hugging Face (no ``datasets`` install). The ``n`` lowest-rated Codeforces problems of a split that
  read stdin and write stdout, and that the grader can judge: up to five of the dataset's own Python 3 solutions
  must all pass every chosen test. That drops most problems that accept several answers (CodeContests has no
  checkers, so different right solutions print different answers) and ones a correct Python solution can't do in
  time. A problem whose references all print the same one of several valid answers still gets through, so a few
  rewards of 0 may be right answers the grader can't recognise. Tests: public, then private,
  then generated, up to ``max_tests``.
- ``smoke``: three tiny problems, for checking the pipeline in a couple of minutes.

The grader compares whitespace-separated tokens, with YES and NO in any case (as Codeforces accepts them).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PYTHON3 = 3  # CodeContests' language id for Python 3 solutions
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset=deepmind/code_contests"


@dataclass
class Task:
    id: str
    source: str
    statement: str
    #: Shown to the model (``examples/``).
    examples: list[tuple[str, str]]
    #: Judged on; the examples come first.
    tests: list[tuple[str, str]]
    #: Seconds per test, as the problem states it.
    time_limit: float = 2.0
    rating: int | None = None
    meta: dict = field(default_factory=dict)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")[:60] or "task"


def _pairs(tests: dict | None) -> list[tuple[str, str]]:
    if not tests:
        return []
    return list(zip(tests.get("input") or [], tests.get("output") or [], strict=False))


def codecontests(split: str, n: int, max_rating: int, max_tests: int, cache: Path) -> list[Task]:
    import httpx
    import pyarrow.parquet as pq

    cache.mkdir(parents=True, exist_ok=True)
    files = [f for f in httpx.get(PARQUET_API, timeout=60).json()["parquet_files"] if f["split"] == split]
    if not files:
        raise SystemExit(f"CodeContests has no split {split!r} (test, valid, train)")
    columns = ["name", "description", "public_tests", "private_tests", "generated_tests", "cf_rating", "time_limit",
               "input_file", "output_file", "solutions"]  # fmt: skip
    rows: list[dict] = []
    for index, entry in enumerate(files):
        path = cache / f"code_contests-{split}-{index}.parquet"
        if not path.exists():
            print(f"downloading CodeContests {split} ({entry['size'] / 1e6:.0f} MB) to {path}")
            with httpx.stream("GET", entry["url"], timeout=600, follow_redirects=True) as response:
                response.raise_for_status()
                with path.with_suffix(".part").open("wb") as sink:
                    for chunk in response.iter_bytes():
                        sink.write(chunk)
            path.with_suffix(".part").rename(path)
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    eligible = [
        r for r in rows
        if not r["input_file"] and not r["output_file"] and _pairs(r["public_tests"]) and 0 < (r["cf_rating"] or 0) <= max_rating
    ]  # fmt: skip
    eligible.sort(key=lambda r: (r["cf_rating"], r["name"]))
    # Whether a reference solution passes a problem's tests, by name: slow to find out, so kept between runs.
    judged_path = cache / f"judgeable-{split}-{max_tests}.json"
    judged: dict[str, str] = json.loads(judged_path.read_text()) if judged_path.exists() else {}
    tasks: list[Task] = []
    skipped: dict[str, str] = {}
    for r in eligible:
        if len(tasks) == n:
            break
        examples = _pairs(r["public_tests"])
        tests: list[tuple[str, str]] = []
        for pair in examples + _pairs(r["private_tests"]) + _pairs(r["generated_tests"]):
            if pair not in tests:
                tests.append(pair)
        limit = r["time_limit"] or {}
        seconds = (limit.get("seconds") or 0) + (limit.get("nanos") or 0) / 1e9
        task = Task(
            id=slug(r["name"]),
            source=f"codecontests/{split}",
            statement=r["description"],
            examples=examples,
            tests=tests[:max_tests],
            time_limit=seconds or 2.0,
            rating=r["cf_rating"],
            meta={"name": r["name"]},
        )
        if r["name"] not in judged:
            judged[r["name"]] = _judgeable(task, r["solutions"] or {})
            judged_path.write_text(json.dumps(judged, indent=1))
        if judged[r["name"]] == "ok":
            tasks.append(task)
        else:
            skipped[task.id] = judged[r["name"]]
    if skipped:
        print(
            f"skipped {len(skipped)} problems the grader can't judge: "
            + ", ".join(f"{k} ({v})" for k, v in skipped.items())
        )
    if len(tasks) < n:
        print(f"note: only {len(tasks)} judgeable CodeContests {split} problems rated <= {max_rating}")
    return tasks


def _judgeable(task: Task, solutions: dict) -> str:
    """ "ok" when the first five (or fewer) reference Python 3 solutions all pass every test, else why not."""
    sources = [
        code
        for lang, code in zip(solutions.get("language") or [], solutions.get("solution") or [], strict=False)
        if lang == PYTHON3
    ]
    if not sources:
        return "no Python 3 reference"
    for index, code in enumerate(sources[:5], 1):
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / "solution.py").write_text(code)
            verdict = grade(task, Path(work))["verdict"]
        if verdict != "accepted":
            return f"reference {index} of {min(len(sources), 5)}: {verdict}"
    return "ok"


def smoke() -> list[Task]:
    def task(id_: str, statement: str, tests: list[tuple[str, str]]) -> Task:
        return Task(id=id_, source="smoke", statement=statement, examples=tests[:2], tests=tests)

    return [
        task(
            "sum-of-list",
            "The first line holds n (1 <= n <= 1000). The second holds n integers, each between -10^9 and 10^9. "
            "Print their sum.",
            [
                ("3\n1 2 3\n", "6\n"),
                ("1\n-5\n", "-5\n"),
                ("4\n1000000000 1000000000 1000000000 1000000000\n", "4000000000\n"),
            ],
        ),
        task(
            "fizzbuzz-n",
            "Given n (1 <= n <= 100), print the numbers 1 to n, one per line, but print Fizz for multiples of 3, "
            "Buzz for multiples of 5 and FizzBuzz for multiples of both.",
            [
                ("5\n", "1\n2\nFizz\n4\nBuzz\n"),
                ("15\n", "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\nFizz\n13\n14\nFizzBuzz\n"),
            ],
        ),
        task(
            "second-largest",
            "The first line holds n (2 <= n <= 10^5). The second holds n integers. Print the second largest distinct "
            "value, or -1 if all values are equal.",
            [("4\n3 1 4 4\n", "3\n"), ("3\n7 7 7\n", "-1\n"), ("2\n-1 -2\n", "-2\n")],
        ),
    ]


def write_workspace(task: Task, work: Path) -> None:
    """What Codex starts from: the statement and the examples."""
    (work / "examples").mkdir(parents=True, exist_ok=True)
    (work / "problem.md").write_text(task.statement)
    for index, (given, expected) in enumerate(task.examples, 1):
        (work / "examples" / f"{index}.in").write_text(given)
        (work / "examples" / f"{index}.out").write_text(expected)


def _tokens(text: str) -> list[str]:
    return [token.upper() if token.upper() in ("YES", "NO") else token for token in text.split()]


def grade(task: Task, work: Path) -> dict:
    """Runs ``work/solution.py`` on every test until one fails."""
    solution = work / "solution.py"
    total = len(task.tests)
    if not solution.exists():
        return {"reward": 0.0, "passed": 0, "total": total, "verdict": "no solution.py"}
    timeout = task.time_limit * 3 + 1  # Python against limits set for C++
    for index, (given, expected) in enumerate(task.tests):
        try:
            run = subprocess.run(
                ["python3", "solution.py"],
                input=given,
                capture_output=True,
                text=True,
                cwd=work,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"reward": 0.0, "passed": index, "total": total, "verdict": f"test {index + 1}: time limit"}
        if run.returncode != 0:
            error = (run.stderr.strip().splitlines() or [""])[-1][:120]
            verdict = f"test {index + 1}: exit {run.returncode}" + (f" ({error})" if error else "")
            return {"reward": 0.0, "passed": index, "total": total, "verdict": verdict}
        if _tokens(run.stdout) != _tokens(expected):
            return {"reward": 0.0, "passed": index, "total": total, "verdict": f"test {index + 1}: wrong answer"}
    return {"reward": 1.0, "passed": total, "total": total, "verdict": "accepted"}
