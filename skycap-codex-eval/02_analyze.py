"""The graph structure of an eval run's records, checked: is every rollout what a trainer can use, and where it
isn't, why.

Each record is rebuilt with skycap's own reader (``skycap.record.load``) and checked against invariants that must
hold for every trajectory, whatever the model did:

- ``tokens``      every node on every path has its tokens (token mode records them all);
- ``prompt``      each call's prompt is exactly the tokens of the path above it: the ancestors' tokens plus the
                  model node's scaffold (``usage.prompt_tokens`` = their count), and its completion is the rest
                  of the node (``usage.completion_tokens``);
- ``logprobs``    every sampled token has a finite logprob <= 0;
- ``dialect``     every call came in on ``openai_responses``;
- ``samples``     the training rows (``skycap.samples.build_samples``) cover every leaf, and every sampled token is
                  trained in exactly one row.

and, separately, for what is expected of a clean Codex rollout, with each deviation explained:

- one path (one root, no branch points);
- every call after the first bridged (it extended the previous call's tokens);
- no client-authored assistant node (every model turn Codex sent back matched what was sampled).

A fork or an unbridged call is not an invariant failure: it is classified (resampled, a model turn the client
changed, a turn that ended inside <think>, a history edit) and reported. The exit code is 1 when an invariant
fails, 0 otherwise.

    bash skyrl-test/skycap-codex-eval/02_analyze.sh <run>          # on the VM, or on the Mac after pull.sh
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from skycap import record
from skycap.graph import MessageGraph, Node
from skycap.samples import build_samples

INVARIANTS = ("tokens", "prompt", "logprobs", "dialect", "samples")


def sampled(node: Node) -> int:
    tokens = node.tokens
    return len(tokens.token_ids) - tokens.sampled_start if tokens and tokens.sampled_start is not None else 0


def reasoning_only(message: dict[str, Any]) -> bool:
    return bool(message.get("reasoning_content")) and not message.get("content") and not message.get("tool_calls")


def describe(node: Node) -> str:
    """One line for a node in a tree."""
    m = node.message
    if node.role == "assistant":
        parts = []
        if m.get("reasoning_content"):
            parts.append(f"reasoning {len(m['reasoning_content'])}ch")
        if m.get("content"):
            parts.append(f"text {json.dumps(m['content'][:50], ensure_ascii=False)}")
        for call in m.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(f"{function.get('name')}()")
        what = ", ".join(parts) or "empty"
    else:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        what = json.dumps((text or "")[:60], ensure_ascii=False)
    out = f"[{node.id}] {node.role} ({node.author}) {what}"
    if node.tokens is not None:
        out += f"  {len(node.tokens.token_ids)} tok"
    for call in node.calls:
        bridged = {True: "bridged", False: "UNBRIDGED", None: "first call"}[call.bridged]
        out += f"  <- call: {call.finish_reason}, {bridged}"
    return out


def tree(graph: MessageGraph, limit: int = 80) -> list[str]:
    """The graph as text. A straight run of nodes stays at one indent; only a branch point indents."""
    lines: list[str] = []

    def walk(node_id: int, first: str, rest: str) -> None:
        while len(lines) < limit:
            lines.append(first + describe(graph.nodes[node_id]))
            children = graph.children(node_id)
            if len(children) != 1:
                break
            node_id, first = children[0], rest
        else:
            return
        for index, child in enumerate(children):
            last = index == len(children) - 1
            walk(child, rest + ("└─ " if last else "├─ "), rest + ("   " if last else "│  "))

    for root in graph.roots():
        walk(root, "", "")
    if len(lines) >= limit:
        lines.append(f"... ({len(graph.nodes)} nodes)")
    return lines


def check_invariants(graph: MessageGraph) -> dict[str, list[str]]:
    """Invariant name -> problems (empty when it holds)."""
    problems: dict[str, list[str]] = {name: [] for name in INVARIANTS}
    for path in graph.paths():
        missing = [node for node in path if graph.nodes[node].tokens is None]
        if missing:
            problems["tokens"].append(f"path to {path[-1]}: nodes {missing} have no tokens")
    for node in graph:
        if node.author != "model":
            continue
        tokens = node.tokens
        if tokens is None or tokens.sampled_start is None:
            problems["tokens"].append(f"model node {node.id} has no sampled tokens")
            continue
        above = sum(len(graph.nodes[a].tokens.token_ids) for a in graph.path_to(node.id)[:-1] if graph.nodes[a].tokens)
        for call in node.calls:
            usage = call.usage or {}
            if usage.get("prompt_tokens") != above + tokens.sampled_start:
                problems["prompt"].append(
                    f"node {node.id}: prompt_tokens {usage.get('prompt_tokens')} != path tokens {above} + scaffold "
                    f"{tokens.sampled_start}"
                )
            if usage.get("completion_tokens") != sampled(node):
                problems["prompt"].append(
                    f"node {node.id}: completion_tokens {usage.get('completion_tokens')} != {sampled(node)} sampled"
                )
            if call.dialect != "openai_responses":
                problems["dialect"].append(f"node {node.id}: call recorded as {call.dialect}")
        logprobs = (tokens.logprobs or [])[tokens.sampled_start :]
        if len(logprobs) != sampled(node):
            problems["logprobs"].append(f"node {node.id}: {len(logprobs)} logprobs for {sampled(node)} sampled tokens")
        bad = [v for v in logprobs if not math.isfinite(v) or v > 1e-6]
        if bad:
            problems["logprobs"].append(f"node {node.id}: {len(bad)} logprobs not finite or > 0, e.g. {bad[:3]}")
    samples = build_samples(graph)
    if sorted(s.leaf for s in samples) != sorted(graph.leaves()):
        problems["samples"].append(
            f"rows end at {sorted(s.leaf for s in samples)}, leaves are {sorted(graph.leaves())}"
        )
    trained = sum(sum(s.loss_mask or []) for s in samples)
    total = sum(sampled(node) for node in graph if node.author == "model")
    if trained != total:
        problems["samples"].append(f"{trained} tokens trained across rows, {total} sampled")
    return problems


def counterpart(graph: MessageGraph, node: Node) -> Node | None:
    """The node with the same message in the other branch of a fork: under its parent's counterpart (or its own
    parent, if that is shared), with the same match hash. A re-rendered history is a chain of such copies."""
    if node.parent is None:
        siblings = graph.roots()
    else:
        above = counterpart(graph, graph.nodes[node.parent])
        siblings = graph.children(above.id if above is not None else node.parent)
    for sibling in siblings:
        other = graph.nodes[sibling]
        if other.id != node.id and other.match_hash == node.match_hash:
            return other
    return None


def descendants(graph: MessageGraph, node_id: int) -> list[int]:
    out, stack = [], list(graph.children(node_id))
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(graph.children(current))
    return sorted(out)


def why_rerendered(model: Node) -> str:
    """Why the renderer couldn't extend this model turn, as far as the record says."""
    if any(call.finish_reason == "length" for call in model.calls):
        return "it was cut off (length)"
    if reasoning_only(model.message):
        return "it ended inside <think> (no answer, no tool call)"
    return "the renderer declined to extend it"


def classify_unbridged(graph: MessageGraph, node: Node) -> str:
    path = [graph.nodes[a] for a in graph.path_to(node.id)[:-1]]
    if any(a.author == "model" for a in path):
        model = [a for a in path if a.author == "model"][-1]
        return f"rendered from messages though node {model.id} was on its path ({why_rerendered(model)})"
    copies = [a for a in path if a.role == "assistant"]
    if not copies:
        return "no model turn before it to extend (a new conversation in the same trajectory)"
    original = counterpart(graph, copies[-1])
    if original is not None and original.author == "model":
        return f"could not extend node {original.id}, because {why_rerendered(original)}; the history was re-rendered"
    return f"a model turn came back changed (node {copies[-1].id}), so there was nothing to extend"


def changed(copy: Node, model: list[Node]) -> str:
    """How the client's version of a model turn differs from the sampled one (a model sibling, if there is one)."""
    if not model:
        return "has no sampled sibling to compare with"
    original = model[0].message
    missing = sorted(k for k, v in original.items() if v and not copy.message.get(k))
    differs = sorted(k for k, v in copy.message.items() if v and original.get(k) and original.get(k) != v)
    parts = [f"lacks {', '.join(missing)}" if missing else "", f"differs in {', '.join(differs)}" if differs else ""]
    return f"(vs node {model[0].id}): " + ("; ".join(p for p in parts if p) or "same fields, other spelling")


def classify_fork(graph: MessageGraph, point: int) -> str:
    children = [graph.nodes[c] for c in graph.children(point)]
    model = [c for c in children if c.author == "model"]
    client = [c for c in children if c.author != "model"]
    if model and not client:
        return f"resampled: {len(model)} samples of the same prompt (a request sent again without a retry marker)"
    reasons = []
    for copy in client:
        original = counterpart(graph, copy)
        below = [graph.nodes[d] for d in [copy.id, *descendants(graph, copy.id)]]
        unbridged = [d for d in below if any(c.bridged is False for c in d.calls)]
        if original is not None and unbridged:
            reasons.append(
                f"node {copy.id} on is a re-rendered copy of node {original.id}'s branch: the call at node "
                f"{unbridged[0].id} {classify_unbridged(graph, unbridged[0])}"
            )
        elif copy.role == "assistant":
            reasons.append(f"a model turn came back changed: node {copy.id} {changed(copy, model)}")
        else:
            reasons.append(f"history edited: the client sent a different {copy.role} message (node {copy.id})")
    return "; ".join(reasons)


def analyze(record_dir: Path, trajectory_id: str) -> dict[str, Any]:
    trajectory = record.load(record_dir, trajectory_id)
    graph = trajectory.graph
    calls = sorted(((c, n) for n in graph for c in n.calls), key=lambda pair: pair[0].t_start)
    problems = check_invariants(graph) if trajectory.capture.get("mode") == "tokens" else {}
    forks = [{"at": point, "why": classify_fork(graph, point)} for point in graph.branch_points()]
    unbridged = [{"node": n.id, "why": classify_unbridged(graph, n)} for c, n in calls if c.bridged is False]
    copies = [n.id for n in graph if n.author != "model" and n.role == "assistant"]
    tools = Counter(
        (call.get("function") or {}).get("name") for n in graph if n.author == "model" for call in n.message.get("tool_calls") or []
    )  # fmt: skip
    # Codex answers a tool call it refuses with an error message instead of command output ("Chunk ID: ...").
    tool_errors = Counter(
        content[:60]
        for n in graph
        if n.role == "tool"
        and isinstance(content := n.message.get("content"), str)
        and not content.startswith("Chunk ID")
        and trajectory.meta.get("codex")
    )
    clean = len(graph.roots()) == 1 and not forks and not unbridged and not copies
    broken = [f"{name}: {p}" for name, ps in problems.items() for p in ps]
    return {
        "id": trajectory.id,
        "task": trajectory.meta.get("task_id"),
        "repeat": trajectory.meta.get("repetition_id"),
        "reward": trajectory.annotations.get("reward"),
        "stop_reason": trajectory.annotations.get("stop_reason"),
        "verdict": trajectory.annotations.get("verdict"),
        "nodes": len(graph.nodes),
        "calls": len(calls),
        "roots": len(graph.roots()),
        "paths": len(graph.paths()),
        "forks": forks,
        "unbridged": unbridged,
        "client_assistant_nodes": copies,
        "finish_reasons": [c.finish_reason for c, _ in calls],
        "prompt_tokens_max": max((c.usage or {}).get("prompt_tokens", 0) for c, _ in calls) if calls else 0,
        "sampled_tokens": sum(sampled(n) for n in graph if n.author == "model"),
        "tools": dict(tools),
        "tool_errors": dict(tool_errors),
        "failures": [f"{f.status} {f.error[:120]}" for f in trajectory.failures],
        "invariants": {name: not ps for name, ps in problems.items()},
        "broken": broken,
        "shape": "clean" if clean else "forked" if forks else "unbridged" if unbridged else "irregular",
        "tree": tree(graph),
    }


def report(run: str, rows: list[dict[str, Any]], record_dir: Path) -> str:
    n = len(rows)
    out = [f"# skycap Codex eval: {run}", "", f"Records: `{record_dir}`. {n} trajectories.", ""]
    shapes = Counter(r["shape"] for r in rows)
    accepted = sum(1 for r in rows if r["reward"])
    out += [
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Trajectories | {n} |",
        f"| Accepted (reward 1) | {accepted} ({accepted / n:.0%}) |" if n else "| Accepted | 0 |",
        f"| Clean (one path, every call after the first bridged, no client copy of a model turn) | {shapes['clean']} |",
        f"| Forked | {shapes['forked']} |",
        f"| Unbridged without a fork | {shapes['unbridged']} |",
        f"| Calls | {sum(r['calls'] for r in rows)} (median {sorted(r['calls'] for r in rows)[n // 2] if n else 0} per rollout) |",
        f"| Truncated calls (finish_reason length) | {sum(r['finish_reasons'].count('length') for r in rows)} |",
        f"| Stop reasons | {dict(Counter(r['stop_reason'] for r in rows))} |",
        f"| Recorded failures | {sum(len(r['failures']) for r in rows)} |",
        "",
        "## Invariants",
        "",
        "Must hold for every trajectory, whatever the model did.",
        "",
        "| Invariant | Holds in |",
        "|---|---|",
    ]
    for name in INVARIANTS:
        holds = sum(1 for r in rows if r["invariants"].get(name, True))
        out.append(f"| `{name}` | {holds}/{n}{'' if holds == n else ' **FAILS**'} |")
    broken = [(r["id"], b) for r in rows for b in r["broken"]]
    if broken:
        out += ["", "Broken:", ""] + [f"- `{tid}`: {b}" for tid, b in broken[:50]]
    out += ["", "## Every trajectory", "", "| Task | Rep | Reward | Stop | Calls | Paths | Unbridged | Shape | Trajectory |",
            "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in sorted(rows, key=lambda r: (str(r["task"]), r["repeat"] or 0)):
        out.append(
            f"| {r['task']} | {r['repeat']} | {r['reward']} | {r['stop_reason']} | {r['calls']} | {r['paths']} | "
            f"{len(r['unbridged'])} | {r['shape']} | `{r['id']}` |"
        )
    odd = [r for r in rows if r["forks"] or r["unbridged"] or r["client_assistant_nodes"]]
    if odd:
        out += ["", "## Forks and unbridged calls, explained", ""]
        for r in odd:
            out.append(f"**`{r['id']}`** ({r['task']} r{r['repeat']})")
            out += [f"- fork at node {f['at']}: {f['why']}" for f in r["forks"]]
            out += [f"- unbridged call at node {u['node']}: {u['why']}" for u in r["unbridged"]]
            out.append("")
            out += ["```", *r["tree"], "```", ""]
    clean = [r for r in rows if r["shape"] == "clean"]
    if clean:
        example = max(clean, key=lambda r: r["calls"])
        out += ["", f"## A clean trajectory: `{example['id']}` ({example['task']}, {example['calls']} calls)", "",
                "One root, one path; every call after the first extends the previous one token for token.", "",
                "```", *example["tree"], "```"]  # fmt: skip
    tools = Counter()
    errors = Counter()
    for r in rows:
        tools.update(r["tools"])
        errors.update(r["tool_errors"])
    out += ["", "## What the model did", "", "| Tool | Calls |", "|---|---|"]
    out += [f"| `{name}` | {count} |" for name, count in tools.most_common()]
    if errors:
        out += [
            "",
            "Tool calls Codex answered with an error instead of running them:",
            "",
            "| Codex's answer | Count |",
            "|---|---|",
        ]
        out += [f"| {text}… | {count} |" for text, count in errors.most_common()]
    by_task: dict[str, list[float]] = {}
    for r in rows:
        by_task.setdefault(str(r["task"]), []).append(r["reward"] or 0)
    out += ["", "## Per task", "", "| Task | Accepted |", "|---|---|"]
    out += [f"| {task} | {int(sum(v))}/{len(v)} |" for task, v in sorted(by_task.items())]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", type=Path, help="the run's record directory (records/<run>)")
    parser.add_argument("--out", type=Path, help="where analysis.md and analysis.json go (default: runs/<run>/)")
    args = parser.parse_args()
    record_dir = args.records.resolve()
    run = record_dir.name
    out_dir = args.out or record_dir.parent.parent / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(record.list_ids(record_dir))
    if not ids:
        raise SystemExit(f"no records in {record_dir}")
    rows = [analyze(record_dir, tid) for tid in ids]
    (out_dir / "analysis.json").write_text(json.dumps(rows, indent=1))
    text = report(run, rows, record_dir)
    (out_dir / "analysis.md").write_text(text)
    failing = [r["id"] for r in rows if r["broken"]]
    shapes = Counter(r["shape"] for r in rows)
    print(f"{len(rows)} trajectories: {dict(shapes)}; invariants broken in {len(failing)}")
    print(f"Report: {out_dir / 'analysis.md'}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
