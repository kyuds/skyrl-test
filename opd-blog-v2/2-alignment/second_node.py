"""Shares one answer-generation job (traces.sh) with a second node, then merges the second node's answers back.

generate.py gives engine w of E the requests [w::E] and samples them in order, in --chunks parts of ceil(len / chunks),
appending each finished part to <out>.shard<w>.part. So the first node works through every engine's share from the
front, and a second node can take the back of every share:

  split  (anywhere) writes the requests of every engine's share from part --from-part on, for the second node to sample
         with generate.py (any number of engines, its own --out).
  merge  (on the first node, with its job stopped) appends the second node's answers to the first node's .part files,
         skipping requests already saved there. Started again, the job finds every request saved, so its engines sample
         nothing and it goes straight on to grading and tokenizing.

A request counts only with all its samples; one with fewer, in either file, is left for the job to sample again.

    python3 second_node.py split <eval dir>/requests.jsonl --engines 4 --chunks 8 --from-part 4 --out <file>
    python3 second_node.py merge <eval dir> <second node's responses.jsonl> --engines 4
"""

import argparse
import json
import os
import subprocess
import sys


def read_requests(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def read_answers(path):
    """id -> list of answer lines (as JSON strings), for complete lines only."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut by a crash
            out.setdefault(x["id"], []).append(json.dumps(x) + "\n")
    return out


def split(args) -> None:
    requests = read_requests(args.requests)
    with open(args.requests) as f:
        lines = f.readlines()
    picked = []
    for w in range(args.engines):
        share = list(range(len(requests)))[w :: args.engines]
        size = -(-len(share) // args.chunks)
        picked += share[args.from_part * size :]
    with open(args.out, "w") as f:
        for i in sorted(picked):
            f.write(lines[i])
    print(f"{len(picked)} of {len(requests)} requests (parts {args.from_part} to {args.chunks - 1} of each of "
          f"{args.engines} engines) -> {args.out}")


def merge(args) -> None:
    d = args.eval_dir.rstrip("/")
    running = subprocess.run(["pgrep", "-f", f"generate.py.*{d}/"], capture_output=True, text=True).stdout.split()
    if running:
        sys.exit(f"generate.py is still running on {d} (pids {' '.join(running)}): stop the job first")
    requests = read_requests(os.path.join(d, "requests.jsonl"))
    want = {r["id"]: int(r["n"]) for r in requests}
    owner = {r["id"]: i % args.engines for i, r in enumerate(requests)}
    theirs = {i: ls for i, ls in read_answers(args.answers).items() if len(ls) == want.get(i)}
    total_missing = 0
    for w in range(args.engines):
        part = os.path.join(d, f"responses.jsonl.shard{w}.part")
        if os.path.exists(os.path.join(d, f"responses.jsonl.shard{w}")):
            sys.exit(f"engine {w}'s shard is already finished; nothing to merge into")
        mine = {i: ls for i, ls in read_answers(part).items() if len(ls) == want.get(i)}
        added = {i: ls for i, ls in theirs.items() if owner[i] == w and i not in mine}
        with open(part + ".tmp", "w") as f:
            for ls in list(mine.values()) + list(added.values()):
                f.writelines(ls)
        os.replace(part + ".tmp", part)
        share = sum(1 for i in owner.values() if i == w)
        missing = share - len(mine) - len(added)
        total_missing += missing
        print(f"engine {w}: {len(mine)} saved here, {len(added)} added from the second node, {missing} of {share} "
              f"still to sample")
    print(f"done: {total_missing} requests left for the job to sample")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("requests")
    s.add_argument("--engines", type=int, required=True)
    s.add_argument("--chunks", type=int, required=True)
    s.add_argument("--from-part", type=int, required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("eval_dir")
    m.add_argument("answers")
    m.add_argument("--engines", type=int, required=True)
    args = parser.parse_args()
    split(args) if args.cmd == "split" else merge(args)


if __name__ == "__main__":
    main()
