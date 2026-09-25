"""Prove which `skyrl` the driver AND a Ray worker import: this checkout (the PR branch) or a copy
baked into the image. Run from the SkyRL repo root, after 00_env.sh:

    uv run --isolated --extra fsdp skyrl-test/anyscale-pr2198/check_code_path.py

Both lines must show this checkout's path, the branch's commit, and dispatcher=True (the module
skyrl.train.eval.dispatcher exists only on PR 2198). A worker line pointing into site-packages, or
dispatcher=False, means Ray workers are not running in the driver's uv environment: check that
RAY_RUNTIME_ENV_HOOK is exported (00_env.sh does it) and that you ran from the repo root.
"""

import importlib.util
import os
import subprocess
import sys


def describe():
    import skyrl  # noqa: F401  -- resolved inside whichever process runs this

    repo = os.path.dirname(os.path.dirname(skyrl.__file__))
    try:
        commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # no .git next to an installed copy
        commit = "no-git"
    return {
        "skyrl": skyrl.__file__,
        "commit": commit,
        "dispatcher": importlib.util.find_spec("skyrl.train.eval.dispatcher") is not None,
        "python": sys.executable,
    }


def fmt(d):
    return f"{d['skyrl']}  commit={d['commit']}  dispatcher={d['dispatcher']}  python={d['python']}"


print("driver:", fmt(describe()))

import ray  # noqa: E402

ray.init(logging_level="ERROR")
print("worker:", fmt(ray.get(ray.remote(describe).remote())))
