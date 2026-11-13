"""The opd-blog-v2 entrypoint: registers the kit's two environments, then runs SkyRL's GRPO or OPD trainer.

    cd ~/SkyRL
    uv run --isolated --extra fsdp --with math-verify==0.9.0 python skyrl-test/opd-blog-v2/common/kit_main.py grpo <overrides>
    uv run --isolated --extra fsdp --with math-verify==0.9.0 python skyrl-test/opd-blog-v2/common/kit_main.py opd <overrides>

``grpo`` is ``skyrl.train.entrypoints.main_base`` and ``opd`` is ``skyrl.train.entrypoints.main_opd``, each with the
environments below registered first. The run scripts call this; nothing else in SkyRL changes.

- ``opd_v2_math``: a math answer graded with math-verify, reward 1 or 0. SkyRL's ``aime`` env compares normalized
  strings, which suits DAPO-17k's integer answers but not Skywork-OR1's: in a 100-row sample of its math split, 38
  answers were fractions, percentages or expressions, such as ``\\frac{13}{4}``, which a correct ``3.25`` would miss.
  The reward trains the GRPO teacher; under OPD it is only logged, as a train-time accuracy.
- ``opd_v2_noreward``: reward 0, for MOPD's code and IF rows. OPD needs no reward, so they need no sandbox or verifier.

The classes live in this file on purpose. Ray pickles the entrypoint, and what it refers to in ``__main__``, by value,
so the Ray workers need no import path to this kit; ``skyrl-test`` is not an importable name. math-verify reaches the
workers through ``uv run --with``, which Ray's uv hook repeats for them.
"""

import logging
import re
import sys
from typing import Any, Dict, List, Optional

import ray
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

MATH_ENV = "opd_v2_math"
NO_REWARD_ENV = "opd_v2_noreward"
_ANSWER_LINE = re.compile(r"(?i)answer\s*:\s*([^\n]+)")


def _last_boxed(text: str) -> Optional[str]:
    """The content of the last ``\\boxed{...}`` (or ``\\fbox{...}``), braces balanced; ``None`` if there is none."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def final_answer(response: str) -> Optional[str]:
    """The answer a response gives, or ``None``.

    A thinking response is graded on what follows its last ``</think>``; one that never closes its thinking was cut
    off and gives no answer. The math prompts ask for both forms (M2RL's ``Answer: $Answer`` line and the sources'
    ``\\boxed{}``), so the last ``\\boxed{}`` wins and the last ``Answer:`` line is the fallback.
    """
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    elif "<think>" in response:
        return None
    boxed = _last_boxed(response)
    if boxed is not None:
        return boxed.strip()
    lines = _ANSWER_LINE.findall(response)
    if not lines:
        return None
    answer = lines[-1].strip().strip("$").strip()
    inner = _last_boxed(answer)
    return (inner if inner is not None else answer).strip() or None


def _normalize(text: str) -> str:
    return re.sub(r"\\text\{([^}]*)\}|\\left|\\right|\\!|\s|\$", r"\1", text).rstrip(".")


def _drop_percent(text: str) -> str:
    """``20\\%`` as ``20``. SkyRL pins ANTLR 4.9.3 (through omegaconf), and math-verify's parser for that version
    leaves a percentage unparsed, so ``20\\%`` would match nothing but itself."""
    return text.replace("\\%", "").replace("%", "").strip()


def is_correct(response: str, ground_truth: str) -> bool:
    """Whether the response's final answer equals the ground truth, symbolically (math-verify) or as a string."""
    pred = final_answer(response)
    if pred is None:
        return False
    pred, ground_truth = _drop_percent(pred), _drop_percent(ground_truth)
    if _normalize(pred) == _normalize(ground_truth):
        return True
    return _symbolic_equal(ground_truth, pred)


# math-verify runs in child processes with a hard deadline. It never returns on some gold answers of M2RL's math set:
# 6 of Qwen3-32B's 44,112 answers to it hung the grader on 2026-11-11, on 4 prompts whose golds are 3^{2^{2018}}-1
# (sympy builds the integer), (10001111101)_{2}, a sum to 2016 with 10^{2017} in it and a binomial sum. A run grading
# in-process would stall for good the first time one of them met an answer that is not the same string. math-verify's
# own timeouts use signal.alarm, which works only on a process's main thread (the generator grades from executor
# threads), and no signal interrupts a long C call such as a huge integer power. So the parse and the compare run in a
# child, on its own main thread with math-verify's timeouts on, and a child that does not answer within
# VERIFY_DEADLINE_S is killed and the answer counted wrong. VERIFY_CHILDREN children serve the grading threads.
VERIFY_DEADLINE_S = 10
VERIFY_CHILDREN = 8
_VERIFY_CHILD = r"""
import json, sys
from math_verify import parse, verify
for line in sys.stdin:
    gold_text, pred_text = json.loads(line)
    try:
        gold = parse("\\boxed{" + gold_text + "}", parsing_timeout=3)
        target = parse("\\boxed{" + pred_text + "}", parsing_timeout=3)
        ok = bool(gold and target and verify(gold, target, timeout_seconds=3))
    except BaseException:
        ok = False
    sys.stdout.write("1\n" if ok else "0\n")
    sys.stdout.flush()
"""


class _ProcessLocal(dict):
    """State of the process that grades, filled on first use. Ray pickles this module's functions by value, with what
    they refer to; this pickles as empty, so a copy starts its own children instead of carrying a queue and processes."""

    def __reduce__(self):
        return (_ProcessLocal, ())


_VERIFIERS = _ProcessLocal()


def _symbolic_equal(gold_text: str, pred_text: str) -> bool:
    """math-verify's verdict on two answer strings, from a child process; False if it fails or takes too long."""
    import json
    import queue
    import select
    import subprocess

    slots = _VERIFIERS.get("slots")
    if slots is None:
        fresh: "queue.Queue" = queue.Queue()
        for _ in range(VERIFY_CHILDREN):
            fresh.put(None)  # a child is started on the slot's first use
        slots = _VERIFIERS.setdefault("slots", fresh)
    child = slots.get()
    answer = ""
    try:
        if child is None or child.poll() is not None:
            child = subprocess.Popen(
                [sys.executable, "-c", _VERIFY_CHILD],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        child.stdin.write(json.dumps([gold_text, pred_text]) + "\n")
        child.stdin.flush()
        if select.select([child.stdout], [], [], VERIFY_DEADLINE_S)[0]:
            answer = child.stdout.readline().strip()
    except (OSError, ValueError):
        answer = ""
    finally:
        if answer not in ("0", "1") and child is not None:
            child.kill()
            child.wait()
            child = None
        slots.put(child)
    return answer == "1"


class MathVerifyEnv(BaseTextEnv):
    """One math answer, graded by ``is_correct``: reward 1 or 0, done after one step."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = str(extras["reward_model"]["ground_truth"])

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward = 1.0 if is_correct(action, self.ground_truth) else 0.0
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"acc": reward})


class NoRewardEnv(BaseTextEnv):
    """Reward 0, done after one step: a row OPD scores with its teacher alone."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})


def register_envs() -> None:
    """Register both environments in this process, once."""
    # The registry is looked up through its module, here and now. A module-level `registry` name would be pickled
    # by value with the Ray entrypoint, so a worker would check the driver's copy, which already has both envs, and
    # never register them in its own registry ("No registered env with id: opd_v2_math").
    from skyrl_gym.envs import registration

    # math-verify warns on every call that its timeouts are off; that is deliberate here (see is_correct).
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    for name, cls in ((MATH_ENV, MathVerifyEnv), (NO_REWARD_ENV, NoRewardEnv)):
        if name not in registration.registry:
            registration.register(id=name, entry_point=cls)


@ray.remote(num_cpus=1)
def kit_entrypoint(mode: str, cfg) -> None:
    register_envs()
    if mode == "grpo":
        from skyrl.train.entrypoints.main_base import BasePPOExp

        BasePPOExp(cfg).run()
    else:
        from skyrl.train.entrypoints.main_opd import OPDExp

        OPDExp(cfg).run()


def build_config(mode: str, overrides: List[str]):
    """The validated config for ``mode``, exactly as the run would build it. check_configs.py uses this too."""
    from skyrl.train.utils import validate_cfg

    register_envs()
    if mode == "grpo":
        from skyrl.train.config import SkyRLTrainConfig

        cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
        validate_cfg(cfg)
    elif mode == "opd":
        from skyrl.train.opd.config import OPDExpConfig, validate_opd_cfg

        cfg = OPDExpConfig.from_cli_overrides(overrides)
        validate_cfg(cfg)
        validate_opd_cfg(cfg)
    else:
        raise SystemExit(f"mode must be grpo or opd, got {mode!r}")
    return cfg


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: kit_main.py grpo|opd <skyrl overrides...>")
    mode, overrides = sys.argv[1], sys.argv[2:]
    cfg = build_config(mode, overrides)

    from skyrl.train.utils import initialize_ray

    initialize_ray(cfg)
    ray.get(kit_entrypoint.remote(mode, cfg))


if __name__ == "__main__":
    main()
