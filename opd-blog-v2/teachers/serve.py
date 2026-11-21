"""Serves OPD teachers from a node of their own, for runs on other nodes to score against.

    cd ~/SkyRL
    uv run --isolated --extra fsdp python skyrl-test/opd-blog-v2/teachers/serve.py \
        --port qwen3_32b=31000 --status-file ~/opd-store/teachers/<job>.status -- <SkyRL overrides>

teachers/serve.sh builds the command from config.sh's V2_TEACHER_SERVERS. The overrides name each teacher as
``trainer.teachers.<key>`` with ``backend=skyrl``, as a run that launches its own teacher would.

Each teacher is launched by SkyRL's own ``launch_teacher`` (``skyrl/train/opd/teacher_launch.py``): vLLM server groups
with ``build_frozen_vllm_cli_args`` and a vllm-router in front, the deployment a run gets for a teacher on its own
GPUs. Ray's runtime environment is SkyRL's (``initialize_ray``), as in a run. One thing differs: SkyRL gives the router
a free port at random, and here it gets the fixed port of ``--port``, so runs on other nodes find it at
``http://<this node's internal IP>:<port>`` (``trainer.teachers.<name>.backend=vllm``, ``server_url``). The router and
the servers listen on every interface, as SkyRL starts them.

Before reporting a teacher up, it scores one token through the router with SkyRL's own preflight, the check a run
makes. Then, every minute, it checks the router and every server's ``/health``, and writes one status line to
``--status-file`` (``common/job_report.sh`` shows it). A teacher that fails its checks for 5 minutes in a row ends the
job with an error, so the job shows as failed rather than serving a broken teacher.

The functions live in this file, not an importable module: Ray pickles what ``__main__`` defines by value
(``common/kit_main.py`` says more).
"""

import argparse
import asyncio
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import ray

CHECK_EVERY_S = 60
FAILS_ALLOWED = 5


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def write_status(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text + "\n")
    os.replace(tmp, path)


@ray.remote(num_cpus=1)
def serve(cfg, ports, status_file):
    import skyrl.backends.skyrl_train.inference_servers.setup as setup_module
    from skyrl.train.opd.config import teacher_logprob_width, teacher_topk
    from skyrl.train.opd.teacher_client import SkyRLTeacherClient
    from skyrl.train.opd.teacher_launch import launch_teacher, teacher_start_ports

    # The router's port. create_inference_servers builds its arguments with this module's build_router_args, which
    # asks for a free port; the wrapper puts the teacher's fixed port in its place. VLLMRouter then reserves it, and
    # would take the next free port if it were busy: the check below refuses that.
    build_router_args = setup_module.build_router_args
    port_for = {}

    def router_args_on_fixed_port(*args, **kwargs):
        router_args = build_router_args(*args, **kwargs)
        router_args.port = port_for["now"]
        return router_args

    setup_module.build_router_args = router_args_on_fixed_port

    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    write_status(status_file, "starting")
    starts = teacher_start_ports(cfg)
    held = {}
    for name, teacher in cfg.trainer.teachers.items():
        ie = teacher.inference_engine
        port_for["now"] = ports[name]
        write_status(status_file, f"loading {name}")
        client, setup = launch_teacher(cfg, teacher, starts[name])
        if not setup.proxy_url.rstrip("/").endswith(f":{ports[name]}"):
            raise RuntimeError(f"{name}'s router came up at {setup.proxy_url}, not on port {ports[name]}: the port is busy")
        scorer = SkyRLTeacherClient(client, server_setup=setup, max_concurrency=1)

        async def preflight():
            try:
                await scorer.preflight(teacher_logprob_width(cfg), teacher_topk(cfg))
            finally:
                await scorer.aclose()

        asyncio.run(preflight())
        held[name] = (setup, scorer)
        print(
            f"teacher {name}: {teacher.model}, {ie.num_engines} engine(s) x {ie.tensor_parallel_size} GPU(s), "
            f"max_model_len {ie.engine_init_kwargs.get('max_model_len')}, at {setup.proxy_url} "
            f"(servers {list(setup.server_urls)})",
            flush=True,
        )

    fails = {name: 0 for name in held}
    while True:
        parts = []
        for name, (setup, _) in held.items():
            urls = [setup.proxy_url] + list(setup.server_urls)
            down = [url for url in urls if not healthy(url)]
            fails[name] = fails[name] + 1 if down else 0
            if fails[name] >= FAILS_ALLOWED:
                raise RuntimeError(f"teacher {name}: {down} failed /health {FAILS_ALLOWED} minutes in a row")
            parts.append(f"{name}:{ports[name]} {'ok' if not down else f'{len(down)} down'}")
        stamp = datetime.now(timezone.utc).strftime("%H:%MZ")
        write_status(status_file, f"serving {', '.join(parts)} ({stamp})")
        time.sleep(CHECK_EVERY_S)


def build_config(overrides):
    """The teachers' config, built and checked as a run builds and checks its teachers. check_configs.py uses this."""
    from skyrl.train.opd.config import OPDExpConfig, _validate_teachers

    cfg = OPDExpConfig.from_cli_overrides(overrides)
    _validate_teachers(cfg)
    for name, teacher in cfg.trainer.teachers.items():
        if teacher.backend != "skyrl":
            raise SystemExit(f"trainer.teachers.{name}: a served teacher is launched here, so its backend is skyrl")
    return cfg


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", action="append", required=True, help="<teacher name>=<router port>, once per teacher")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("overrides", nargs="*", help="SkyRL overrides, after --")
    args = parser.parse_args(argv)
    ports = {}
    for item in args.port:
        name, _, port = item.partition("=")
        ports[name] = int(port)
    return args, ports


def main() -> None:
    args, ports = parse_args(sys.argv[1:])
    cfg = build_config(args.overrides)
    if set(ports) != set(cfg.trainer.teachers):
        raise SystemExit(f"--port names {sorted(ports)} but the overrides name {sorted(cfg.trainer.teachers)}")

    from skyrl.train.utils import initialize_ray

    initialize_ray(cfg)
    ray.get(serve.remote(cfg, ports, args.status_file))


if __name__ == "__main__":
    main()
