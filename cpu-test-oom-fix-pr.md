## What

`example_dummy_config()` now builds its `DataConfig` with `data.dataloader.num_workers=0`, so no CPU test spawns dataloader worker processes over dummy data.

## Why

The `SkyRL-Train-CPU` job intermittently fails in `tests/backends/skyrl_train/utils/test_ppo_utils.py` with `ray.exceptions.OutOfMemoryError`: the 16 GB runner reaches Ray's 95% memory threshold and the memory monitor kills the two registry actors (0.01 GB each, the only Ray workers around), so every registry test that then reaches them fails. Example: https://github.com/NovaSky-AI/SkyRL/actions/runs/35277987080/job/105396628699

The memory itself is held by ~30 half-gigabyte `python -c "from multiprocessing.spawn ..."` interpreters, which Ray cannot kill because they are not Ray workers. They come from the test dataloaders:

- `build_dataloader` uses the `spawn` start method with `num_workers=8`, the value `SkyRLTrainConfig.__post_init__` derives when nothing sets it, and the dummy test config set nothing. Any CPU test that iterates a dataloader over a handful of in-memory rows therefore spawns eight fresh interpreters, each importing torch and skyrl.
- `StatefulDataLoader` caches its worker iterator on the loader object, so the workers live as long as the trainer, not just the current iteration. A test that leaves the training loop early (an exception mid-step, an early stop) pins them until the cyclic GC happens to run.

A process census over both CPU test trees with the cyclic GC disabled found a single leak on the async-eval branch (a step-crash test that exits the loop by exception, 8 workers / 4 GB); on `main` today the loop-driving tests (`test_rl_callbacks.py`) simply pay the eight-interpreter spawn on every `train()` call. Whether a run tips over the threshold depends on GC timing, which is why it is flaky.

## Why zero rather than fewer workers

`num_workers=0` is a documented mode ("in-process loading that never respawns workers at epoch boundaries"): `build_dataloader` sets `multiprocessing_context=None` for it and `StatefulDataLoader` uses `_StatefulSingleProcessDataLoaderIter`, with `state_dict` / `load_state_dict` working the same way. Verified through `build_dataloader`: iterate, checkpoint mid-epoch, load into a fresh loader, and it replays exactly the remaining batches, with no child process. The only test that checks worker settings (`test_build_dataloader_worker_config`) builds its own config from CLI overrides and is unaffected.

## Verification

- `tests/train/` on `main` with this change: 960 passed.
- A per-test process census over the loop-driving tests shows no spawned worker processes at all; before, `train()`-based tests spawned eight per iterated dataloader.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
