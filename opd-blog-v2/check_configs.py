"""Builds every job's config with SkyRL's own parser and validators, then checks what SkyRL leaves to the GPUs.

Called by check_configs.sh with one file per job: the mode on the first line (opd or sft), one override per line
after it. An eval or traces job prints a one-line plan instead, shown as it is. It needs a SkyRL checkout with its
.venv (the Mac's is enough: no GPU, no data, nothing downloaded).
"""

import os
import sys

NUM_GPUS = 8  # every node


def load_kit_main(kit_dir):
    import importlib.util

    spec = importlib.util.spec_from_file_location("kit_main", os.path.join(kit_dir, "common", "kit_main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sft(overrides):
    from skyrl.train.config.sft_config import SFTConfig, build_skyrl_config_for_sft

    cfg = SFTConfig.from_cli_overrides(overrides)
    build_skyrl_config_for_sft(cfg)  # validates
    problems = []
    gpus = cfg.placement.num_gpus_per_node
    if cfg.batch_size % (gpus * cfg.micro_train_batch_size_per_gpu):
        problems.append(f"batch {cfg.batch_size} does not split into {gpus} GPUs x {cfg.micro_train_batch_size_per_gpu}")
    o = cfg.optimizer_config
    summary = (
        f"sft  {cfg.run_name}\n"
        f"     {cfg.model.path} on {gpus} GPUs ({cfg.strategy}); data {cfg.pretokenized_dataset_paths}\n"
        f"     batch {cfg.batch_size} (micro {cfg.micro_train_batch_size_per_gpu}/GPU, packed={cfg.remove_microbatch_padding}), "
        f"{cfg.num_epochs} epoch(s), max {cfg.max_training_steps or '-'} steps, max_length {cfg.max_length}; "
        f"lr {o.lr:g} {o.scheduler}, warmup {o.num_warmup_steps}, betas {list(o.adam_betas)}, wd {o.weight_decay:g}\n"
        f"     W&B {cfg.project_name}; checkpoint every {cfg.ckpt_interval}, export every {cfg.hf_save_interval} to {cfg.export_path}"
    )
    return summary, problems


def check(kit_main, path):
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    mode, overrides = lines[0], lines[1:]
    if mode.split()[0] in ("eval", "traces"):
        return mode, []
    if mode == "sft":
        return check_sft(overrides)
    cfg = kit_main.build_config(mode, overrides)
    t, g = cfg.trainer, cfg.generator
    problems = []

    student = t.placement.policy_num_gpus_per_node
    if g.inference_engine.num_engines * g.inference_engine.tensor_parallel_size != student:
        problems.append("the student's engines do not fill its GPUs (colocated)")
    teacher_gpus = 0
    teachers = {}
    if mode == "opd":
        for name, teacher in t.teachers.items():
            ie = teacher.inference_engine
            gpus = ie.num_engines * ie.tensor_parallel_size * ie.pipeline_parallel_size * ie.data_parallel_size
            teacher_gpus += gpus
            teachers[name] = (teacher.model, gpus)
            need = t.max_prompt_length + g.sampling_params.max_generate_length + 1
            if ie.engine_init_kwargs.get("max_model_len", 0) < need:
                problems.append(f"teacher {name}'s max_model_len is below prompt + response + 1 = {need}")
    if student + teacher_gpus > NUM_GPUS:
        problems.append(f"needs {student} + {teacher_gpus} GPUs, the node has {NUM_GPUS}")
    rollouts = t.train_batch_size * g.n_samples_per_prompt
    if rollouts % student or (t.policy_mini_batch_size * g.n_samples_per_prompt) % student:
        problems.append(f"{rollouts} rollouts do not split over {student} student GPUs")
    longest = t.max_prompt_length + max(g.sampling_params.max_generate_length, g.eval_sampling_params.max_generate_length)
    student_len = g.inference_engine.engine_init_kwargs.get("max_model_len")
    if student_len is not None and student_len < longest + 1:
        problems.append(f"the student's max_model_len {student_len} is below its longest sequence {longest} + 1")

    updates = t.train_batch_size // t.policy_mini_batch_size * t.update_epochs_per_batch
    summary = (
        f"{mode:4} {t.run_name}\n"
        f"     student {t.policy.model.path}  on {student} GPUs; "
        + (", ".join(f"teacher {n}={m} on {k}" for n, (m, k) in teachers.items()) or "no teacher")
        + f"\n     {t.train_batch_size} prompts x {g.n_samples_per_prompt}, {updates} optimizer step(s) per step, "
        f"{t.max_training_steps} steps, lr {t.policy.optimizer_config.lr:g}, warmup "
        f"{t.policy.optimizer_config.num_warmup_steps}, betas {list(t.policy.optimizer_config.adam_betas)}, wd "
        f"{t.policy.optimizer_config.weight_decay:g}, loss {t.algorithm.policy_loss_type}/{t.algorithm.loss_reduction}"
        f"\n     answers {g.sampling_params.max_generate_length} tokens at T={g.sampling_params.temperature}; eval "
        f"{g.eval_n_samples_per_prompt} x {g.eval_sampling_params.max_generate_length} tokens at "
        f"T={g.eval_sampling_params.temperature}, every {t.eval_interval}; template kwargs {dict(g.chat_template_kwargs)}"
        f"\n     W&B {t.project_name}; exports every {t.hf_save_interval} to {t.export_path}"
    )
    if mode == "opd":
        o = t.algorithm.opd
        clips = {k: getattr(o, k) for k in ("divergence_clip", "advantage_clip_min", "advantage_clip_max") if getattr(o, k) is not None}
        if o.advantage_log_compress:
            clips["advantage_log_compress"] = True
        if clips:
            summary += f"\n     teacher signal: {clips}"
    return summary, problems


def main():
    kit_dir, paths = sys.argv[1], sys.argv[2:]
    kit_main = load_kit_main(kit_dir)
    failed = 0
    for path in paths:
        try:
            summary, problems = check(kit_main, path)
        except (Exception, SystemExit) as e:  # a config SkyRL refuses
            failed += 1
            print(f"FAIL {os.path.basename(path)}: {type(e).__name__}: {e}\n")
            continue
        print(summary)
        for p in problems:
            print(f"     PROBLEM: {p}")
        failed += bool(problems)
        print()
    print(f"{len(paths) - failed} of {len(paths)} configs pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
