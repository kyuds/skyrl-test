# Shared by every script in opd-blog-v2, on your Mac and on the VMs. Sourced, so nothing here may call gcloud, and it
# has to work on the Mac's bash 3.2. The VMs read this file from GitHub: to change a setting, edit it here and push.

# ---- code ----
# The VMs check out this exact SkyRL commit, including after a preemption, so every run and every resume use the same
# code. It is the head of kyuds/opd-mopd (PR #2679, "[OPD][5/6] Implement MOPD") on 2026-11-09, which carries the
# whole OPD stack up to MOPD. The stack is not merged, so it is pinned by commit, not by branch. To move it, change it
# here, then run common/check_configs.sh on your Mac against the new commit before pushing.
SKYRL_REPO=https://github.com/NovaSky-AI/SkyRL.git
SKYRL_COMMIT=988d977f4c8b6fe67316e56307b7f555672255cf
KIT_REPO=https://github.com/kyuds/skyrl-test.git
KIT_BRANCH=opd-blog-v2

# ---- where results go ----
V2_PROJECT=opd-blog-v2            # W&B project; smoke tests go to opd-blog-v2-smoke
V2_HF_REPO=kyuds/opd-blog-v2      # one Hub repo, one folder per model: <run>_step<N>

# ---- models ----
M2RL_SFT=Jackwang111/M2RL-SFT
M2RL_MATH=Jackwang111/M2RL-RL_Math
M2RL_CODE=Jackwang111/M2RL-RL_Coding
M2RL_IF=Jackwang111/M2RL-RL_IF
QWEN3_8B=Qwen/Qwen3-8B
QWEN3_32B=Qwen/Qwen3-32B

# ---- the runs: <node> <run name> <script, relative to opd-blog-v2/> ----
# status.sh reads this. Phase 1.2 waits for the GRPO run on the same node, because its teacher is that run's export.
V2_RUNS="${V2_RUNS:-
b200-1 grpo_qwen3_8b_nothink                   grpo-qwen3-8b/run.sh
b200-1 p1_opd_qwen3_8b_from_grpo               phase1-opd/run_qwen3_8b.sh
h100-1 p1_opd_m2rl_sft_from_rl_math            phase1-opd/run_m2rl.sh
b200-2 p2_opd_m2rl_sft_from_qwen3_32b           phase2-opd/run.sh
h100-2 p3_mopd_m2rl_sft_from_math_code_if      phase3-mopd/run.sh
}"
