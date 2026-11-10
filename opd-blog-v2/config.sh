# Shared by every script in opd-blog-v2, on your Mac and on the VMs. Sourced, so nothing here may call gcloud, and it
# has to work on the Mac's bash 3.2. The VMs read this file from GitHub: to change a setting, edit it here and push.

# ---- code ----
# The VMs check out this exact SkyRL commit, including after a preemption, so every run and every resume use the same
# code. It is the head of kyuds/opd-mopd (PR #2679, "[OPD][5/6] Implement MOPD") on 2026-11-09, which carries the
# whole OPD stack up to MOPD. The stack is not merged, so it is pinned by commit, not by branch. To move it, change it
# here, then run check_configs.sh on your Mac against the new commit before pushing.
SKYRL_REPO=https://github.com/NovaSky-AI/SkyRL.git
SKYRL_COMMIT=988d977f4c8b6fe67316e56307b7f555672255cf
KIT_REPO=https://github.com/kyuds/skyrl-test.git
KIT_BRANCH=opd-blog-v2

# ---- where results go ----
V2_PROJECT=opd-blog-v2            # W&B project of the training runs; evals log to opd-blog-v2-evals, smoke tests
                                  # to opd-blog-v2-smoke
V2_HF_REPO=kyuds/opd-blog-v2      # one Hub repo, one folder per model: SFT outputs as <job>, OPD exports as <job>_step<N>

# ---- models ----
QWEN3_8B=Qwen/Qwen3-8B             # section 1's teacher, thinking mode
QWEN3_8B_BASE=Qwen/Qwen3-8B-Base   # section 1's student
QWEN3_32B=Qwen/Qwen3-32B           # section 2's misaligned teacher, thinking mode
M2RL_SFT=Jackwang111/M2RL-SFT      # the student of sections 2 and 3
M2RL_MATH=Jackwang111/M2RL-RL_Math # section 2's aligned teacher (RL'd from M2RL-SFT), and section 3's math teacher
M2RL_CODE=Jackwang111/M2RL-RL_Coding
M2RL_IF=Jackwang111/M2RL-RL_IF

# ---- the jobs ----
# One line per job: <stage> <node> <job> <deps> <command, relative to opd-blog-v2/, with its arguments>.
#   stage    orchestrate.sh run <stage> starts the stage's jobs. Stages are the README's plan, not barriers: a job
#            starts once its own deps are done.
#   node     the VM it runs on. A node runs its jobs of a stage one after another, in this order.
#   job      unique. It names the W&B run, the log (~/opd-store/logs/<job>.log) and the Hub folder of its output.
#   deps     jobs that must be done first, comma-separated, or -. Checked on every node, so a dep can run elsewhere:
#            SFT outputs go through the Hub, which is how an eval on an H100 finds a model trained on a B200.
# common/job.sh runs a job by name and records it in ~/opd-store/jobs/; a job that is done is never run again.
# Every OPD run waits for its teacher's eval and its student's (README.md, "The plan").
V2_JOBS="${V2_JOBS:-
1 b200-1 sft_1a_qwen3_8b_base_ot3             -                                       1-tml/sft.sh
1 b200-2 eval_qwen3_32b_math                  -                                       0-evals/run.sh eval_qwen3_32b_math Qwen/Qwen3-32B qwen math
1 b200-2 traces_2c_qwen3_32b                  -                                       2-alignment/traces.sh
1 h100-1 eval_m2rl_sft_math                   -                                       0-evals/run.sh eval_m2rl_sft_math Jackwang111/M2RL-SFT m2rl math
1 h100-2 eval_m2rl_rl_math_math               -                                       0-evals/run.sh eval_m2rl_rl_math_math Jackwang111/M2RL-RL_Math m2rl math
1 h100-3 eval_qwen3_8b_math                   -                                       0-evals/run.sh eval_qwen3_8b_math Qwen/Qwen3-8B qwen math
1 h100-4 eval_m2rl_sft_code_if                -                                       0-evals/run.sh eval_m2rl_sft_code_if Jackwang111/M2RL-SFT m2rl code,if
1 h100-4 eval_m2rl_rl_coding_code             -                                       0-evals/run.sh eval_m2rl_rl_coding_code Jackwang111/M2RL-RL_Coding m2rl code
1 h100-4 eval_m2rl_rl_if_if                   -                                       0-evals/run.sh eval_m2rl_rl_if_if Jackwang111/M2RL-RL_IF m2rl if
2 b200-2 sft_2c_m2rl_sft_warmup               traces_2c_qwen3_32b                     2-alignment/sft_2c.sh
2 b200-2 opd_2c_m2rl_warmup_from_qwen3_32b    sft_2c_m2rl_sft_warmup,eval_qwen3_32b_math,eval_m2rl_sft_math 2-alignment/opd.sh 2c
2 b200-2 eval_opd_2c_final                    opd_2c_m2rl_warmup_from_qwen3_32b       0-evals/run.sh eval_opd_2c_final run:opd_2c_m2rl_warmup_from_qwen3_32b m2rl math
2 h100-1 opd_2b_m2rl_sft_from_qwen3_32b       eval_qwen3_32b_math,eval_m2rl_sft_math  2-alignment/opd.sh 2b
2 h100-1 eval_opd_2b_final                    opd_2b_m2rl_sft_from_qwen3_32b          0-evals/run.sh eval_opd_2b_final run:opd_2b_m2rl_sft_from_qwen3_32b m2rl math
2 h100-2 opd_2a_m2rl_sft_from_m2rl_rl_math    eval_m2rl_rl_math_math,eval_m2rl_sft_math 2-alignment/opd.sh 2a
2 h100-2 eval_opd_2a_final                    opd_2a_m2rl_sft_from_m2rl_rl_math       0-evals/run.sh eval_opd_2a_final run:opd_2a_m2rl_sft_from_m2rl_rl_math m2rl math
2 h100-3 opd_2d_m2rl_sft_from_qwen3_32b_clip  eval_qwen3_32b_math,eval_m2rl_sft_math  2-alignment/opd.sh 2d
2 h100-3 eval_opd_2d_final                    opd_2d_m2rl_sft_from_qwen3_32b_clip     0-evals/run.sh eval_opd_2d_final run:opd_2d_m2rl_sft_from_qwen3_32b_clip m2rl math
2 h100-4 mopd_3_m2rl_sft_from_math_code_if    eval_m2rl_sft_math,eval_m2rl_rl_math_math,eval_m2rl_sft_code_if,eval_m2rl_rl_coding_code,eval_m2rl_rl_if_if 3-mopd/run.sh
2 h100-4 eval_mopd_3_final                    mopd_3_m2rl_sft_from_math_code_if       0-evals/run.sh eval_mopd_3_final run:mopd_3_m2rl_sft_from_math_code_if m2rl all
3 b200-1 eval_sft_1a                          sft_1a_qwen3_8b_base_ot3                0-evals/run.sh eval_sft_1a kit:sft_1a_qwen3_8b_base_ot3 qwen math
3 b200-1 opd_1b_ot3sft8b_from_qwen3_8b        sft_1a_qwen3_8b_base_ot3,eval_qwen3_8b_math 1-tml/opd.sh
3 b200-1 eval_opd_1b_final                    opd_1b_ot3sft8b_from_qwen3_8b           0-evals/run.sh eval_opd_1b_final run:opd_1b_ot3sft8b_from_qwen3_8b qwen math
}"

# Fewer nodes: move a node's jobs onto another one, e.g. with 3 H100s (README.md, "Fewer nodes"):
#   export V2_NODE_MAP="h100-4=h100-2"
# Each moved job then runs after the jobs already listed for that node in the same stage.
V2_NODE_MAP="${V2_NODE_MAP:-}"

# The job table with V2_NODE_MAP applied and comments and blank lines dropped: stage node job deps command...
# Sorted by stage; within a stage a moved job goes after the jobs the table already lists for its new node.
v2_jobs() {
  echo "$V2_JOBS" | awk -v map="$V2_NODE_MAP" '
    BEGIN { n = split(map, pairs, " "); for (i = 1; i <= n; i++) { split(pairs[i], kv, "="); to[kv[1]] = kv[2] } }
    NF == 0 || $1 ~ /^#/ { next }
    { moved = ($2 in to) ? 1 : 0; if (moved) $2 = to[$2]; printf "%s\t%d\t%06d\t%s\n", $1, moved, NR, $0 }' |
    sort -t "$(printf '\t')" -k1,1n -k2,2n -k3,3n | cut -f4-
}
# The command of one job (everything after its deps), or nothing.
v2_job_command() { v2_jobs | awk -v j="$1" '$3 == j { $1 = $2 = $3 = $4 = ""; sub(/^ +/, ""); print; exit }'; }
v2_job_deps()    { v2_jobs | awk -v j="$1" '$3 == j { print ($4 == "-" ? "" : $4); exit }' | tr ',' ' '; }
