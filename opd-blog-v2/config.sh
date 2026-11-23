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

# ---- teacher servers (teachers/README.md) ----
# 1b and MOPD score with teachers served from a node of their own, so their students have all 8 GPUs of their nodes.
# Each line is one served model: SkyRL's own teacher launch (vLLM servers plus a router) kept up by the job
# serve_teachers_<node>, reached by the runs at http://<node's internal IP>:<port>.
#   <key> <node> <router port> <model> <engines> <GPUs per engine>
# Qwen3-8B serves 1b, whose student is on b200-2, from h100-6: it served from h100-2 until h100-2 was preempted on
# 2026-11-13 at 08:03Z with europe-west2-b out of capacity, and the RL teachers moved to h100-2 in exchange. The RL
# teachers serve MOPD's student on h100-4. Section 2 serves no teacher: each run keeps its teacher on 4 of its node's
# 8 GPUs (2-alignment/opd.sh), so 2a, 2b and 2d run at once, one per node.
V2_TEACHER_SERVERS="${V2_TEACHER_SERVERS:-
qwen3_8b      h100-6  31000  $QWEN3_8B   8  1
rl_math    h100-2  31001  $M2RL_MATH  4  1
rl_coding  h100-2  31002  $M2RL_CODE  2  1
rl_if      h100-2  31003  $M2RL_IF    2  1
}"
# The longest sequence a served teacher scores: a 2,048-token prompt and a 16,384-token answer, sections 2 and 3's
# lengths, plus one. A run with longer ones is refused (common/run_lib.sh, v2_remote_teacher).
V2_TEACHER_MAX_LEN=18433
# The top-K overlap metrics of every OPD run (SkyRL's trainer.algorithm.opd.topk_metrics; diagnostics only, the loss
# is unchanged): each step, on V2_TOPK_ROWS rows per teacher, the share of the student's top V2_TOPK tokens that are
# also in the teacher's top V2_TOPK, and the teacher advantage on those shared tokens. Logged as
# opd/teacher/<name>/overlap_ratio and overlap_token_advantage. K=16 is the paper's. 64 rows, evenly spaced over a
# teacher's rows (all of them when it has fewer), by choice on 2026-11-11: a quarter of a 1,024-answer batch, four times
# the OPD stack's plan of 16 (plan/opd-entrypoint/overlap-metrics.md), for a less noisy curve. The count is per
# teacher, so MOPD measures up to 192 rows. Each measured row's teacher sends its top K over the whole sequence as
# JSON, roughly 5 MB at 12,000 tokens (an estimate), so about 0.3 GB a step per teacher. A served teacher must allow K
# logprobs, so serve.sh passes the same two values to its servers.
V2_TOPK_ROWS=64
V2_TOPK=16
# One field of a served model's line: 2 node, 3 port, 4 model, 5 engines, 6 GPUs per engine.
v2_teacher_field() { echo "$V2_TEACHER_SERVERS" | awk -v k="$1" -v c="$2" '$1 == k {print $c; exit}'; }

# ---- the jobs ----
# One line per job: <stage> <node> <job> <deps> <command, relative to opd-blog-v2/, with its arguments>.
#   stage    orchestrate.sh run <stage> starts the stage's jobs. Stages are the README's plan, not barriers: a job
#            starts once its own deps are done.
#   node     the VM it runs on. A node runs its jobs of a stage one after another, in this order.
#   job      unique. It names the W&B run, the log (~/opd-store/logs/<job>.log) and the Hub folder of its output.
#   deps     jobs that must be done first, comma-separated, or -. Checked on every node, so a dep can run elsewhere:
#            SFT outputs go through the Hub, which is how an eval on an H100 finds a model trained on a B200.
# common/job.sh runs a job by name and records it in ~/opd-store/jobs/; a job that is done is never run again.
# Every OPD run waits for its teacher's eval and its student's (README.md, "The plan"). A serve_teachers_* job never
# ends: it serves until stopped. The runs that score against it wait, at launch, until it answers.
# eval_sft_1a_tml and eval_opd_1b_final_tml (section 1's models at Thinking Machines' sampling, 0-evals/README.md)
# sit in b200-2's stage 2, ahead of 2c, so one chain runs them and then 2c; 2b follows 2c there. Section 3 (MOPD and
# its teachers) is in stage 3: a serve_teachers_* job never ends, so nothing could follow it.
# 2d (opd_2d_m2rl_sft_from_qwen3_32b_clip, h100-2) has no rows: it was stopped at step 56 on 2026-11-23, having
# collapsed as 2b did (2-alignment/README.md, "2d's clip"), and its step-50 export is on the Hub. Its job is
# unfinished, so a row would start it again. Nor have 2e and 2f (h100-2), each stopped at OPD step 1 or 2 on
# 2026-11-23 because its warm-up's student failed the screen (2-alignment/README.md, "Repairing 2c's warm-up").
# eval_sft_2c_warmup_math_temperature_1 asks whether that student's loops go away at temperature 1.0 (same section); its
# model comes from the Hub, so it waits for nothing. 2g follows it on h100-2: its own answers from Qwen3-32B, sampled
# at temperature 1.0, then its warm-up, then its OPD.
V2_JOBS="${V2_JOBS:-
1 b200-2 sft_1a_qwen3_8b_base_ot3             -                                       1-tml/sft.sh
1 b200-2 eval_qwen3_32b_math                  -                                       0-evals/run.sh eval_qwen3_32b_math Qwen/Qwen3-32B qwen math
1 b200-2 traces_2c_qwen3_32b                  -                                       2-alignment/traces.sh 2c
1 h100-1 eval_m2rl_sft_math                   -                                       0-evals/run.sh eval_m2rl_sft_math Jackwang111/M2RL-SFT m2rl math
1 h100-2 eval_m2rl_rl_math_math               -                                       0-evals/run.sh eval_m2rl_rl_math_math Jackwang111/M2RL-RL_Math m2rl math
1 h100-3 eval_qwen3_8b_math                   -                                       0-evals/run.sh eval_qwen3_8b_math Qwen/Qwen3-8B qwen math
1 h100-4 eval_m2rl_sft_code_if                -                                       0-evals/run.sh eval_m2rl_sft_code_if Jackwang111/M2RL-SFT m2rl code,if
1 h100-4 eval_m2rl_rl_coding_code             -                                       0-evals/run.sh eval_m2rl_rl_coding_code Jackwang111/M2RL-RL_Coding m2rl code
1 h100-4 eval_m2rl_rl_if_if                   -                                       0-evals/run.sh eval_m2rl_rl_if_if Jackwang111/M2RL-RL_IF m2rl if
2 b200-2 sft_2c_m2rl_sft_warmup               traces_2c_qwen3_32b                     2-alignment/sft_2c.sh 2c
2 b200-2 eval_sft_1a_tml                      sft_1a_qwen3_8b_base_ot3                0-evals/run.sh eval_sft_1a_tml kit:sft_1a_qwen3_8b_base_ot3 tml math
2 b200-2 eval_opd_1b_final_tml                opd_1b_ot3sft8b_from_qwen3_8b           0-evals/run.sh eval_opd_1b_final_tml run:opd_1b_ot3sft8b_from_qwen3_8b tml math
2 b200-2 opd_2c_m2rl_warmup_from_qwen3_32b    sft_2c_m2rl_sft_warmup,eval_qwen3_32b_math,eval_m2rl_sft_math 2-alignment/opd.sh 2c
2 b200-2 eval_opd_2c_final                    opd_2c_m2rl_warmup_from_qwen3_32b       0-evals/run.sh eval_opd_2c_final run:opd_2c_m2rl_warmup_from_qwen3_32b m2rl math
2 b200-2 opd_2b_m2rl_sft_from_qwen3_32b       eval_qwen3_32b_math,eval_m2rl_sft_math  2-alignment/opd.sh 2b
2 b200-2 eval_opd_2b_final                    opd_2b_m2rl_sft_from_qwen3_32b          0-evals/run.sh eval_opd_2b_final run:opd_2b_m2rl_sft_from_qwen3_32b m2rl math
2 h100-4 opd_2a_m2rl_sft_from_m2rl_rl_math    eval_m2rl_rl_math_math,eval_m2rl_sft_math 2-alignment/opd.sh 2a
2 h100-4 eval_opd_2a_final                    opd_2a_m2rl_sft_from_m2rl_rl_math       0-evals/run.sh eval_opd_2a_final run:opd_2a_m2rl_sft_from_m2rl_rl_math m2rl math
2 h100-2 eval_sft_2c_warmup_math_temperature_1 -                                     0-evals/run.sh eval_sft_2c_warmup_math_temperature_1 kit:sft_2c_m2rl_sft_warmup m2rl_temperature_1 math
2 h100-2 traces_2g_qwen3_32b_temperature_1    -                                       2-alignment/traces.sh 2g
2 h100-2 sft_2g_m2rl_sft_warmup_temperature_1 traces_2g_qwen3_32b_temperature_1       2-alignment/sft_2c.sh 2g
2 h100-2 opd_2g_m2rl_warmup_temperature_1_from_qwen3_32b sft_2g_m2rl_sft_warmup_temperature_1,eval_qwen3_32b_math,eval_m2rl_sft_math 2-alignment/opd.sh 2g
2 h100-2 eval_opd_2g_final                    opd_2g_m2rl_warmup_temperature_1_from_qwen3_32b 0-evals/run.sh eval_opd_2g_final run:opd_2g_m2rl_warmup_temperature_1_from_qwen3_32b m2rl math
3 h100-2 serve_teachers_h100_2                -                                       teachers/serve.sh h100-2
3 h100-4 mopd_3_m2rl_sft_from_math_code_if    eval_m2rl_sft_math,eval_m2rl_rl_math_math,eval_m2rl_sft_code_if,eval_m2rl_rl_coding_code,eval_m2rl_rl_if_if 3-mopd/run.sh
3 h100-4 eval_mopd_3_final                    mopd_3_m2rl_sft_from_math_code_if       0-evals/run.sh eval_mopd_3_final run:mopd_3_m2rl_sft_from_math_code_if m2rl all
3 h100-6 serve_teachers_h100_6                -                                       teachers/serve.sh h100-6
3 h100-6 eval_sft_1a                          sft_1a_qwen3_8b_base_ot3                0-evals/run.sh eval_sft_1a kit:sft_1a_qwen3_8b_base_ot3 qwen math
3 b200-2 opd_1b_ot3sft8b_from_qwen3_8b        sft_1a_qwen3_8b_base_ot3,eval_qwen3_8b_math 1-tml/opd.sh
3 b200-2 eval_opd_1b_final                    opd_1b_ot3sft8b_from_qwen3_8b           0-evals/run.sh eval_opd_1b_final run:opd_1b_ot3sft8b_from_qwen3_8b qwen math
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
