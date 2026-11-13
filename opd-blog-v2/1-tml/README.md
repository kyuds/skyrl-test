# Section 1: on-policy distillation as Thinking Machines ran it

Thinking Machines' [post][blog] SFTs Qwen3-8B-Base on 400k OpenThoughts-3 prompts (60% on AIME 2024), then distills it
from Qwen3-8B in about 150 steps (70%). This section recreates both steps.

| Job | Stage | Node | What | Target |
|---|---|---|---|---|
| `sft_1a_qwen3_8b_base_ot3` | 1 | b200-1 | 1a: SFT of Qwen3-8B-Base on 384k OpenThoughts-3 rows | 60% AIME 2024 |
| `eval_sft_1a` | 3 | b200-1 | 1a at 64 samples (`../0-evals`), before 1b | 60% |
| `opd_1b_ot3sft8b_from_qwen3_8b` | 3 | b200-1 | 1b: OPD of 1a from Qwen3-8B | 70% |
| `eval_opd_1b_final` | 3 | b200-1 | 1b's last export at 64 samples | 70% |

The teacher is Qwen3-8B in thinking mode. The post presents Qwen3-32B, but says it actually used Qwen3-8B because it
performed slightly better.

**Which version of the cookbook.** The cookbook's distillation recipes moved to Qwen3.5-9B on 2026-06-16
([`ca232b0844`][qwen35]). Every value below comes from [`330b73d8d7`][c330] (2025-10-27), the commit that
published these recipes the day of the blog. The last Qwen3 version, [`58e781e78e`][c58e] (2026-05-27), has the same
values; it only adds an explicit `temperature: float = 1.0` (the Tinker default the older code already sampled at),
`max_steps`, and the SFT checkpoint the OPD command starts from. The Tinker SDK is the [commit][tinker] of 2025-10-17
(release 0.2.2), the API the blog's runs used.

## 1a: SFT

Script: [`sft.sh`](sft.sh), block `HYPERPARAMETERS`. SkyRL's SFT trainer, FSDP, full fine-tuning.

| Setting | Value | Source |
|---|---|---|
| Base | `Qwen/Qwen3-8B-Base` | [blog]; [recipe] `model_name` |
| Data | OpenThoughts-3, every domain (math, code, science), shuffled, first 384,000 | [recipe] streams the whole set |
| Rows | 384,000 = 3000 steps × 128 | [recipe] `max_prompts = 128 * 3000` |
| Length | 16,384 tokens, longer rows cut | [recipe] `max_length=16384` |
| Batch | 128 rows, 1 epoch | [recipe] `batch_size=128`, `num_epochs=1` |
| Learning rate | 1e-4 (the recipe's 1e-3 default is for LoRA) | [readme-sft]: "1e-4 with full finetuning" |
| Schedule | linear to 0, no warmup | [recipe] `lr_schedule="linear"`; [sched] `1 - step / total_steps` |
| Optimizer | Adam, betas 0.9 and 0.95, eps 1e-8 | [train] `adam_beta1`, `adam_beta2`, `adam_eps` |
| Weight decay | 0 | [tinker-adam]: Tinker's `AdamParams` has no weight decay |
| Grad clip | 1.0 | not in Tinker's API, so unknown; a guard here |
| Loss | summed over the answer tokens | [recipe] `TrainOnWhat.ALL_ASSISTANT_MESSAGES`; Tinker's `cross_entropy` is a sum, as SkyRL's |
| Prompt format | Qwen3's chat template; the answer starts `<think>\n` | [recipe] `renderer_name="qwen3"` |

**The starting model.** Qwen3-8B-Base is packaged as a text completer: its files end a sequence at `<|endoftext|>`
only and allow 32,768 tokens of context. 1a's answers, like its teacher's, end at `<|im_end|>`, and the evals need
more than 32,768 tokens. So before training, `../common/fix_base_settings.py` gives the Base weights Qwen3-8B's tokenizer, end
tokens and 40,960-token context; the post-trained model's files differ from the Base model's in exactly those, and its
chat template renders prompts the same way. SkyRL's export keeps the starting model's files, so 1a's output inherits
the fix; the job checks that before it uploads. Thinking Machines needed none of this: their sampler is told to stop
at `<|im_end|>`, and their evals are not limited by the model's config.

Infrastructure: micro-batch 2 rows per GPU (packed), checkpoints every 250 steps (the last kept), exports at 1000,
2000 and 3000 steps. The last export goes to the Hub as `kyuds/opd-blog-v2/sft_1a_qwen3_8b_base_ot3`, where other
nodes fetch it.

The data is described in `../data/README.md`, including the unfinished answers OpenThoughts-3 keeps (62% to 68% of
one code and one math shard, 12% of one science shard).

## 1b: OPD

Script: [`opd.sh`](opd.sh), block `HYPERPARAMETERS`.

| Setting | Value | Source |
|---|---|---|
| Student | 1a's output | [c58e] README: OPD starts from the SFT checkpoint |
| Teacher | `Qwen/Qwen3-8B`, thinking, served from h100-4: 8 engines × 1 GPU | [blog]; [opd] `teacher_model` |
| Prompts | DeepMath-103K, the `question` column alone, cut at 1,024 tokens | [opd] `dataset="deepmath"`; [data] `row["question"]`, `max_prompt_tokens=1024` |
| Batch | 512 prompts × 4 samples, one update a step | [readme-opd] `groups_per_batch=512`; [opd] `group_size=4`, `num_substeps=1`; [blog] 150 steps is about 77K prompts |
| Steps | 150 | [blog] |
| Learning rate | 5e-5, constant (1e-4 is for LoRA) | [readme-opd]: "5e-5 with full finetuning"; [rl] no schedule |
| Optimizer | Adam, betas 0.9 and 0.95, eps 1e-8 | [rl] `AdamParams(... beta1=0.9, beta2=0.95, eps=1e-8)` |
| Weight decay | 0 | [tinker-adam] |
| Grad clip | 1.0 | not in Tinker's API, so unknown; a guard here |
| Rollouts | 4,096 tokens, temperature 1.0, top-p 1.0, no top-k | [opd] `max_tokens=4096`; [tinker-sampling] defaults (the recipe sets none) |
| Advantage | reverse KL per token: teacher log-prob minus student log-prob, coefficient 1.0, no discount, no task reward | [opd] `kl_penalty_coef=1.0`, `kl_discount_factor=0.0`; [kl] |
| Loss | importance sampling, token mean | [opd] `loss_fn="importance_sampling"`; Tinker sums over tokens, the same up to a constant per step |
| Prompt format | Qwen3's chat template, thinking | see "Differences" |
| In-training eval | AIME 2024 and 2025, 8 samples, 32,768 tokens, temperature 0.6, top-p 0.95, top-k 20, every 5 steps | this kit's, at [qwen3]'s settings |

Infrastructure: all 8 GPUs of b200-2 for the student (training and generation colocated); the teacher is served
from h100-4 ([`../teachers/`](../teachers/README.md); before 2026-11-12, 4 GPUs each on one node); micro-batch 2;
checkpoints every 10 steps (the last 2 kept), exports every 50, and the last export goes to the Hub.

## Run by hand

The orchestrator runs these (`../README.md`). By hand on a node, from `~/SkyRL`, through the job runner:

```bash
nohup bash skyrl-test/opd-blog-v2/common/job.sh opd_1b_ot3sft8b_from_qwen3_8b > /dev/null 2>&1 < /dev/null &
```

```bash
tail -f ~/opd-store/logs/opd_1b_ot3sft8b_from_qwen3_8b.log
```

Smoke tests (2 tiny steps; nothing uploaded). The OPD one runs before 1a exists, with the Base model (settings fixed) as
its student:

```bash
bash skyrl-test/opd-blog-v2/1-tml/sft.sh --smoke
```

```bash
bash skyrl-test/opd-blog-v2/1-tml/opd.sh --smoke
```

## What to watch

- **1a:** `train/loss` on W&B project `opd-blog-v2`. Thinking Machines saw AIME rise log-linearly with SFT prompts, so
  the exports at 1000 and 2000 steps should score well below 60%, and 3000 near it.
- **1b:** `eval/aime-2024/avg_score` (the target), `opd/teacher/_/reverse_kl` (should fall), `opd/truncation_rate`
  (most rollouts hit 4,096 tokens at first; that is expected here) and `opd/repetition_rate`.

## Differences from Thinking Machines

- Their 400k subset and their shuffle are not public: same size and recipe, different rows.
- Gradient clipping at 1.0: Tinker's API of the time has no clipping or weight-decay field, so what its server did is
  unknown. No weight decay is assumed; the clip is a guard that rarely binds at these rates.
- **The OPD prompt format.** At `330b73d8d7` the OPD command names `Qwen/Qwen3-8B-Base` with no renderer, and a Base
  model name picks the plain `User:`/`Assistant:` renderer ([model-info]), not the `qwen3` one its SFT used. From
  2026-02 ([`631aa1884e`][renderer]) the cookbook takes the renderer from the SFT checkpoint instead, which is `qwen3`.
  This kit uses Qwen3's chat format, the one 1a trains on. Their `qwen3` renderer also puts `<think>\n` in the prompt;
  here the student writes it.
- The student's log-probabilities, in the advantage and the importance ratio, come from the sampler in Tinker and from
  a trainer forward pass in SkyRL. With one update a step, the only difference is sampler/trainer numerical mismatch.
- Unfinished OpenThoughts-3 answers get no end token; the cookbook's renderer would append one if the row is shorter
  than 16,384 tokens. Almost every unfinished row is cut at 16,384, so few rows differ.
- The answer starts `<think>\n` (Qwen3's layout) instead of OpenThoughts-3's `<think> `, so the student's format matches
  its teacher's.
- Their AIME settings are not in the blog; this kit uses the Qwen settings of `../0-evals` for every section 1 model.
- Their training ran on Tinker (LoRA for most results; the 60% and 70% are full fine-tuning per the blog).

## Risks

- 1a starts at 1e-4 with no warmup, as the cookbook does. If `train/loss` spikes in the first 50 steps, stop the job,
  delete its `ckpts/<job>`, `exports/<job>` and `jobs/<job>.*` on the node, set `SFT_WARMUP=30` in `sft.sh`, push, and
  run the stage again.
- 5e-5 is high for full fine-tuning; it is the cookbook's number. If 1b's reverse KL climbs or its eval drops in the
  first 25 steps, do the same with `LR=1e-5` in `opd.sh`.
- 1a is on the critical path: about 15 hours on a B200 node is an estimate (5 billion tokens), not measured.

[blog]: https://thinkingmachines.ai/blog/on-policy-distillation/
[c330]: https://github.com/thinking-machines-lab/tinker-cookbook/tree/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation
[c58e]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/58e781e78ebc58620c0ef0bc13ce7a45db691a2a/tinker_cookbook/recipes/distillation/README.md
[qwen35]: https://github.com/thinking-machines-lab/tinker-cookbook/commit/ca232b0844
[renderer]: https://github.com/thinking-machines-lab/tinker-cookbook/commit/631aa1884e
[recipe]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/off_policy_reasoning.py#L39-L101
[readme-sft]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/README.md#L17
[readme-opd]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/README.md#L30-L39
[train]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/supervised/train.py#L49-L69
[sched]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/utils/lr_scheduling.py#L11
[opd]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/recipes/distillation/on_policy_distillation.py#L50-L72
[data]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/distillation/datasets.py#L170-L221
[rl]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/rl/train.py#L116
[kl]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/distillation/train_on_policy.py#L81-L95
[model-info]: https://github.com/thinking-machines-lab/tinker-cookbook/blob/330b73d8d763b0f65bcf04e643cc6f082e002554/tinker_cookbook/model_info.py#L63-L71
[tinker]: https://github.com/thinking-machines-lab/tinker/tree/2c9f564c9f7d9236a0659da89781c09c6509b950
[tinker-adam]: https://github.com/thinking-machines-lab/tinker/blob/2c9f564c9f7d9236a0659da89781c09c6509b950/src/tinker/types/optim_step_request.py#L11-L21
[tinker-sampling]: https://github.com/thinking-machines-lab/tinker/blob/2c9f564c9f7d9236a0659da89781c09c6509b950/src/tinker/types/sampling_params.py#L18-L24
[qwen3]: https://arxiv.org/abs/2505.09388
