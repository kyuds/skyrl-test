"""Fixes a Qwen3 Base model's settings files so it can be run like a chat model: copies the weights, and takes the
settings files (tokenizer, generation config, end token and context length in config.json) from the post-trained Qwen3
of the same size. The weights are not changed. Training does not need this; running the model afterwards does.

    cd ~/SkyRL && uv run --isolated --extra fsdp python skyrl-test/opd-blog-v2/common/fix_base_settings.py \\
        --weights Qwen/Qwen3-8B-Base --tokenizer-from Qwen/Qwen3-8B --out ~/opd-store/models/<name>

Why: a Qwen3 Base model is packaged as a text completer, not a chat model, in two ways that matter once its teacher
is a post-trained Qwen3 (whose answers end at <|im_end|>) and its evals are 32k tokens long:
  1. Its generation config ends a sequence at <|endoftext|> (151643) only. A chat answer ends at <|im_end|> (151645).
     vLLM stops at the generation config's end tokens, so a Base model, or one SFT'd to end its answers at <|im_end|>,
     writes <|im_end|> and carries on, often into a made-up next turn, until it happens to write <|endoftext|> or hits
     the length limit. (Thinking Machines' renderer avoids this by passing <|im_end|> as a stop to the sampler.)
  2. Its config says 32,768 positions, the post-trained Qwen3's 40,960. vLLM refuses a context longer than the
     config's, and the evals need the prompt + 32,768 tokens (38,912 for Qwen's AIME).
The post-trained Qwen3 of the same size has the same weights layout and differs in exactly these files and fields
(checked 2026-11-10), and its chat template renders a one-turn prompt exactly like the Base template does (checked
2026-11-09: '<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n' from both), so the swap changes nothing
else. Used once, on Qwen3-8B-Base before 1a's SFT (run_lib.sh, v2_fix_base): SkyRL's export keeps the starting
model's config, generation config and tokenizer, so 1a's output inherits the fix.

Only the weights and the architecture config are taken from --weights; every tokenizer file comes from
--tokenizer-from, so a stale chat_template.jinja in an export cannot override the post-trained template.
"""

import argparse
import json
import os
import shutil
import time

from huggingface_hub import snapshot_download

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
]
WEIGHT_FILES = ("config.json", "model.safetensors.index.json")


def local_dir(source: str, patterns) -> str:
    """A local folder holding ``source``: the folder itself, or a Hub snapshot of the files matching ``patterns``."""
    if os.path.isdir(source):
        return source
    return snapshot_download(source, allow_patterns=list(patterns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="an HF export folder or a Hub id")
    parser.add_argument("--tokenizer-from", required=True, help="a post-trained Qwen3 (Hub id or folder)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = os.path.expanduser(args.out)

    weights = local_dir(args.weights, ["*.safetensors", *WEIGHT_FILES])
    tok = local_dir(args.tokenizer_from, [*TOKENIZER_FILES, "config.json"])
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    copied = 0
    for name in sorted(os.listdir(weights)):
        if name.endswith(".safetensors") or name in WEIGHT_FILES:
            shutil.copy2(os.path.join(weights, name), os.path.join(tmp, name))
            copied += 1
    if not os.path.exists(os.path.join(tmp, "config.json")) or copied < 2:
        raise SystemExit(f"{weights} has no config.json and safetensors to copy")
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(tok, name)):
            shutil.copy2(os.path.join(tok, name), os.path.join(tmp, name))

    # The architecture config keeps the weights' values except the two that Qwen3's own Base and post-trained configs
    # differ in (checked 2026-11-10 for 4B and 8B): the end token and the context, 32,768 on Base, 40,960 post-trained.
    # Both follow the tokenizer's model. Without the longer context, vLLM refuses the evals' prompt + 32,768 tokens.
    with open(os.path.join(tok, "config.json")) as f:
        tok_cfg = json.load(f)
    cfg_path = os.path.join(tmp, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    for key in ("eos_token_id", "max_position_embeddings"):
        cfg[key] = tok_cfg[key]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # The check that matters: the copy ends a turn at <|im_end|> and renders the prompt the OPD student sees.
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(tmp)
    prompt = t.apply_chat_template(
        [{"role": "user", "content": "1+1?"}], add_generation_prompt=True, enable_thinking=True, tokenize=False
    )
    assert t.eos_token == "<|im_end|>", t.eos_token
    assert prompt == "<|im_start|>user\n1+1?<|im_end|>\n<|im_start|>assistant\n", repr(prompt)
    with open(os.path.join(tmp, "generation_config.json")) as f:
        gen_eos = json.load(f).get("eos_token_id")
    assert 151645 in (gen_eos if isinstance(gen_eos, list) else [gen_eos]), gen_eos

    with open(os.path.join(tmp, "kit_provenance.json"), "w") as f:
        json.dump(
            {"weights": args.weights, "tokenizer_from": args.tokenizer_from, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            f,
            indent=2,
        )
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    print(f"wrote {out}: weights from {args.weights}, tokenizer from {args.tokenizer_from}")


if __name__ == "__main__":
    main()
