# requires-python = ">=3.12"
# dependencies = [
#     "mlx-lm~=0.17.1",
# ]
# ///

import argparse
import time

import mlx_lm
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import KVCache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instant Apply, from https://www.cursor.com/blog/instant-apply"
    )
    _ = parser.add_argument(
        "model", type=str, help="Example: mlx-community/Meta-Llama-3.1-8B-8bit"
    )
    _ = parser.add_argument("target", type=str, help="Example: sample_target.py")
    _ = parser.add_argument("edit", type=str, help="Example: sample_edit.py")
    _ = parser.add_argument("--speculation-lookahead", type=int, default=64)
    _ = parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    model, tokenizer = mlx_lm.load(args.model)
    with open(args.target) as target_file, open(args.edit) as edit_file:
        target, edit = target_file.read(), edit_file.read()
    target_tokens, edit_tokens = tokenizer.encode(target), tokenizer.encode(edit)
    target_edit_dist = list(range(len(target_tokens) + 1))
    edit_edit_dist = list(range(len(edit_tokens) + 1))

    try:  # instruct model
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Apply to the following file:\n```\n{target}\n```\nthe following edit:\n```\n{edit}\n```\nRespond with only the full modified file (no omissions), Markdown fenced. The content from the edit MUST replace the content from the target where applicable.",
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
    except ValueError:  # base model
        prompt = tokenizer.encode(
            f"The original source code was:\n```\n{target}\n```\nAfter applying the following edit:\n```\n{edit}\n```\nthe new code was the following, which differs from the original code where indicated by the edit:"
        )
    prompt = mx.array(prompt)[None]
    prompt_len = prompt.shape[1]
    cache = create_cache(model)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    tic = time.perf_counter()
    prompt_time = float("inf")
    token = 0
    n_tokens = 0

    for n in range(args.max_tokens):
        draft: list[int] = []
        target_idx = target_edit_dist.index(min(target_edit_dist))
        if target_idx > 0 and token == target_tokens[target_idx - 1]:
            draft = target_tokens[target_idx:]
        else:
            edit_idx = edit_edit_dist.index(min(edit_edit_dist))
            if edit_idx > 0 and token == edit_tokens[edit_idx - 1]:
                draft = edit_tokens[edit_idx:]
            else:
                # to recover quickly from the LLM deleting a large chunk of text
                # (otherwise keeps drafting from pre-deletion position due to edit dist)
                target_idx = min(
                    (i for i, t in enumerate(target_tokens) if t == token),
                    default=0,
                    key=lambda i: target_edit_dist[i + 1],
                )
                draft = target_tokens[target_idx + 1 :]
        draft = draft[: args.speculation_lookahead] or [0]
        draft_toks = mx.array(draft)[None]
        input_toks = mx.concatenate([prompt, draft_toks[:, :-1]], axis=-1)
        logits = model(input_toks, cache=cache)
        logits = logits[:, prompt.shape[1] - 1 :, :]
        output_toks = logits.argmax(axis=-1)
        n_accepted = (output_toks == draft_toks).astype(mx.uint8).cummin().sum().item()
        n_used = min(n_accepted + 1, len(draft))
        break_flag = False
        for i in range(n_used):
            prompt = output_toks[:, i : i + 1]
            token = prompt.item()
            detokenizer.add_token(token)
            n_tokens += 1
            if token == tokenizer.eos_token_id:
                break_flag = True
                break
            update_edit_dists(target_edit_dist, target_tokens, token)
            update_edit_dists(edit_edit_dist, edit_tokens, token)
        if break_flag:
            break
        for c in cache:
            drop_from_cache(c, len(draft) - n_used)
        print(detokenizer.last_segment, end="", flush=True)
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()

    gen_time = time.perf_counter() - tic
    print(detokenizer.last_segment)
    print(f"Prompt processing: {prompt_len / prompt_time} tokens-per-second")
    print(f"Generation: {n_tokens / gen_time} tokens-per-second")


def create_cache(model: nn.Module) -> list[KVCache]:
    if hasattr(model, "make_cache"):
        return model.make_cache()
    else:
        kv_heads = (
            [model.n_kv_heads] * len(model.layers)
            if isinstance(model.n_kv_heads, int)
            else model.n_kv_heads
        )
        return [KVCache(model.head_dim, n) for n in kv_heads]


def drop_from_cache(cache: KVCache, n: int):
    if n >= cache.offset:
        cache.keys = cache.values = None
        cache.offset = 0
    elif n > 0:
        cache.offset -= n


def update_edit_dists(edit_dist: list[int], tokens: list[int], token: int) -> None:
    prev = edit_dist[0]
    edit_dist[0] += 1
    for i in range(len(tokens)):
        cur = edit_dist[i + 1]
        edit_dist[i + 1] = (
            prev if token == tokens[i] else 1 + min(prev, cur, edit_dist[i])
        )
        prev = cur


if __name__ == "__main__":
    main()
