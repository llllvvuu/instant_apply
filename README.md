# Instant Apply Proof-of-Concept.

Implementation of Cursor's [Instant Apply](https://www.cursor.com/blog/instant-apply) feature.

Achieves close to max parallelism during generation (generation tok/s approx 70% of prefill tok/s on my laptop).

Only MLX (Apple devices) supported for now.

## Usage

### With `pip`
```sh
pip install -r requirements.txt  # preferably in a venv
python instant_apply_mlx.py
```

### With `uv`
```sh
./instant_apply_mlx.py
```

## How it works

The core logic is in these lines:
```python
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
```

This generally mispredicts only once per hunk, which enables turning up the speculative lookahead to the point of FLOPS saturation.
