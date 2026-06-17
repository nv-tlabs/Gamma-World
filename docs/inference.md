# Inference

Gamma-World ships three models, selected with `--mode`:

| mode | description | denoising |
| --- | --- | --- |
| `bidirectional` | full-context teacher | ~35 steps, single-shot |
| `causal` | block-causal student | ~35 steps, KV-cache |
| `causal_few_step` | distilled streaming student | 4 steps `[1000,750,500,250]`, KV-cache |

The released checkpoints are single network `.safetensors` files at
[`chijw/Gamma-World`](https://huggingface.co/chijw/Gamma-World)
(`bidirectional/`, `causal/`, `causal-few-step/`). The VAE (`tokenizer.pth`) and the
Cosmos-Reason1 text encoder are downloaded automatically from Hugging Face.

## Input

Inference reads sample folders from `--eval-dir`. Each sample folder contains:

- `first_frame.png`: horizontally-concatenated player views
- `prompt.txt`: text prompt for that sample
- `action_left.json` and `action_right.json` for two-player samples, or `action_0.json`, `action_1.json`, ... for N-player samples

An action JSON has one row per frame:

```json
{
  "keyboard": [[0, 1, 0, ...], ...],
  "camera":   [[0.0, 0.0], ...]
}
```

`keyboard` is a 23-dim binary vector per frame; `camera` is a 2-dim yaw/pitch delta per frame.

## Run

The bundled example data under `data/` stores each sample as a horizontally-concatenated
two-player `first_frame.png`, per-player actions, a prompt, and an optional ground-truth video.
The inference entry reads these sample folders directly:

```bash
torchrun --nproc_per_node=1 scripts/inference.py \
    --mode causal_few_step \
    --eval-dir data \
    --n-players 2 \
    --output results/examples
```

This processes all sample folders under `data/` and writes each result to
`results/examples/<sample>/generated.mp4`. Use `--max-eval-samples 1` to run a single sample.
Each sample uses its own `prompt.txt`; pass `--prompt "..."` to override all eval samples.

Each mode defaults to its public checkpoint. `--checkpoint` accepts a local path or an `hf://` URI
when you want to override it. Four players work without retraining when each sample contains four
views in `first_frame.png` and `action_0.json` through `action_3.json`. The output is written to
`<output>/<sample>/generated.mp4`.

### Options

| flag | default | notes |
| --- | --- | --- |
| `--num-frames` | 189 | frames per view |
| `--height` / `--width` | 320 / 480 | per-view resolution |
| `--guidance` | 5.0 | classifier-free guidance |
| `--num-steps` | per-mode | denoising steps (ignored by `causal_few_step`) |
| `--seed` | 1 | |
| `--eval-dir` | required | directory of example samples with `first_frame.png`, `prompt.txt`, and action JSONs |
| `--n-players` | 2 | number of players/views in each `--eval-dir` sample |
| `--max-eval-samples` | all | limit the number of sample folders to run |
| `--prompt` | sample `prompt.txt` | override the prompt for all eval samples |
| `--vae` / `--text-encoder` | public HF | override with a local path or `hf://` URI |

## Notes

- `causal_few_step` is the fastest (4 steps) and runs comfortably on one 80 GB GPU.
- `bidirectional` is full-context (no KV-cache); at 189 frames it needs FlashAttention or
  context-parallel inference (`--context-parallel-size N` with `torchrun --nproc_per_node=N`).
