# Training

Gamma-World is trained in three stages:

```
bidirectional teacher  ->  causal student  ->  DMD / few-step student
```

Use the Cosmos-Predict2.5 2B pre-trained checkpoint as the training initialization checkpoint.
Training reads sharded DCP checkpoints, so first download the HF checkpoint and convert it to DCP.

## 1. Convert a checkpoint to DCP

`scripts/convert_checkpoint_to_dcp.py` turns the pre-trained `.pt` checkpoint into a model-only
DCP directory:

```bash
PRETRAINED_CKPT=$(uvx "hf>=1.3.5" download nvidia/Cosmos-Predict2.5-2B \
    --repo-type model \
    --revision 15a82a2ec231bc318692aa0456a36537c806e7d4 \
    base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt)

python scripts/convert_checkpoint_to_dcp.py \
    --input "$PRETRAINED_CKPT" \
    --output /path/to/converted_pretrained_dcp
```

This writes `<output>/model/` (the DCP) and `<output>/conversion.json`. Keys are normalized to the
`net.` prefix the trainer expects; the result is a model-only init checkpoint — it carries no
optimizer/scheduler/trainer state, so resume it with `checkpoint.load_training_state=False`.

## 2. Data

The training configs default to a synthetic `mock` data source so a run can start without data.
For real training, point the dataloader at your own data and override the dataset root; the
multi-player dataloader expects per-view frames plus per-frame keyboard/camera actions (see
`gamma_world/_src/gamma_world/datasets/`). The released models use 320×480 per view, 189 frames,
2 players.

## 3. Launch

All three stages share one trainer entry and differ only by `--config`, the experiment, and how the
init checkpoint is supplied.

### Bidirectional teacher

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config gamma_world/_src/gamma_world/configs/causal_cosmos2/config.py \
    -- experiment=bidirectional \
       checkpoint.load_path=/path/to/converted_pretrained_dcp \
       checkpoint.load_training_state=False \
       dataloader_train.data_root=/path/to/data
```

### Causal student

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config gamma_world/_src/gamma_world/configs/causal_cosmos2/config.py \
    -- experiment=causal \
       checkpoint.load_path=/path/to/converted_causal_dcp \
       checkpoint.load_training_state=False \
       dataloader_train.data_root=/path/to/data
```

### DMD / few-step student

DMD initializes three networks from converted DCP directories (note the trailing `/model`):

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config gamma_world/_src/gamma_world/configs/self_forcing/config.py \
    -- experiment=causal_few_step \
       model.config.net_ckpt=/path/to/converted_causal_dcp/model \
       model.config.net_real_score_ckpt=/path/to/converted_bidirectional_dcp/model \
       model.config.net_fake_score_ckpt=/path/to/converted_bidirectional_dcp/model \
       dataloader_train.data_root=/path/to/data
```
