# Architecture Rules

## Entry & dispatch
- `src/main.py` → `@hydra.main` → dispatches to `run_train / run_optuna / run_infer` in `src/train.py`.
- Mode controlled by `cfg.mode` (debug | train | optuna | infer).

## Method layering
```
methods/<name>/
  model.py      # pure nn.Module(s): Encoder / Decoder / Head
  lit_module.py # LightningModule: *_step + configure_optimizers
  config.yaml   # hyperparams + _target_ → LitModule class path
```
- `instantiate(cfg.method)` → LightningModule passed directly to `Trainer.fit`.
- Multiple methods co-exist; registry in `methods/registry.py`.

## Data layer
- One DataModule per dataset; no multi-DataModule composition.
- `data/preprocess.py` provides `transform_raw_to_tensor(raw) -> (X, y)`.
- DataModule implements `setup(stage)` → `train_ds / val_ds / test_ds`.

## Inference layer
- `inference/pipeline.py`: load ckpt → `trainer.predict()` → postprocess → analyzers.
- `inference/analyzers.py`: confusion_matrix, per_class_metrics, tsne_embedding, calibration.
- Artifacts uploaded to MLflow or written to shared dir.

## Extension points (LLM/Diffusion)
- Swap `model.py` only; keep predict contract (`{"pred": Tensor, ...}`).
- Trainer strategy (deepspeed/fsdp) configured in `configs/trainer/` only.
