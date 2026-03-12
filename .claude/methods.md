# Methods Rules (applies to src/methods/**)

## model.py
- Contains ONLY `nn.Module` subclasses (Encoder, Decoder, Head, etc.).
- No cfg, no logger, no training logic, no hydra imports.
- Hyperparams passed as plain Python args to `__init__`.
- Use LayerNorm (not BatchNorm) to handle any batch size.

## lit_module.py
- Inherits `pl.LightningModule`.
- Implements: `training_step`, `validation_step`, `test_step`, `predict_step`, `configure_optimizers`.
- `predict_step` MUST return: `{"pred": Tensor, "prob": Tensor, "meta": {...}}`.
- Call `self.save_hyperparameters()` for necessary hyperparams; do not store full cfg.
- Use `ReduceLROnPlateau` scheduler (avoids `max_epochs` dependency).

## config.yaml (per method)
- Must contain `_target_` pointing to `src.methods.<name>.lit_module.<Class>`.
- Declare all hyperparams with sensible defaults.

## registry.py
- `@register("name")` decorator maps string name → LitModule factory.
- Used for dynamic method lookup; do not hard-code method names elsewhere.

## Adding a new method
1. Create `methods/<name>/` with `model.py`, `lit_module.py`, `config.yaml`.
2. Register with `@register("<name>")`.
3. Add `configs/method/<name>.yaml` mirroring the per-method config.
