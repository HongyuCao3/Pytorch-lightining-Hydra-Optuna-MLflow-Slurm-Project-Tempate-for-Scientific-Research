# Data Rules (applies to src/data/**)

## DataModule contract
- Must implement `setup(stage)` creating `train_ds`, `val_ds`, `test_ds`.
- DataLoaders returned by `train_dataloader / val_dataloader / test_dataloader`.
- Batch shape must be compatible with `model.forward` input.

## Preprocessing
- All raw→tensor conversion goes through `data/preprocess.py::transform_raw_to_tensor`.
- DataModule calls preprocess; never preprocess inline in model/lit_module.
- Output format: `(X_tensor: FloatTensor, y_tensor: LongTensor)`.

## Constraints
- No dataset mixing / multi-datamodule composition.
- Do not add dataset-specific logic to preprocess.py; keep it generic.
