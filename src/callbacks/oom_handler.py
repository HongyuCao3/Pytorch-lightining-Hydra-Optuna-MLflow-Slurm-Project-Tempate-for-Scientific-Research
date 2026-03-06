"""OOM (Out-Of-Memory) handler callback.

Injected into Trainer callbacks list in train.run_train().

Behavior
--------
- on_train_batch_start : pre-clear CUDA cache to reduce fragmentation
- on_exception         : if CUDA OOM, clear cache + save emergency checkpoint
                         + log reason as a text artifact
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from src.utils.logging import get_logger

log = get_logger(__name__)


class OOMHandler(pl.Callback):
    """Callback to handle CUDA out-of-memory errors gracefully."""

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        is_oom = isinstance(exception, RuntimeError) and "out of memory" in str(exception).lower()
        if not is_oom:
            return

        log.error("CUDA Out-Of-Memory detected. Attempting recovery...")

        # 1. Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info("CUDA cache cleared.")

        # 2. Save emergency checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = f"checkpoints/emergency_oom_{timestamp}.ckpt"
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            trainer.save_checkpoint(ckpt_path)
            log.info(f"Emergency checkpoint saved: {ckpt_path}")
        except Exception as save_err:
            log.warning(f"Could not save emergency checkpoint: {save_err}")
            ckpt_path = None

        # 3. Write failure record
        record = {
            "event": "oom",
            "timestamp": timestamp,
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "emergency_checkpoint": ckpt_path,
            "error": str(exception),
        }
        record_path = Path(f"checkpoints/oom_record_{timestamp}.json")
        record_path.write_text(json.dumps(record, indent=2))
        log.info(f"OOM record written to {record_path}")
