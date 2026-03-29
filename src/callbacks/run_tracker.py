"""RunTrackerCallback: lightweight epoch-level observability.

This callback is a **pure data accumulator** — it never writes files or calls
MLflow.  Its sole job is to record convergence, best-metric, and per-epoch
timing so that ``run_train`` can build a structured summary after
``trainer.fit()`` returns.

Collected fields (all public, read by ``run_train``):

* ``best_value``       — best monitored metric seen so far (float | None)
* ``best_step``        — epoch at which best_value was first achieved (int | None)
* ``convergence_step`` — first epoch where the metric crossed
                         ``convergence_threshold`` (int | None).
                         Stays None when no threshold is configured.
* ``epoch_wall_times`` — list of per-epoch elapsed seconds (list[float])
* ``status``           — "running" | "completed" | "failed"
"""
from __future__ import annotations

import time
from typing import Optional

import pytorch_lightning as pl

from src.utils.logging import get_logger

log = get_logger(__name__)


class RunTrackerCallback(pl.Callback):
    """Track convergence, best metric, and epoch timing without model-level I/O.

    Parameters
    ----------
    monitor:
        Name of the metric to watch, e.g. ``"val_loss"`` or ``"val_acc"``.
    mode:
        ``"min"`` if lower is better (loss), ``"max"`` if higher is better
        (accuracy).  Controls how ``best_value`` and ``convergence_step``
        are updated.
    convergence_threshold:
        Optional float.  When the monitored metric first crosses this value
        (≤ threshold for ``mode="min"``, ≥ threshold for ``mode="max"``),
        ``convergence_step`` is set to the current epoch index.  Leave as
        ``None`` to disable convergence detection (``convergence_step`` will
        always be ``None`` in the run summary).
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        convergence_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.monitor = monitor
        self.mode = mode
        self.convergence_threshold = convergence_threshold

        # ------------------------------------------------------------------
        # Public state — read by run_train after trainer.fit() returns.
        # ------------------------------------------------------------------
        self.best_value: Optional[float] = None
        self.best_step: Optional[int] = None
        self.convergence_step: Optional[int] = None
        self.epoch_wall_times: list[float] = []
        self.status: str = "running"

        # Internal helpers
        self._epoch_start: Optional[float] = None
        self._is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    # ------------------------------------------------------------------
    # Epoch-level metric tracking
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update best_value / best_step and optionally record convergence_step."""
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return

        val_f = float(val)
        epoch = trainer.current_epoch

        # Track best seen so far
        if self.best_value is None or self._is_better(val_f, self.best_value):
            self.best_value = val_f
            self.best_step = epoch

        # Check convergence threshold (only set once, at first crossing)
        if (
            self.convergence_step is None
            and self.convergence_threshold is not None
        ):
            crossed = (
                val_f <= self.convergence_threshold
                if self.mode == "min"
                else val_f >= self.convergence_threshold
            )
            if crossed:
                self.convergence_step = epoch
                log.info(
                    f"[RunTracker] Convergence reached at epoch {epoch}: "
                    f"{self.monitor}={val_f:.6f} "
                    f"(threshold={self.convergence_threshold})"
                )

    # ------------------------------------------------------------------
    # Per-epoch wall-time recording
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Record the wall-clock start of each training epoch."""
        self._epoch_start = time.monotonic()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Compute and store the elapsed time for the completed epoch."""
        if self._epoch_start is not None:
            elapsed = round(time.monotonic() - self._epoch_start, 3)
            self.epoch_wall_times.append(elapsed)
            self._epoch_start = None

    # ------------------------------------------------------------------
    # Terminal lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Mark the run as completed when fit() exits normally."""
        self.status = "completed"
        log.info(
            f"[RunTracker] fit complete | "
            f"best_{self.monitor}={self.best_value} at epoch={self.best_step} | "
            f"convergence_step={self.convergence_step}"
        )

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        """Mark the run as failed when fit() raises an exception."""
        self.status = "failed"
        log.warning(
            f"[RunTracker] fit raised {type(exception).__name__}: "
            f"marking status='failed'."
        )
