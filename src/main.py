"""Hydra entry point: dispatches to run_train / run_optuna / run_infer."""
from __future__ import annotations

from typing import Optional

import hydra
from omegaconf import DictConfig

from src.train import run_eval, run_infer, run_optuna, run_train
from src.utils.logging import get_logger

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point.

    Dispatch based on cfg.mode.name:
      train / debug  -> run_train
      optuna         -> run_optuna
      infer          -> run_infer      (visual artifacts)
      eval           -> run_eval       (scalar metrics, checkpoint re-eval)
    """
    mode = cfg.mode.name
    log.info(f"Mode: {mode} | Dataset: {cfg.dataset._target_} | Method: {cfg.method._target_}")

    if mode in ("train", "debug"):
        metrics = run_train(cfg)
        # Return primary metric for Hydra sweeper / multirun. Sourced from
        # cfg.experiment.monitor so it matches ModelCheckpoint, EarlyStopping
        # and Optuna — never val_loss by default.
        monitor = cfg.experiment.monitor
        return float(metrics.get(monitor, 0.0))

    elif mode == "optuna":
        run_optuna(cfg)
        return None

    elif mode == "infer":
        run_infer(cfg)
        return None

    elif mode == "eval":
        run_eval(cfg)
        return None

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Choose from: train, debug, optuna, infer, eval."
        )


if __name__ == "__main__":
    main()
