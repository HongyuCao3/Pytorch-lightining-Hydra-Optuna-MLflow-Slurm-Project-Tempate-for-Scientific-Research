"""Small CLI adapters: bridge argparse/click CLIs to train.run_* functions.

Usage examples (without Hydra, for quick scripting):
    python -m src.entrypoints train --config configs/config.yaml
    python -m src.entrypoints infer --checkpoint checkpoints/best.ckpt
"""
from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf

from src.utils.logging import get_logger

log = get_logger(__name__)


def _load_cfg(config_path: str):
    return OmegaConf.load(config_path)


def cmd_train(args: argparse.Namespace) -> None:
    from src.train import run_train

    cfg = _load_cfg(args.config)
    metrics = run_train(cfg)
    log.info("Training complete. Metrics: %s", metrics)


def cmd_infer(args: argparse.Namespace) -> None:
    from src.train import run_infer

    cfg = _load_cfg(args.config)
    OmegaConf.update(cfg, "inference.checkpoint_path", args.checkpoint)
    OmegaConf.update(cfg, "mode.name", "infer")
    run_infer(cfg)


def cmd_optuna(args: argparse.Namespace) -> None:
    from src.train import run_optuna

    cfg = _load_cfg(args.config)
    OmegaConf.update(cfg, "mode.name", "optuna")
    run_optuna(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research template CLI adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Run training pipeline")
    p_train.add_argument("--config", default="configs/config.yaml")
    p_train.set_defaults(func=cmd_train)

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Run inference pipeline")
    p_infer.add_argument("--config", default="configs/config.yaml")
    p_infer.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p_infer.set_defaults(func=cmd_infer)

    # --- optuna ---
    p_optuna = subparsers.add_parser("optuna", help="Run Optuna hyperparameter search")
    p_optuna.add_argument("--config", default="configs/config.yaml")
    p_optuna.set_defaults(func=cmd_optuna)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
