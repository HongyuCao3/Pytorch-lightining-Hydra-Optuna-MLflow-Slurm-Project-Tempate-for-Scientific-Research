"""Export a Lightning checkpoint to a pure PyTorch state_dict.

Usage
-----
    python -m src.scripts.export_ckpt <ckpt_path> [--output <out_path>]

The exported .pt file can be loaded with torch.load() without requiring
pytorch_lightning to be installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_ckpt(ckpt_path: str, output_path: str | None = None) -> str:
    """Convert Lightning checkpoint to pure state_dict .pt file.

    Parameters
    ----------
    ckpt_path   : path to the .ckpt file
    output_path : destination path; defaults to <stem>_exported.pt

    Returns
    -------
    output_path : path to the written .pt file
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip PL's "model." prefix added by save_hyperparameters
    cleaned: dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith("model.") else k
        cleaned[new_key] = v

    if output_path is None:
        output_path = str(Path(ckpt_path).with_suffix("")) + "_exported.pt"

    torch.save(cleaned, output_path)
    print(f"Exported {len(cleaned)} tensors -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lightning checkpoint to state_dict.")
    parser.add_argument("ckpt_path", type=str, help="Path to .ckpt file")
    parser.add_argument("--output", type=str, default=None, help="Output .pt file path")
    args = parser.parse_args()
    export_ckpt(args.ckpt_path, args.output)


if __name__ == "__main__":
    main()
