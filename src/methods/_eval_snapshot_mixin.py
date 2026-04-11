"""Eval-snapshot mixin for Lightning modules.

Embeds the eval-critical subset of the training config into every saved
checkpoint so that post-hoc ``run_eval`` can verify the reload conditions
match the original training conditions.

Design rationale
----------------
A sidecar YAML next to the checkpoint gets orphaned whenever the checkpoint
file is moved, renamed, or shared. Embedding the snapshot *inside* the
Lightning checkpoint makes it unforgeable and travels with the file.

Writer side (training)
~~~~~~~~~~~~~~~~~~~~~~
``run_train`` calls ``model.attach_eval_snapshot({...})`` right after
instantiating the LightningModule. On the next ``trainer.save_checkpoint`` call
PyTorch Lightning invokes ``on_save_checkpoint`` (the mixin hook), which copies
the attached dict into ``checkpoint["eval_snapshot"]``.

Reader side (post-hoc eval)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``run_eval`` uses :func:`read_eval_snapshot` to load the snapshot directly from
the checkpoint file without instantiating the module, then compares the
recorded fields against ``cfg.dataset`` / ``cfg.method``. A mismatch triggers
either a warning (default) or an exception depending on
``cfg.eval.strict_snapshot``.

Checkpoints saved by older template versions (pre-snapshot) simply lack the
key — :func:`read_eval_snapshot` returns ``None`` and ``run_eval`` degrades to
a warning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.utils.logging import get_logger

log = get_logger(__name__)


# Schema version for forward compatibility. Bump when fields are added/removed.
EVAL_SNAPSHOT_VERSION = 1

# Keys whose equality is required for two runs to be considered "same conditions".
EVAL_SNAPSHOT_KEYS = (
    "dataset_target",
    "split_seed",
    "val_split",
    "test_split",
    "method_target",
)


class EvalSnapshotMixin:
    """Mixin for :class:`pytorch_lightning.LightningModule` subclasses.

    Adds:
      * :meth:`attach_eval_snapshot` — setter called by ``run_train``.
      * :meth:`on_save_checkpoint`  — PL hook that embeds the snapshot.

    Safe to mix in even when ``attach_eval_snapshot`` is never called: the
    hook simply omits the key in that case (the resulting checkpoint stays
    backwards-compatible with old checkpoints).
    """

    # Private storage — intentionally NOT persisted via save_hyperparameters()
    # because the snapshot must travel under its own dedicated checkpoint key,
    # not be mixed into hparams.
    _eval_snapshot: Optional[Dict[str, Any]] = None

    def attach_eval_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Attach an eval-critical config snapshot to this module instance.

        Parameters
        ----------
        snapshot
            Flat dict of primitive values. Expected keys are enumerated by
            ``EVAL_SNAPSHOT_KEYS``; extra keys are permitted and preserved.
            A ``snapshot_version`` field is added automatically.
        """
        if not isinstance(snapshot, dict):
            raise TypeError(
                f"attach_eval_snapshot expects a dict, got {type(snapshot).__name__}"
            )
        stored = dict(snapshot)
        stored.setdefault("snapshot_version", EVAL_SNAPSHOT_VERSION)
        self._eval_snapshot = stored
        log.debug(f"Eval snapshot attached: {sorted(stored.keys())}")

    # PyTorch Lightning hook — called by Trainer on every save_checkpoint.
    # We intentionally do NOT override on_load_checkpoint: readers use
    # ``read_eval_snapshot`` to pull the key without module instantiation.
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        if self._eval_snapshot is not None:
            checkpoint["eval_snapshot"] = dict(self._eval_snapshot)


# ---------------------------------------------------------------------------
# Reader helpers — used by run_eval
# ---------------------------------------------------------------------------

def read_eval_snapshot(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Return ``checkpoint["eval_snapshot"]`` or ``None`` if the key is absent.

    Uses ``torch.load(weights_only=False)`` because the snapshot dict contains
    primitive Python values only, but is stored inside a pickle container.
    Returns ``None`` (not raise) for missing files, missing keys, or any load
    error — the caller decides whether to warn or block.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        log.warning(f"read_eval_snapshot: file not found: {checkpoint_path}")
        return None
    try:
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        log.warning(f"read_eval_snapshot: torch.load failed: {e}")
        return None
    snap = raw.get("eval_snapshot") if isinstance(raw, dict) else None
    if snap is None:
        log.info(
            f"Checkpoint at {checkpoint_path} has no eval_snapshot "
            "(older template version or non-train save)."
        )
        return None
    return snap


def compare_eval_snapshot(
    snapshot: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> Dict[str, tuple]:
    """Return a dict of ``{key: (snapshot_value, current_value)}`` for mismatches.

    Only compares the canonical ``EVAL_SNAPSHOT_KEYS``. A ``None`` snapshot
    returns an empty dict (caller treats this as "unknown, warn-only").
    """
    if snapshot is None:
        return {}
    diffs: Dict[str, tuple] = {}
    for key in EVAL_SNAPSHOT_KEYS:
        snap_val = snapshot.get(key)
        cur_val = current.get(key)
        if snap_val != cur_val:
            diffs[key] = (snap_val, cur_val)
    return diffs
