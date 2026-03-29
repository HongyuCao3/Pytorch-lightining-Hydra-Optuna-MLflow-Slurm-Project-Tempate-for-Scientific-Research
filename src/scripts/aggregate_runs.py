"""Aggregate MLflow experiment runs into a summary CSV.

This script reads all runs in a named MLflow experiment and writes a flat CSV
where each row corresponds to one run.  It also attempts to download the
per-run ``run_summary.json`` artifact (produced by ``run_train``) to surface
the structured observability fields alongside the standard MLflow params and
metrics.

Usage
-----
Basic aggregation (all FINISHED runs)::

    python -m src.scripts.aggregate_runs --experiment research-template

Write to a custom output file::

    python -m src.scripts.aggregate_runs \\
        --experiment research-template \\
        --output results/all_runs.csv

Multi-seed aggregation (group by a param, compute mean/std)::

    python -m src.scripts.aggregate_runs \\
        --experiment research-template \\
        --group-by method.learning_rate \\
        --output results/lr_sweep.csv
    # produces results/lr_sweep.csv  (per-run rows)
    #          results/lr_sweep.grouped.csv  (aggregated rows)

Include failed runs::

    python -m src.scripts.aggregate_runs \\
        --experiment research-template \\
        --status-filter ALL

Dependencies
------------
Only ``mlflow`` and the Python standard library are required.  No pandas.

Output columns
--------------
run_id, run_name, status, start_time,
summary.trial_id, summary.seed, summary.final_metric, summary.best_metric,
summary.convergence_step, summary.wall_time, summary.status, summary.traceback,
param.<key> (one column per logged param),
metric.<key> (one column per logged metric)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core aggregation logic
# ---------------------------------------------------------------------------

def _download_run_summary(client: Any, run_id: str, tmp_dir: str) -> Optional[Dict[str, Any]]:
    """Download and parse run_summary.json from MLflow artifact storage.

    Returns the parsed dict, or ``None`` when the artifact is absent or
    cannot be downloaded (e.g. artifact store not reachable).

    Parameters
    ----------
    client:
        An ``mlflow.tracking.MlflowClient`` instance.
    run_id:
        The MLflow run ID whose artifacts to search.
    tmp_dir:
        Temporary local directory to download the artifact into.
    """
    try:
        local_path = client.download_artifacts(run_id, "summary/run_summary.json", tmp_dir)
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _flatten_run(run: Any, run_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a single MLflow run object into a flat dict row.

    Parameters
    ----------
    run:
        An ``mlflow.entities.Run`` object returned by ``MlflowClient.search_runs``.
    run_summary:
        Parsed ``run_summary.json`` content, or ``None`` if unavailable.

    Returns
    -------
    dict
        Flat key→value mapping ready for ``csv.DictWriter``.
    """
    row: Dict[str, Any] = {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name or "",
        "status": run.info.status,
        "start_time": (
            str(run.info.start_time)
            if run.info.start_time is None
            else _ms_to_iso(run.info.start_time)
        ),
    }

    # Structured fields from our run_summary.json artifact
    for field in (
        "trial_id", "seed", "final_metric", "best_metric",
        "convergence_step", "wall_time", "status", "traceback",
    ):
        row[f"summary.{field}"] = (
            run_summary.get(field) if run_summary else ""
        )

    # All logged MLflow params (prefixed to avoid collision)
    for k, v in run.data.params.items():
        row[f"param.{k}"] = v

    # All logged MLflow metrics — last recorded value only
    for k, v in run.data.metrics.items():
        row[f"metric.{k}"] = v

    return row


def _ms_to_iso(ms: int) -> str:
    """Convert a UNIX millisecond timestamp to an ISO-8601 string."""
    import datetime
    return datetime.datetime.fromtimestamp(ms / 1000.0).isoformat()


def _write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Write a list of flat dicts to a CSV file.

    Collects all keys across all rows to build the header, so rows with
    different param/metric keys are handled gracefully (missing values → "").

    Parameters
    ----------
    rows:
        List of flat dicts (one per run).
    output_path:
        Destination CSV file path.
    """
    if not rows:
        print(f"[aggregate_runs] No rows to write → {output_path}")
        return

    # Build a stable, ordered set of all column names
    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Fill missing columns with empty string
            writer.writerow({k: row.get(k, "") for k in all_keys})

    print(f"[aggregate_runs] Wrote {len(rows)} rows → {output_path}")


def _write_grouped_csv(
    rows: List[Dict[str, Any]],
    group_by_param: str,
    output_path: str,
) -> None:
    """Group rows by a param key and write mean/std of final_metric.

    The grouped output is written to ``<output_path>.grouped.csv`` (the
    ``.grouped`` suffix is always appended, independent of the original
    extension).

    Parameters
    ----------
    rows:
        All per-run rows already written to the flat CSV.
    group_by_param:
        Column name as it appears in the flat CSV, e.g.
        ``"param.method.learning_rate"``.  If the ``"param."`` prefix is
        absent it is added automatically.
    output_path:
        Base path for the grouped CSV; ``.grouped.csv`` is appended.
    """
    # Normalise the group-by key to include the "param." prefix
    if not group_by_param.startswith("param."):
        group_by_param = f"param.{group_by_param}"

    groups: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        group_key = str(row.get(group_by_param, ""))
        fm = row.get("summary.final_metric", "")
        try:
            groups[group_key].append(float(fm))
        except (TypeError, ValueError):
            pass  # skip rows without a valid final_metric

    grouped_rows: List[Dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        n = len(values)
        mean = statistics.mean(values) if n > 0 else None
        std = statistics.stdev(values) if n > 1 else None
        grouped_rows.append(
            {
                group_by_param: key,
                "n_runs": n,
                "mean_final_metric": round(mean, 6) if mean is not None else "",
                "std_final_metric": round(std, 6) if std is not None else "",
            }
        )

    grouped_path = output_path + ".grouped.csv"
    _write_csv(grouped_rows, grouped_path)


def aggregate_runs(
    experiment: str,
    tracking_uri: str = "mlruns",
    output: str = "runs_summary.csv",
    group_by: Optional[str] = None,
    status_filter: str = "FINISHED",
) -> None:
    """Fetch all MLflow runs from an experiment and write a summary CSV.

    Parameters
    ----------
    experiment:
        MLflow experiment name (e.g. ``"research-template"``).
    tracking_uri:
        MLflow tracking URI; defaults to the local ``mlruns/`` directory.
    output:
        Path for the output CSV file.
    group_by:
        Optional param name to group rows by (enables grouped CSV output).
    status_filter:
        One of ``"FINISHED"``, ``"FAILED"``, or ``"ALL"`` (case-insensitive).
        Controls which runs are included.
    """
    import mlflow
    import mlflow.tracking

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Locate experiment
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        sys.exit(
            f"[aggregate_runs] ERROR: experiment '{experiment}' not found at "
            f"tracking_uri='{tracking_uri}'.\n"
            f"Available experiments: "
            + ", ".join(
                e.name for e in client.search_experiments()
            )
        )

    # Fetch all runs (no server-side filter; we filter locally for clarity)
    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        max_results=50000,
    )

    status_filter_upper = status_filter.upper()
    if status_filter_upper != "ALL":
        all_runs = [r for r in all_runs if r.info.status == status_filter_upper]

    print(
        f"[aggregate_runs] Experiment '{experiment}': "
        f"{len(all_runs)} runs after filter '{status_filter_upper}'"
    )

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for run in all_runs:
            run_summary = _download_run_summary(client, run.info.run_id, tmp_dir)
            rows.append(_flatten_run(run, run_summary))

    _write_csv(rows, output)

    if group_by and rows:
        _write_grouped_csv(rows, group_by, output)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MLflow experiment runs into a summary CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name (e.g. 'research-template').",
    )
    parser.add_argument(
        "--tracking-uri",
        default="mlruns",
        help="MLflow tracking URI.  Defaults to local 'mlruns/'.",
    )
    parser.add_argument(
        "--output",
        default="runs_summary.csv",
        help="Output CSV file path.  Defaults to 'runs_summary.csv'.",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        metavar="PARAM_KEY",
        help=(
            "Param key to group rows by for multi-seed aggregation, e.g. "
            "'method.learning_rate'.  Produces an additional '<output>.grouped.csv'."
        ),
    )
    parser.add_argument(
        "--status-filter",
        default="FINISHED",
        choices=["FINISHED", "FAILED", "ALL"],
        help="Include only runs with this MLflow status.  Defaults to FINISHED.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    aggregate_runs(
        experiment=args.experiment,
        tracking_uri=args.tracking_uri,
        output=args.output,
        group_by=args.group_by,
        status_filter=args.status_filter,
    )
