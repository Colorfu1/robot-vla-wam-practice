"""Select an official StackCube checkpoint from fixed development evaluations."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

_CHECKPOINT_RE = re.compile(r"^(?P<step>[0-9]+)\.pt$")


def _mean_metric(payload: dict[str, Any], key: str) -> float:
    evaluation = payload.get("evaluation", {})
    metric = evaluation.get(key, {})
    value = metric.get("mean", metric.get("rate"))
    if value is None:
        raise ValueError(f"evaluation payload has no evaluation.{key}.mean")
    return float(value)


def _candidates(train_run: Path, evaluation_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    checkpoint_dir = train_run / "checkpoints"
    for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
        match = _CHECKPOINT_RE.match(checkpoint.name)
        if match is None:
            continue
        step = int(match.group("step"))
        metrics_path = evaluation_root / f"checkpoint_{step:06d}" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"missing exact development evaluation for checkpoint {checkpoint}: {metrics_path}"
            )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "metrics": str(metrics_path),
                "success_once": _mean_metric(payload, "success_once"),
                "success_at_end": _mean_metric(payload, "success_at_end"),
            }
        )
    if not candidates:
        raise FileNotFoundError(f"no numeric checkpoints found under {checkpoint_dir}")
    return candidates


def select_checkpoint(
    *,
    train_run: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
    selection_source: str = "exact fixed CPU development evaluation",
) -> dict[str, Any]:
    train_run = Path(train_run).expanduser().resolve()
    evaluation_root = Path(evaluation_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    candidates = _candidates(train_run, evaluation_root)
    ranking = sorted(
        candidates,
        key=lambda item: (
            -item["success_once"],
            -item["success_at_end"],
            item["step"],
        ),
    )
    selected = ranking[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_metrics = Path(selected["metrics"])
    shutil.copyfile(selected_metrics, output_dir / "metrics.json")
    payload = {
        "schema_version": 1,
        "selection_protocol": {
            "evaluation_split": "development",
            "primary_metric": "success_once",
            "tie_break": ["success_at_end", "earliest_checkpoint"],
            "selection_source": selection_source,
        },
        "selected": selected,
        "ranking": ranking,
        "selected_metrics_path": str(output_dir / "metrics.json"),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-source",
        default="exact fixed CPU development evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = select_checkpoint(
        train_run=args.train_run,
        evaluation_root=args.evaluation_root,
        output_dir=args.output_dir,
        selection_source=args.selection_source,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
