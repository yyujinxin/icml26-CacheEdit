#!/usr/bin/env python3
"""Summarize quantization-error probe data from a CacheEdit timing report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _score(row: dict[str, Any]) -> tuple:
    rmse = _as_float(row.get("rmse"))
    metadata_ratio = _as_float(row.get("metadata_over_original_ratio"))
    return (
        rmse if rmse is not None else 999.0,
        metadata_ratio if metadata_ratio is not None else 999.0,
    )


def build_rows(timings: dict[str, Any]) -> list[dict[str, Any]]:
    summary = (timings.get("compression") or {}).get("summary") or {}
    by_quant = summary.get("quant_error_probe_by_quantization") or {}
    rows = []
    for quantization, values in by_quant.items():
        sampled_numel = int(values.get("sampled_numel", 0) or 0)
        original_bytes = int(values.get("original_bytes", 0) or 0)
        metadata_bytes = int(values.get("metadata_bytes", 0) or 0)
        outlier_extra_metadata_bytes = int(
            values.get("outlier_extra_metadata_bytes", 0) or 0
        )
        row = {
            "quantization": quantization,
            "record_count": values.get("record_count"),
            "skipped_count": values.get("skipped_count"),
            "sampled_numel": sampled_numel,
            "original_numel": values.get("original_numel"),
            "original_mib": original_bytes / 1024.0 / 1024.0,
            "metadata_mib": metadata_bytes / 1024.0 / 1024.0,
            "outlier_extra_metadata_mib": (
                outlier_extra_metadata_bytes / 1024.0 / 1024.0
            ),
            "metadata_over_original_ratio": values.get(
                "metadata_over_original_ratio"
            ),
            "rmse": values.get("rmse"),
            "mae": values.get("mae"),
            "max_abs": values.get("max_abs"),
            "relative_rmse": values.get("relative_rmse"),
        }
        rows.append(row)
    rows.sort(key=_score)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    timings = _read_json(args.timings)
    rows = build_rows(timings)
    if not rows:
        raise SystemExit(
            "No quant_error_probe_by_quantization data found in timings report"
        )

    csv_output = args.csv_output or args.timings.with_name(
        "quant_error_summary.csv"
    )
    json_output = args.json_output or args.timings.with_name(
        "quant_error_summary.json"
    )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_output, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"quant error csv -> {csv_output}")
    print(f"quant error json -> {json_output}")
    best = rows[0]
    print(
        "best quantization by rmse: "
        f"{best['quantization']} "
        f"rmse={best['rmse']} "
        f"relative_rmse={best['relative_rmse']} "
        f"metadata_ratio={best['metadata_over_original_ratio']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
