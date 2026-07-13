#!/usr/bin/env python3
"""Summarize full-dataset no-cache/cache/compressed experiments to CSV/JSON/XLSX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PAIR_NAMES = (
    "cache_vs_baseline",
    "compressed_vs_baseline",
    "compressed_vs_cache",
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, str):
        if value == "Infinity":
            return float("inf")
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _sum(values: list[float]) -> float:
    return float(sum(float(v) for v in values if math.isfinite(float(v))))


def _flatten_times(timings: dict[str, Any]) -> list[float]:
    flat = []
    for values in (timings.get("per_image_round_times") or {}).values():
        if isinstance(values, list):
            flat.extend(float(v) for v in values if isinstance(v, (int, float)))
    return flat


def _mode_timing_row(name: str, timings: dict[str, Any]) -> dict[str, Any]:
    per_image = timings.get("per_image_round_times") or {}
    flat = _flatten_times(timings)
    return {
        "mode": name,
        "complete": timings.get("complete"),
        "num_images": timings.get("num_images", len(per_image)),
        "num_images_with_timing": sum(1 for v in per_image.values() if v),
        "num_rounds_with_timing": len(flat),
        "avg_round_time_s": timings.get("avg_round_time") or _mean(flat),
        "total_round_time_s": _sum(flat),
        "min_round_time_s": min(flat) if flat else None,
        "max_round_time_s": max(flat) if flat else None,
    }


def _compression_summary(timings: dict[str, Any]) -> dict[str, Any]:
    return ((timings.get("compression") or {}).get("summary") or {})


def _quality_summaries(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for pair in metrics.get("pairs", []):
        summary = pair.get("summary") or {}
        name = summary.get("name")
        if name:
            result[str(name)] = summary
    return result


def _record_image_id(filename: str) -> str:
    match = re.match(r"([^_]+)_r\d+_", filename)
    if match:
        return match.group(1)
    return filename.split("_r", 1)[0]


def _quality_record_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair in metrics.get("pairs", []):
        pair_name = (pair.get("summary") or {}).get("name")
        for record in pair.get("records") or []:
            row = {"pair": pair_name, "image_idx": _record_image_id(record["file"])}
            row.update(record)
            rows.append(row)
    return rows


def _per_image_quality_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in _quality_record_rows(metrics):
        key = (str(row["pair"]), str(row["image_idx"]))
        bucket = buckets.setdefault(key, {"psnr": [], "ssim": [], "lpips": []})
        for metric in ("psnr", "ssim", "lpips"):
            value = _safe_float(row.get(metric))
            if value is not None:
                bucket[metric].append(value)

    rows = []
    for (pair, image_idx), values in sorted(buckets.items()):
        rows.append(
            {
                "pair": pair,
                "image_idx": image_idx,
                "num_rounds": max(len(v) for v in values.values()) if values else 0,
                "mean_psnr": _mean(values["psnr"]),
                "mean_ssim": _mean(values["ssim"]),
                "mean_lpips": _mean(values["lpips"]),
            }
        )
    return rows


def _round_time_rows(mode_timings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    image_ids = sorted(
        {
            str(image_id)
            for timings in mode_timings.values()
            for image_id in (timings.get("per_image_round_times") or {})
        }
    )
    rows = []
    for image_id in image_ids:
        max_rounds = 0
        for timings in mode_timings.values():
            max_rounds = max(
                max_rounds,
                len((timings.get("per_image_round_times") or {}).get(image_id) or []),
            )
        for round_idx in range(max_rounds):
            row = {"image_idx": image_id, "round": round_idx}
            for mode, timings in mode_timings.items():
                values = (timings.get("per_image_round_times") or {}).get(image_id) or []
                row[f"{mode}_round_time_s"] = (
                    values[round_idx] if round_idx < len(values) else None
                )
            rows.append(row)
    return rows


def _mode_summary_rows(
    mode_timings: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [_mode_timing_row(mode, timings) for mode, timings in mode_timings.items()]
    quality = _quality_summaries(metrics)
    cache_only_comp = _compression_summary(mode_timings["cache_only"])
    compressed_comp = _compression_summary(mode_timings["cache_compressed"])

    compressed_original_mib = _safe_float(compressed_comp.get("original_mib")) or 0.0
    compressed_fallback_mib = (
        _safe_float(compressed_comp.get("uncompressed_fallback_mib")) or 0.0
    )
    cache_only_recorded_mib = _safe_float(cache_only_comp.get("original_mib"))
    cache_only_space_mib = cache_only_recorded_mib
    cache_only_space_source = "recorded"
    if not cache_only_space_mib:
        cache_only_space_mib = compressed_original_mib + compressed_fallback_mib
        cache_only_space_source = "estimated_from_compressed_original_plus_fallback"

    compressed_total_mib = _safe_float(compressed_comp.get("compressed_total_mib")) or 0.0
    compressed_actual_mib = compressed_total_mib + compressed_fallback_mib

    for row in rows:
        mode = row["mode"]
        if mode == "cache_only":
            row.update(
                {
                    "cache_space_mib": cache_only_space_mib,
                    "cache_space_source": cache_only_space_source,
                    "compression_enabled": False,
                }
            )
            pair = quality.get("cache_vs_baseline", {})
        elif mode == "cache_compressed":
            row.update(
                {
                    "cache_space_mib": compressed_actual_mib,
                    "cache_space_source": "compressed_total_plus_uncompressed_fallback",
                    "compression_enabled": compressed_comp.get("enabled"),
                    "compression_codec": compressed_comp.get("codec"),
                    "compression_rc_mode": compressed_comp.get("rc_mode"),
                    "compression_const_qp": compressed_comp.get("const_qp"),
                    "compression_const_qp_intra": compressed_comp.get(
                        "const_qp_intra"
                    ),
                    "compression_const_qp_inter_p": compressed_comp.get(
                        "const_qp_inter_p"
                    ),
                    "compression_const_qp_inter_b": compressed_comp.get(
                        "const_qp_inter_b"
                    ),
                    "compression_bitrate_mbps": compressed_comp.get("bitrate_mbps"),
                    "compression_bitrate_max_multiplier": compressed_comp.get(
                        "bitrate_max_multiplier"
                    ),
                    "compression_codec_preset": compressed_comp.get("codec_preset"),
                    "compression_codec_tuning": compressed_comp.get("codec_tuning"),
                    "compression_codec_spatial_aq": compressed_comp.get(
                        "codec_spatial_aq"
                    ),
                    "compression_codec_temporal_aq": compressed_comp.get(
                        "codec_temporal_aq"
                    ),
                    "compression_codec_target_quality": compressed_comp.get(
                        "codec_target_quality"
                    ),
                    "compression_gop_length": compressed_comp.get(
                        "configured_gop_length"
                    ),
                    "compression_gop_start_layer": compressed_comp.get(
                        "configured_gop_start_layer"
                    ),
                    "compression_frame_interval_p": compressed_comp.get(
                        "configured_frame_interval_p"
                    ),
                    "compression_quant_group_size": compressed_comp.get(
                        "quant_group_size"
                    ),
                    "compression_quant_outlier_ratio": compressed_comp.get(
                        "quant_outlier_ratio"
                    ),
                    "compression_codec_residual_ratio": compressed_comp.get(
                        "codec_residual_ratio"
                    ),
                    "compression_quality_steps": compressed_comp.get(
                        "quality_steps"
                    ),
                    "compression_quality_streams": json.dumps(
                        compressed_comp.get("quality_streams") or [],
                        ensure_ascii=False,
                    ),
                    "compression_success_count_by_profile": json.dumps(
                        compressed_comp.get("success_count_by_profile") or {},
                        ensure_ascii=False,
                    ),
                    "compression_success_count": compressed_comp.get("success_count"),
                    "compression_failure_count": compressed_comp.get("failure_count"),
                    "compressed_payload_mib": compressed_comp.get(
                        "compressed_payload_mib"
                    ),
                    "compressed_auxiliary_mib": compressed_comp.get(
                        "compressed_auxiliary_mib"
                    ),
                    "compressed_total_mib": compressed_comp.get("compressed_total_mib"),
                    "uncompressed_fallback_mib": compressed_comp.get(
                        "uncompressed_fallback_mib"
                    ),
                    "original_mib": compressed_comp.get("original_mib"),
                    "payload_compression_ratio": compressed_comp.get(
                        "payload_compression_ratio"
                    ),
                    "total_compression_ratio_success_only": compressed_comp.get(
                        "total_compression_ratio"
                    ),
                    "actual_compression_ratio_including_fallback": (
                        (compressed_original_mib + compressed_fallback_mib)
                        / compressed_actual_mib
                        if compressed_actual_mib > 0
                        else None
                    ),
                    "total_compression_time_s": compressed_comp.get(
                        "total_compression_time_s"
                    ),
                    "total_decompression_time_s": compressed_comp.get(
                        "total_decompression_time_s"
                    ),
                }
            )
            pair = quality.get("compressed_vs_baseline", {})
            cache_pair = quality.get("compressed_vs_cache", {})
            row.update(
                {
                    "vs_cache_psnr": cache_pair.get("mean_psnr"),
                    "vs_cache_ssim": cache_pair.get("mean_ssim"),
                    "vs_cache_lpips": cache_pair.get("mean_lpips"),
                }
            )
        else:
            row.update({"cache_space_mib": None, "cache_space_source": "not_applicable"})
            pair = {}
        row.update(
            {
                "vs_baseline_psnr": pair.get("mean_psnr"),
                "vs_baseline_ssim": pair.get("mean_ssim"),
                "vs_baseline_lpips": pair.get("mean_lpips"),
                "quality_num_common": pair.get("num_common"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cell_ref(row_idx: int, col_idx: int) -> str:
    letters = ""
    col = col_idx
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row_idx}"


def _sheet_xml(rows: list[dict[str, Any]]) -> str:
    if not rows:
        rows = [{"empty": ""}]
    headers = sorted({key for row in rows for key in row})
    all_rows = [headers] + [[row.get(h) for h in headers] for row in rows]
    xml_rows = []
    for r_idx, row in enumerate(all_rows, 1):
        cells = []
        for c_idx, value in enumerate(row, 1):
            ref = _cell_ref(r_idx, c_idx)
            if value is None:
                cells.append(f'<c r="{ref}"/>')
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def _write_xlsx(path: Path, sheets: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_items = list(sheets.items())
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx, _ in enumerate(sheet_items, 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for idx, (name, _) in enumerate(sheet_items, 1):
        safe_name = escape(name[:31])
        workbook_sheets.append(
            f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook_rels.append("</Relationships>")
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        for idx, (_, rows) in enumerate(sheet_items, 1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cache_quality_metrics_28step_full_dataset"),
    )
    parser.add_argument("--baseline-name", default="baseline_no_cache")
    parser.add_argument("--cache-only-name", default="cache_only")
    parser.add_argument("--compressed-name", default="cache_compressed")
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()

    report_dir = args.report_dir or (args.output_root / "paper_report")
    metrics_path = args.metrics_json or (args.output_root / "quality_metrics.json")

    mode_paths = {
        "no_cache": args.output_root / args.baseline_name,
        "cache_only": args.output_root / args.cache_only_name,
        "cache_compressed": args.output_root / args.compressed_name,
    }
    mode_timings = {
        mode: _read_json(path / "timings.json") for mode, path in mode_paths.items()
    }
    metrics = _read_json(metrics_path)

    mode_summary = _mode_summary_rows(mode_timings, metrics)
    quality_records = _quality_record_rows(metrics)
    per_image_quality = _per_image_quality_rows(metrics)
    round_times = _round_time_rows(mode_timings)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "mode_summary.csv", mode_summary)
    _write_csv(report_dir / "quality_records.csv", quality_records)
    _write_csv(report_dir / "per_image_quality.csv", per_image_quality)
    _write_csv(report_dir / "round_times.csv", round_times)
    with open(report_dir / "full_dataset_report.json", "w") as f:
        json.dump(
            {
                "mode_summary": mode_summary,
                "quality_records": quality_records,
                "per_image_quality": per_image_quality,
                "round_times": round_times,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    _write_xlsx(
        report_dir / "full_dataset_report.xlsx",
        {
            "mode_summary": mode_summary,
            "per_image_quality": per_image_quality,
            "quality_records": quality_records,
            "round_times": round_times,
        },
    )

    print(f"report dir -> {report_dir}")
    print(f"xlsx -> {report_dir / 'full_dataset_report.xlsx'}")
    for row in mode_summary:
        print(
            f"{row['mode']}: avg_round={row.get('avg_round_time_s')}s "
            f"cache_space={row.get('cache_space_mib')}MiB "
            f"PSNR={row.get('vs_baseline_psnr')} SSIM={row.get('vs_baseline_ssim')} "
            f"LPIPS={row.get('vs_baseline_lpips')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
