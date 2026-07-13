#!/usr/bin/env python3
"""Summarize compression parameter sweep metrics and timing reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RUN_RE = re.compile(
    r"(?:codec_(?P<codec>[a-z0-9]+))?"
    r"(?:_br(?P<br>[0-9]+(?:p[0-9]+)?))?"
    r".*?gop(?P<gop>\d+)_p(?P<p>\d+)_qg(?P<qg>-?\d+)"
    r"(?:_o(?P<outlier>[0-9]+(?:p[0-9]+)?))?"
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _pair_summary(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    for pair in metrics.get("pairs", []):
        summary = pair.get("summary", {})
        if summary.get("name") == name:
            return summary
    return {}


def _parse_run_name(name: str) -> dict[str, Any]:
    match = RUN_RE.search(name)
    if not match:
        return {
            "codec": None,
            "bitrate_mbps": None,
            "gop_length": None,
            "frame_interval_p": None,
            "quant_group_size": None,
            "quant_outlier_ratio": None,
        }
    bitrate = match.group("br")
    outlier = match.group("outlier")
    return {
        "codec": match.group("codec"),
        "bitrate_mbps": (
            float(bitrate.replace("p", ".")) if bitrate is not None else None
        ),
        "gop_length": int(match.group("gop")),
        "frame_interval_p": int(match.group("p")),
        "quant_group_size": int(match.group("qg")),
        "quant_outlier_ratio": (
            float(outlier.replace("p", ".")) if outlier is not None else None
        ),
    }


def _score(row: dict[str, Any]) -> tuple:
    psnr = row.get("compressed_vs_cache_psnr")
    ssim = row.get("compressed_vs_cache_ssim")
    lpips = row.get("compressed_vs_cache_lpips")
    ratio = row.get("total_compression_ratio")
    return (
        float(psnr) if isinstance(psnr, (int, float)) else -1.0,
        float(ssim) if isinstance(ssim, (int, float)) else -1.0,
        -(float(lpips) if isinstance(lpips, (int, float)) else 999.0),
        float(ratio) if isinstance(ratio, (int, float)) else -1.0,
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _passes_quality_gate(
    row: dict[str, Any],
    *,
    min_psnr: float,
    min_ssim: float,
    max_lpips: float,
    max_peak_reserved_gib: float | None,
) -> bool:
    psnr = _as_float(row.get("compressed_vs_cache_psnr"))
    ssim = _as_float(row.get("compressed_vs_cache_ssim"))
    lpips = _as_float(row.get("compressed_vs_cache_lpips"))
    peak_reserved = _as_float(row.get("max_cuda_peak_reserved_gib"))
    failure_count = row.get("failure_count")
    if failure_count not in (0, None):
        return False
    if psnr is None or ssim is None or lpips is None:
        return False
    if (
        max_peak_reserved_gib is not None
        and peak_reserved is not None
        and peak_reserved > max_peak_reserved_gib
    ):
        return False
    return psnr >= min_psnr and ssim >= min_ssim and lpips <= max_lpips


def _ratio_score(row: dict[str, Any]) -> tuple:
    ratio = _as_float(row.get("total_compression_ratio"))
    payload_ratio = _as_float(row.get("payload_compression_ratio"))
    peak_reserved = _as_float(row.get("max_cuda_peak_reserved_gib"))
    lpips = _as_float(row.get("compressed_vs_cache_lpips"))
    psnr = _as_float(row.get("compressed_vs_cache_psnr"))
    return (
        ratio if ratio is not None else -1.0,
        payload_ratio if payload_ratio is not None else -1.0,
        -(peak_reserved if peak_reserved is not None else 999.0),
        -(lpips if lpips is not None else 999.0),
        psnr if psnr is not None else -1.0,
    )


def build_rows(output_root: Path, metrics_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(metrics_dir.glob("*.json")):
        run_name = metrics_path.stem
        metrics = _read_json(metrics_path)
        timings_path = output_root / run_name / "timings.json"
        timings = _read_json(timings_path) if timings_path.is_file() else {}
        comp = (timings.get("compression") or {}).get("summary") or {}
        cuda_memory = timings.get("cuda_memory") or {}
        cache_pair = _pair_summary(metrics, "compressed_vs_cache")
        baseline_pair = _pair_summary(metrics, "compressed_vs_baseline")
        parsed = _parse_run_name(run_name)

        row = {
            "run_name": run_name,
            **parsed,
            "codec": comp.get("codec", parsed.get("codec")),
            "bitrate_mbps": comp.get(
                "bitrate_mbps",
                parsed.get("bitrate_mbps"),
            ),
            "gop_length": comp.get(
                "configured_gop_length",
                parsed.get("gop_length"),
            ),
            "gop_start_layer": comp.get("configured_gop_start_layer", 0),
            "frame_interval_p": comp.get(
                "configured_frame_interval_p",
                parsed.get("frame_interval_p"),
            ),
            "quant_group_size": comp.get(
                "quant_group_size",
                parsed.get("quant_group_size"),
            ),
            "num_common": cache_pair.get("num_common"),
            "compressed_vs_cache_psnr": cache_pair.get("mean_psnr"),
            "compressed_vs_cache_ssim": cache_pair.get("mean_ssim"),
            "compressed_vs_cache_lpips": cache_pair.get("mean_lpips"),
            "compressed_vs_baseline_psnr": baseline_pair.get("mean_psnr"),
            "compressed_vs_baseline_ssim": baseline_pair.get("mean_ssim"),
            "compressed_vs_baseline_lpips": baseline_pair.get("mean_lpips"),
            "success_count": comp.get("success_count"),
            "failure_count": comp.get("failure_count"),
            "rc_mode": comp.get("rc_mode"),
            "const_qp": comp.get("const_qp"),
            "const_qp_intra": comp.get("const_qp_intra"),
            "const_qp_inter_p": comp.get("const_qp_inter_p"),
            "const_qp_inter_b": comp.get("const_qp_inter_b"),
            "bitrate_max_multiplier": comp.get("bitrate_max_multiplier"),
            "codec_preset": comp.get("codec_preset"),
            "codec_tuning": comp.get("codec_tuning"),
            "codec_spatial_aq": comp.get("codec_spatial_aq"),
            "codec_temporal_aq": comp.get("codec_temporal_aq"),
            "codec_target_quality": comp.get("codec_target_quality"),
            "quality_steps": json.dumps(
                comp.get("quality_steps") or [],
                ensure_ascii=False,
            ),
            "quality_streams": json.dumps(
                comp.get("quality_streams") or [],
                ensure_ascii=False,
            ),
            "quality_codec": comp.get("quality_codec"),
            "quality_rc_mode": comp.get("quality_rc_mode"),
            "quality_const_qp": comp.get("quality_const_qp"),
            "quality_bitrate_mbps": comp.get("quality_bitrate_mbps"),
            "success_count_by_profile": json.dumps(
                comp.get("success_count_by_profile") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "quant_outlier_ratio": comp.get(
                "quant_outlier_ratio",
                parsed.get("quant_outlier_ratio"),
            ),
            "codec_residual_ratio": comp.get("codec_residual_ratio"),
            "avg_round_time_s": timings.get("avg_round_time"),
            "payload_compression_ratio": comp.get("payload_compression_ratio"),
            "total_compression_ratio": comp.get("total_compression_ratio"),
            "compressed_payload_mib": comp.get("compressed_payload_mib"),
            "compressed_auxiliary_mib": comp.get("compressed_auxiliary_mib"),
            "compressed_total_mib": comp.get("compressed_total_mib"),
            "total_compression_time_s": comp.get("total_compression_time_s"),
            "total_decompression_time_s": comp.get("total_decompression_time_s"),
            "max_cuda_peak_allocated_gib": cuda_memory.get(
                "max_peak_allocated_gib"
            ),
            "max_cuda_peak_reserved_gib": cuda_memory.get(
                "max_peak_reserved_gib"
            ),
            "cuda_memory_source": cuda_memory.get("source"),
            "success_count_by_quantization": json.dumps(
                comp.get("success_count_by_quantization") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "success_count_by_quantization_variant": json.dumps(
                comp.get("success_count_by_quantization_variant") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        rows.append(row)
    rows.sort(key=_score, reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs/compression_quant_sweep_28step"),
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Default: <output-root>/metrics",
    )
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--recommendation-output", type=Path, default=None)
    parser.add_argument(
        "--min-psnr",
        type=float,
        default=41.0,
        help="Quality gate for compressed_vs_cache PSNR.",
    )
    parser.add_argument(
        "--min-ssim",
        type=float,
        default=0.994,
        help="Quality gate for compressed_vs_cache SSIM.",
    )
    parser.add_argument(
        "--max-lpips",
        type=float,
        default=0.004,
        help="Quality gate for compressed_vs_cache LPIPS.",
    )
    parser.add_argument(
        "--max-peak-reserved-gib",
        type=float,
        default=None,
        help=(
            "Optional memory gate based on max torch.cuda reserved memory "
            "per GPU after runtime peak reset."
        ),
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir or (args.output_root / "metrics")
    rows = build_rows(args.output_root, metrics_dir)
    if not rows:
        raise SystemExit(f"No metrics JSON files found under {metrics_dir}")

    for row in rows:
        row["passes_quality_gate"] = _passes_quality_gate(
            row,
            min_psnr=args.min_psnr,
            min_ssim=args.min_ssim,
            max_lpips=args.max_lpips,
            max_peak_reserved_gib=args.max_peak_reserved_gib,
        )

    csv_output = args.csv_output or (args.output_root / "sweep_summary.csv")
    json_output = args.json_output or (args.output_root / "sweep_summary.json")
    recommendation_output = (
        args.recommendation_output
        or (args.output_root / "recommended_config.json")
    )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    recommendation_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_output, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    best_quality = rows[0]
    gated_rows = [row for row in rows if row["passes_quality_gate"]]
    best_ratio_gated = max(gated_rows, key=_ratio_score) if gated_rows else None
    recommendation = {
        "quality_gate": {
            "min_psnr": args.min_psnr,
            "min_ssim": args.min_ssim,
            "max_lpips": args.max_lpips,
            "max_peak_reserved_gib": args.max_peak_reserved_gib,
            "requires_failure_count": 0,
        },
        "best_quality": best_quality,
        "best_ratio_under_quality_gate": best_ratio_gated,
        "recommended_for_quality": {
            "compression_codec": best_quality.get("codec"),
            "compression_bitrate": best_quality.get("bitrate_mbps"),
            "compression_rc_mode": best_quality.get("rc_mode"),
            "compression_const_qp": best_quality.get("const_qp"),
            "compression_const_qp_intra": best_quality.get("const_qp_intra"),
            "compression_const_qp_inter_p": best_quality.get(
                "const_qp_inter_p"
            ),
            "compression_const_qp_inter_b": best_quality.get(
                "const_qp_inter_b"
            ),
            "compression_bitrate_max_multiplier": best_quality.get(
                "bitrate_max_multiplier"
            ),
            "compression_codec_preset": best_quality.get("codec_preset"),
            "compression_codec_tuning": best_quality.get("codec_tuning"),
            "compression_codec_spatial_aq": best_quality.get(
                "codec_spatial_aq"
            ),
            "compression_codec_temporal_aq": best_quality.get(
                "codec_temporal_aq"
            ),
            "compression_codec_target_quality": best_quality.get(
                "codec_target_quality"
            ),
            "compression_quality_steps": best_quality.get("quality_steps"),
            "compression_quality_streams": best_quality.get("quality_streams"),
            "compression_quality_codec": best_quality.get("quality_codec"),
            "compression_quality_rc_mode": best_quality.get("quality_rc_mode"),
            "compression_quality_const_qp": best_quality.get(
                "quality_const_qp"
            ),
            "compression_gop_length": best_quality.get("gop_length"),
            "compression_gop_start_layer": best_quality.get("gop_start_layer"),
            "compression_frame_interval_p": best_quality.get("frame_interval_p"),
            "compression_quant_group_size": best_quality.get(
                "quant_group_size"
            ),
            "compression_quant_outlier_ratio": best_quality.get(
                "quant_outlier_ratio"
            ),
            "compression_codec_residual_ratio": best_quality.get(
                "codec_residual_ratio"
            ),
        },
        "recommended_for_ratio_under_quality_gate": (
            None
            if best_ratio_gated is None
            else {
                "compression_codec": best_ratio_gated.get("codec"),
                "compression_bitrate": best_ratio_gated.get("bitrate_mbps"),
                "compression_rc_mode": best_ratio_gated.get("rc_mode"),
                "compression_const_qp": best_ratio_gated.get("const_qp"),
                "compression_const_qp_intra": best_ratio_gated.get(
                    "const_qp_intra"
                ),
                "compression_const_qp_inter_p": best_ratio_gated.get(
                    "const_qp_inter_p"
                ),
                "compression_const_qp_inter_b": best_ratio_gated.get(
                    "const_qp_inter_b"
                ),
                "compression_bitrate_max_multiplier": best_ratio_gated.get(
                    "bitrate_max_multiplier"
                ),
                "compression_codec_preset": best_ratio_gated.get(
                    "codec_preset"
                ),
                "compression_codec_tuning": best_ratio_gated.get(
                    "codec_tuning"
                ),
                "compression_codec_spatial_aq": best_ratio_gated.get(
                    "codec_spatial_aq"
                ),
                "compression_codec_temporal_aq": best_ratio_gated.get(
                    "codec_temporal_aq"
                ),
                "compression_codec_target_quality": best_ratio_gated.get(
                    "codec_target_quality"
                ),
                "compression_quality_steps": best_ratio_gated.get(
                    "quality_steps"
                ),
                "compression_quality_streams": best_ratio_gated.get(
                    "quality_streams"
                ),
                "compression_quality_codec": best_ratio_gated.get(
                    "quality_codec"
                ),
                "compression_quality_rc_mode": best_ratio_gated.get(
                    "quality_rc_mode"
                ),
                "compression_quality_const_qp": best_ratio_gated.get(
                    "quality_const_qp"
                ),
                "compression_gop_length": best_ratio_gated.get("gop_length"),
                "compression_gop_start_layer": best_ratio_gated.get(
                    "gop_start_layer"
                ),
                "compression_frame_interval_p": best_ratio_gated.get(
                    "frame_interval_p"
                ),
                "compression_quant_group_size": best_ratio_gated.get(
                    "quant_group_size"
                ),
                "compression_quant_outlier_ratio": best_ratio_gated.get(
                    "quant_outlier_ratio"
                ),
                "compression_codec_residual_ratio": best_ratio_gated.get(
                    "codec_residual_ratio"
                ),
            }
        ),
    }
    with open(recommendation_output, "w") as f:
        json.dump(recommendation, f, indent=2, ensure_ascii=False)

    print(f"summary csv -> {csv_output}")
    print(f"summary json -> {json_output}")
    print(f"recommendation json -> {recommendation_output}")
    print(
        "best by compressed_vs_cache quality: "
        f"{best_quality['run_name']} "
        f"PSNR={best_quality['compressed_vs_cache_psnr']} "
        f"SSIM={best_quality['compressed_vs_cache_ssim']} "
        f"LPIPS={best_quality['compressed_vs_cache_lpips']} "
        f"total_ratio={best_quality['total_compression_ratio']}"
    )
    memory_gate = (
        "none"
        if args.max_peak_reserved_gib is None
        else f"<={args.max_peak_reserved_gib}GiB"
    )
    print(
        "quality gate: "
        f"PSNR>={args.min_psnr}, SSIM>={args.min_ssim}, "
        f"LPIPS<={args.max_lpips}, "
        f"max_peak_reserved_gib={memory_gate}, "
        "failures=0"
    )
    if best_ratio_gated is None:
        print("best ratio under quality gate: none")
    else:
        print(
            "best ratio under quality gate: "
            f"{best_ratio_gated['run_name']} "
            f"PSNR={best_ratio_gated['compressed_vs_cache_psnr']} "
            f"SSIM={best_ratio_gated['compressed_vs_cache_ssim']} "
            f"LPIPS={best_ratio_gated['compressed_vs_cache_lpips']} "
            f"total_ratio={best_ratio_gated['total_compression_ratio']} "
            f"peak_reserved={best_ratio_gated['max_cuda_peak_reserved_gib']}GiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
