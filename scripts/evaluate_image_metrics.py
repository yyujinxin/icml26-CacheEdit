#!/usr/bin/env python3
"""Evaluate generated image quality between experiment output directories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((ref - pred) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.view(1, 1, size, size)


def _ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    """Compute mean RGB SSIM with an 11x11 Gaussian window."""
    x = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
    channels = x.shape[1]
    window = _gaussian_window().repeat(channels, 1, 1, 1)
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(value.mean().item())


def _try_load_lpips(net: str, device: torch.device):
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        return None, f"lpips import failed: {exc}"

    try:
        model = lpips.LPIPS(net=net).to(device).eval()
        return model, None
    except Exception as exc:  # pragma: no cover - depends on optional weights
        return None, f"lpips model init failed: {exc}"


def _lpips_value(model, ref: np.ndarray, pred: np.ndarray, device: torch.device) -> float:
    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (tensor * 2.0 - 1.0).to(device)

    with torch.inference_mode():
        value = model(to_tensor(ref), to_tensor(pred))
    return float(value.detach().cpu().reshape(-1)[0].item())


def _pngs(directory: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(directory.glob("*.png"))}


def _finite_mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    if all(math.isinf(v) and v > 0 for v in values):
        return float("inf")
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _json_safe(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return "NaN"
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _evaluate_pair(
    ref_dir: Path,
    pred_dir: Path,
    *,
    name: str,
    lpips_model,
    lpips_device: torch.device,
) -> dict:
    ref_files = _pngs(ref_dir)
    pred_files = _pngs(pred_dir)
    common = sorted(set(ref_files) & set(pred_files))
    missing_in_pred = sorted(set(ref_files) - set(pred_files))
    extra_in_pred = sorted(set(pred_files) - set(ref_files))

    records = []
    for filename in common:
        ref = _load_rgb(ref_files[filename])
        pred = _load_rgb(pred_files[filename])
        if ref.shape != pred.shape:
            raise ValueError(
                f"shape mismatch for {filename}: ref={ref.shape}, pred={pred.shape}"
            )

        record = {
            "file": filename,
            "psnr": _psnr(ref, pred),
            "ssim": _ssim(ref, pred),
            "lpips": None,
        }
        if lpips_model is not None:
            record["lpips"] = _lpips_value(lpips_model, ref, pred, lpips_device)
        records.append(record)

    summary = {
        "name": name,
        "ref_dir": str(ref_dir),
        "pred_dir": str(pred_dir),
        "num_common": len(common),
        "missing_in_pred": missing_in_pred,
        "extra_in_pred": extra_in_pred,
        "mean_psnr": _finite_mean(r["psnr"] for r in records),
        "mean_ssim": _finite_mean(r["ssim"] for r in records),
        "mean_lpips": _finite_mean(r["lpips"] for r in records),
    }
    return {"summary": summary, "records": records}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--compressed-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip LPIPS even if the optional lpips package is installed.",
    )
    return parser.parse_args()


def main() -> int:
    args = get_args()
    lpips_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_model = None
    lpips_status = "disabled by --no-lpips" if args.no_lpips else "enabled"
    if not args.no_lpips:
        lpips_model, lpips_error = _try_load_lpips(args.lpips_net, lpips_device)
        if lpips_error:
            lpips_status = lpips_error
        else:
            lpips_status = f"enabled: net={args.lpips_net}, device={lpips_device}"

    results = {
        "lpips_status": lpips_status,
        "pairs": [
            _evaluate_pair(
                args.baseline_dir,
                args.cache_dir,
                name="cache_vs_baseline",
                lpips_model=lpips_model,
                lpips_device=lpips_device,
            ),
            _evaluate_pair(
                args.baseline_dir,
                args.compressed_dir,
                name="compressed_vs_baseline",
                lpips_model=lpips_model,
                lpips_device=lpips_device,
            ),
            _evaluate_pair(
                args.cache_dir,
                args.compressed_dir,
                name="compressed_vs_cache",
                lpips_model=lpips_model,
                lpips_device=lpips_device,
            ),
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(_json_safe(results), f, indent=2, ensure_ascii=False)

    print(f"metrics -> {args.output}")
    print(f"LPIPS: {lpips_status}")
    for pair in results["pairs"]:
        summary = pair["summary"]
        print(
            f"{summary['name']}: n={summary['num_common']} "
            f"PSNR={summary['mean_psnr']} "
            f"SSIM={summary['mean_ssim']} "
            f"LPIPS={summary['mean_lpips']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
