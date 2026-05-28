import os
import json
import argparse
from copy import deepcopy

from Flux_cache_multi_round import (
    load_full_metadata,
    batch_evaluation_multi_round,
)
from Flux_cache import cache_edit_init
from Flux_utils import ActivationCacheManager


def get_args():
    parser = argparse.ArgumentParser(
        description="Sweep threshold from 0.7 to 1.0 (step 0.1), run first N samples for each threshold."
    )

    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=16.0)

    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--warmup_step", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.97,
                        help="Base threshold, will be overwritten during sweep.")
    parser.add_argument("--cache_interval", type=int, default=5)

    parser.add_argument("--evaluation", action="store_true", default=True)

    parser.add_argument("--model_path", type=str,
                        default="/home/dataset-local/chenxueqing/model/black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--image_path", type=str,
                        default="/home/dataset-local/chenxueqing/datasets/test")
    parser.add_argument("--output_dir", type=str,
                        default="/home/dataset-local/chenxueqing/result/Flux/kontext-bench-test/CacheEdit/multi-round-threshold-sweep")

    parser.add_argument("--num_samples", type=int, default=10,
                        help="How many samples from dataset head for each threshold.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use (accepted for compatibility; current sweep runs single process).")
    parser.add_argument("--threshold_start", type=float, default=0.85,
                        help="Threshold sweep start value.")
    parser.add_argument("--threshold_end", type=float, default=0.85,
                        help="Threshold sweep end value (inclusive).")
    parser.add_argument("--threshold_step", type=float, default=0.01,
                        help="Threshold sweep step size.")

    return parser.parse_args()


def threshold_values(start=0.7, end=1.0, step=0.01):
    vals = []
    cur = start
    while cur <= end + 1e-8:
        vals.append(round(cur, 2))
        cur += step
    return vals


def main():
    args = get_args()

    all_metadata = load_full_metadata(args.image_path)
    if not all_metadata:
        raise RuntimeError(f"Empty metadata in {args.image_path}")

    head_n = min(args.num_samples, len(all_metadata))
    metadata_head = all_metadata[:head_n]

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "num_total": len(all_metadata),
        "num_used_each_threshold": head_n,
        "thresholds": [],
    }

    print(f"Loaded metadata: total={len(all_metadata)}, used_each_threshold={head_n}")

    thresholds = threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)

    for th in thresholds:
        run_args = deepcopy(args)
        run_args.threshold = th
        run_args.use_cache = True
        run_args.evaluation = True

        threshold_tag = f"{th:.2f}".replace(".", "p")
        run_out_dir = os.path.join(args.output_dir, f"threshold_{threshold_tag}")
        run_args.output_dir = run_out_dir
        os.makedirs(run_out_dir, exist_ok=True)

        print("=" * 80)
        print(f"[SWEEP] threshold={th:.2f}, output_dir={run_out_dir}")

        pipeline = cache_edit_init(run_args.model_path, run_args.device)
        ActivationCacheManager.set_parameters(args=run_args)

        try:
            batch_evaluation_multi_round(
                pipeline,
                run_args,
                metadata_slice=metadata_head,
                rank=0,
            )
            status = "ok"
            error = None
        except Exception as exc:
            status = "failed"
            error = str(exc)
            print(f"[SWEEP] threshold={th:.1f} failed: {error}")

        try:
            del pipeline
        except Exception:
            pass

        summary["thresholds"].append({
            "threshold": th,
            "output_dir": run_out_dir,
            "status": status,
            "error": error,
        })

    summary_path = os.path.join(args.output_dir, "threshold_sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Sweep finished. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
