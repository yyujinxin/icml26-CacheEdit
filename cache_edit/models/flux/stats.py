"""Flux-specific stats collection and key-token visualization."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

try:
    from PIL import ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


@dataclass
class FluxKeyTokenStatsCollector:
    """
    Flux 中 key_token_indices 的统计收集器。

    维度:
      - step
      - layer
      - stream: "single" / "double"
      - count: 当前 key_token_indices 的元素个数

    内部同时维护：
      stats[stream][step] = [count_layer0, count_layer1, ...]

    用途:
      - 终端打印 report()
      - 导出 Excel：RawData / Summary / {stream}_Pivot / {stream}_StepAvg
    """

    records: List[Dict] = field(default_factory=list)
    stats: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)

    def record(
        self,
        manager_cls,
        step: int,
        layer_idx: int,
        stream: str,
    ) -> None:
        """
        在每次 update_key_token_indices 之后调用记录统计。

        Args:
            manager_cls: FluxCacheManager 实例（或具备 key_token_indices 属性的对象）
            step: 当前 step
            layer_idx: 当前 layer 索引
            stream: "single" 或 "double"
        """
        kt = getattr(manager_cls, "key_token_indices", None)
        if kt is None:
            count = 0
        else:
            try:
                count = int(kt.numel())
            except Exception:
                count = 0

        self.records.append(
            {
                "step": int(step),
                "layer": int(layer_idx),
                "stream": str(stream),
                "count": int(count),
            }
        )

        stream = str(stream)
        if stream not in self.stats:
            self.stats[stream] = {}
        if step not in self.stats[stream]:
            self.stats[stream][step] = []

        lst = self.stats[stream][step]
        if len(lst) < layer_idx:
            lst.extend([0] * (layer_idx - len(lst)))
        if len(lst) == layer_idx:
            lst.append(count)
        else:
            lst[layer_idx] = count

    def report(self) -> None:
        """终端按 stream 分组打印每个 step 各层的 key_token 数量。"""
        print("\n===== Key Token Indices Stats (step-wise) =====")
        if not self.stats:
            print("No stats collected.")
            print("==============================================\n")
            return

        for stream, step_dict in self.stats.items():
            print(f"[Stream={stream}]")
            sorted_steps = sorted(step_dict.items(), key=lambda x: x[0])
            for i, (step, counts) in enumerate(sorted_steps):
                total = sum(counts)
                avg = total / max(len(counts), 1)
                layer_parts = [f"L{idx}:{c}" for idx, c in enumerate(counts)]
                layer_str = " ".join(layer_parts)
                print(
                    f"  Step={step} | {layer_str} | "
                    f"Total={total} Avg={avg:.2f}"
                )
                if i != len(sorted_steps) - 1:
                    print("  " + "-" * 43)
            print()

        print("==============================================\n")

    def _to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["step", "layer", "stream", "count"])
        df = pd.DataFrame(self.records)
        df.sort_values(by=["stream", "step", "layer"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _add_charts_to_excel(
        self, writer: pd.ExcelWriter, df: pd.DataFrame
    ) -> None:
        """为每个 stream 生成 step × layer 透视表 + 折线图 + 条件格式。"""
        workbook = writer.book
        streams = df["stream"].unique().tolist()
        for stream in streams:
            df_s = df[df["stream"] == stream]

            pivot = df_s.pivot_table(
                index="step", columns="layer", values="count", aggfunc="mean"
            ).sort_index()

            sheet_name_pivot = f"{stream}_Pivot"
            if len(sheet_name_pivot) > 31:
                sheet_name_pivot = sheet_name_pivot[:31]
            pivot.to_excel(writer, sheet_name=sheet_name_pivot)
            worksheet_pivot = writer.sheets[sheet_name_pivot]

            n_rows, n_cols = pivot.shape
            if n_rows > 0 and n_cols > 0:
                first_row = 1
                first_col = 1
                last_row = first_row + n_rows - 1
                last_col = first_col + n_cols - 1
                worksheet_pivot.conditional_format(
                    first_row,
                    first_col,
                    last_row,
                    last_col,
                    {
                        "type": "3_color_scale",
                        "min_color": "#FFFFFF",
                        "mid_color": "#FFD966",
                        "max_color": "#FF0000",
                    },
                )

            df_step = df_s.groupby("step")["count"].mean().reset_index()

            sheet_name_stepavg = f"{stream}_StepAvg"
            if len(sheet_name_stepavg) > 31:
                sheet_name_stepavg = sheet_name_stepavg[:31]
            df_step.to_excel(writer, sheet_name=sheet_name_stepavg, index=False)
            worksheet_stepavg = writer.sheets[sheet_name_stepavg]

            chart = workbook.add_chart({"type": "line"})
            n = len(df_step)
            if n > 0:
                chart.add_series(
                    {
                        "name": f"{stream} avg count per step",
                        "categories": [sheet_name_stepavg, 1, 0, n, 0],
                        "values": [sheet_name_stepavg, 1, 1, n, 1],
                    }
                )
                chart.set_title({"name": f"{stream} Avg Count per Step"})
                chart.set_x_axis({"name": "Step"})
                chart.set_y_axis({"name": "Avg Count"})
                chart.set_legend({"position": "bottom"})
                worksheet_stepavg.insert_chart("D2", chart)

    def save_to_excel(self, filepath: str = "flux_key_token_stats.xlsx") -> None:
        """
        写入 Excel：
          - RawData: 原始记录
          - Summary: (stream, step) 汇总
          - {stream}_Pivot: step × layer 透视 + 色阶
          - {stream}_StepAvg: per-step 平均 + 折线图
        """
        df = self._to_dataframe()

        if not df.empty:
            df_summary = (
                df.groupby(["stream", "step"])
                .agg(
                    total_count=("count", "sum"),
                    avg_count=("count", "mean"),
                    max_count=("count", "max"),
                    min_count=("count", "min"),
                )
                .reset_index()
                .sort_values(["stream", "step"])
            )
        else:
            df_summary = pd.DataFrame(
                columns=[
                    "stream",
                    "step",
                    "total_count",
                    "avg_count",
                    "max_count",
                    "min_count",
                ]
            )

        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="RawData", index=False)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            if not df.empty:
                self._add_charts_to_excel(writer, df)

        print(f"[FluxKeyTokenStatsCollector] Excel saved to: {filepath}")

    def reset(self) -> None:
        """重置统计。"""
        self.records.clear()
        self.stats.clear()


def visualize_key_tokens_on_image(
    key_token_indices: torch.Tensor,
    image,
    img_token_len: int,
    save_path: str,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 2,
) -> None:
    """
    把 key token 对应的图像 patch 映射回原图并圈出。

    Args:
        key_token_indices: 1D Tensor
        image: PIL.Image 或具有 .size 属性的对象
        img_token_len: 当前生成图像对应的 token 数
        save_path: 可视化结果保存路径
        outline_color: RGB 框线颜色
        outline_width: 框线宽度
    """
    if not _HAS_PIL:
        print("[visualize_key_tokens_on_image] PIL is not installed, skip.")
        return
    if key_token_indices is None or image is None or img_token_len is None:
        return

    if isinstance(key_token_indices, torch.Tensor):
        indices = key_token_indices.detach().flatten().to("cpu").long()
    else:
        indices = torch.as_tensor(key_token_indices, dtype=torch.long).flatten()

    if indices.numel() == 0:
        return
    if not hasattr(image, "size"):
        return

    width, height = image.size
    if width <= 0 or height <= 0 or img_token_len <= 0:
        return

    grid_h = int(round(math.sqrt(img_token_len * height / max(width, 1))))
    grid_h = max(1, grid_h)
    grid_w = int(math.ceil(img_token_len / grid_h))
    grid_w = max(1, grid_w)

    if grid_w * grid_h != img_token_len:
        candidates = []
        for h in range(1, int(math.sqrt(img_token_len)) + 1):
            if img_token_len % h == 0:
                w = img_token_len // h
                candidates.append((h, w))
                if h != w:
                    candidates.append((w, h))
        if candidates:
            grid_h, grid_w = min(
                candidates, key=lambda hw: abs((hw[1] / hw[0]) - (width / height))
            )
        else:
            grid_h, grid_w = 1, img_token_len

    valid = indices[(indices >= 0) & (indices < img_token_len)]
    if valid.numel() == 0:
        return

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    patch_w = width / grid_w
    patch_h = height / grid_h

    for idx in valid.tolist():
        row = idx // grid_w
        col = idx % grid_w
        x0 = int(round(col * patch_w))
        y0 = int(round(row * patch_h))
        x1 = int(round((col + 1) * patch_w)) - 1
        y1 = int(round((row + 1) * patch_h)) - 1
        draw.rectangle(
            [x0, y0, max(x0, x1), max(y0, y1)],
            outline=outline_color,
            width=outline_width,
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img_draw.save(save_path)


def append_key_token_ratio_with_edit_ratio(
    image_filename: Optional[str],
    cur_round: int,
    cur_step: int,
    key_token_num: int,
    img_token_len: int,
    output_csv: str,
    edit_ratio_summary_candidates: Optional[List[str]] = None,
) -> None:
    """
    将 key_token_ratio 与 image_edit_ratio 写入 CSV（如果给定 summary 文件可以找到匹配条目）。

    Args:
        image_filename: 当前正在处理的图像文件名（用于解析 image_id）
        cur_round: 当前轮次
        cur_step: 当前 step
        key_token_num: 当前 key_token 数
        img_token_len: 当前图像总 token 数
        output_csv: 输出 CSV 路径
        edit_ratio_summary_candidates: 可选的参考 summary CSV 路径列表
    """
    if key_token_num < 0 or img_token_len <= 0:
        return

    key_token_ratio = float(key_token_num) / float(img_token_len)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    image_id = None
    if image_filename:
        m = re.match(r"^(\d{4})", image_filename)
        if m:
            image_id = m.group(1)

    image_edit_ratio: object = ""
    ratio_gap: object = ""

    summary_path: Optional[str] = None
    if edit_ratio_summary_candidates:
        for p in edit_ratio_summary_candidates:
            if os.path.isfile(p):
                summary_path = p
                break

    if summary_path is not None and image_id is not None:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = str(row.get("id", "")).strip()
                    rround = str(row.get("round", "")).strip()
                    if rid == image_id and rround == str(cur_round):
                        if row.get("changed_ratio", "") not in (None, ""):
                            image_edit_ratio = float(row["changed_ratio"])
                        elif row.get("changed_percent", "") not in (None, ""):
                            image_edit_ratio = (
                                float(row["changed_percent"]) / 100.0
                            )
                        break
        except Exception:
            image_edit_ratio = ""

    if isinstance(image_edit_ratio, float):
        ratio_gap = key_token_ratio - image_edit_ratio

    existing_keys = set()
    existing_rows = []
    if os.path.isfile(output_csv):
        try:
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    k = (
                        str(r.get("image_id", "")).strip(),
                        str(r.get("round", "")).strip(),
                        str(r.get("step", "")).strip(),
                    )
                    existing_keys.add(k)
                    existing_rows.append(r)
        except Exception:
            existing_keys = set()
            existing_rows = []

    new_key = (
        image_id if image_id is not None else "",
        str(cur_round),
        str(cur_step),
    )
    if new_key in existing_keys:
        return

    fieldnames = [
        "image_id",
        "round",
        "step",
        "image_filename",
        "key_token_num",
        "img_token_len",
        "key_token_ratio",
        "image_edit_ratio",
        "ratio_gap_key_minus_edit",
        "edit_ratio_summary_path",
    ]

    row_obj = {
        "image_id": image_id if image_id is not None else "",
        "round": cur_round,
        "step": cur_step,
        "image_filename": image_filename if image_filename is not None else "",
        "key_token_num": key_token_num,
        "img_token_len": img_token_len,
        "key_token_ratio": key_token_ratio,
        "image_edit_ratio": image_edit_ratio,
        "ratio_gap_key_minus_edit": ratio_gap,
        "edit_ratio_summary_path": summary_path if summary_path is not None else "",
    }

    merged = existing_rows + [row_obj]
    try:
        merged.sort(
            key=lambda r: (
                int(str(r.get("image_id", "0") or 0)),
                int(str(r.get("round", "0") or 0)),
                int(str(r.get("step", "0") or 0)),
            )
        )
    except Exception:
        pass

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def infer_image_id_from_csv_by_round(csv_path: str, rounds_per_image: int = 7) -> str:
    """
    稳健规则：扫描整个 CSV，按 (image_id 数值最大者) 当前；
    若该 id 的 max_round >= rounds_per_image，则推断已切到下一个 image_id。

    Returns:
        4 位 zero-padded image_id 字符串
    """
    if not os.path.isfile(csv_path):
        return "0000"

    id_to_max_round: Dict[str, int] = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = str(row.get("image_id", "")).strip()
                rround = str(row.get("round", "")).strip()
                if re.fullmatch(r"\d{4}", rid) and re.fullmatch(r"\d+", rround):
                    rr = int(rround)
                    if rid not in id_to_max_round or rr > id_to_max_round[rid]:
                        id_to_max_round[rid] = rr
    except Exception:
        return "0000"

    if not id_to_max_round:
        return "0000"

    current_id = max(id_to_max_round.keys(), key=lambda x: int(x))
    max_round_for_current = id_to_max_round[current_id]

    if max_round_for_current >= rounds_per_image:
        return f"{int(current_id) + 1:04d}"
    return current_id
