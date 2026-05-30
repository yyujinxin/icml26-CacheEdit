import torch
import inspect
import numpy as np
import torch.nn.functional as F
from PIL import ImageDraw
from bisect import bisect_right
from dataclasses import dataclass, field
from diffusers import FluxKontextPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux import FluxPipelineOutput
from diffusers.utils import BaseOutput, is_torch_xla_available
from typing import Optional, Union, List, Dict, Any, Callable, Tuple, Set, Literal, Dict
import pandas as pd
# from __future__ import annotations
from torch import Tensor
import math
import os

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
    
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor





@dataclass
class FluxKeyTokenStatsCollector:
    """
    统计 FLUX 中 key_token_indices 的信息。

    维度:
      - step
      - layer
      - stream: "single" / "double"
      - count:  当前 key_token_indices 的元素个数

    同时维护:
      stats[stream][step] = [count_layer0, count_layer1, ...]
    用于:
      - 终端打印 report()
      - 导出 Excel 并生成基础图表
    """

    # 原始记录（给 DataFrame / Excel 用）
    records: List[Dict] = field(default_factory=list)

    # 结构化统计：
    # stream -> { step -> [count_layer0, count_layer1, ...] }
    stats: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)

    def record(
        self,
        manager_cls,
        step: int,
        layer_idx: int,
        stream: str,
    ):
        """
        在每次 update_key_token_indices 之后调用，用于记录当前统计。
        - manager_cls: ActivationCacheManager（类或者单例）
        - step: 当前 step
        - layer_idx: 当前 layer 索引
        - stream: "single" 或 "double"
        """

        # 你的 FLUX 没有 mode，就假设 ActivationCacheManager.key_token_indices
        # 要么是 Tensor，要么是 None
        kt = getattr(manager_cls, "key_token_indices", None)

        if kt is None:
            count = 0
        else:
            try:
                count = int(kt.numel())
            except Exception:
                count = 0

        # 1) 记录到 records
        self.records.append(
            {
                "step": int(step),
                "layer": int(layer_idx),
                "stream": str(stream),
                "count": int(count),
            }
        )

        # 2) 记录到结构化 stats
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

    # ========== 终端 report ==========

    def report(self):
        """
        在所有 step 跑完后调用，按 stream 分组输出：

        ===== Key Token Indices Stats (step-wise) =====
        [Stream=double]
          Step=0 | L0:10 L1:12 ... | Total=xx Avg=yy
          -------------------------------------------
          Step=1 | L0: 8 L1:11 ... | Total=xx Avg=yy
          -------------------------------------------
        [Stream=single]
          ...
        ===============================================
        """
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

    # ========== DataFrame & Excel ==========

    def _to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["step", "layer", "stream", "count"])
        df = pd.DataFrame(self.records)
        df.sort_values(by=["stream", "step", "layer"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _add_charts_to_excel(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """
        为每个 stream（single/double）生成：
          - step × layer 透视表（带 3 色条件格式）
          - per-step 平均 count 表 + 折线图
        """
        workbook = writer.book

        streams = df["stream"].unique().tolist()
        for stream in streams:
            df_s = df[df["stream"] == stream]

            # 1) step × layer 透视
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
                first_row = 1  # 0 行是列名
                first_col = 1  # 0 列是 index(step)
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

            # 2) per-step 平均 count + 折线图
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
                        "name":       f"{stream} avg count per step",
                        "categories": [sheet_name_stepavg, 1, 0, n, 0],  # step
                        "values":     [sheet_name_stepavg, 1, 1, n, 1],  # avg count
                    }
                )
                chart.set_title({"name": f"{stream} Avg Count per Step"})
                chart.set_x_axis({"name": "Step"})
                chart.set_y_axis({"name": "Avg Count"})
                chart.set_legend({"position": "bottom"})
                worksheet_stepavg.insert_chart("D2", chart)

    def save_to_excel(self, filepath: str = "flux_key_token_stats.xlsx"):
        """
        写入 Excel：
          - RawData        : 原始记录
          - Summary        : (stream, step) 汇总
          - {stream}_Pivot : step × layer 透视 + 色阶
          - {stream}_StepAvg : per-step 平均 + 折线图
        """
        df = self._to_dataframe()

        # Summary: (stream, step) 聚合
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

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="RawData", index=False)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            if not df.empty:
                self._add_charts_to_excel(writer, df)

        print(f"[FluxKeyTokenStatsCollector] Excel saved to: {filepath}")
stats_collector = FluxKeyTokenStatsCollector()


StreamType = Literal["double", "single"]

class ActivationCacheManager:
    """
    负责多轮图像编辑时的：
    - 轮次与 step 管理
    - 激活缓存的写入与复用
    - key token 索引与 token 重排
    """

    def __init__(
        self,
        use_activation_cache: bool = True,
        cache_steps: Optional[Set[int]] = None,   # None = 所有 step 都缓存
        cache_device: torch.device = torch.device("cuda:1"),
        num_num_gpus: int = 1,
        total_step_num: int = 30,
        threshold: float = 0.97,
        cache_interval: int = 5,                 # 用于自动生成 cache_steps
    ):
        # 配置
        self.use_activation_cache = use_activation_cache
        self.cache_steps = cache_steps
        self.cache_device = cache_device          # 第一块
        self.total_step_num = total_step_num
        self.threshold = threshold
        self.cache_interval = cache_interval

        # 轮次信息
        self.current_round: int = -1
        self.is_round0: bool = self.current_round == 0
        
        # step信息
        self.current_step: int = -1
        self.stream_type: StreamType = "single"

        # per-stream 缓存
        self.prev_cache: Dict[Tuple[StreamType, int, int], Tensor] = {}
        self.new_cache: Dict[Tuple[StreamType, int, int], Tensor] = {}

        # key token 信息
        self.key_token_indices: Optional[Tensor] = None
        
        # ---------- 自动设置 cache_steps ----------
        if self.cache_steps is None:
            self._build_cache_steps_from_interval()
        self._refresh_cache_steps_sorted()
        print("Initialized  cache_steps:", self.cache_steps)
        
    def gpu_has_space(self, device: torch.device, extra_bytes: int, limit_gb: float) -> bool:
        used = torch.cuda.memory_allocated(device)
        # print("GPU", device, "used GB:", used / 1024**3)
        # 预留一点 buffer，防止刚好打满 OOM，比如多加 1GB
        limit_bytes = int(limit_gb * 1024**3)
        buffer_bytes = int(1 * 1024**3)
        return used + extra_bytes + buffer_bytes <= limit_bytes
    
    def _build_cache_steps_from_interval(self):
        """
        根据 total_step_num 和 cache_interval 自动生成 cache_steps，
        例如 total_step_num=40, cache_interval=5 -> {0,5,10,15,20,25,30,35}
        """
        if self.cache_interval <= 0:
            # 做个兜底，间隔非法时就只在 step=0 cache
            self.cache_steps = {0}
            return

        steps = list(range(0, self.total_step_num, self.cache_interval))
        self.cache_steps = set(steps)
        self._refresh_cache_steps_sorted()

    def _refresh_cache_steps_sorted(self):
        if self.cache_steps is None:
            self._cache_steps_sorted = None
        else:
            self._cache_steps_sorted = sorted(self.cache_steps)

    def set_parameters(self, args) -> None:
        self.total_step_num = args.num_inference_steps
        self.threshold = args.threshold
        self.cache_interval = args.cache_interval
        self.num_num_gpus = args.num_gpus
        self._build_cache_steps_from_interval()
        print("Initialized  cache_steps:", self.cache_steps)
    
    # ----------------- 轮次 & step -----------------
    def on_step_start(self, step: int):
        """
        每个 step 开头调用一次。
        step == 0 时认为开启新一轮编辑。
        """
        self.current_step = step
        if step == 0:
            self.current_round += 1
            self.is_round0 = (self.current_round == 0)
            self._rearranged_pe_cache = None
            self._rearranged_pe_cache_version = -1
            self._restore_masks = None
            self._prev_mask_cache = None

    # ----------------- cache 相关 -----------------
    def should_cache(self, step: int) -> bool:
        if not self.use_activation_cache:
            return False
        if self.cache_steps is None:
            return True
        return step in self.cache_steps

    def map_to_group_min(self, step: int) -> int:
        """
        把 step 映射为 self.cache_steps 中 <= step 的最大值。
        如果所有 cache_step 都 > step，则返回 None（你也可以换成 0 或抛异常）。
        """
        if self.cache_steps is None:
            return step
        if not self._cache_steps_sorted:
            return None
        idx = bisect_right(self._cache_steps_sorted, step) - 1
        if idx < 0:
            return None
        return self._cache_steps_sorted[idx]

    def should_reuse(self, step: int) -> bool:
        """
        第二轮及以后，且当前 step 不需要写 cache 时，才复用上一轮缓存。
        """
        return (not self.is_round0) and (not self.should_cache(self.current_step))

    def store_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        tensor: Tensor,
    ):
        """
        写入 new_cache，在合适的时机再 flush 到 prev_cache。
        """
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        key = (stream, self.current_step, layer_idx)
        self.new_cache[key] = tensor.to(self.cache_device)
    
    def maby_store_activation(self, stream: StreamType, layer_idx: int, tensor: Tensor):
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        key = (stream, self.current_step, layer_idx)

        # 预计这个 tensor 占多少显存
        bytes_per_element = tensor.element_size()   # float16=2, float32=4 ...
        extra_bytes = tensor.numel() * bytes_per_element
        # 起始设备：类似 torch.device("cuda:2")
        start_dev: torch.device = self.cache_device
        assert start_dev.type == "cuda", f"cache_device 必须是 cuda 设备，当前是 {start_dev}"

        start_idx = start_dev.index               # int，比如 2
        num_gpus = self.num_num_gpus   # 总 GPU 数量

        # 为不同 GPU 设定限制（按需修改）
        def get_limit_gb(device_idx: int) -> float:
            # 这里简单统一成 40G 机，用 37G 安全线；你可以自定义
            return 77.0

        target_device: torch.device | None = None

        # 从 start_idx 开始，依次尝试 start_idx, start_idx+1, ... , num_gpus-1
        for dev_idx in range(start_idx, num_gpus):
            dev = torch.device(f"cuda:{dev_idx}")
            limit_gb = get_limit_gb(dev_idx)
            if self.gpu_has_space(dev, extra_bytes, limit_gb=limit_gb):
                target_device = dev
                break

        # 如果从 start_idx 到最后一块都不够，再尝试 0 ~ start_idx-1（可选）
        if target_device is None:
            for dev_idx in range(0, start_idx):
                dev = torch.device(f"cuda:{dev_idx}")
                limit_gb = get_limit_gb(dev_idx)
                if self.gpu_has_space(dev, extra_bytes, limit_gb=limit_gb):
                    target_device = dev
                    break

        # 如果所有 GPU 都几乎满了，可以选择 fallback 或直接 return
        if target_device is None:
            print(f"All GPUs 0..{num_gpus-1} are almost full, "
                f"falling back to {start_dev} (may OOM)")
            target_device = start_dev
            # 或者你也可以直接：
            # return
        self.new_cache[key] = tensor.to(target_device)

    def load_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """
        从 prev_cache 读取激活，用 step 映射后的 group_min。
        """
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        step_to_load = self.map_to_group_min(step)
        key = (stream, step_to_load, layer_idx)
        prev = self.prev_cache.get(key, None)
        if prev is None:
            return None
        return prev.to(device)
    
    def load_key_token_ref(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """
        如果你有“关键 token reference”也需要区分两套 cache，这里同样按 mode 读取。
        """
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        prev = self.prev_cache.get(key, None)
        if prev is None:
            return None
        return prev.to(device)

    def load_key_token_cur(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """
        如果你有“关键 token reference”也需要区分两套 cache，这里同样按 mode 读取。
        """
        if not self.use_activation_cache:
            return None
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        cur = self.new_cache.get(key, None)
        if cur is None:
            return None
        return cur.to(device)
    
    def flush_new_cache_after_step(self):
        """
        在每个 step 结尾，根据你的策略把 new_cache 合入 prev_cache。
        这里复刻你原始逻辑：
        - round0: 若 should_cache(step)，立刻合并
        - round>0: 若 should_cache(step+1) 且 step < last_step，则合并
        """
        last_step = self.total_step_num - 1
        if not self.use_activation_cache:
            return

        if self.is_round0:
            if self.should_cache(self.current_step):
                self.prev_cache.update(self.new_cache)
                self.new_cache.clear()
        else:
            if last_step is not None and self.current_step >= last_step:
                return
            if self.should_cache(self.current_step + 1):
                self.prev_cache.update(self.new_cache)
                self.new_cache.clear()

    # ----------------- key token / 重排 -----------------
    @staticmethod
    def rearrange_tensor_with_key_token_indices(
        img: Tensor,
        cos_img: Tensor,
        sin_img: Tensor,
        key_token_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        把关键 token 调到前面，其余 token 顺序保持不变。
        img: (B, L_img, C)
        pe_img: (B, C_pe, L_img, ...)  —— 这里沿着 token 维度重排
        """
        img_key = torch.index_select(img, 1, key_token_indices)
        key_token_indices = key_token_indices.to(cos_img.device)
        cos_img_key = torch.index_select(cos_img, 0, key_token_indices)
        sin_img_key = torch.index_select(sin_img, 0, key_token_indices)

        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False
        img_not_key = img[:, mask, :]
        mask = mask.to(cos_img.device)
        cos_img_not_key = cos_img[ mask, ...]
        sin_img_not_key = sin_img[ mask, ...]

        img_new = torch.cat((img_key, img_not_key), dim=1)
        cos_img = torch.cat((cos_img_key, cos_img_not_key), dim=0)
        sin_img = torch.cat((sin_img_key, sin_img_not_key), dim=0)
        return img_new, cos_img, sin_img

    @staticmethod
    def restore_original_token_order(
        x: Tensor,
        key_token_indices: Tensor,
    ) -> Tensor:
        """
        将前 K 个 token 放回原 key_token_indices 指定的位置。
        x: (B, L, C)
        """
        K = key_token_indices.size(0)
        x_key = x[:, :K, :]
        x_not_key = x[:, K:, :]
        L = x.size(1)

        mask_not_key = torch.ones(L, dtype=torch.bool, device=x.device)
        mask_not_key[key_token_indices] = False
        mask_key = ~mask_not_key

        out = x.clone()
        out[:, mask_not_key, :] = x_not_key
        out[:, mask_key, :] = x_key
        return out

    def precompute_masks(self, total_img_len: int, device: torch.device):
        """在 key_token_indices 更新后调用，预计算各类 mask"""
        if self.key_token_indices is None:
            return
        key_token_indices = self.key_token_indices.to(device)

        # 用于 restore_original_token_order 的 mask
        mask_not_key = torch.ones(total_img_len, dtype=torch.bool, device=device)
        mask_not_key[key_token_indices] = False
        self._restore_masks = (mask_not_key, ~mask_not_key)

        # 用于复用时去除 key tokens 的 mask（在 prev cache 上）
        self._prev_mask_cache = mask_not_key.clone()

    def clear_pe_cache(self):
        """每轮开始或 key_token_indices 更新后清空 PE 缓存"""
        self._rearranged_pe_cache = None
        self._rearranged_pe_cache_version = -1

    def _rearrange_img_only(self, img: Tensor) -> Tensor:
        """只重排 img，不重排 PE（PE 已缓存）"""
        key_token_indices = self.key_token_indices
        if key_token_indices is None:
            return img
        if key_token_indices.device != img.device:
            key_token_indices = key_token_indices.to(img.device)
        img_key = torch.index_select(img, 1, key_token_indices)
        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False
        img_not_key = img[:, mask, :]
        return torch.cat((img_key, img_not_key), dim=1)

    def maybe_rearrange_img_and_pe(
        self,
        img: Tensor,
        pe: Tuple[Tensor, Tensor], # 修改类型提示
        txt_len: int,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], int, bool]: # 修改返回类型提示
        """
        根据当前轮次与是否 should_reuse 决定：
        - 是否对 img / pe_img 做重排
        - 返回 key_token_num 和 should_reuse
        pe: Tuple(cos, sin), each shape (B, C_pe, L_txt + L_img, ...)
        """
        should_reuse = self.should_reuse(self.current_step)
        
        # 1. 解包 PE (Tuple -> Tensor, Tensor)
        cos, sin = pe 

        if should_reuse and self.key_token_indices is not None:
            # 2. 分别切片 (Split Text & Image)
            cos_txt, cos_img = cos[:txt_len, :], cos[txt_len:, :]
            sin_txt, sin_img = sin[:txt_len, :], sin[txt_len:, :]

            # 3. 分别重排 (Rearrange)
            # 这里的逻辑比较微妙：我们需要用同样的 indices 重排 img, cos_img, sin_img。
            # 为了安全（防止 img 在第一次调用后变小，导致第二次调用时索引越界），
            # 我们先对 sin_img 进行重排（此时传入原始 img，忽略返回的 new_img）
            img, cos_img, sin_img = self.rearrange_tensor_with_key_token_indices(
                img, cos_img, sin_img, self.key_token_indices
            )

            # 4. 拼接回完整序列 (Concat)
            cos = torch.cat((cos_txt, cos_img), dim=0)
            sin = torch.cat((sin_txt, sin_img), dim=0)
            
            # 5. 重新打包 PE
            pe = (cos, sin)
            
            key_token_num = self.key_token_indices.size(0)
        else:
            key_token_num = img.size(1)

        return img, pe, key_token_num, should_reuse

    def maybe_restore_img_order(
        self,
        img: Tensor,
    ) -> Tensor:
        """
        若本 step 使用了复用逻辑，则在输出前恢复原 token 顺序。
        """
        if self.should_reuse(self.current_step) and self.key_token_indices is not None:
            img = self.restore_original_token_order(img, self.key_token_indices)
        # if self.should_reuse(self.current_step) and self.key_token_indices is not None:
        #     key_token_indices = self.key_token_indices
        #     if key_token_indices.device != img.device:
        #         key_token_indices = key_token_indices.to(img.device)

        #     if self._restore_masks is not None:
        #         mask_not_key, mask_key = self._restore_masks
        #         if mask_not_key.device != img.device or mask_not_key.numel() != img.size(1):
        #             mask_not_key = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        #             mask_not_key[key_token_indices] = False
        #             mask_key = ~mask_not_key
        #             self._restore_masks = (mask_not_key, mask_key)
        #     else:
        #         mask_not_key = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        #         mask_not_key[key_token_indices] = False
        #         mask_key = ~mask_not_key
        #         self._restore_masks = (mask_not_key, mask_key)

        #     K = key_token_indices.size(0)
        #     x_key = img[:, :K, :]
        #     x_not_key = img[:, K:, :]
        #     out = img.clone()
        #     out[:, mask_not_key, :] = x_not_key
        #     out[:, mask_key, :] = x_key
        #     return out
        return img

    def update_key_token_indices(
        self,
        cur_img: Tensor,
        ref_img: Optional[Tensor],
    ):
        """
        在第二轮及之后、并满足你指定的 step 条件时，调用自定义函数
        compute_key_indices_fn(cur_img[0], ref_img[0]) -> Tensor[indices]
        生成新的 key_token_indices。
        """
        if not self.use_activation_cache:
            return
        if self.is_round0:
            return
        # 仅在 step == 0 时更新（可按需求改）
        if not self.current_step in self.cache_steps or ref_img is None or cur_img is None:
            return
        indices = self.compute_key_indices_fn(cur_img[0], ref_img[0])
        self.key_token_indices = indices.to(cur_img.device)
        
    def compute_key_indices_fn(
        self,
        tensor1: torch.Tensor, 
        tensor2: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算两个tensor中对应行向量的余弦相似度
        
        参数:
            tensor1: shape (n, d)
            tensor2: shape (n, d)
            device: 指定运行的GPU设备（如torch.device("cuda:0")），默认为None（自动选择）
        返回:
            similarities: shape (n,), 每一行的余弦相似度
        """
        # print("Computing key token indices based on cosine similarity...")
        device = tensor2.device
        tensor1 = tensor1.to(device)
        assert tensor1.shape == tensor2.shape, "两个tensor的shape必须相同"
        
        # 计算每行的点积
        dot_product = (tensor1 * tensor2).sum(dim=1)
        
        # 计算每行的范数
        norm1 = tensor1.norm(p=2, dim=1)
        norm2 = tensor2.norm(p=2, dim=1)
        
        # 防止除零
        eps = 1e-8
        similarities = dot_product / (norm1 * norm2 + eps)
        
        # 将相似度归一化到[0, 1]范围
        normalized_similarities = (similarities + 1) / 2

        # 找出相似度小于阈值的布尔掩码 (mask)
        mask = normalized_similarities < self.threshold
        # 从掩码中获取索引ID
        indices = mask.nonzero(as_tuple=True)[0]
        return indices
        
    def reset(self):
        """
        将管理器恢复到初始状态，清空所有缓存并重置计数器。
        
        参数:
            empty_cuda_cache: 是否在清空后显式调用 torch.cuda.empty_cache()
        """
        # 1. 清空缓存字典（释放 Tensor 引用）
        self.prev_cache.clear()
        self.new_cache.clear()

        # 2. 重置轮次和 Step 信息
        self.current_round = -1
        self.is_round0 = True  # 初始状态认为是 Round 0 准备阶段
        self.current_step = -1
        
        # 3. 重置 Key Token 索引
        self.key_token_indices = None
        self._rearranged_pe_cache = None
        self._rearranged_pe_cache_version = -1
        self._restore_masks = None
        self._prev_mask_cache = None
        self._key_indices_version = 0
            
        print("ActivationCacheManager has been reset to initial state.")
        
ActivationCacheManager = ActivationCacheManager()


def visualize_key_tokens_on_image(
    key_token_indices: torch.Tensor,
    image,
    img_token_len: int,
    save_path: str,
    outline_color: tuple = (255, 0, 0),
    outline_width: int = 2,
):
    """
    将 key token 对应的图像 patch 映射到原图并圈出。

    参数:
        key_token_indices: 1D Tensor，来自 update_key_token_indices 的索引。
        image: PIL.Image（或可被转换为 PIL 的对象）。
        img_token_len: 当前“生成图像”对应的 token 数（即 image 序列长度）。
        save_path: 可视化结果保存路径。
    """
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
            grid_h, grid_w = min(candidates, key=lambda hw: abs((hw[1] / hw[0]) - (width / height)))
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
        draw.rectangle([x0, y0, max(x0, x1), max(y0, y1)], outline=outline_color, width=outline_width)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_draw.save(save_path)

# def get_key_token_indices(
#     tensor1: torch.Tensor, 
#     tensor2: torch.Tensor,
#     threshold: float,
# ) -> torch.Tensor:
#     """
#     计算两个tensor中对应行向量的余弦相似度
    
#     参数:
#         tensor1: shape (n, d)
#         tensor2: shape (n, d)
#         device: 指定运行的GPU设备（如torch.device("cuda:0")），默认为None（自动选择）
#     返回:
#         similarities: shape (n,), 每一行的余弦相似度
#     """
#     device = tensor2.device
#     tensor1 = tensor1.to(device)
#     assert tensor1.shape == tensor2.shape, "两个tensor的shape必须相同"
    
#     # 计算每行的点积
#     dot_product = (tensor1 * tensor2).sum(dim=1)
    
#     # 计算每行的范数
#     norm1 = tensor1.norm(p=2, dim=1)
#     norm2 = tensor2.norm(p=2, dim=1)
    
#     # 防止除零
#     eps = 1e-8
#     similarities = dot_product / (norm1 * norm2 + eps)
    
#     # 将相似度归一化到[0, 1]范围
#     normalized_similarities = (similarities + 1) / 2

#     # 找出相似度小于阈值的布尔掩码 (mask)
#     mask = normalized_similarities < threshold
#     # 从掩码中获取索引ID
#     indices = mask.nonzero(as_tuple=True)[0]
#     return indices


@torch.no_grad()
def pipeline_call(
    self,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_ip_adapter_image: Optional[PipelineImageInput] = None,
    negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    max_area: int = 1024**2,
    _auto_resize: bool = True,
):
    multiple_of = self.vae_scale_factor * 2

    # 1. Preprocess image
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        img = image[0] if isinstance(image, list) else image
        image_height, image_width = self.image_processor.get_default_height_width(img)
        aspect_ratio = image_width / image_height
        if _auto_resize:
            # Kontext is trained on specific resolutions, using one of them is recommended
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of
        image = self.image_processor.resize(image, image_height, image_width)
        image = self.image_processor.preprocess(image, image_height, image_width)
        height, width = image.shape[-2], image.shape[-1]

    else:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width  or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

    # 2. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 3. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = self.prepare_latents(
        image,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
        negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
    ):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

    elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
        negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
    ):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

    if self.joint_attention_kwargs is None:
        self._joint_attention_kwargs = {}

    image_embeds = None
    negative_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
        negative_image_embeds = self.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image,
            negative_ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            if image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
