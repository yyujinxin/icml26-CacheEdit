import math
import torch
from torch import Tensor
import inspect
import torch.nn.functional as F
from dataclasses import dataclass, field
from diffusers.utils import BaseOutput
from typing import Optional, Union, List, Dict, Tuple, Set, Literal

from collections import defaultdict
import pandas as pd



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
    

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

StreamType = Literal["double", "single"]

# @dataclass
# class KeyTokenStatsCollector:
#     """
#     统计各个 layer、各个 mode ("cond"/"uncond") 的 key_token_indices 数量（numel）。
#     适配结构：
#         self.key_token_indices: Dict[str, Optional[Tensor]] = {"cond": None, "uncond": None}
#     即：对某个 mode，整块就是一个 Tensor 或 None，而不是按 layer 存字典。
#     """
#     # stats[mode] = [count_at_layer_0, count_at_layer_1, ...]
#     stats: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))

#     def record(self, manager_cls, layer_idx: int):
#         """
#         在每个 layer 更新完 QwenActivationCacheManager.key_token_indices 之后调用。

#         manager_cls: QwenActivationCacheManager（类或实例，只要有
#                      current_mode 和 key_token_indices 两个属性）
#         layer_idx: 当前更新到第几层（用来保证长度对应）
#         """
#         mode = manager_cls.current_mode  # "cond" 或 "uncond"
#         kt: Optional["Tensor"] = manager_cls.key_token_indices.get(mode, None)

#         if kt is None:
#             count = 0
#         else:
#             # kt 是一个 Tensor，直接用 numel 统计元素数量
#             count = int(kt.numel())

#         # 确保 stats[mode] 的长度与 layer_idx 对齐（允许中间跳层的情况）
#         lst = self.stats[mode]
#         if len(lst) < layer_idx:
#             # 如果外部逻辑中 layer 是严格从 0 递增，可以不需要这段，但保险起见补齐
#             lst.extend([0] * (layer_idx - len(lst)))
#         if len(lst) == layer_idx:
#             lst.append(count)
#         else:
#             # 若同一层多次更新，就覆盖最后一次为准
#             lst[layer_idx] = count

#     def report(self):
#         """
#         紧凑输出，每个 mode 一行：
#         Mode=cond | L0:10 L1:12 ... | Total=xx Avg=yy
#         """
#         print("\n===== Key Token Indices Stats (compact) =====")
#         if not self.stats:
#             print("No stats collected.")
#             print("=============================================\n")
#             return

#         for mode, counts in self.stats.items():
#             total = sum(counts)
#             avg = total / max(len(counts), 1)

#             # layer 部分组装成一行字符串
#             layer_parts = [f"L{idx}:{c}" for idx, c in enumerate(counts)]
#             layer_str = " ".join(layer_parts)

#             print(
#                 f"Mode={mode} | {layer_str} | "
#                 f"Total={total} Avg={avg:.2f}"
#             )
#         print("=============================================\n")
@dataclass
class KeyTokenStatsCollector:
    """
    记录:
        step, layer, mode, count (= key_token_indices[mode].numel())
    同时维护:
        stats[mode][step] = [count_layer0, count_layer1, ...]
    用于终端打印(report) 和 Excel 分析。
    """

    # 原始逐条记录：适合做 DataFrame / Excel / 图表
    records: List[Dict] = field(default_factory=list)

    # 结构化统计：适合终端可读性输出
    stats: Dict[str, Dict[int, List[int]]] = field(
        default_factory=lambda: {}
    )

    def record(self, manager_cls, step: int, layer_idx: int):
        """
        在【每个 step 的每一层】调用一次：
        在调用完 update_key_token_indices 之后记录当前 mode 的统计。
        """
        mode = manager_cls.current_mode   # "cond" / "uncond"
        kt: Optional["Tensor"] = manager_cls.key_token_indices.get(mode, None)
        count = int(kt.numel()) if kt is not None else 0

        # 1) 记录到 records（给 DataFrame / Excel 用）
        self.records.append(
            {
                "step": int(step),
                "layer": int(layer_idx),
                "mode": str(mode),
                "count": int(count),
            }
        )

        # 2) 同时维护 stats（给 report 终端输出用）
        if mode not in self.stats:
            self.stats[mode] = {}
        if step not in self.stats[mode]:
            self.stats[mode][step] = []

        lst = self.stats[mode][step]
        # 确保长度和 layer_idx 对齐
        if len(lst) < layer_idx:
            lst.extend([0] * (layer_idx - len(lst)))
        if len(lst) == layer_idx:
            lst.append(count)
        else:
            lst[layer_idx] = count

    # ========== 终端打印：可读性好的 summary ==========

    def report(self):
        """
        在所有 step 跑完后调用，紧凑但可读性较好的输出：

        ===== Key Token Indices Stats (step-wise) =====
        [Mode=cond]
          Step=0 | L0:10 L1:12 ... | Total=xx Avg=yy
          -------------------------------------------
          Step=1 | L0: 8 L1:11 ... | Total=xx Avg=yy
          -------------------------------------------
        [Mode=uncond]
          Step=0 | ...
        ===============================================
        """
        print("\n===== Key Token Indices Stats (step-wise) =====")
        if not self.stats:
            print("No stats collected.")
            print("==============================================\n")
            return

        for mode, step_dict in self.stats.items():
            print(f"[Mode={mode}]")
            # 按 step 排序输出
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
                # 同一个 mode 内部 step 之间加分隔线（最后一个 step 不加）
                if i != len(sorted_steps) - 1:
                    print("  " + "-" * 43)
            print()  # mode 之间空一行

        print("==============================================\n")

    # ========== DataFrame & Excel ==========

    def _to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["step", "layer", "mode", "count"])
        df = pd.DataFrame(self.records)
        df.sort_values(by=["mode", "step", "layer"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _add_charts_to_excel(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """
        使用 xlsxwriter 在 Excel 中增加图表和色阶：
        - 每个 mode 一张 step×layer 透视表 + 3 色条件格式（类似热力图）
        - 每个 mode 一张“per-step 平均 count 折线图”
        """
        workbook = writer.book

        modes = df["mode"].unique().tolist()
        for mode in modes:
            df_m = df[df["mode"] == mode]

            # step × layer 透视：后面做色阶
            pivot = df_m.pivot_table(
                index="step", columns="layer", values="count", aggfunc="mean"
            ).sort_index()
            sheet_name_pivot = f"{mode}_Pivot"
            pivot.to_excel(writer, sheet_name=sheet_name_pivot)
            worksheet_pivot = writer.sheets[sheet_name_pivot]

            # 条件格式：3 色 scale，模拟“热力图”
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

            # 每个 mode 做一个 per-step 平均 count 表 + 折线图
            df_step = df_m.groupby("step")["count"].mean().reset_index()
            sheet_name_summary = f"{mode}_StepAvg"
            df_step.to_excel(writer, sheet_name=sheet_name_summary, index=False)
            worksheet_summary = writer.sheets[sheet_name_summary]

            chart = workbook.add_chart({"type": "line"})
            n = len(df_step)
            if n > 0:
                chart.add_series(
                    {
                        "name":       f"{mode} avg count per step",
                        "categories": [sheet_name_summary, 1, 0, n, 0],  # step
                        "values":     [sheet_name_summary, 1, 1, n, 1],  # avg count
                    }
                )
                chart.set_title({"name": f"{mode} - Avg Count per Step"})
                chart.set_x_axis({"name": "Step"})
                chart.set_y_axis({"name": "Avg Count"})
                chart.set_legend({"position": "bottom"})
                worksheet_summary.insert_chart("D2", chart)

    def save_to_excel(self, filepath: str = "/home/chenxueqing/image-edit-round-reuse/result/QwenImageEdit/analysis/key_token_stats.xlsx"):
        """
        写入 Excel：
          - RawData          : 原始记录
          - Summary          : 模式 × step 汇总 total/avg/max/min
          - {mode}_Pivot     : step × layer 透视 + 色阶
          - {mode}_StepAvg   : per-step 平均 + 折线图
        """
        df = self._to_dataframe()

        # 汇总：mode × step
        if not df.empty:
            df_summary = (
                df.groupby(["mode", "step"])
                .agg(
                    total_count=("count", "sum"),
                    avg_count=("count", "mean"),
                    max_count=("count", "max"),
                    min_count=("count", "min"),
                )
                .reset_index()
                .sort_values(["mode", "step"])
            )
        else:
            df_summary = pd.DataFrame(
                columns=["mode", "step", "total_count", "avg_count", "max_count", "min_count"]
            )

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            # 原始数据
            df.to_excel(writer, sheet_name="RawData", index=False)
            # 汇总
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            if not df.empty:
                self._add_charts_to_excel(writer, df)

        print(f"[KeyTokenStatsCollector] Excel saved to: {filepath}")
stats_collector = KeyTokenStatsCollector()

        
class QwenActivationCacheManager:
    def __init__(
        self,
        use_activation_cache: bool = True,
        cache_steps: Optional[Set[int]] = None,   # None = 所有 step 都缓存
        cache_device: torch.device = torch.device("cuda:1"),
        cache_device2: Optional[torch.device] = torch.device("cuda:3"),
        total_step_num: int = 30,
        num_gpus: int = 2,
        threshold: float = 0.99,
        cache_interval: int = 5,                 # 用于自动生成 cache_steps
    ):
        # 配置
        self.use_activation_cache = use_activation_cache
        self.cache_steps = cache_steps
        self.cache_device = cache_device          # 第一块
        self.cache_device2 = cache_device2        # 第二块
        self.num_gpus = num_gpus
        self.total_step_num = total_step_num
        self.threshold = threshold
        self.cache_interval = cache_interval
        self.current_mode: str = "cond"  # "cond" or "uncond"

        # 轮次 / step
        self.current_round = -1
        self.current_step = -1
        self.get_key_token_indices_step = 0

        # ===== cache：按 mode 分两套 =====
        # key: (stream, step, layer_idx) ；最外层再按 mode 分桶
        self.stream_type: StreamType = "double"
        self.prev_cache: Dict[str, Dict[Tuple[StreamType, int, int], Tensor]] = {
            "cond": {},
            "uncond": {},
        }
        self.new_cache: Dict[str, Dict[Tuple[StreamType, int, int], Tensor]] = {
            "cond": {},
            "uncond": {},
        }

        # ===== 关键 token：按 mode 分两套 =====
        self.key_token_indices: Dict[str, Optional[Tensor]] = {
            "cond": None,
            "uncond": None,
        }


        # ---------- 自动设置 cache_steps ----------
        if self.cache_steps is None:
            self._build_cache_steps_from_interval()
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

    # --------- 配置更新时，允许动态重建 cache_steps ---------
    def set_parameters(
        self,
        num_inference_steps: int,
        threshold: float,
        cache_device: torch.device,
        cache_interval: Optional[int] = None,
    ):
        self.total_step_num = num_inference_steps
        self.threshold = threshold
        self.cache_device = cache_device
        if cache_interval is not None:
            self.cache_interval = cache_interval
            self._build_cache_steps_from_interval()

    # --------- 外部接口：设置当前是 cond / uncond ---------
    def set_mode(self, mode: str):
        """
        mode: "cond" or "uncond"
        """
        assert mode in ("cond", "uncond")
        self.current_mode = mode
        
    # --------- 配置 ---------
    def set_parameters(self, num_inference_steps: int, threshold: float, cache_device: torch.device):
        self.total_step_num = num_inference_steps
        self.threshold = threshold
        self.cache_device = cache_device

    # --------- 轮次 / step ---------
    @property
    def is_round0(self) -> bool:
        return self.current_round == 0

    def on_step_start(self, step: int):
        self.current_step = step
        if step == 0:
            self.current_round += 1

    # --------- cache ---------
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
        prev = None
        for s in self.cache_steps:
            if s <= step:
                prev = s
            else:
                break
        return prev

    def should_reuse(self, step: int) -> bool:
        return (not self.is_round0) and (not self.should_cache(step))

    # --------- store_activation --------
    def maby_store_activation(self, stream: StreamType, layer_idx: int, tensor: Tensor):
        if not (self.use_activation_cache and self.should_cache(self.current_step)):
            return
        mode = self.current_mode
        key = (stream, self.current_step, layer_idx)

        if self.new_cache[mode] is None:
            self.new_cache[mode] = {}

        # 预计这个 tensor 占多少显存
        bytes_per_element = tensor.element_size()   # float16=2, float32=4 ...
        extra_bytes = tensor.numel() * bytes_per_element

        import torch

        # 起始设备：类似 torch.device("cuda:2")
        start_dev: torch.device = self.cache_device
        assert start_dev.type == "cuda", f"cache_device 必须是 cuda 设备，当前是 {start_dev}"

        start_idx = start_dev.index               # int，比如 2
        num_gpus = self.num_gpus     # 总 GPU 数量

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

        self.new_cache[mode][key] = tensor.to(target_device)


    def load_activation(
        self,
        stream: StreamType,
        layer_idx: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> Optional[Tensor]:
        """
        读取当前 mode (cond/uncond) 对应的 prev_cache；
        复用逻辑自动区分两套。
        """
        if not self.use_activation_cache:
            return None
        mode = self.current_mode
        step = step if step is not None else self.current_step
        step_to_load = self.map_to_group_min(step)
        key = (stream, step_to_load, layer_idx)
        prev = self.prev_cache[mode].get(key, None)
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
        mode = self.current_mode
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        prev = self.prev_cache[mode].get(key, None)
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
        mode = self.current_mode
        step = step if step is not None else self.current_step
        key = (stream, step, layer_idx)
        cur = self.new_cache[mode].get(key, None)
        if cur is None:
            return None
        return cur.to(device)

    def flush_new_cache_after_step(self):
        """
        把 new_cache[*][...] 刷到 prev_cache[*][...]。
        两套 cache 同步更新。
        """
        last_step = self.total_step_num - 1
        if not self.use_activation_cache:
            return

        if self.is_round0:
            if self.should_cache(self.current_step):
                for mode in ("cond", "uncond"):
                    self.prev_cache[mode].update(self.new_cache[mode])
                    self.new_cache[mode].clear()
        else:
            if self.current_step >= last_step:
                return
            if self.should_cache(self.current_step + 1):
                for mode in ("cond", "uncond"):
                    self.prev_cache[mode].update(self.new_cache[mode])
                    self.new_cache[mode].clear()

    # --------- token 重排 / 还原（只针对 image tokens + img RoPE） ---------

    @staticmethod
    def rearrange_tensor_with_key_token_indices(
        img: Tensor,
        img_freqs: Tensor,
        modulate_index: Tensor,
        key_token_indices: Tensor,
    ):
        """
        把关键 image tokens 调到前面，同时对 RoPE 的 img_freqs 做相同重排。
        img: (B, L_img, C)
        img_freqs: (L_img, D_freq)
        """
        # 重排 hidden_states
        img_key = torch.index_select(img, 1, key_token_indices)
        mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
        mask[key_token_indices] = False
        img_not_key = img[:, mask, :]
        img_new = torch.cat((img_key, img_not_key), dim=1)

        # 重排 RoPE freqs
        key_idx_freq = key_token_indices.to(img_freqs.device)
        freqs_key = torch.index_select(img_freqs, 0, key_idx_freq)
        mask_f = torch.ones(img_freqs.size(0), dtype=torch.bool, device=img_freqs.device)
        mask_f[key_idx_freq] = False
        freqs_not_key = img_freqs[mask_f, :]
        freqs_new = torch.cat((freqs_key, freqs_not_key), dim=0)
        
        # 重排 modulate_index
        if modulate_index is not None:
            modulate_index_key = torch.index_select(modulate_index, 1, key_token_indices)
            mask = torch.ones(img.size(1), dtype=torch.bool, device=img.device)
            mask[key_token_indices] = False
            modulate_index_not_key = modulate_index[:, mask]
            modulate_index_new = torch.cat((modulate_index_key, modulate_index_not_key), dim=1)
        else:
            modulate_index_new = None
    
        return img_new, freqs_new, modulate_index_new

    @staticmethod
    def restore_original_token_order(x: Tensor, key_token_indices: Tensor) -> Tensor:
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

    def maybe_rearrange_img_and_rope(
        self,
        img: Tensor,
        img_freqs: Tensor,
        modulate_index: Tensor,
    ):
        """
        若 should_reuse(step) 且当前 mode 对应的 key_token_indices 存在，
        则重排 image hidden_states + img_freqs，否则不动。
        返回：img_new, img_freqs_new, key_token_num, should_reuse_flag
        """
        should_reuse = self.should_reuse(self.current_step)
        cur_indices = self.key_token_indices[self.current_mode]
        if should_reuse and cur_indices is not None:
            img, img_freqs, modulate_index = self.rearrange_tensor_with_key_token_indices(
                img, img_freqs, modulate_index, cur_indices
            )
            key_token_num = cur_indices.size(0)
        else:
            key_token_num = img.size(1)
        return img, img_freqs, modulate_index, key_token_num, should_reuse

    def maybe_restore_img_order(self, img: Tensor) -> Tensor:
        cur_indices = self.key_token_indices[self.current_mode]
        if self.should_reuse(self.current_step) and cur_indices is not None:
            img = self.restore_original_token_order(img, cur_indices)
        return img

    # --------- 关键 token 选择 ---------
    @torch.no_grad()
    def compute_key_indices_fn(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        """
        基于两轮同层 hidden_states 的余弦相似度，选出变化较大的 token 下标。
        tensor1, tensor2: (L, C)
        """
        device = tensor2.device
        tensor1 = tensor1.to(device)
        assert tensor1.shape == tensor2.shape

        dot = (tensor1 * tensor2).sum(dim=1)
        n1 = tensor1.norm(p=2, dim=1)
        n2 = tensor2.norm(p=2, dim=1)
        eps = 1e-8
        sim = dot / (n1 * n2 + eps)
        norm_sim = (sim + 1) / 2  # [0, 1]
        # print("norm_sim:", norm_sim)
        # print("self.threshold.size:", self.threshold.shape)
        mask = norm_sim < self.threshold
        idx = mask.nonzero(as_tuple=True)[0]
        return idx

    def update_key_token_indices(self, cur_img: Tensor, ref_img: Tensor):
        """
        在 round>0 且 current_step==0 时，利用当前轮 / 上一轮最后一层输出，更新 key_token_indices。
        注意：现在会把 index 写入当前 mode 对应的那一份。
        cur_img, ref_img: (B, L, C)
        """
        # print("Updating key token indices...", "threshold:", self.threshold)
        if not self.use_activation_cache or self.is_round0:
            return
        if not self.current_step in self.cache_steps or ref_img is None:
            return
        idx = self.compute_key_indices_fn(cur_img[0], ref_img[0])
        # print("mode", self.current_mode, "key token indices found:", idx.shape)
        self.key_token_indices[self.current_mode] = idx.to(cur_img.device)
        # print("current_mode: ", self.current_mode,"Updated key_token_indices:", self.key_token_indices[self.current_mode].shape)

    def reset(self):
        """
        恢复运行时状态到初始值：
        - 轮次 / step 相关变量
        - 所有 cache（prev_cache / new_cache）
        - key_token_indices
        - current_mode 恢复为 'cond'
        """
        # 轮次 / step
        self.current_round = -1
        self.current_step = -1
        self.get_key_token_indices_step = 0

        # cache 清空
        self.prev_cache = {
            "cond": {},
            "uncond": {},
        }
        self.new_cache = {
            "cond": {},
            "uncond": {},
        }

        # 关键 token 索引清空
        self.key_token_indices = {
            "cond": None,
            "uncond": None,
        }

        # mode 复位
        self.current_mode = "cond"
        
QwenActivationCacheManager = QwenActivationCacheManager()

# def create_kernel(kernel_size=3, kernel_type='square'):
#     """
#     Create a morphological operation kernel (structuring element).
    
#     Args:
#         kernel_size (int): The size of the kernel, default is 3x3.
#         kernel_type (str): Type of the kernel, either 'square' or 'cross'.
    
#     Returns:
#         torch.Tensor: The kernel matrix.
#     """
#     if kernel_type == 'square':
#         # Square-shaped kernel
#         kernel = torch.ones(1, 1, kernel_size, kernel_size)
#     elif kernel_type == 'cross':
#         # Cross-shaped kernel
#         kernel = torch.zeros(1, 1, kernel_size, kernel_size)
#         mid = kernel_size // 2
#         kernel[0, 0, mid, :] = 1  # Horizontal line
#         kernel[0, 0, :, mid] = 1  # Vertical line
#     else:
#         raise ValueError("kernel_type must be 'square' or 'cross'")
    
#     return kernel


# def morphological_erosion(image, kernel):
#     """
#     Morphological erosion operation.
    
#     Args:
#         image (torch.Tensor): Input binary image [H, W].
#         kernel (torch.Tensor): Structuring element.
    
#     Returns:
#         torch.Tensor: Eroded image.
#     """
#     # Convert the image to a 4D tensor [1, 1, H, W]
#     if image.dim() == 2:
#         image = image.unsqueeze(0).unsqueeze(0)
    
#     # Perform erosion using convolution
#     # Erosion: the center pixel is 1 only if all pixels covered by the kernel are 1
#     kernel_size = kernel.shape[-1]
#     padding = kernel_size // 2
    
#     # Convert kernel and image to float type
#     kernel = kernel.float()
#     image = image.float()
    
#     # Perform convolution
#     conv_result = F.conv2d(image, kernel, padding=padding)
    
#     # Erosion condition: convolution result equals the number of ones in the kernel
#     kernel_sum = kernel.sum()
#     eroded = (conv_result == kernel_sum).float()
    
#     return eroded.squeeze()


# def morphological_dilation(image, kernel):
#     """
#     Morphological dilation operation.
    
#     Args:
#         image (torch.Tensor): Input binary image [H, W].
#         kernel (torch.Tensor): Structuring element.
    
#     Returns:
#         torch.Tensor: Dilated image.
#     """
#     # Convert the image to a 4D tensor [1, 1, H, W]
#     if image.dim() == 2:
#         image = image.unsqueeze(0).unsqueeze(0)
    
#     kernel_size = kernel.shape[-1]
#     padding = kernel_size // 2
    
#     # Convert kernel and image to float type and move kernel to the same device as image
#     kernel = kernel.float().to(image)
#     image = image.float()
    
#     # Perform convolution
#     conv_result = F.conv2d(image, kernel, padding=padding)
    
#     # Dilation condition: convolution result greater than 0
#     dilated = (conv_result > 0).float()
    
#     return dilated.squeeze()


# def remove_scattered_points(binary_matrix, kernel_size=3, kernel_type='square'):
#     """
#     Remove isolated points in a binary matrix.
    
#     Args:
#         binary_matrix (torch.Tensor): Input binary matrix [H, W].
#         kernel_size (int): Size of the kernel.
#         kernel_type (str): Type of the kernel.
    
#     Returns:
#         torch.Tensor: Processed matrix with scattered points removed.
#     """
#     # Create structuring elements
#     erosion_kernel = create_kernel(3, 'cross').to(binary_matrix)
#     dilation_kernel = create_kernel(5, kernel_type).to(binary_matrix)
    
#     # First, perform erosion
#     eroded = morphological_erosion(binary_matrix, erosion_kernel)
    
#     # Then, perform dilation
#     result = morphological_dilation(eroded, dilation_kernel)

#     return result


# def ids_scatter(gathered_latent, ids, src) -> torch.Tensor:
#     """
#     Scatter gathered latent vectors back into their original positions.
    
#     Args:
#         gathered_latent (torch.Tensor): [batch_size, k, dim] - latent vectors to scatter.
#         ids (torch.Tensor): [batch_size, k] - target indices where to place the latent vectors.
#         src (torch.Tensor): [batch_size, seq_length, dim] - target matrix to store the scattered vectors.
    
#     Returns:
#         torch.Tensor: Updated target matrix [batch_size, seq_length, dim] with scattered values.
#     """
#     B, K, D = gathered_latent.shape

#     # Use scatter to place gathered_latent back to the corresponding positions
#     src[torch.arange(B).unsqueeze(1), ids] = gathered_latent

#     return src  # [B, seq_length, D]


# def ids_gather(latent, ids, rope=False, condition_length=None) -> torch.Tensor:
#     """
#     Gather specific latent vectors from a sequence based on indices.
    
#     Args:
#         latent (torch.Tensor): [batch_size, seq_length, dim] - input sequence of latent vectors.
#         ids (torch.Tensor): [batch_size, k] - indices of the positions to gather.
#         rope (bool): Optional, not used here (placeholder for future use).
#         condition_length: Optional, not used here (placeholder for future use).
    
#     Returns:
#         torch.Tensor: Gathered latent vectors [batch_size, k, dim].
#     """
#     B, K = ids.shape

#     # Create batch indices for advanced indexing
#     batch_indices = torch.arange(B, device=latent.device).unsqueeze(1).expand(-1, K)
    
#     if rope:
#         return latent[batch_indices, ids]  # [B, K, D]

#     # Gather the latent vectors at the specified positions
#     return latent[batch_indices, ids, :]  # [B, K, D]


# def token_selector(tensor1, tensor2, k, similarity_type='cosine', height=-1, width=-1,
#                    erosion_dilation=False, kernel_size=5, kernel_type='square',
#                    patch_size=2, vae_scale_factor=8):
#     """
#     Select k similar positions along the seq_length dimension from two tensors.
    
#     Args:
#         tensor1 (torch.Tensor): [batch_size, seq_length, dim]
#         tensor2 (torch.Tensor): [batch_size, seq_length, dim]
#         k (int): Number of similar positions to select.
#         similarity_type (str): Method to compute similarity ('cosine', 'dot', 'euclidean', 'mse', 'diff_std').
#         height (int): Height of the 2D feature map (for erosion/dilation).
#         width (int): Width of the 2D feature map (for erosion/dilation).
#         erosion_dilation (bool): Whether to apply morphological erosion and dilation to remove scattered points.
#         kernel_size (int): Kernel size for morphological operations.
#         kernel_type (str): Type of kernel for morphological operations ('square' or 'cross').
#         patch_size (int): Patch size used in reshaping.
#         vae_scale_factor (int): Scaling factor for reshaping.

#     Returns:
#         indices (torch.Tensor): [batch_size, k] - indices of selected positions (edit region).
#         unselected_indices (torch.Tensor): [batch_size, seq_length - k] - indices of unselected positions (non-edit region).
#     """
#     batch_size, seq_length, dim = tensor1.shape

#     # Compute similarity matrix
#     if similarity_type == 'cosine':
#         # Cosine similarity
#         tensor1_norm = F.normalize(tensor1, dim=-1)
#         tensor2_norm = F.normalize(tensor2, dim=-1)
#         similarity = torch.sum(tensor1_norm * tensor2_norm, dim=-1)  # [batch_size, seq_length]
#     elif similarity_type == 'dot':
#         # Dot product similarity
#         similarity = torch.sum(tensor1 * tensor2, dim=-1)  # [batch_size, seq_length]
#     elif similarity_type == 'euclidean':
#         # Euclidean distance converted to similarity
#         distance = torch.norm(tensor1 - tensor2, dim=-1)
#         similarity = -distance  # smaller distance = higher similarity
#         similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
#     elif similarity_type == 'mse':
#         # Mean squared error converted to similarity
#         diff = tensor1 - tensor2
#         similarity = -torch.mean(diff ** 2, dim=-1)
#     elif similarity_type == 'diff_std':
#         # Standard deviation of differences
#         diff = tensor1 - tensor2
#         similarity = torch.std(diff, dim=-1)
#     else:
#         raise ValueError("similarity_type must be 'cosine', 'dot', 'euclidean', 'mse', or 'diff_std'")

#     # Threshold selection
#     selected_mask = similarity <= k  # [batch_size, seq_length]

#     if erosion_dilation:
#         # Reshape to 2D mask for morphological processing
#         selected_mask = selected_mask.float().squeeze().reshape(
#             height // (patch_size * vae_scale_factor),
#             width // (patch_size * vae_scale_factor)
#         )
#         # Remove isolated points
#         selected_mask = remove_scattered_points(selected_mask, kernel_size, 'square')
#         selected_mask = selected_mask.bool().flatten().unsqueeze(0)

#     # Get indices of selected positions
#     unselected_indices = torch.arange(seq_length, device=tensor1.device).unsqueeze(0).expand(batch_size, -1)
#     indices = unselected_indices[selected_mask].unsqueeze(0)  # selected indices
#     n_selected = indices.shape[1]

#     # Get indices of unselected positions
#     unselected_mask = ~selected_mask
#     unselected_indices = unselected_indices[unselected_mask].view(batch_size, seq_length - n_selected)

#     return indices, unselected_indices  # [edit region, unedit region]


# class Manager:

#     def __init__(self) -> None:
#         # model config
#         self.patch_size = 2
#         self.vae_scale_factor = 8
#         self.inference_step = 28
#         self.txt_length = None
#         self.height = None
#         self.width = None
#         self.latent_length = 0
#         self.condition_latent = None
#         self.condition_length = 0
#         self.latent_ids = None

#         # regione config
#         self.warmup_step = 8
#         self.post_step = 0
#         self.erosion_dilation = False
#         self.threshold = None
#         self.cache_threshold = 0
#         self.refresh_step = []

#         # realtime data
#         self.current_step = 0
#         self.edited_ids = None
#         self.unedited_ids = None
#         self.unedited_latent = None
#         self.prev_refresh_step = None
#         self.next_refresh_step = None
#         self.refresh_step_real_time = []
#         self.next_estimate = None

#     def set_parameters(self, args) -> None:
#         assert args.warmup_step >= 1 and args.num_inference_steps == 28, "Changing the inference step requires fitting a new gamma"
#         self.inference_step = args.num_inference_steps
#         self.warmup_step = args.warmup_step
#         self.post_step = args.post_step
#         self.threshold = args.threshold
#         self.cache_threshold = args.cache_threshold
#         self.erosion_dilation = args.erosion_dilation
#         self.refresh_step = sorted([int(item) for item in args.refresh_step.split(',')])
#         assert min(self.refresh_step) > self.warmup_step + 1 and max(self.refresh_step) <= self.inference_step - self.post_step - 1
#         has_adjacent = lambda nums: any(abs(nums[i] - nums[i+1]) == 1 for i in range(len(nums)-1))
#         assert not has_adjacent(self.refresh_step), "Refresh steps must not be adjacent."
#         self.refresh_step.append(self.inference_step - self.post_step + 1)


#     def step(self, latent, latent_ids) -> torch.Tensor:
#         self.current_step += 1

#         if self.current_step == self.warmup_step:
#             self.unedited_latent = ids_gather(latent, self.unedited_ids)
#             latent = ids_gather(latent, self.edited_ids)
#             latent_ids = ids_gather(latent_ids.unsqueeze(0), self.edited_ids, rope=True, condition_length=self.condition_length).squeeze(0)

#         elif self.current_step == self.inference_step - self.post_step:
#             final_latent = torch.zeros_like(self.condition_latent)
#             final_latent = ids_scatter(latent, self.edited_ids, final_latent)
#             final_latent = ids_scatter(self.unedited_latent, self.unedited_ids, final_latent)
#             latent = final_latent
#             latent_ids = self.latent_ids
#             self.prev_refresh_step = None

#         # gather
#         elif self.prev_refresh_step != None and self.current_step == self.prev_refresh_step:
#             final_latent = torch.zeros_like(self.condition_latent)
#             final_latent = ids_scatter(latent, self.edited_ids, final_latent)
#             final_latent = ids_scatter(self.unedited_latent, self.unedited_ids, final_latent)
#             latent = final_latent
#             latent_ids = self.latent_ids

#         # scatter
#         elif self.prev_refresh_step != None and self.current_step == self.prev_refresh_step + 1:
#             self.unedited_latent = ids_gather(latent, self.unedited_ids)
#             latent = ids_gather(latent, self.edited_ids)
#             latent_ids = ids_gather(latent_ids.unsqueeze(0), self.edited_ids, rope=True, condition_length=self.condition_length).squeeze(0)
#             self.prev_refresh_step = self.next_refresh_step

#         return latent, latent_ids

#     def refresh(
#         self,
#         latents,
#         image_latents,
#         latent_ids,
#         patch_size=2,
#         vae_scale_factor=8,
#         height=None,
#         width=None
#     ) -> None:
#         self.width = width
#         self.height = height
#         self.patch_size = patch_size
#         self.vae_scale_factor = vae_scale_factor
#         self.latent_length = latents.size(1)
#         self.condition_latent = image_latents
#         self.condition_length = image_latents.size(1) if image_latents is not None else 0
#         self.current_step = 0
#         self.prev_refresh_step = None
#         self.next_refresh_step = None
#         self.edited_ids = None
#         self.unedited_ids = None
#         self.unedited_latent = None
#         self.latent_ids = latent_ids
#         self.refresh_step_real_time = list(self.refresh_step)
        
#         self.next_estimate = None

# MANAGER = Manager()
