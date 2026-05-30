"""Base statistics collector for tracking cache and token metrics."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class BaseStatsCollector(ABC):
    """
    统计收集器抽象基类。

    该类用于收集和分析多轮图像编辑过程中的各种统计信息，包括：
    - 关键 token 数量统计
    - 缓存命中率
    - 性能指标

    子类需要实现具体的统计逻辑和报告格式。

    Attributes:
        records: 原始记录列表，用于生成 DataFrame 和导出
        stats: 结构化统计数据，用于终端输出
        enabled: 是否启用统计收集
    """

    # 原始记录（适合 DataFrame / Excel / 图表）
    records: List[Dict[str, Any]] = field(default_factory=list)

    # 结构化统计（适合终端可读性输出）
    stats: Dict[str, Any] = field(default_factory=dict)

    # 是否启用统计
    enabled: bool = True

    @abstractmethod
    def record(self, *args, **kwargs) -> None:
        """
        记录一条统计数据。

        该方法在每个需要统计的时间点调用，记录当前状态。
        具体参数由子类定义。
        """
        pass

    @abstractmethod
    def report(self) -> None:
        """
        生成并打印统计报告。

        该方法在统计收集完成后调用，输出人类可读的统计摘要。
        """
        pass

    def to_dataframe(self) -> pd.DataFrame:
        """
        将统计记录转换为 pandas DataFrame。

        Returns:
            pd.DataFrame: 包含所有统计记录的数据框

        Examples:
            >>> collector = MyStatsCollector()
            >>> # ... 收集数据 ...
            >>> df = collector.to_dataframe()
            >>> df.to_csv("stats.csv")
        """
        if not self.records:
            return pd.DataFrame()
        return self._to_dataframe()

    @abstractmethod
    def _to_dataframe(self) -> pd.DataFrame:
        """
        子类实现的 DataFrame 转换逻辑。

        Returns:
            pd.DataFrame: 格式化的数据框
        """
        pass

    def save_to_csv(self, filepath: str) -> None:
        """
        将统计数据保存为 CSV 文件。

        Args:
            filepath: 输出文件路径

        Examples:
            >>> collector.save_to_csv("stats.csv")
        """
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False)
            print(f"✓ Stats saved to {filepath}")
        else:
            print("⚠ No stats to save")

    def save_to_excel(self, filepath: str, sheet_name: str = "Stats") -> None:
        """
        将统计数据保存为 Excel 文件。

        Args:
            filepath: 输出文件路径
            sheet_name: Excel 工作表名称

        Examples:
            >>> collector.save_to_excel("stats.xlsx", sheet_name="KeyTokens")
        """
        df = self.to_dataframe()
        if not df.empty:
            df.to_excel(filepath, sheet_name=sheet_name, index=False)
            print(f"✓ Stats saved to {filepath}")
        else:
            print("⚠ No stats to save")

    def reset(self) -> None:
        """重置所有统计数据。"""
        self.records.clear()
        self.stats.clear()

    def enable(self) -> None:
        """启用统计收集。"""
        self.enabled = True

    def disable(self) -> None:
        """禁用统计收集。"""
        self.enabled = False

    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要。

        Returns:
            Dict[str, Any]: 包含关键统计指标的字典
        """
        return {
            "total_records": len(self.records),
            "enabled": self.enabled,
        }

    def __len__(self) -> int:
        """返回记录数量。"""
        return len(self.records)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"records={len(self.records)}, "
            f"enabled={self.enabled})"
        )


@dataclass
class KeyTokenStatsCollector(BaseStatsCollector):
    """
    关键 token 统计收集器。

    用于记录和分析关键 token 的数量变化，支持：
    - 按 step、layer、mode/stream 分组统计
    - 终端可读报告
    - DataFrame 导出

    Attributes:
        stats: 结构化统计，格式为 {mode/stream: {step: [count_per_layer]}}
    """

    # 重写 stats 类型提示
    stats: Dict[str, Dict[int, List[int]]] = field(default_factory=dict)

    def record(
        self,
        step: int,
        layer_idx: int,
        count: int,
        group_key: str = "default",
    ) -> None:
        """
        记录关键 token 统计。

        Args:
            step: 当前推理步骤
            layer_idx: 层索引
            count: 关键 token 数量
            group_key: 分组键（如 "cond"/"uncond" 或 "single"/"double"）
        """
        if not self.enabled:
            return

        # 1) 记录到 records（给 DataFrame 用）
        self.records.append({
            "step": int(step),
            "layer": int(layer_idx),
            "group": str(group_key),
            "count": int(count),
        })

        # 2) 记录到结构化 stats（给 report 用）
        if group_key not in self.stats:
            self.stats[group_key] = {}
        if step not in self.stats[group_key]:
            self.stats[group_key][step] = []

        lst = self.stats[group_key][step]
        # 确保长度和 layer_idx 对齐
        if len(lst) < layer_idx:
            lst.extend([0] * (layer_idx - len(lst)))
        if len(lst) == layer_idx:
            lst.append(count)
        else:
            lst[layer_idx] = count

    def report(self) -> None:
        """
        打印关键 token 统计报告。

        输出格式：
        ===== Key Token Stats (step-wise) =====
        [Group=cond]
          Step=0 | L0:10 L1:12 ... | Total=xx Avg=yy
          -------------------------------------------
          Step=1 | L0: 8 L1:11 ... | Total=xx Avg=yy
        ===============================================
        """
        print("\n===== Key Token Stats (step-wise) =====")
        if not self.stats:
            print("No stats collected.")
            print("===========================================\n")
            return

        for group_key, step_dict in self.stats.items():
            print(f"[Group={group_key}]")
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
                # 同一组内 step 之间加分隔线
                if i != len(sorted_steps) - 1:
                    print("  " + "-" * 43)
            print()  # 组之间空一行

        print("===========================================\n")

    def _to_dataframe(self) -> pd.DataFrame:
        """
        转换为 DataFrame。

        Returns:
            pd.DataFrame: 包含 step, layer, group, count 列的数据框
        """
        df = pd.DataFrame(self.records)
        if not df.empty:
            df.sort_values(by=["group", "step", "layer"], inplace=True)
        return df

    def get_summary(self) -> Dict[str, Any]:
        """
        获取统计摘要。

        Returns:
            Dict[str, Any]: 包含总记录数、分组数、平均 token 数等
        """
        summary = super().get_summary()
        if self.records:
            df = self.to_dataframe()
            summary.update({
                "num_groups": len(self.stats),
                "avg_count": df["count"].mean(),
                "max_count": df["count"].max(),
                "min_count": df["count"].min(),
            })
        return summary
