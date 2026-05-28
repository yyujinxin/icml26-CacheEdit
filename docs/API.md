# API 文档

CacheEdit 的公共 API 参考。导入路径以 `cache_edit` 为根。

> 约定：以下签名中的默认值与源码保持一致；标注 ⚙️ 的为工厂/便捷函数。

---

## config 配置

`cache_edit.config`

所有配置类继承自 `BaseConfig`，基于 dataclass，提供加载/合并/校验能力。

### BaseConfig

| 方法 | 说明 |
| --- | --- |
| `from_dict(data)` | 从字典创建（自动忽略未知键，递归处理嵌套 dataclass） |
| `from_yaml(path)` | 从 YAML 加载（需 `pyyaml`） |
| `from_json(path)` | 从 JSON 加载 |
| `from_file(path)` | 按扩展名自动选择 `.yaml/.yml/.json` |
| `merge(other)` | 合并另一个配置或字典；仅显式给出的字段覆盖当前值 |
| `apply_env_overrides(prefix="CACHEEDIT")` | 应用环境变量覆盖，返回新实例 |
| `to_dict()` | 递归转字典 |
| `validate()` | 子类校验钩子，非法时抛 `ValueError` |

**环境变量规则**：`{PREFIX}_{FIELD}`，嵌套字段用双下划线，如 `CACHEEDIT_CACHE__THRESHOLD`、`CACHEEDIT_MODEL__DEVICE`。

### QwenConfig

```python
from cache_edit.config import QwenConfig
cfg = QwenConfig.from_yaml("configs/qwen_default.yaml")
cfg = cfg.apply_env_overrides()
cfg.validate()
```

字段分组：`model` (`QwenModelConfig`)、`cache` (`QwenCacheConfig`)、
`pipeline` (`QwenPipelineConfig`)、`output_dir`。

- `QwenModelConfig`: `model_path`, `device`, `dtype`, `device_map`
- `QwenCacheConfig`: `threshold`, `cache_interval`, `enable_stats`, `use_activation_cache`
- `QwenPipelineConfig`: `num_inference_steps`, `guidance_scale`, `height`, `width`, `max_sequence_length`

校验：`threshold ∈ [0,1]`，`cache_interval > 0`，`num_inference_steps > 0`，
`dtype ∈ {float32, float16, bfloat16}`。

### FluxConfig

字段分组：`model`、`cache`、`pipeline`、`viz` (`FluxVizConfig`)、`output_dir`。

- `FluxCacheConfig`: Qwen 字段 + `num_gpus`
- `FluxPipelineConfig`: Qwen 字段 + `true_cfg`
- `FluxVizConfig`: `enable`, `gen_dir`, `viz_out_dir`, `csv_out_path`,
  `edit_ratio_summary_candidates`, `rounds_per_image`, `ref_layer_idx` (默认 37),
  `ref_stream` (默认 "single")

---

## core 核心

`cache_edit.core`

### BaseCacheManager（抽象基类）

```python
BaseCacheManager(
    use_activation_cache=True,
    cache_steps=None,            # None -> 按 cache_interval 自动生成
    cache_device=torch.device("cuda"),
    total_step_num=30,
    threshold=0.95,
    cache_interval=5,
)
```

关键属性/方法：

| 成员 | 说明 |
| --- | --- |
| `current_round` / `current_step` | 轮次/步索引（初始 -1） |
| `is_round0` (property) | 是否第一轮 |
| `on_step_start(step)` | 每步开始时调用；`step==0` 视为新一轮 |
| `should_cache(step)` | 该步是否应写入缓存 |
| `should_reuse(step)` | 该步是否应复用缓存 |
| `store_activation(...)` / `get_activation(...)` | 写入/读取激活 |
| `flush_new_to_prev()` | 将本轮新缓存提交为上一轮缓存 |
| `clear_cache()` | 清空缓存 |
| `set_parameters(...)` | 运行时调整参数 |
| `get_stats()` / `__repr__()` | 统计信息 |

### StatsCollector

- `BaseStatsCollector` — 统计基类
- `KeyTokenStatsCollector` — 记录关键 token 比例，支持 `to_dataframe()`、`summary()`、`report()`、`reset()`

---

## models.qwen

`cache_edit.models.qwen`

| 符号 | 说明 |
| --- | --- |
| `QwenCacheManager` | Qwen 缓存管理器，支持 `cond`/`uncond` 模式隔离 |
| `QwenRegionAwareScheduler` | 区域感知调度器 |
| `QwenDoubleStreamCacheAttnProcessor` | 带缓存的注意力处理器 |
| ⚙️ `init_qwen_pipeline(...)` | 构建带缓存的 Qwen pipeline |
| ⚙️ `create_default_cache_manager(...)` | 默认 Qwen 缓存管理器 |

```python
init_qwen_pipeline(
    model_path,
    device="cuda",
    dtype=torch.bfloat16,
    cache_manager=None,           # None -> 不使用缓存
    use_region_aware_scheduler=True,
    use_cache_processor=True,
)

create_default_cache_manager(
    num_inference_steps=30,
    threshold=0.99,
    cache_interval=5,
    cache_device=None,            # None -> cuda:0 或 cpu
    num_gpus=1,
)
```

---

## models.flux

`cache_edit.models.flux`

| 符号 | 说明 |
| --- | --- |
| `FluxCacheManager` | Flux 缓存管理器，含关键 token 重排/复原逻辑 |
| `FluxAttnCacheProcessor` | 带缓存的注意力处理器 |
| `CacheFluxKontextPipeline` | Flux Kontext pipeline |
| `FluxCacheVizConfig` | 关键 token 可视化配置 |
| `FluxKeyTokenStatsCollector` | 关键 token 统计 |
| `PREFERRED_KONTEXT_RESOLUTIONS` | 推荐分辨率列表 |
| ⚙️ `init_flux_pipeline(...)` | 构建带缓存的 Flux pipeline |
| ⚙️ `create_default_cache_manager(...)` | 默认 Flux 缓存管理器 |

辅助函数：`cache_flux_transformer_2d_forward`、`cache_flux_transformer_block_forward`、
`cache_flux_single_transformer_block_forward`、`append_key_token_ratio_with_edit_ratio`、
`infer_image_id_from_csv_by_round`、`visualize_key_tokens_on_image`。

`FluxCacheManager` 关键方法：`map_to_group_min(step)`、`store_activation/load_activation`、
`flush_new_cache_after_step`、`compute_key_indices_fn(a, b)`、`update_key_token_indices(cur, ref)`、
`rearrange_tensor_with_key_token_indices(...)`、`restore_original_token_order(...)`、`reset()`。

---

## utils 工具

`cache_edit.utils`

| 函数 | 签名 | 说明 |
| --- | --- | --- |
| `calculate_shift` | `(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16)` | flow-match 时间步偏移 |
| `retrieve_timesteps` | `(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs)` | 获取时间步序列 |
| `calculate_dimensions` | `(target_area, ratio) -> (width, height, actual_area)` | 按面积+比例算 32 对齐尺寸 |

---

## cli 命令行

`cache_edit.cli`

- `build_parser() -> argparse.ArgumentParser` — 构建解析器
- `main(argv=None) -> int` — 入口，返回退出码

子命令见 [README CLI 参考](../README.md#cli-参考) 与 [教程](TUTORIAL.md)。
