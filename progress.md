# CacheEdit 重构进度报告

## 检查点：v0.1.0-checkpoint

**日期**: 2026-05-28
**分支**: `refactor/restructure`
**Git Tag**: `v0.1.0-checkpoint`

---

## 已完成的工作

### ✅ 阶段 0：准备工作

#### 步骤 0.1：环境准备
- ✅ 创建分支 `refactor/restructure`
- ✅ 备份原始代码到 `legacy/` 目录
- ✅ 提交备份（commit: `1cb1a3e`）

#### 步骤 0.2：依赖梳理
- ✅ 创建 `requirements.txt`
- ✅ 创建 `pyproject.toml`（含 black、ruff、mypy、pytest 配置）
- ✅ 创建 `setup.py`
- ✅ 提交配置（commit: `7e9e5d0`）

### ✅ 阶段 1：提取公共代码

#### 步骤 1.1：核心工具模块
- ✅ `cache_edit/utils/scheduler_utils.py`（150+ 行）
  - `calculate_shift()`
  - `retrieve_timesteps()`
  - `FlowMatchEulerDiscreteSchedulerOutput`
- ✅ `cache_edit/utils/image_utils.py`（50+ 行）
  - `calculate_dimensions()`
- ✅ 提交（commit: `5fe95ac`）

#### 步骤 1.2：缓存管理器基类
- ✅ `cache_edit/core/cache_manager.py`（238 行）
  - `BaseCacheManager` 抽象基类
  - 标准接口：`store_activation()`, `get_activation()`
  - 公共方法：`on_step_start()`, `should_cache()`, `should_reuse()` 等
- ✅ 提交（commit: `0fc8c6b`）

#### 步骤 1.3：统计收集器基类
- ✅ `cache_edit/core/stats_collector.py`（281 行）
  - `BaseStatsCollector` 抽象基类
  - `KeyTokenStatsCollector` 具体实现
  - 支持 CSV/Excel 导出
- ✅ 提交（commit: `c003213`）

### 🚧 阶段 2：重构模型特定代码（进行中）

#### 步骤 2.1：重构 Qwen 模块（部分完成）
- ✅ 创建 `cache_edit/models/qwen/` 目录
- ✅ `cache_edit/models/qwen/cache_manager.py`（252 行）
  - `QwenCacheManager` 继承 `BaseCacheManager`
  - 支持 cond/uncond 双模式
  - 多 GPU 显存管理
- ✅ 提交（commit: `c6819aa`）

---

## 未完成的工作

### 阶段 2 剩余任务

#### 步骤 2.1：Qwen 模块（剩余）
- ⏳ `cache_edit/models/qwen/scheduler.py` - `RegionEFlowMatchEulerDiscreteScheduler`
- ⏳ `cache_edit/models/qwen/processor.py` - `RegionEQwenDoubleStreamAttnProcessor2_0`
- ⏳ `cache_edit/models/qwen/pipeline.py` - `CacheEditQwenImageEditPipeline`
- ⏳ `cache_edit/models/qwen/stats.py` - Qwen 特定统计

#### 步骤 2.2：Flux 模块（未开始）
- ⏳ `cache_edit/models/flux/cache_manager.py`
- ⏳ `cache_edit/models/flux/scheduler.py`
- ⏳ `cache_edit/models/flux/processor.py`
- ⏳ `cache_edit/models/flux/pipeline.py`
- ⏳ `cache_edit/models/flux/stats.py`

### 阶段 3：配置管理和 CLI（未开始）
- ⏳ 步骤 3.1：实现配置系统
- ⏳ 步骤 3.2：实现 CLI 工具

### 阶段 4：测试和文档（未开始）
- ⏳ 步骤 4.1：编写单元测试
- ⏳ 步骤 4.2：编写文档

### 阶段 5：优化和发布（未开始）
- ⏳ 步骤 5.1：代码质量优化
- ⏳ 步骤 5.2：打包和发布

---

## 代码统计

| 模块 | 文件 | 行数 |
|------|------|------|
| utils | scheduler_utils.py | ~150 |
| utils | image_utils.py | ~60 |
| core | cache_manager.py | 238 |
| core | stats_collector.py | 281 |
| models/qwen | cache_manager.py | 252 |
| **总计** | | **~981** |

外加：
- `plan.md`: 1022 行重构计划
- `requirements.txt`: 20 行
- `pyproject.toml`: 100+ 行
- `setup.py`: 10 行

---

## 提交历史

```
c6819aa feat: 创建 Qwen 缓存管理器
c003213 feat: 创建统计收集器基类
0fc8c6b feat: 创建缓存管理器基类
5fe95ac feat: 创建核心工具模块
7e9e5d0 chore: 添加项目依赖配置文件
1cb1a3e backup: 备份原始代码到 legacy/ 目录
```

---

## 项目结构

```
icml26-CacheEdit/
├── cache_edit/                    # 新架构（重构目标）
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cache_manager.py       ✅ BaseCacheManager
│   │   └── stats_collector.py     ✅ BaseStatsCollector
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── scheduler_utils.py     ✅ 调度器工具
│   │   └── image_utils.py         ✅ 图像工具
│   ├── models/
│   │   ├── __init__.py
│   │   └── qwen/
│   │       ├── __init__.py
│   │       └── cache_manager.py   ✅ QwenCacheManager
│   ├── evaluation/                ⏳ 待实现
│   ├── config/                    ⏳ 待实现
│   └── cli/                       ⏳ 待实现
│
├── legacy/                        # 原始代码备份
│   ├── Qwen-image-edit-plus/
│   └── Flux-kontext/
│
├── Qwen-image-edit-plus/          # 原始代码（待迁移）
├── Flux-kontext/                  # 原始代码（待迁移）
│
├── plan.md                        # 重构计划文档
├── progress.md                    # 本进度报告
├── requirements.txt               # 依赖
├── pyproject.toml                 # 项目配置
└── setup.py                       # 安装脚本
```

---

## 验证结果

所有已完成的模块都通过了导入和基本功能测试：

```bash
✓ calculate_shift(256): 0.5
✓ calculate_shift(4096): 1.15
✓ calculate_dimensions(1024*1024, 1.0): (1024, 1024, 1048576)
✓ BaseCacheManager imported successfully
✓ KeyTokenStatsCollector created and records
✓ QwenCacheManager created and modes work
```

---

## 如何恢复工作

```bash
# 1. 切换到重构分支
git checkout refactor/restructure

# 2. 激活 conda 环境
conda activate cacheedit

# 3. 查看最新检查点
git log --oneline v0.1.0-checkpoint

# 4. 继续重构（按 plan.md 中的步骤）
# 下一步：步骤 2.1 剩余部分 - 迁移 Qwen Pipeline/Processor/Scheduler
```

---

## 下一步建议

按 `plan.md` 继续执行：

1. **优先级最高**：完成步骤 2.1 剩余部分
   - Qwen Scheduler
   - Qwen Attention Processor
   - Qwen Pipeline

2. **然后**：执行步骤 2.2 - 重构 Flux 模块（可参考 Qwen 模式）

3. **最后**：阶段 3-5 - 配置、测试、文档、发布

每个组件预计耗时：
- Scheduler: 0.5 天
- Processor: 1 天
- Pipeline: 1-2 天
- Flux 模块: 2-3 天
- 配置 + CLI: 1-2 天
- 测试 + 文档: 2-3 天

**预计剩余时间**: 8-12 天

---

**报告生成时间**: 2026-05-28
**当前分支**: `refactor/restructure`
**检查点 Tag**: `v0.1.0-checkpoint`
