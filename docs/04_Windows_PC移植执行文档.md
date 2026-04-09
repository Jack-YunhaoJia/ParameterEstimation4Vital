# preset-first 生产流程 Windows PC 移植执行文档

## 1. 文档目标

本文档用于指导把当前仓库已经落地的 `preset-first` 数据生产主线移植到 Windows PC，并且以最小风险完成第一轮可运行验证。

本次移植的目标不是“一次性把所有历史链路都搬过去”，而是先让下面这条主线在 Windows 上跑通：

`generate_pilot_manifest.py -> run_fidelity_spike.py -> run_pilot.py`

移植完成的验收标准是：

1. Windows 可以成功加载 Vital VST3 并执行真实 preset 渲染
2. `pilot` 生产可以输出 `preset_corpus_dataset.h5`
3. `resume` 可以在中断后继续执行
4. 生成的数据不是“零 embedding 占位版”

## 2. 迁移范围与边界

本轮只迁移当前主线：

- `scripts/generate_pilot_manifest.py`
- `scripts/run_fidelity_spike.py`
- `scripts/run_pilot.py`
- `src/preset_dataset_producer.py`
- `src/renderer_backend.py`
- `src/audio_renderer.py`
- `src/multi_condition_renderer.py`

本轮不作为首要目标的内容：

- 旧版 `parameter-first` 大规模生产链
- `parallel_producer.py` 相关历史恢复脚本
- `run_phase0.py` / `run_phase1.py` 的全链路 Windows 适配

原因很简单：当前仓库的业务主线已经是 `preset-first`，而旧链路还保留了更多 `mps` 默认值、历史配置和并行实现，先迁移它们会显著扩大风险面。

## 3. 先确认的事实

当前这次移植不是“需要重写架构”的问题，更像“把现有主线从 macOS 默认假设改成平台中立”。

已经确认的关键事实：

- `pathlib`、HDF5、预设扫描、预设解析、schema 提取、route mask、preset 写回这些核心逻辑本身是跨平台友好的
- `pedalboard` 官方文档明确支持在 macOS、Windows、Linux 上加载 VST3 插件
- Steinberg 官方文档明确说明 Windows 下 VST3 的标准安装目录是 `C:\Program Files\Common Files\VST3`

这意味着当前最主要的工作不是重做 renderer，而是修正仓库里的平台默认值、CLI 参数、环境说明和验收流程。

参考资料：

- [Pedalboard 文档](https://spotify.github.io/pedalboard/)
- [Pedalboard API: `load_plugin`](https://spotify.github.io/pedalboard/reference/pedalboard.html)
- [Steinberg: VST plug-in locations on Windows](https://helpcenter.steinberg.de/hc/en-us/articles/115000177084-VST-plug-in-locations-on-Windows)

## 4. 当前阻塞点清单

### P0 阻塞点

这些问题不先改掉，Windows 上即使“脚本能启动”，最终也很容易得到不可用产物。

1. `src/preset_dataset_producer.py`
   当前 `ProducerConfig.embedding_device` 默认值是 `"mps"`。在 Windows 上这通常不可用，结果会导致 `EmbeddingExtractor` 初始化失败，producer 退回 1024 维零向量占位，最终数据集看起来能产出，但实际上 embedding 无效。

2. `scripts/run_pilot.py`
   当前没有 `--embedding-device`、`--vital-root`、`--vst-path` 这类平台适配入口，同时报错示例是 macOS 的 `export VITAL_VST_PATH=...`。

3. `scripts/run_fidelity_spike.py`
   同样只有 macOS 风格的环境变量示例，缺少 Windows 友好的执行入口和提示。

4. `scripts/run_pilot_no_render.py`
   当前硬编码了 `Path("/Users/jack/Music/Vital")`，这在 Windows 上无法直接复用。

5. `README.md` 与部分现有文档
   当前环境描述仍以 macOS 为已验证环境，Windows 迁移完成后需要补充交叉平台说明。

6. 测试中的默认值假设
   至少已有测试把默认 embedding 设备写死成了 `mps`，Windows 适配后这些断言需要同步修改。

### P1 建议项

这些不是首个 blocker，但做了会显著降低 Windows 首轮试跑成本。

1. 为平台相关逻辑增加统一 helper，而不是把设备判断散落在多个脚本中
2. 把 `run_pilot_no_render.py` 也一起参数化，作为 Windows 环境下的兜底 dry-run 工具
3. 增加最小化的 Windows smoke test 步骤，避免一上来直接长时间生产

## 5. 推荐的实现原则

移植时请严格遵循下面几条原则：

1. 不改 HDF5 结构，不改 `param_values` 编码契约
2. 不改 `preset-first` 的业务流程顺序
3. 先让 `pilot` 可跑，再谈 `canary/full`
4. 首轮 Windows 验证优先使用 `cpu` embedding，先排除 CUDA 驱动变量
5. 任何一步只要出现“脚本能跑但 embedding 全零”，都应判定为移植失败

## 6. 代码改造任务单

### 任务 A：把 embedding 设备改成平台中立

目标文件：

- `src/preset_dataset_producer.py`

执行要求：

1. 把 `ProducerConfig.embedding_device` 从固定字符串 `"mps"` 改成 `str | None = None`
2. 增加统一的设备解析逻辑，建议规则如下：
   - 如果用户显式传入 `cpu` / `cuda` / `mps`，则优先使用显式值
   - 如果用户未传值，则按 `cuda -> mps -> cpu` 自动选择
3. 如果用户显式要求一个当前不可用的设备，应给出清晰报错，而不是静默回退
4. 在日志中明确打印“请求设备”和“实际解析后的设备”

建议实现形式：

```python
def resolve_embedding_device(requested: str | None) -> str:
    if requested is not None:
        return validate_requested_device(requested)

    import torch

    if torch.cuda.is_available():
        return "cuda"

    has_mps = hasattr(torch.backends, "mps")
    if has_mps and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
```

验收标准：

- 在没有 MPS 的环境下不会默认退回零 embedding
- Windows 首轮可用 `cpu` 明确运行

### 任务 B：给主脚本补平台参数入口

目标文件：

- `scripts/run_pilot.py`
- `scripts/run_fidelity_spike.py`
- `scripts/run_pilot_no_render.py`

执行要求：

1. `run_pilot.py` 增加这些 CLI 参数：
   - `--vst-path`
   - `--embedding-device`
   - `--vital-root`
   - `--corpus-dir`
2. `run_fidelity_spike.py` 至少增加：
   - `--vst-path`
   - `--corpus-manifest` 或继续沿用 `--manifest`
3. `run_pilot_no_render.py` 去掉任何 `/Users/jack/...` 的硬编码路径，改成 CLI 参数或环境变量
4. 环境变量优先级建议：
   - CLI 参数
   - 环境变量
   - 默认值
5. 报错提示同时给出 PowerShell 和 bash 示例

推荐的 Windows PowerShell 提示格式：

```powershell
$env:VITAL_VST_PATH = 'C:\Program Files\Common Files\VST3\Vital.vst3'
```

推荐的 bash 提示格式：

```bash
export VITAL_VST_PATH=/Library/Audio/Plug-Ins/VST3/Vital.vst3
```

验收标准：

- Windows 用户无需修改源码即可从命令行传入 VST 路径和 embedding 设备
- `run_pilot_no_render.py` 可以在没有任何 macOS 用户目录的情况下运行

### 任务 C：更新平台文档与运行提示

目标文件：

- `README.md`
- `docs/01_数据生产流水线技术文档.md`
- `docs/02_训练数据集使用指南.md`

执行要求：

1. 保留“当前主线已在 macOS 验证”的事实
2. 新增“Windows 移植中 / Windows 执行方式”说明
3. 补充 PowerShell 环境变量示例
4. 明确说明：
   - Windows 首轮建议 `--embedding-device cpu`
   - `pedalboard` + VST3 是迁移保留方案，不需要先换 backend

验收标准：

- 新同事只看文档就能知道 Windows 首轮怎么配置、怎么跑、怎么判断失败

### 任务 D：补测试，防止回归到 MPS-only

目标文件：

- `tests/test_preset_dataset_producer.py`
- 新增与平台辅助函数相关的测试文件

执行要求：

1. 删除“默认设备必须是 `mps`”这种平台绑定断言
2. 增加设备解析逻辑测试：
   - 显式 `cpu`
   - 显式 `cuda` 但不可用
   - 显式 `mps` 但不可用
   - 自动解析到 `cpu`
   - 自动解析到 `cuda`
3. 补一个最小脚本参数测试，至少验证 `--embedding-device cpu` 能传到 producer 配置

验收标准：

- 在 macOS 和 Windows 上，测试都不会因为“默认设备写死”而失败

## 7. Windows 执行顺序

下面是推荐的实际执行顺序。不要跳步骤，尤其不要在没有做 smoke test 的情况下直接开长时间生产。

### 阶段 1：Windows 环境准备

推荐使用 Windows 11 x64 + Python 3.10/3.11。

PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

如果首轮只做稳定性验证，建议先用 CPU 版 PyTorch，不要在第一步就把 CUDA 驱动问题混进来。

如果 `muq` 首次运行需要下载模型，请确保机器有网络，或提前准备 HuggingFace 缓存。

### 阶段 2：Vital VST3 路径确认

先确认 Vital VST3 已正确安装。

Windows 下 VST3 标准目录应优先检查：

```text
C:\Program Files\Common Files\VST3
```

典型环境变量设置：

```powershell
$env:VITAL_VST_PATH = 'C:\Program Files\Common Files\VST3\Vital.vst3'
```

如果波表目录扫描也要启用，请同时准备 `VITAL_ROOT`，但不要在文档里硬编码一个未经确认的 Windows 用户路径。应以实际安装目录为准，由执行者传参或设置环境变量。

### 阶段 3：插件加载 smoke test

先验证 `pedalboard` 能否真正加载插件，不要直接跑整条生产链。

PowerShell 示例：

```powershell
python -c "import os; from pedalboard import load_plugin; p = load_plugin(os.environ['VITAL_VST_PATH']); print(p.is_instrument, len(p.parameters))"
```

验收标准：

- 命令成功返回
- `is_instrument` 为真
- 能打印出非零数量的参数

如果这一步失败，不要继续跑后面的脚本。先排查：

1. Python 是否为 64 位
2. Vital 是否为 64 位 VST3
3. `VITAL_VST_PATH` 是否真的指向 `.vst3`
4. 插件是否安装在标准 VST3 目录中

### 阶段 4：单元测试与脚本干跑

先跑不依赖实际渲染的测试，再跑最小脚本。

```powershell
pytest tests/test_preset_dataset_producer.py
pytest tests/test_renderer_backend.py
pytest tests/test_preset_render_audit.py
```

然后生成当前 pilot manifest：

```powershell
python scripts\generate_pilot_manifest.py
```

验收标准：

- `pilot_manifest.json` 成功生成
- 当前 9 个 preset 都被扫描出来

### 阶段 5：先跑 Fidelity Spike

这是 Windows 移植的第一道真实渲染验证门。

```powershell
python scripts\run_fidelity_spike.py --manifest pilot_manifest.json --vst-path $env:VITAL_VST_PATH
```

注意：

- 该脚本退出码为 `0` 表示当前 backend 可以继续做 pilot
- 退出码为 `2` 表示渲染保真上建议换 backend，但这不等价于“Windows 移植失败”
- 退出码为 `1` 才应优先按“移植/环境错误”处理

对 Windows 移植而言，这一步的最低验收是：

1. 脚本能完成全部 preset 的审计
2. 不发生纯平台层面的崩溃
3. 能拿到完整审计报告

### 阶段 6：先用 CPU 跑 pilot 生产

首轮建议明确指定 CPU，先把平台问题和 GPU 问题分开。

```powershell
python scripts\run_pilot.py --output-dir experiments\pilot_run_win --embedding-device cpu --vst-path $env:VITAL_VST_PATH
```

验收标准：

1. 能成功生成：
   - `checkpoint.json`
   - `corpus_schema.json`
   - `preset_corpus_dataset.h5`
   - `production_summary.json`
2. 至少一个 split 中存在非空样本
3. `preset_corpus_dataset.h5` 的结构与当前 macOS 主线一致
4. embedding 不是零向量占位

可用下面的快速检查命令：

```powershell
python -c "import h5py, numpy as np; hf = h5py.File(r'experiments\pilot_run_win\preset_corpus_dataset.h5', 'r'); e = hf['train']['embeddings'][:]; print(e.shape, float(e.std()))"
```

如果标准差接近 `0.0`，应优先怀疑 `EmbeddingExtractor` 没真正初始化成功。

### 阶段 7：做一次 interrupt / resume drill

不要把“resume 代码存在”误当成“resume 真的可用”。要做一次真实演练。

执行方式：

1. 启动 `run_pilot.py`
2. 等到至少已经写出 `checkpoint.json` 和部分 `shard`
3. 人为中断进程
4. 重新执行同一命令，但不要加 `--no-resume`

验收标准：

1. 能跳过已完成的 base preset
2. 能继续处理剩余 preset
3. 最终 HDF5 可以正常 finalize
4. 不出现明显的重复样本爆增

### 阶段 8：再决定是否切到 CUDA 或更大规模

只有在 CPU 版 `pilot` 完全通过后，才建议继续做：

1. `--embedding-device cuda` 的二次验证
2. `canary` 模式验证
3. 更大规模 preset corpus 生产

## 8. 明确的完成定义

只有同时满足下面这些条件，才算 Windows 移植完成：

1. `pedalboard.load_plugin()` 可以在 Windows 成功加载 Vital VST3
2. `run_fidelity_spike.py` 能完整跑完真实 pilot 语料
3. `run_pilot.py` 能输出完整 HDF5
4. `resume` 演练通过
5. 生成的 embedding 不是零向量占位
6. 没有保留任何 `/Users/jack/...`、`/Library/Audio/...`、`"mps"` 默认值这类 macOS-only 关键阻塞项

## 9. 常见失败与处理建议

### 情况 1：插件加载失败

优先检查：

1. `VITAL_VST_PATH` 是否指向真实的 `.vst3`
2. Python 与插件是否同为 64 位
3. Vital 是否正确安装在 Windows 标准 VST3 目录

### 情况 2：`run_pilot.py` 跑完了，但 embedding 全零

这通常说明：

1. `EmbeddingExtractor` 初始化失败
2. 设备选择逻辑仍然回到了无效分支
3. 当前产物只是“流程联调占位版”

处理原则：

- 这应判定为移植失败，不要继续进入 canary/full

### 情况 3：`run_fidelity_spike.py` 返回 2

这通常是业务上的 backend fidelity gate，不一定是 Windows 兼容性问题。

处理原则：

- 先区分“平台执行成功但 fidelity 不达标”与“脚本本身跑不通”

### 情况 4：wavetable 分类不完整

如果 Windows 上暂时拿不到正确的 Vital 内容目录：

1. 不要硬编码一个猜测路径
2. 允许先把部分波表归到 `embedded`
3. 但要在日志和 summary 里如实记录

## 10. 移植后的下一步

Windows `pilot` 验证完成后，推荐顺序是：

1. 补 README 的跨平台运行说明
2. 视情况增加 Windows CI smoke test
3. 再决定是否继续迁移旧版 `parameter-first` 链路

如果还要继续迁移旧链路，应单独开一个任务，不建议和这次 `preset-first` Windows 迁移混在一起。
