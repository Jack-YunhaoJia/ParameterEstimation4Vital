# Vital 合成器参数估计

从音频信号反向估计 [Vital](https://vital.audio/) 合成器参数的研究项目。

## 项目简介

本项目构建了一个自动化数据生产流水线，通过随机采样 Vital 合成器的 45 维参数空间，渲染音频并提取 MuQ embedding，生成大规模的（参数, embedding）配对训练数据集。最终目标是训练一个从音频 embedding 反推合成器参数的回归模型。

### 数据集统计

| 指标 | 值 |
|------|-----|
| 有效样本数 | 94,586 |
| 参数维度 | 45 |
| Embedding 维度 | 1024 (MuQ) |
| 预设数量 | 17,007 |
| 渲染条件 | 6 (3 音高 × 2 力度) |
| 训练/验证/测试 | 75,668 / 9,458 / 9,460 |

## 项目结构

```
├── src/                        # 核心模块
│   ├── smart_sampler.py        # LHS + 分层效果器开关采样
│   ├── preset_generator.py     # Vital .vital 预设文件生成
│   ├── audio_renderer.py       # pedalboard + Vital VST3 渲染 (含参数映射)
│   ├── multi_condition_renderer.py  # 多条件渲染 (6 MIDI 条件)
│   ├── audio_preprocessor.py   # 音频预处理 (重采样/归一化/过滤)
│   ├── embedding_extractor.py  # MuQ embedding 提取
│   ├── quality_validator.py    # 质量验证 + 近似重复检测
│   ├── distribution_analyzer.py # 分布分析
│   ├── parallel_producer.py    # 并行生产调度器
│   ├── training_data.py        # CORE_PARAMS 定义 + HDF5 读写
│   ├── preset_parser.py        # Vital 预设解析
│   ├── parameter_regressor.py  # 参数回归模型
│   ├── discriminator.py        # 区分能力评估
│   └── pipeline.py             # Phase 0/1 流水线
├── scripts/                    # 运行脚本
│   ├── run_production.py       # 生产流水线入口
│   ├── run_phase0.py           # Phase 0: 数据生成
│   ├── run_phase1.py           # Phase 1: 模型训练
│   └── debug_render_test.py    # 渲染调试工具
├── tests/                      # 单元测试 (361 tests)
├── configs/                    # 配置文件
│   ├── default.yaml            # 默认配置
│   └── production.yaml         # 生产配置
├── docs/                       # 文档
│   ├── 01_数据生产流水线技术文档.md
│   ├── 02_训练数据集使用指南.md
│   └── 03_参数映射问题发现与解决.md
└── experiments/                # 实验输出
    └── production_run/         # 生产运行结果
        ├── production_dataset.h5   # HDF5 数据集
        ├── audio/                  # 102,042 个 WAV 文件
        └── presets/                # 17,007 个 .vital 预设
```

## 环境要求

- Python >= 3.10
- macOS (需要 Vital VST3 插件: `/Library/Audio/Plug-Ins/VST3/Vital.vst3`)
- [Vital](https://vital.audio/) 合成器已安装

## 安装

```bash
pip install -e .

# 开发依赖 (测试)
pip install -e ".[dev]"
```

## 快速开始

### 运行生产流水线

```bash
# 完整生产 (约 4 小时, M4 Pro)
python3 scripts/run_production.py --output-dir experiments/production_run -y

# 小规模测试
python3 scripts/run_production.py --output-dir experiments/test_run --target-samples 100 -y

# 断点恢复
python3 scripts/run_production.py --output-dir experiments/production_run --resume
```

### 使用数据集

```python
import h5py

with h5py.File("production_dataset.h5", "r") as f:
    train_params = f["train"]["params"][:]          # (75668, 45)
    train_embeds = f["train"]["embeddings"][:]      # (75668, 1024)
    param_names = [n.decode() for n in f["metadata"]["param_names"][:]]
```

详细使用方法见 [训练数据集使用指南](docs/02_训练数据集使用指南.md)。

### 运行测试

```bash
pytest
```

## 技术文档

- [数据生产流水线技术文档](docs/01_数据生产流水线技术文档.md) — 流水线架构、参数空间、预处理流程
- [训练数据集使用指南](docs/02_训练数据集使用指南.md) — 数据集格式、加载方法、训练建议
- [参数映射问题发现与解决](docs/03_参数映射问题发现与解决.md) — VST3 参数映射的逆向工程过程

## 关键技术点

### 参数映射

pedalboard 的 `raw_value` 始终是 [0, 1] 归一化值，但 Vital 内部使用多种映射关系：
- **直接映射**：`raw = vital_value`（范围本身是 [0, 1]）
- **线性归一化**：`raw = (value - min) / (max - min)`
- **Power-law**：包络时间 `time = 32 × raw^4`，逆映射 `raw = (time/32)^0.25`
- **MIDI note**：滤波器截止频率 `raw = (midi_note - 8) / 128`

详见 [参数映射文档](docs/03_参数映射问题发现与解决.md)。

## License

Research use only.

## 安装

```bash
pip install -e .

# 开发依赖 (测试)
pip install -e ".[dev]"
```

## 快速开始

### 运行生产流水线

```bash
# 完整生产 (约 4 小时, M4 Pro)
python3 scripts/run_production.py --output-dir experiments/production_run -y

# 小规模测试
python3 scripts/run_production.py --output-dir experiments/test_run --target-samples 100 -y

# 断点恢复
python3 scripts/run_production.py --output-dir experiments/production_run --resume
```

### 使用数据集

```python
import h5py

with h5py.File("production_dataset.h5", "r") as f:
    train_params = f["train"]["params"][:]          # (75668, 45)
    train_embeds = f["train"]["embeddings"][:]      # (75668, 1024)
    param_names = [n.decode() for n in f["metadata"]["param_names"][:]]
```

详细使用方法见 [训练数据集使用指南](docs/02_训练数据集使用指南.md)。

### 运行测试

```bash
pytest
```

## 技术文档

- [数据生产流水线技术文档](docs/01_数据生产流水线技术文档.md) — 流水线架构、参数空间、预处理流程
- [训练数据集使用指南](docs/02_训练数据集使用指南.md) — 数据集格式、加载方法、训练建议
- [参数映射问题发现与解决](docs/03_参数映射问题发现与解决.md) — VST3 参数映射的逆向工程过程

## 关键技术点

### 参数映射

pedalboard 的 `raw_value` 始终是 [0, 1] 归一化值，但 Vital 内部使用多种映射关系：
- **直接映射**：`raw = vital_value`（范围本身是 [0, 1]）
- **线性归一化**：`raw = (value - min) / (max - min)`
- **Power-law**：包络时间 `time = 32 × raw^4`，逆映射 `raw = (time/32)^0.25`
- **MIDI note**：滤波器截止频率 `raw = (midi_note - 8) / 128`

详见 [参数映射文档](docs/03_参数映射问题发现与解决.md)。

## License

Research use only.
