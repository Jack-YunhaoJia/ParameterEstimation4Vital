# Vital 参数估计与预设语料生产

从音频反推 [Vital](https://vital.audio/) 合成器参数的研究项目。当前仓库同时保留两条数据生产路线：

- `preset-first`：以现成 `.vital` 预设为输入，做解析、路由掩码增强、渲染、预处理、embedding 提取和 HDF5 落盘
- `parameter-first`：基于 45 维 `CORE_PARAMS` 随机采样的旧版生产链，用于历史数据集和 baseline

当前主线已经切换到 `preset-first`。仓库根目录下的 [presets](presets/) 和 [pilot_manifest.json](pilot_manifest.json) 提供了第一轮 `pilot` 小样本语料。

## 当前状态

### 主线能力

- 扫描真实 `.vital` 预设并验证 JSON 可解析性
- 对预设执行 `Fidelity Spike` 审计，统计 `applied / unsupported / modulation_gap / wavetable_gap`
- 从真实预设语料自动提取 `CorpusSchema`
- 对路由图做 mask 增强，生成 `variant_0 + masked variants`
- 用固定 6 个 MIDI 条件渲染音频
- 预处理音频并提取 1024 维 MuQ embedding
- 将全部可训练字段编码到单个 `float32 param_values` 稠密矩阵
- 以 `base_preset_id` 为分组单位做 train / val / test 划分
- 通过 `checkpoint.json + split shards` 支持 resume

### 当前 pilot 语料

- 语料目录：`presets/`
- manifest：`pilot_manifest.json`
- 当前样本数：9 个真实 preset
- 典型用途：先跑 `Fidelity Spike`，再跑 `pilot` 端到端生产，最后做 interrupt/resume drill

### 当前实现注意事项

- 当前 `preset-first` 渲染使用 `MultiConditionRenderer` 的默认 6 条件，每条条件默认 `2.0s`；自适应时长目前仍主要在旧版 `parameter-first` 流水线中
- `param_values` 是唯一监督矩阵；categorical 字段会先编码成 `category_id` 再写入 `float32`
- `present_mask` 用来区分“字段原本存在”与“由默认值填充”
- `route_edge_schema_json` 当前还是占位元数据
- schema 中已经预留 modulation / wavetable 扩展字段，但当前 sample-level 编码仍主要基于 `preset.settings`
- 如果 `EmbeddingExtractor` 初始化失败，当前 producer 会回退到 1024 维零向量占位，这只适合流程联调，不适合正式数据集

## 项目结构

```text
├── src/
│   ├── preset_corpus_scanner.py
│   ├── preset_introspector.py
│   ├── preset_render_audit.py
│   ├── preset_schema_extractor.py
│   ├── route_graph_builder.py
│   ├── route_mask_augmenter.py
│   ├── mutated_preset_writer.py
│   ├── renderer_backend.py
│   ├── wavetable_catalog.py
│   ├── preset_dataset_producer.py
│   ├── preset_parser.py
│   ├── audio_renderer.py
│   ├── multi_condition_renderer.py
│   ├── audio_preprocessor.py
│   ├── embedding_extractor.py
│   ├── parallel_producer.py
│   ├── training_data.py
│   └── parameter_regressor.py
├── scripts/
│   ├── generate_pilot_manifest.py
│   ├── run_fidelity_spike.py
│   ├── run_pilot.py
│   ├── run_production.py
│   └── run_phase1.py
├── presets/
├── docs/
│   ├── 01_数据生产流水线技术文档.md
│   ├── 02_训练数据集使用指南.md
│   └── 03_参数映射问题发现与解决.md
├── tests/
└── .kiro/specs/preset-corpus-pipeline/
```

## 环境要求

- Python >= 3.10
- macOS
- 已安装 Vital VST3
- 环境变量 `VITAL_VST_PATH` 指向 Vital 插件路径

示例：

```bash
export VITAL_VST_PATH=/Library/Audio/Plug-Ins/VST3/Vital.vst3
```

## 安装

```bash
pip install -e .

# 开发与测试依赖
pip install -e ".[dev]"
```

## 快速开始

### 1. 生成 pilot manifest

```bash
python3 scripts/generate_pilot_manifest.py
```

输出：

- `pilot_manifest.json`

### 2. 跑 Fidelity Spike

```bash
python3 scripts/run_fidelity_spike.py
```

可选参数：

```bash
python3 scripts/run_fidelity_spike.py \
  --manifest pilot_manifest.json \
  --mod-gap-threshold 0.3 \
  --wt-gap-threshold 0.3
```

### 3. 跑 preset-first pilot 生产

```bash
python3 scripts/run_pilot.py
```

可选参数：

```bash
python3 scripts/run_pilot.py \
  --output-dir experiments/pilot_run

python3 scripts/run_pilot.py \
  --output-dir experiments/pilot_run \
  --no-resume
```

典型输出目录：

```text
experiments/pilot_run/
├── audio/
├── variants/
├── shards/
│   ├── train/
│   ├── val/
│   └── test/
├── checkpoint.json
├── corpus_schema.json
├── preset_corpus_dataset.h5
└── production_summary.json
```

### 4. 旧版 parameter-first 生产

```bash
python3 scripts/run_production.py \
  --config configs/adaptive_10k.yaml \
  --output-dir experiments/adaptive_10k \
  --yes
```

### 5. 旧版 Phase 1 训练

```bash
python3 scripts/run_phase1.py
```

## 当前数据集结构概览

当前 `preset-first` 数据集文件为：

- `preset_corpus_dataset.h5`

每个 split 当前包含：

- `param_values`
- `present_mask`
- `embeddings`
- `base_preset_id`
- `variant_id`
- `route_mask_json`
- `sample_id`
- `midi_note`
- `midi_velocity`
- `midi_duration`
- `rms_db`
- `split_local_index`

metadata 当前包含：

- `param_names`
- `param_types`
- `param_value_encoding`
- `default_values`
- `corpus_min`
- `corpus_max`
- `presence_ratio`
- `category_values_json`
- `route_edge_schema_json`
- `corpus_schema_json`
- `run_mode`
- `checkpoint_manifest_json`

详细说明见 [docs/01_数据生产流水线技术文档.md](docs/01_数据生产流水线技术文档.md) 和 [docs/02_训练数据集使用指南.md](docs/02_训练数据集使用指南.md)。

## 文档索引

- [docs/01_数据生产流水线技术文档.md](docs/01_数据生产流水线技术文档.md)
  - 当前 `preset-first` 数据结构
  - 生产方法逻辑
  - run mode、checkpoint、shard、HDF5 finalize
- [docs/02_训练数据集使用指南.md](docs/02_训练数据集使用指南.md)
  - `preset_corpus_dataset.h5` 结构解析
  - categorical 解码方法
  - PyTorch / h5py 使用示例
- [docs/03_参数映射问题发现与解决.md](docs/03_参数映射问题发现与解决.md)
  - Vital 参数映射逆向过程
- [docs/04_Windows_PC移植执行文档.md](docs/04_Windows_PC移植执行文档.md)
  - Windows PC 移植执行步骤
  - 平台阻塞点、验收门槛与 resume drill

## 测试

```bash
pytest
```

也可以按模块运行：

```bash
pytest tests/test_preset_dataset_producer.py
pytest tests/test_preset_render_audit.py
pytest tests/test_route_mask_augmenter.py
```

## License

Research use only.
