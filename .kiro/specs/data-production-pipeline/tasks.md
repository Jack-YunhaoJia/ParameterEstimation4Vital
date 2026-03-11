# 实现计划：实验数据生产与预处理流水线

## 概述

按模块依赖关系分阶段实现：先构建独立的基础模块（音频预处理、智能采样、分布分析），再实现依赖现有组件的模块（多条件渲染、质量验证），最后实现顶层编排模块（并行生产器）和入口脚本。每个模块包含核心实现和可选的属性/单元测试子任务。所有新模块作为独立文件添加到 `src/`，通过组合方式与现有 `AudioRenderer`、`EmbeddingExtractor`、`PresetGenerator` 协作。

## 任务

- [x] 1. 实现 AudioPreprocessor（音频预处理 - 需求1）
  - [x] 1.1 实现 AudioPreprocessor 核心逻辑
    - 在 `src/audio_preprocessor.py` 中实现 `PreprocessConfig` 和 `PreprocessResult` dataclass
    - 实现 `AudioPreprocessor.__init__(config)` 方法，接受可选的 `PreprocessConfig`
    - 实现 `compute_rms_db(audio)` 静态方法：计算 RMS 能量值（dBFS），全零音频返回 -inf
    - 实现 `compute_clipping_ratio(audio, threshold=0.99)` 静态方法：计算绝对值超过阈值的样本比例
    - 实现 `remove_dc_offset(audio)` 静态方法：减去信号均值
    - 实现 `peak_normalize(audio, target_db=-1.0)` 静态方法：峰值归一化至目标 dBFS
    - 实现 `resample(audio, orig_sr, target_sr)` 静态方法：使用 `scipy.signal.resample_poly` 进行抗混叠重采样
    - 实现 `trim_tail_silence(audio, sample_rate)` 方法：裁剪尾部 RMS 低于 -50 dBFS 的静音段，保留最少 0.5 秒
    - 实现 `process(audio, sample_rate)` 方法：按顺序执行静音检测 → 削波检测 → DC 偏移消除 → 峰值归一化 → 重采样 → 尾部裁剪，返回 `PreprocessResult`
    - _需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 1.2 编写属性测试：音频过滤决策正确性
    - **Property 1: 音频过滤决策正确性**
    - 使用 hypothesis 生成随机音频信号，验证 RMS < -60 dBFS 时标记为静音过滤，削波比例 > 10% 时标记为削波过滤，否则通过
    - **验证: 需求 1.1, 1.2**

  - [ ]* 1.3 编写属性测试：DC 偏移消除后均值为零
    - **Property 2: DC 偏移消除后均值为零**
    - 使用 hypothesis 生成带 DC 偏移的随机音频，验证 `remove_dc_offset` 后 |mean| < 1e-7
    - **验证: 需求 1.3**

  - [ ]* 1.4 编写属性测试：峰值归一化目标精度
    - **Property 3: 峰值归一化目标精度**
    - 使用 hypothesis 生成随机非零音频，验证归一化后峰值在 -1.0 dBFS ± 0.01 dB 范围内
    - **验证: 需求 1.4**

  - [ ]* 1.5 编写属性测试：重采样保持时长不变
    - **Property 4: 重采样保持时长不变**
    - 使用 hypothesis 生成随机长度和采样率的音频，验证重采样后样本数 = round(len * target_sr / orig_sr)，误差不超过 1 个样本
    - **验证: 需求 1.5**

  - [ ]* 1.6 编写属性测试：尾部裁剪最小时长保证
    - **Property 5: 尾部裁剪最小时长保证**
    - 使用 hypothesis 生成带静音尾部的音频，验证裁剪后长度 ≥ min_duration_sec × sample_rate
    - **验证: 需求 1.6**

- [x] 2. 实现 SmartSampler（智能参数采样 - 需求2）
  - [x] 2.1 实现 SmartSampler 核心逻辑
    - 在 `src/smart_sampler.py` 中实现 `SamplingReport` dataclass
    - 实现 `SmartSampler.__init__(seed=42)` 方法，导入 `CORE_PARAMS`、`EFFECT_SWITCH_NAMES`、`EFFECT_SWITCH_INDICES` 从 `src/training_data.py`
    - 实现 `sample_lhs(n)` 方法：使用 `scipy.stats.qmc.LatinHypercube` 对连续参数进行 LHS 采样，离散参数（filter_model, filter_style）使用均匀离散采样后取整，效果器开关四舍五入为 0/1，返回 (n, 45) 矩阵
    - 实现 `sample_stratified_switches(n)` 方法：按活跃效果器数量 k（0-9）分层，每层样本数与 C(9,k) 成正比，连续参数仍使用 LHS
    - 实现 `sample(n, strategy="lhs_stratified")` 统一入口方法
    - 实现 `generate_report(params)` 方法：计算每个参数维度的 KS 检验统计量和 p-value，统计效果器开关分布
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 2.2 编写属性测试：LHS 采样边际均匀性
    - **Property 6: LHS 采样边际均匀性**
    - 使用 hypothesis 生成 n ≥ 100 的采样，验证每个连续参数维度的 KS 检验 p-value > 0.01
    - **验证: 需求 2.1**

  - [ ]* 2.3 编写属性测试：分层采样效果器比例
    - **Property 7: 分层采样效果器比例**
    - 使用 hypothesis 生成随机采样数量，验证各层样本数与 C(9,k) 成正比，偏差不超过 1
    - **验证: 需求 2.2, 2.3**

  - [ ]* 2.4 编写属性测试：离散参数值域约束
    - **Property 8: 离散参数值域约束**
    - 使用 hypothesis 生成采样矩阵，验证效果器开关为 0.0/1.0，filter_model ∈ {0..5}，filter_style ∈ {0..3}，所有参数在 [min, max] 范围内
    - **验证: 需求 2.4**

  - [ ]* 2.5 编写属性测试：采样种子可复现性
    - **Property 9: 采样种子可复现性**
    - 使用 hypothesis 生成随机种子，验证相同种子两次调用 sample() 产生完全相同的矩阵
    - **验证: 需求 2.5**

- [x] 3. 检查点 - 确保基础模块测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 4. 实现 MultiConditionRenderer（多条件渲染 - 需求3）
  - [x] 4.1 实现 MultiConditionRenderer 核心逻辑
    - 在 `src/multi_condition_renderer.py` 中实现 `MidiCondition`、`MultiConditionResult` dataclass
    - 定义 `DEFAULT_CONDITIONS` 列表：C3_v80、C3_v120、C4_v80、C4_v120、C5_v80、C5_v120（3 音高 × 2 力度，时长 2.0 秒）
    - 实现 `MultiConditionRenderer.__init__(renderer, conditions=None)` 方法，接受现有 `AudioRenderer` 实例
    - 实现 `render_preset(preset_path, output_dir, preset_id)` 方法：为单个预设渲染所有条件，输出文件名格式 `{preset_id}_{condition_label}.wav`，失败条件记录到 `failed_conditions` 列表，不影响其他条件
    - _需求: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 4.2 编写属性测试：多条件渲染文件命名
    - **Property 10: 多条件渲染文件命名**
    - 使用 hypothesis 生成随机 preset_id 和条件标签，验证输出文件名为 `{preset_id}_{condition_label}.wav`
    - **验证: 需求 3.3**

  - [ ]* 4.3 编写属性测试：条件渲染失败容错
    - **Property 11: 条件渲染失败容错**
    - 使用 hypothesis 模拟随机条件失败，验证成功条件数 + 失败条件数 = 总条件数
    - **验证: 需求 3.5**

- [x] 5. 实现 QualityValidator（数据质量验证 - 需求4）
  - [x] 5.1 实现 QualityValidator 核心逻辑
    - 在 `src/quality_validator.py` 中实现 `SampleQualityResult` 和 `DatasetQualityReport` dataclass
    - 实现 `QualityValidator.__init__` 方法，接受静音/削波/频谱熵/近重复/PCA 坍缩阈值参数
    - 实现 `compute_spectral_entropy(audio, sample_rate)` 方法：使用 FFT 计算功率谱密度，然后计算归一化熵
    - 实现 `validate_sample(audio, sample_rate, sample_id)` 方法：执行静音检测、削波检测、频谱熵检测，返回 `SampleQualityResult`
    - 实现 `detect_near_duplicates(embeddings, threshold=0.999)` 方法：计算 pairwise cosine similarity，返回超过阈值的样本对数
    - 实现 `validate_embeddings(embeddings, target_samples, valid_mask)` 方法：执行 PCA 坍缩检测（前 10 主成分 > 95% 方差时警告）、近重复检测、有效样本不足检测（< 80% 目标时警告），返回 `DatasetQualityReport`
    - _需求: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 5.2 编写属性测试：频谱熵过滤
    - **Property 12: 频谱熵过滤**
    - 使用 hypothesis 生成纯正弦波和随机噪声，验证低熵音频被标记为 `low_entropy`
    - **验证: 需求 4.1**

  - [ ]* 5.3 编写属性测试：PCA 坍缩警告阈值
    - **Property 13: PCA 坍缩警告阈值**
    - 使用 hypothesis 生成不同秩的随机矩阵，验证前 10 主成分累积方差 > 0.95 时 `pca_collapse_warning = True`
    - **验证: 需求 4.2**

  - [ ]* 5.4 编写属性测试：近重复检测正确性
    - **Property 14: 近重复检测正确性**
    - 构造含重复向量的矩阵，验证 `detect_near_duplicates` 返回正确的近重复对数
    - **验证: 需求 4.3**

  - [ ]* 5.5 编写属性测试：有效样本不足警告
    - **Property 15: 有效样本不足警告**
    - 使用 hypothesis 生成随机目标数和有效数，验证 V < 0.8 * T 时 `insufficient_samples_warning = True`
    - **验证: 需求 4.5**

- [x] 6. 检查点 - 确保预处理和验证模块测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 7. 实现 DistributionAnalyzer（分布分析 - 需求8）
  - [x] 7.1 实现 DistributionAnalyzer 核心逻辑
    - 在 `src/distribution_analyzer.py` 中实现 `DistributionReport` dataclass
    - 实现 `DistributionAnalyzer.__init__(diversity_threshold=0.95)` 方法
    - 实现 `analyze_embeddings(embeddings)` 方法：使用 sklearn PCA 降维，输出前 50 个主成分方差解释比和累积比；计算 pairwise cosine similarity 分布统计（均值、标准差、最小值、最大值、分位数）
    - 实现 `analyze_parameters(params)` 方法：对每个参数维度计算分布统计，使用 `scipy.stats.kstest` 与均匀分布进行 KS 检验
    - 实现 `generate_report(embeddings, params)` 方法：组合 embedding 分析和参数分析，当平均 cosine similarity > 0.95 时设置 `diversity_warning = True`
    - 实现 `save_report(report, output_path)` 方法：将 `DistributionReport` 序列化为 JSON 文件
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 7.2 编写属性测试：PCA 输出不变量
    - **Property 21: PCA 输出不变量**
    - 使用 hypothesis 生成随机 (N, 1024) 矩阵（N ≥ 50），验证输出 min(50, N, 1024) 个方差解释比，每个在 [0, 1]，累积比单调递增且 ≤ 1.0
    - **验证: 需求 8.1**

  - [ ]* 7.3 编写属性测试：Cosine similarity 统计与多样性警告
    - **Property 22: Cosine similarity 统计与多样性警告**
    - 使用 hypothesis 生成随机 embedding 矩阵，验证 min ≤ mean ≤ max，std ≥ 0，值在 [-1, 1]，mean > 0.95 时 `diversity_warning = True`
    - **验证: 需求 8.2, 8.5**

  - [ ]* 7.4 编写属性测试：参数 KS 检验范围
    - **Property 23: 参数 KS 检验范围**
    - 使用 hypothesis 生成随机参数矩阵，验证 KS statistic ∈ [0, 1]，pvalue ∈ [0, 1]，覆盖所有 45 个参数
    - **验证: 需求 8.3**

  - [ ]* 7.5 编写属性测试：分布报告 round-trip
    - **Property 24: 分布报告 round-trip**
    - 使用 hypothesis 生成随机 `DistributionReport`，验证 JSON 保存后重新加载数值等价
    - **验证: 需求 8.4**

- [x] 8. 检查点 - 确保分布分析模块测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 9. 实现 ParallelProducer（并行生产 - 需求5,6,7）
  - [x] 9.1 实现 ParallelProducer 断点续传与资源估算
    - 在 `src/parallel_producer.py` 中实现 `ProductionConfig`、`SampleStatus`、`ProductionSummary` dataclass
    - 实现 `ParallelProducer.__init__` 方法，接受 `vital_vst_path`、`ProductionConfig`、`AudioPreprocessor`、`SmartSampler`、`QualityValidator`、`DistributionAnalyzer` 实例
    - 实现 `_save_checkpoint(statuses, path)` 方法：将样本状态列表保存为 JSON 文件
    - 实现 `_load_checkpoint(path)` 方法：从 JSON 文件加载样本状态列表
    - 实现 `estimate_resources(n_presets, n_conditions)` 方法：计算 WAV 存储估算（N × SR × D × 4 / 1e9 GB）、HDF5 大小估算、时间估算（N × render_time / n_workers + N × embed_time）
    - 实现预设数量计算逻辑：`ceil(target_samples / n_conditions / (1 - filter_margin))`
    - _需求: 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4_

  - [x] 9.2 实现 ParallelProducer 多进程渲染与 GPU 批量 embedding
    - 实现 `_render_worker(tasks, vital_vst_path)` 静态方法：每个 worker 加载独立的 Vital VST3 实例，渲染任务列表中的预设，返回 (sample_id, success, error_msg) 列表
    - 实现多进程渲染调度逻辑：使用 `multiprocessing.Pool(n_workers)` 分配渲染任务，处理进程崩溃时重新分配未完成任务
    - 实现 GPU 批量 embedding 提取逻辑：将预处理后的音频按 `embedding_batch_size` 分批，使用 MPS 设备批量推理
    - 实现每 100 个样本更新 checkpoint 和输出进度日志（已完成数、失败数、ETA）
    - _需求: 5.1, 5.2, 5.5, 5.6_

  - [x] 9.3 实现 ParallelProducer 完整生产流水线与 HDF5 保存
    - 实现 `produce(output_dir, resume=False)` 方法：按顺序执行参数采样 → 预设生成 + 多条件渲染（多进程）→ 音频预处理 → Embedding 提取（GPU 批量）→ 质量验证 → 数据集保存 → 分布分析
    - 实现断点恢复逻辑：resume=True 时加载 checkpoint，跳过 status 为 "embedded" 和 "failed" 的样本，仅处理 "pending" 和 "rendered"
    - 实现 `save_production_hdf5` 方法：保存 train/val/test 分组（80/10/10），包含 params、embeddings、midi_notes、midi_velocities、midi_durations、audio_stats 子组；metadata 组包含 param_names、param_ranges、sampling_strategy、seed、production_timestamp、vital_version、production_config（完整 YAML 字符串）
    - 实现增量生产模式：在已有 HDF5 基础上追加新样本，合并后重新划分
    - 生成 `ProductionSummary` 和人类可读的 `production_summary.json` 报告
    - _需求: 5.4, 6.1, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [ ]* 9.4 编写属性测试：断点续传 round-trip
    - **Property 16: 断点续传 round-trip**
    - 使用 hypothesis 生成随机样本状态列表，验证 JSON 保存后重新加载等价
    - **验证: 需求 5.3**

  - [ ]* 9.5 编写属性测试：恢复时按状态过滤
    - **Property 17: 恢复时按状态过滤**
    - 使用 hypothesis 生成混合状态的 checkpoint，验证恢复时仅选择 "pending" 和 "rendered" 状态的样本
    - **验证: 需求 5.4**

  - [ ]* 9.6 编写属性测试：预设数量计算
    - **Property 18: 预设数量计算**
    - 使用 hypothesis 生成随机目标/条件/余量，验证预设数 = ceil(T / C / (1 - M))
    - **验证: 需求 6.2**

  - [ ]* 9.7 编写属性测试：资源估算计算
    - **Property 19: 资源估算计算**
    - 使用 hypothesis 生成随机样本数/采样率/时长/并行度，验证存储和时间估算与手动计算一致
    - **验证: 需求 6.3, 6.4**

  - [ ]* 9.8 编写属性测试：HDF5 数据集完整性
    - **Property 20: HDF5 数据集完整性**
    - 使用 hypothesis 生成随机数据集，验证 HDF5 保存/加载 round-trip 和所有必需字段存在
    - **验证: 需求 7.1, 7.2, 7.3**

- [x] 10. 检查点 - 确保并行生产模块测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 11. 创建生产配置和入口脚本
  - [x] 11.1 创建 `configs/production.yaml` 生产配置文件
    - 定义生产规模配置：target_samples=100000、filter_margin=0.02、n_presets=17000
    - 定义采样策略配置：strategy="lhs_stratified"、seed=42
    - 定义多条件渲染配置：6 个 MIDI 条件（C3/C4/C5 × v80/v120）
    - 定义音频预处理配置：silence_threshold_db=-60.0、clipping_threshold=0.99、target_peak_db=-1.0、target_sample_rate=16000
    - 定义质量验证配置：spectral_entropy_threshold=1.0、near_duplicate_threshold=0.999、pca_collapse_threshold=0.95
    - 定义并行生产配置（M4 Pro 优化）：n_workers=11、embedding_batch_size=32、embedding_device="mps"、checkpoint_interval=100
    - 定义分布分析配置：n_pca_components=50、diversity_threshold=0.95
    - _需求: 5.1, 5.2, 6.1, 6.2_

  - [x] 11.2 创建 `scripts/run_production.py` 生产流水线入口脚本
    - 解析命令行参数：输出目录、VST 路径、配置文件路径、是否恢复（--resume）、目标样本数覆盖
    - 加载 `configs/production.yaml` 配置，支持命令行参数覆盖
    - 实例化所有组件（AudioPreprocessor、SmartSampler、MultiConditionRenderer、QualityValidator、DistributionAnalyzer、ParallelProducer）
    - 启动前输出存储和时间估算报告，等待用户确认
    - 调用 `ParallelProducer.produce()` 执行完整生产流水线
    - 输出生产摘要到控制台和 `production_summary.json`
    - _需求: 5.1, 6.1, 6.3, 6.4, 7.4_

- [x] 12. 集成与端到端连接
  - [x] 12.1 连接新模块到现有系统
    - 确保 `SmartSampler` 正确导入和使用 `CORE_PARAMS`、`EFFECT_SWITCH_NAMES`、`EFFECT_SWITCH_INDICES` 从 `src/training_data.py`
    - 确保 `MultiConditionRenderer` 正确复用现有 `AudioRenderer` 实例，通过修改 `RenderConfig` 的 MIDI 参数实现多条件渲染
    - 确保 `ParallelProducer` 正确协调所有新旧模块：`PresetGenerator` 生成预设 → `MultiConditionRenderer` 多条件渲染 → `AudioPreprocessor` 预处理 → `EmbeddingExtractor` 提取 embedding → `QualityValidator` 验证 → `DistributionAnalyzer` 分析
    - 验证 HDF5 输出格式与设计文档中的数据模型一致
    - _需求: 3.4, 5.1, 7.1, 7.2_

  - [ ]* 12.2 编写单元测试：生产配置解析和默认值
    - 验证 `configs/production.yaml` 能被正确解析
    - 验证默认 6 条件配置的具体值（C3=48、C4=60、C5=72，力度 80/120）
    - 验证 M4 Pro 默认并行度为 11
    - _需求: 3.2, 5.1_

- [x] 13. 最终检查点 - 确保所有测试通过
  - 确保所有测试通过，如有问题请向用户确认。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP 开发
- 每个任务引用了具体的需求编号以确保可追溯性
- 检查点任务确保增量验证
- 属性测试验证 24 个正确性属性（Property 1-24），单元测试验证具体示例和边界条件
- 所有新模块与现有代码通过组合方式协作，不修改现有 `TrainingDataGenerator`
- 音频渲染和 embedding 提取的测试使用 mock 对象，集成测试需要实际的 Vital VST3 和 MuQ 模型
- 并行渲染测试使用 `multiprocessing.dummy`（线程池）替代真实进程池
- 属性测试使用 `hypothesis` 库，`@settings(max_examples=100)` 配置
- 硬件优化：M4 Pro 12 核 → 11 渲染 worker，MPS GPU → embedding 批量推理
