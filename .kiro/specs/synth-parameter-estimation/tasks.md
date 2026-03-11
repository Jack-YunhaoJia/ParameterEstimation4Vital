# 实现计划：Vital 合成器参数估计系统

## 概述

按依赖关系分阶段实现：先构建共享基础设施（预设解析），再逐步实现 Phase 0 各组件（预设生成 → 音频渲染 → embedding 提取 → 区分评估 → 流水线编排），最后实现 Phase 1（训练数据生成 → 参数回归）。每个组件包含核心实现和可选的属性/单元测试子任务。

## 任务

- [x] 1. 项目结构与基础设施搭建
  - [x] 1.1 创建项目目录结构和包初始化
    - 创建 `src/__init__.py`、`tests/__init__.py`、`scripts/`、`experiments/`、`configs/` 目录
    - 创建 `configs/default.yaml` 默认配置文件（包含 VST 路径、MIDI 参数、采样率等）
    - 创建 `pyproject.toml` 或 `requirements.txt`，声明依赖：pedalboard、torch、hypothesis、numpy、h5py、scikit-learn、pyyaml
    - _需求: 全局_

- [x] 2. 实现 PresetParser（预设解析与序列化 - 需求5）
  - [x] 2.1 实现 VitalPreset 数据类和 PresetParser 核心逻辑
    - 在 `src/preset_parser.py` 中实现 `VitalPreset` dataclass（settings、modulations、extra 字段）
    - 实现 `PresetParser.parse(filepath)` 方法：读取 JSON、提取 settings/modulations/extra、验证 settings 键存在
    - 实现 `PresetParser.serialize(preset, filepath)` 方法：将 VitalPreset 序列化为 JSON 写入文件
    - 实现 `PresetParser.validate_effect_switches(preset)` 方法：验证 9 个 Effect_Switch 键存在
    - 定义 `PresetParseError` 自定义异常，包含文件路径和具体违规描述
    - _需求: 5.1, 5.2, 5.4, 5.5_

  - [ ]* 2.2 编写属性测试：预设解析-序列化 round-trip
    - **Property 9: 预设解析-序列化 round-trip**
    - 使用 hypothesis 生成随机 VitalPreset 对象，验证 serialize → parse 产生等价结果
    - **验证: 需求 5.3**

  - [ ]* 2.3 编写属性测试：无效输入解析拒绝
    - **Property 10: 无效输入解析拒绝**
    - 使用 hypothesis 生成随机非法 JSON 字符串或缺少 settings 键的 JSON，验证抛出 PresetParseError
    - **验证: 需求 5.5**


- [x] 3. 实现 PresetGenerator（预设生成 - 需求1）
  - [x] 3.1 实现 PresetGenerator 核心逻辑
    - 在 `src/preset_generator.py` 中实现 `PresetGenerator` 类
    - 实现 `create_base_patch()` 方法：基于 772 参数默认值模板，设置 osc_1_on=1.0、filter_1_on=1.0、默认波表
    - 实现 `create_effect_variant(effect_name, state)` 方法：深拷贝 Base_Patch，仅修改指定 Effect_Switch
    - 实现 `generate_all_variants(output_dir)` 方法：为 9 个效果器各生成开/关预设，共 18 个文件
    - 对无效 effect_name 抛出 ValueError，错误信息包含无效名和有效列表
    - _需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 3.2 编写属性测试：预设生成结构完整性
    - **Property 1: 预设生成结构完整性**
    - 使用 hypothesis 验证生成的预设 settings 包含所有 772 个参数键，值类型为 float
    - **验证: 需求 1.1**

  - [ ]* 3.3 编写属性测试：效果器变体单一差异
    - **Property 2: 效果器变体单一差异**
    - 使用 hypothesis 随机选择效果器和状态，验证变体与 Base_Patch 仅在目标参数上不同
    - **验证: 需求 1.3, 1.5**

  - [ ]* 3.4 编写属性测试：无效效果器名拒绝
    - **Property 3: 无效效果器名拒绝**
    - 使用 hypothesis 生成不在 EFFECT_SWITCHES 中的随机字符串，验证抛出 ValueError
    - **验证: 需求 1.6**

  - [ ]* 3.5 编写单元测试：Base_Patch 具体值和 18 个预设生成
    - 验证 Base_Patch 的 osc_1_on=1.0、filter_1_on=1.0
    - 验证 generate_all_variants 生成恰好 18 个文件
    - _需求: 1.2, 1.4_

- [x] 4. 检查点 - 确保基础组件测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 5. 实现 AudioRenderer（音频渲染 - 需求2）
  - [x] 5.1 实现 AudioRenderer 核心逻辑
    - 在 `src/audio_renderer.py` 中实现 `RenderConfig` 和 `RenderSummary` dataclass
    - 实现 `AudioRenderer.__init__` 方法：通过 pedalboard 加载 Vital VST3 插件
    - 实现 `render_preset(preset_path, output_path)` 方法：生成 MIDI 事件（C4、力度100、2秒）、通过 VST 渲染、输出 44100Hz 单声道 WAV
    - 实现 `render_batch(preset_dir, output_dir)` 方法：遍历 .vital 文件批量渲染，超时30秒跳过，记录错误日志
    - 输出文件名与源预设同名（.vital → .wav）
    - 渲染完成后返回 RenderSummary（成功数、失败数、失败文件列表）
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 5.2 编写属性测试：音频文件名映射
    - **Property 4: 音频文件名映射**
    - 使用 hypothesis 生成随机 .vital 文件名，验证输出文件名为同名 .wav
    - **验证: 需求 2.4**

  - [ ]* 5.3 编写属性测试：渲染摘要计数一致性
    - **Property 5: 渲染摘要计数一致性**
    - 使用 hypothesis 生成随机成功/失败计数，验证 success + failure = total 且 len(failed_files) = failure_count
    - **验证: 需求 2.6**

  - [ ]* 5.4 编写单元测试：渲染超时和错误处理
    - 使用 mock 模拟 VST 渲染超时（>30秒）场景，验证跳过并记录日志
    - 使用 mock 模拟渲染错误场景，验证继续处理剩余文件
    - _需求: 2.5_


- [x] 6. 实现 EmbeddingExtractor（MuQ Embedding 提取 - 需求3）
  - [x] 6.1 实现 EmbeddingExtractor 核心逻辑
    - 在 `src/embedding_extractor.py` 中实现 `EmbeddingResult` dataclass
    - 实现 `EmbeddingExtractor.__init__` 方法：加载 MuQ 预训练模型到指定设备
    - 实现 `extract(audio_path)` 方法：读取 WAV、检查采样率（不匹配时自动重采样）、通过 MuQ 提取 embedding 向量
    - 实现 `extract_batch(audio_dir)` 方法：批量提取目录下所有 WAV 的 embedding
    - 实现 `save(result, output_path)` 方法：保存为 .npz 文件（文件名到向量映射）
    - 错误处理：模型加载失败抛出 ModelLoadError、音频格式不兼容返回描述性错误
    - _需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 6.2 编写属性测试：Embedding 存储 round-trip
    - **Property 6: Embedding 存储 round-trip**
    - 使用 hypothesis 生成随机文件名到 embedding 向量的映射，验证 save → load 数值等价
    - **验证: 需求 3.4**

- [x] 7. 实现 Discriminator（区分能力评估 - 需求4）
  - [x] 7.1 实现 Discriminator 核心逻辑
    - 在 `src/discriminator.py` 中实现 `EffectDiscriminationResult` 和 `FeasibilityReport` dataclass
    - 实现 `evaluate_effect(on_embeddings, off_embeddings, effect_name)` 方法：计算 cosine similarity 均值、使用 Logistic Regression 二分类评估准确率
    - 实现 `evaluate_all(embeddings)` 方法：对 9 个效果器分别评估，按区分度降序排列，生成可行性报告
    - 可行性判定：≥6 个效果器准确率 ≥75% 为可行；cosine similarity >0.99 标记为"表征无法区分"
    - _需求: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 7.2 编写属性测试：可行性判定阈值正确性
    - **Property 7: 可行性判定阈值正确性**
    - 使用 hypothesis 生成随机准确率向量和 cosine similarity 值，验证可行性判定逻辑和"表征无法区分"标记
    - **验证: 需求 4.5, 4.6**

  - [ ]* 7.3 编写属性测试：区分度评估排序与计算
    - **Property 8: 区分度评估排序与计算**
    - 使用 hypothesis 生成随机 embedding 对，验证 cosine similarity ∈ [-1,1]、准确率 ∈ [0,1]、结果按区分度降序排列
    - **验证: 需求 4.1, 4.2, 4.3**

  - [ ]* 7.4 编写单元测试：cosine similarity 边界判定
    - 测试 cosine similarity = 0.99 和 0.991 的边界情况
    - 验证"表征无法区分"标记的阈值行为
    - _需求: 4.6_

- [x] 8. 检查点 - 确保 Phase 0 核心组件测试通过
  - 确保所有测试通过，如有问题请向用户确认。

- [x] 9. 实现 PipelineOrchestrator（流水线编排 - 需求6）
  - [x] 9.1 实现 PipelineOrchestrator 核心逻辑
    - 在 `src/pipeline.py` 中实现 `PipelineStep` 枚举和 `PipelineResult` dataclass
    - 实现 `PipelineOrchestrator.run(output_base, start_from)` 方法：按顺序执行预设生成 → 音频渲染 → embedding 提取 → 区分评估
    - 实现失败停止逻辑：某步骤异常时停止后续步骤，记录失败步骤名和错误详情
    - 实现恢复执行逻辑：检测中间产物文件是否存在，从指定步骤恢复
    - 创建带时间戳的实验目录，保存所有中间产物和最终报告
    - 输出实验摘要：各步骤耗时、可行性判定结果、下一步建议
    - _需求: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 9.2 编写属性测试：流水线失败停止
    - **Property 11: 流水线失败停止**
    - 使用 hypothesis 模拟随机步骤失败，验证后续步骤不执行且 PipelineResult 包含错误信息
    - **验证: 需求 6.2**

  - [ ]* 9.3 编写属性测试：流水线恢复执行
    - **Property 12: 流水线恢复执行**
    - 使用 hypothesis 随机选择起始步骤，验证之前步骤不执行、之后步骤按序执行
    - **验证: 需求 6.3**

- [x] 10. 实现 Phase 0 运行入口脚本
  - [x] 10.1 创建 `scripts/run_phase0.py` 入口脚本
    - 解析命令行参数（输出目录、VST 路径、恢复步骤等）
    - 加载 `configs/default.yaml` 配置
    - 实例化各组件并调用 PipelineOrchestrator.run()
    - 输出实验摘要到控制台和 report.json
    - _需求: 6.1, 6.4, 6.5_

- [x] 11. 检查点 - Phase 0 完整流水线验证
  - 确保所有测试通过，如有问题请向用户确认。


- [x] 12. 实现 TrainingDataGenerator（训练数据生成 - 需求8）
  - [x] 12.1 实现 TrainingDataGenerator 核心逻辑
    - 在 `src/training_data.py` 中实现 `DatasetMetadata` dataclass
    - 定义 45 个核心参数的名称和有效值域范围（osc_1 参数、filter_1 参数、env_1 ADSR、9 个 Effect_Switch、9 个 dry_wet、其他效果器参数）
    - 实现 `sample_parameters(n)` 方法：在各参数值域内均匀随机采样，返回 (n, 45) 矩阵
    - 实现 `generate_dataset(n_samples, output_dir)` 方法：采样 → 生成预设 → 渲染音频 → 提取 embedding → 按 80/10/10 划分 → 保存 HDF5
    - 失败样本跳过并记录日志，确保最终数据集无缺失值
    - 保存元数据：参数名称、值域范围、采样分布信息
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 12.2 编写属性测试：参数采样值域约束
    - **Property 14: 参数采样值域约束**
    - 使用 hypothesis 验证 sample_parameters 生成的每个参数值在定义的 [min, max] 范围内
    - **验证: 需求 8.1**

  - [ ]* 12.3 编写属性测试：数据集划分比例
    - **Property 16: 数据集划分比例**
    - 使用 hypothesis 生成随机正整数 N，验证 train + val + test = N 且各子集大小与目标比例偏差 ≤1
    - **验证: 需求 8.3**

  - [ ]* 12.4 编写属性测试：数据集存储 round-trip
    - **Property 17: 数据集存储 round-trip**
    - 使用 hypothesis 生成随机参数矩阵和 embedding 矩阵，验证 HDF5 保存后重新加载数值等价
    - **验证: 需求 8.4**

  - [ ]* 12.5 编写属性测试：数据集元数据完整性
    - **Property 18: 数据集元数据完整性**
    - 验证元数据包含所有 45 个参数名称和值域范围，且 min < max
    - **验证: 需求 8.6**

- [x] 13. 实现 ParameterRegressor（参数回归 - 需求7）
  - [x] 13.1 实现 ParameterRegressor MLP 模型
    - 在 `src/parameter_regressor.py` 中实现 `RegressionMetrics` dataclass
    - 实现 `ParameterRegressor(nn.Module)` 类：MLP 架构，输入 1024 维 embedding，输出 45 维归一化参数向量
    - 实现 `forward(embedding)` 方法：前向传播，输出值经 sigmoid 约束到 [0, 1]
    - 实现 `export_preset(predicted_params, parser)` 方法：将归一化参数反映射到原始值域，生成有效 Vital 预设
    - 实现训练循环：MSE 损失 + 可选多频谱损失，Adam 优化器
    - 实现评估逻辑：计算每个参数的 MAE 和整体多频谱损失
    - 添加 Phase 0 可行性门控：若 Phase 0 判定不可行则拒绝启动
    - _需求: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 13.2 编写属性测试：回归模型输入输出维度
    - **Property 13: 回归模型输入输出维度**
    - 使用 hypothesis 生成随机 batch_size，验证输入 (batch, 1024) → 输出 (batch, 45) 且值在 [0, 1]
    - **验证: 需求 7.1**

  - [ ]* 13.3 编写属性测试：预测参数导出有效性
    - **Property 15: 预测参数导出有效性**
    - 使用 hypothesis 生成有效值域内的 45 维参数向量，验证 export_preset 生成可被 PresetParser 解析的预设
    - **验证: 需求 7.5**

- [x] 14. 集成与端到端连接
  - [x] 14.1 连接 Phase 1 组件到整体系统
    - 在 `scripts/run_phase0.py` 中添加 Phase 0 通过后的 Phase 1 提示
    - 创建 `scripts/run_phase1.py` 入口脚本：读取 Phase 0 报告 → 检查可行性 → 生成训练数据 → 训练回归模型 → 输出评估报告
    - 确保 Phase 0 不可行时输出建议调整技术路线的提示并阻止 Phase 1 启动
    - _需求: 7.6, 8.2_

- [x] 15. 最终检查点 - 确保所有测试通过
  - 确保所有测试通过，如有问题请向用户确认。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP 开发
- 每个任务引用了具体的需求编号以确保可追溯性
- 检查点任务确保增量验证
- 属性测试验证通用正确性属性（18 个 Property），单元测试验证具体示例和边界条件
- Phase 0 组件（任务 1-11）必须在 Phase 1 组件（任务 12-14）之前完成
- 音频渲染和 embedding 提取的测试使用 mock 对象，集成测试需要实际的 Vital VST3 和 MuQ 模型
