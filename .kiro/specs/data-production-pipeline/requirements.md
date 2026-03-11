# 需求文档：实验数据生产与预处理流水线

## 简介

本项目旨在强化 Vital 合成器参数估计系统的数据生产和预处理能力。现有系统（`synth-parameter-estimation`）已具备基础的训练数据生成功能（均匀采样 45 个核心参数 → 渲染音频 → 提取 MuQ embedding → HDF5 存储），但存在以下不足：

- 采样策略过于朴素（均匀随机），大量参数组合产生静音、削波或高度相似的音频
- 缺乏音频质量检测和预处理（响度归一化、静音过滤、DC 偏移消除等）
- 仅使用单一 MIDI 输入（C4、力度 100、2 秒），数据多样性不足
- 当前规划 10,000 样本对 45 维参数回归远远不够
- 无并行处理能力，大规模生产效率低
- 缺乏数据质量验证和分布分析手段

本需求文档定义一套完整的数据生产流水线，覆盖智能采样、音频预处理、质量过滤、多条件渲染、并行生产和数据验证，为后续训练实验提供高质量、大规模的数据集。

## 术语表

- **Audio_Preprocessor**：音频预处理模块，负责对渲染后的原始音频执行响度归一化、静音检测、DC 偏移消除、削波检测等处理
- **Smart_Sampler**：智能采样模块，负责使用拉丁超立方采样（LHS）和分层采样策略替代朴素均匀采样，提升参数空间覆盖率
- **Quality_Validator**：数据质量验证模块，负责对生成的音频和 embedding 执行自动化质量检查，过滤退化样本
- **Multi_Condition_Renderer**：多条件渲染模块，负责对同一预设使用不同 MIDI 音符、力度和时长进行渲染，增加数据多样性
- **Parallel_Producer**：并行生产模块，负责协调多进程渲染和 GPU 批量 embedding 提取，支持断点续传
- **Distribution_Analyzer**：分布分析模块，负责分析 embedding 空间分布、检测聚类坍缩和参数覆盖率
- **LHS**：拉丁超立方采样（Latin Hypercube Sampling），一种空间填充采样方法，确保每个参数维度的边际分布均匀
- **LUFS**：响度单位全刻度（Loudness Units relative to Full Scale），广播标准响度度量
- **DC_Offset**：直流偏移，音频信号中的恒定偏移分量，会浪费动态范围并影响后续处理
- **Degenerate_Sample**：退化样本，指产生静音、纯削波、纯 DC 偏移等无有效音频内容的参数组合

## 需求

### 需求 1：音频预处理流水线

**用户故事：** 作为研究人员，我希望渲染后的音频经过标准化预处理，以便消除音量差异、静音片段和信号缺陷对 embedding 质量的影响。

#### 验收标准

1. WHEN 接收到一段渲染后的原始音频时，THE Audio_Preprocessor SHALL 计算该音频的 RMS 能量值，WHEN RMS 低于 -60 dBFS 时，THE Audio_Preprocessor SHALL 将该样本标记为静音并记录到过滤日志
2. WHEN 接收到一段渲染后的原始音频时，THE Audio_Preprocessor SHALL 检测音频中削波样本点（绝对值超过 0.99）的比例，WHEN 削波比例超过 10% 时，THE Audio_Preprocessor SHALL 将该样本标记为过度削波并记录到过滤日志
3. WHEN 接收到一段通过静音和削波检测的音频时，THE Audio_Preprocessor SHALL 移除该音频的 DC 偏移分量（减去信号均值）
4. WHEN 接收到一段通过静音和削波检测的音频时，THE Audio_Preprocessor SHALL 执行峰值归一化，将音频峰值缩放至 -1.0 dBFS
5. WHEN 接收到采样率为 44100 Hz 的音频且目标采样率为 16000 Hz 时，THE Audio_Preprocessor SHALL 使用抗混叠低通滤波器进行重采样，输出 16000 Hz 音频
6. THE Audio_Preprocessor SHALL 对每段处理后的音频执行尾部静音裁剪，移除末尾 RMS 低于 -50 dBFS 的连续静音段，保留最少 0.5 秒的有效音频
7. THE Audio_Preprocessor SHALL 返回一个预处理结果对象，包含：处理后的音频数据、原始 RMS、削波比例、是否被过滤及过滤原因

### 需求 2：智能参数采样策略

**用户故事：** 作为研究人员，我希望采样策略能更高效地覆盖参数空间，以便用更少的样本获得更好的参数空间覆盖率，减少退化样本的比例。

#### 验收标准

1. THE Smart_Sampler SHALL 支持拉丁超立方采样（LHS）模式，确保 45 个参数维度中每个维度的边际分布均匀覆盖其值域
2. THE Smart_Sampler SHALL 对 9 个 Effect_Switch 二值参数执行分层采样，确保每种效果器开关组合（2^9 = 512 种）在数据集中有合理的代表性
3. WHEN 使用分层采样时，THE Smart_Sampler SHALL 将效果器开关组合按活跃效果器数量（0-9 个）分层，每层的样本数量与该层的组合数成正比
4. THE Smart_Sampler SHALL 对连续参数（如 cutoff、resonance）使用 LHS 采样，对离散参数（如 filter_model、filter_style）使用均匀离散采样
5. THE Smart_Sampler SHALL 支持指定随机种子，确保采样结果可复现
6. THE Smart_Sampler SHALL 输出采样覆盖率报告，包含每个参数维度的实际采样分布直方图数据和 KS 检验统计量

### 需求 3：多条件渲染

**用户故事：** 作为研究人员，我希望同一预设能在不同 MIDI 条件下渲染，以便模型学习到参数对不同音高和力度的影响，提升泛化能力。

#### 验收标准

1. THE Multi_Condition_Renderer SHALL 支持为每个预设配置指定多个 MIDI 条件组合，每个条件包含：音符编号（MIDI note）、力度（velocity）、时长（秒）
2. THE Multi_Condition_Renderer SHALL 提供默认的多条件配置：3 个音高（C3=48、C4=60、C5=72）× 2 个力度（80、120）= 6 个条件组合
3. WHEN 为一个预设渲染多个条件时，THE Multi_Condition_Renderer SHALL 为每个条件生成独立的音频文件，文件名包含预设标识和条件标识
4. THE Multi_Condition_Renderer SHALL 在数据集元数据中记录每个样本对应的 MIDI 条件（音符、力度、时长），作为附加特征列
5. IF 某个条件的渲染失败，THEN THE Multi_Condition_Renderer SHALL 跳过该条件并记录日志，该预设的其他条件渲染结果仍保留

### 需求 4：数据质量验证与过滤

**用户故事：** 作为研究人员，我希望生成的数据集经过自动化质量验证，以便确保训练数据中不包含退化样本，且 embedding 分布合理。

#### 验收标准

1. THE Quality_Validator SHALL 对每个渲染后的音频执行以下检查：静音检测（RMS < -60 dBFS）、削波检测（削波比例 > 10%）、频谱单调性检测（频谱熵低于阈值表示音频内容过于简单）
2. THE Quality_Validator SHALL 在所有 embedding 提取完成后，计算 embedding 矩阵的主成分方差解释比，WHEN 前 10 个主成分解释超过 95% 的方差时，THE Quality_Validator SHALL 发出警告，提示 embedding 空间可能存在维度坍缩
3. THE Quality_Validator SHALL 检测 embedding 空间中的近重复样本（cosine similarity > 0.999 的样本对），并在报告中列出近重复样本数量和占比
4. THE Quality_Validator SHALL 生成数据质量报告，包含：总样本数、过滤样本数及各过滤原因的统计、embedding 维度分布摘要、参数覆盖率统计
5. WHEN 过滤后的有效样本数低于目标样本数的 80% 时，THE Quality_Validator SHALL 发出警告，建议调整采样策略或参数值域约束

### 需求 5：并行生产与断点续传

**用户故事：** 作为研究人员，我希望数据生产流水线支持多进程并行和断点续传，以便在合理时间内完成大规模数据集的生成，且中断后无需从头开始。

#### 验收标准

1. THE Parallel_Producer SHALL 支持多进程并行渲染，每个进程加载独立的 Vital VST3 插件实例，进程数量通过配置参数指定（默认为 CPU 核心数 - 1）
2. THE Parallel_Producer SHALL 支持 GPU 批量 embedding 提取，将多段音频组成 batch 送入 MuQ 模型，batch 大小通过配置参数指定（默认为 32）
3. THE Parallel_Producer SHALL 维护一个进度状态文件（JSON 格式），记录每个样本的处理状态（pending、rendered、embedded、failed），支持从中断点恢复
4. WHEN 从断点恢复时，THE Parallel_Producer SHALL 跳过已完成的样本，仅处理状态为 pending 或 rendered 的样本
5. THE Parallel_Producer SHALL 每处理 100 个样本后更新一次进度状态文件，并输出进度日志，包含已完成数、失败数、预估剩余时间（ETA）
6. IF 某个渲染进程崩溃，THEN THE Parallel_Producer SHALL 将该进程负责的未完成样本重新分配给其他进程，并记录崩溃日志

### 需求 6：生产规模规划与存储估算

**用户故事：** 作为研究人员，我希望系统提供明确的数据规模规划和存储估算，以便我合理分配计算资源和存储空间。

#### 验收标准

1. THE Parallel_Producer SHALL 支持通过配置文件指定目标样本总量，默认目标为 100,000 个有效样本（过滤后）
2. WHEN 使用多条件渲染（6 个条件/预设）时，THE Parallel_Producer SHALL 生成约 17,000 个预设配置 × 6 个条件 = 102,000 个音频样本，预留约 2% 的退化样本过滤余量
3. THE Parallel_Producer SHALL 在启动前输出存储估算报告，包含：原始 WAV 文件总大小估算（每个 2s@44100Hz 单声道 32-bit float ≈ 352KB，100K 样本 ≈ 34GB）、HDF5 数据集大小估算（参数矩阵 + embedding 矩阵，100K 样本 ≈ 420MB）、临时文件空间需求
4. THE Parallel_Producer SHALL 在启动前输出时间估算报告，基于单样本渲染时间（约 3 秒）和 embedding 提取时间（约 0.5 秒/样本 GPU、约 2 秒/样本 CPU）计算总生产时间
5. THE Parallel_Producer SHALL 支持增量生产模式，允许在已有数据集基础上追加新样本，合并后重新划分训练/验证/测试集

### 需求 7：数据集元数据与可追溯性

**用户故事：** 作为研究人员，我希望数据集包含完整的元数据和生产过程记录，以便追溯每个样本的生成条件和复现实验结果。

#### 验收标准

1. THE Parallel_Producer SHALL 在 HDF5 数据集中保存以下元数据：参数名称列表、每个参数的值域范围、采样策略名称（LHS/uniform/stratified）、随机种子、生产时间戳、Vital VST3 插件版本信息
2. THE Parallel_Producer SHALL 为每个样本记录：参数向量、MIDI 条件（音符、力度、时长）、预处理前的原始音频统计（RMS、峰值、削波比例）、预处理后的音频统计、embedding 向量
3. THE Parallel_Producer SHALL 将完整的生产配置（YAML 格式）作为字符串嵌入 HDF5 文件的 metadata 组中
4. THE Parallel_Producer SHALL 生成一份人类可读的生产摘要报告（JSON 格式），包含：总样本数、有效样本数、过滤统计、参数覆盖率、embedding 分布摘要、总耗时、各阶段耗时

### 需求 8：分布分析与可视化数据导出

**用户故事：** 作为研究人员，我希望能分析生成数据集的 embedding 分布和参数覆盖情况，以便评估数据集质量并指导后续采样策略调整。

#### 验收标准

1. THE Distribution_Analyzer SHALL 对 embedding 矩阵执行 PCA 降维，输出前 50 个主成分的方差解释比和累积方差解释比
2. THE Distribution_Analyzer SHALL 计算 embedding 空间的 pairwise cosine similarity 分布统计（均值、标准差、最小值、最大值、分位数）
3. THE Distribution_Analyzer SHALL 对每个参数维度计算实际采样值的分布统计（均值、标准差、最小值、最大值），并与目标均匀分布进行 KS 检验
4. THE Distribution_Analyzer SHALL 将所有分析结果保存为结构化 JSON 文件，包含数值数据和统计摘要，供外部可视化工具使用
5. WHEN embedding 空间的平均 pairwise cosine similarity 超过 0.95 时，THE Distribution_Analyzer SHALL 在报告中发出"embedding 多样性不足"的警告
