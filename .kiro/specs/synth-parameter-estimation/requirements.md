# 需求文档：Vital 合成器参数估计系统

## 简介

本项目旨在构建一个从音频信号反向估计 Vital 合成器参数的系统。核心思路是利用 MuQ（腾讯 AI Lab 的音乐音频表征模型）提取音频 embedding，训练参数-音频匹配模型（双塔架构），逐步实现从效果器开关识别到完整参数回归的能力。

项目采用分阶段策略，Phase 0 为可行性验证实验，验证 MuQ 音频表征能否区分 Vital 效果器的开关状态。该阶段是整个项目的前置门控——若表征无法区分基本的效果器开/关差异，后续更细粒度的参数回归将不可行。

## 术语表

- **Vital**：一款闭源波表合成器 VST 插件，预设文件为 JSON 格式（.vital 扩展名），包含 772 个连续参数和 64 个调制槽位
- **MuQ**：腾讯 AI Lab 开源的音乐音频理解表征模型（Music Understanding with Quantization），用于提取音频的语义级 embedding
- **Preset_Generator**：预设生成模块，负责以编程方式构造 Vital 预设 JSON 文件，控制特定参数的开/关状态
- **Audio_Renderer**：音频渲染模块，负责将 Vital 预设文件渲染为音频波形（通过 Vital CLI 或 headless 模式）
- **Embedding_Extractor**：嵌入提取模块，负责调用 MuQ 模型从音频中提取固定维度的 embedding 向量
- **Discriminator**：区分能力评估模块，负责通过 cosine similarity 或线性分类器量化 embedding 对效果器状态的区分能力
- **Effect_Switch**：Vital 中的效果器开关参数，取值为 0.0（关）或 1.0（开），共 9 个：chorus_on、compressor_on、delay_on、distortion_on、eq_on、flanger_on、phaser_on、reverb_on、filter_fx_on
- **Base_Patch**：基础音色配置，指固定 osc_1 使用默认波表、filter_1 开启的最小化预设，作为实验的控制变量
- **Embedding**：MuQ 模型输出的固定维度浮点向量，表示一段音频的语义特征
- **Parameter_Regressor**：参数回归模块（Phase 1+），负责从 embedding 预测连续参数值

## 需求

### 需求 1：基础预设生成（Phase 0）

**用户故事：** 作为研究人员，我希望能以编程方式生成控制变量的 Vital 预设文件，以便系统地比较效果器开/关对音频的影响。

#### 验收标准

1. THE Preset_Generator SHALL 生成符合 Vital JSON 格式规范的预设文件，包含所有 772 个连续参数的默认值
2. THE Preset_Generator SHALL 以 Base_Patch 作为基础配置，固定 osc_1_on 为 1.0、osc_1 使用默认波表、filter_1_on 为 1.0
3. WHEN 指定某个 Effect_Switch 参数名和目标状态（0.0 或 1.0）时，THE Preset_Generator SHALL 生成仅该 Effect_Switch 与 Base_Patch 不同的预设文件
4. THE Preset_Generator SHALL 为每个 Effect_Switch（共 9 个）分别生成开启和关闭两个预设，共计 18 个预设文件
5. WHEN 生成预设文件时，THE Preset_Generator SHALL 确保除目标 Effect_Switch 外的所有参数值与 Base_Patch 完全一致
6. IF 指定的参数名不在已知的 Effect_Switch 列表中，THEN THE Preset_Generator SHALL 返回明确的错误信息，包含无效参数名和有效参数列表

### 需求 2：音频渲染（Phase 0）

**用户故事：** 作为研究人员，我希望能将预设文件批量渲染为音频文件，以便后续提取 embedding 进行分析。

#### 验收标准

1. WHEN 提供一个有效的 Vital 预设文件路径时，THE Audio_Renderer SHALL 使用固定的 MIDI 输入（单音符 C4、力度 100、时长 2 秒）渲染出对应的音频文件
2. THE Audio_Renderer SHALL 输出采样率为 44100 Hz、单声道、16-bit 或 32-bit float 的 WAV 格式音频文件
3. WHEN 提供一个包含多个预设文件的目录路径时，THE Audio_Renderer SHALL 批量渲染该目录下所有 .vital 文件，并将输出音频保存到指定的输出目录
4. THE Audio_Renderer SHALL 为每个输出音频文件使用与源预设文件相同的基础文件名（扩展名替换为 .wav）
5. IF 渲染过程中 Vital 引擎返回错误或超时（超过 30 秒），THEN THE Audio_Renderer SHALL 记录错误日志并跳过该预设，继续处理剩余文件
6. WHEN 渲染完成后，THE Audio_Renderer SHALL 输出渲染摘要，包含成功数量、失败数量和失败文件列表

### 需求 3：MuQ Embedding 提取（Phase 0）

**用户故事：** 作为研究人员，我希望能从渲染的音频中提取 MuQ embedding，以便量化分析音频表征的区分能力。

#### 验收标准

1. WHEN 提供一个 WAV 音频文件路径时，THE Embedding_Extractor SHALL 调用 MuQ 模型提取该音频的 embedding 向量
2. THE Embedding_Extractor SHALL 输出固定维度的浮点向量（维度由 MuQ 模型架构决定），并以 NumPy .npy 格式保存
3. WHEN 提供一个包含多个 WAV 文件的目录路径时，THE Embedding_Extractor SHALL 批量提取所有音频的 embedding
4. THE Embedding_Extractor SHALL 将所有 embedding 汇总为一个结构化数据文件（包含文件名到 embedding 的映射），以便后续分析使用
5. IF MuQ 模型加载失败或音频文件格式不兼容，THEN THE Embedding_Extractor SHALL 返回描述性错误信息，包含失败原因和文件路径
6. THE Embedding_Extractor SHALL 在提取前验证输入音频的采样率与 MuQ 模型要求一致，WHEN 采样率不匹配时，THE Embedding_Extractor SHALL 自动重采样至目标采样率

### 需求 4：效果器开关区分能力评估（Phase 0）

**用户故事：** 作为研究人员，我希望能量化评估 MuQ embedding 对效果器开关状态的区分能力，以便判断该技术路线的可行性。

#### 验收标准

1. WHEN 提供开启和关闭两组 embedding 时，THE Discriminator SHALL 计算两组 embedding 之间的 cosine similarity 均值
2. THE Discriminator SHALL 对每个 Effect_Switch（共 9 个）分别计算区分度指标，并输出按区分度排序的结果表
3. THE Discriminator SHALL 使用线性分类器（如 Logistic Regression）对每个 Effect_Switch 进行二分类评估，报告分类准确率
4. WHEN 所有 9 个 Effect_Switch 的评估完成后，THE Discriminator SHALL 生成汇总报告，包含每个效果器的 cosine similarity、分类准确率和总体结论
5. THE Discriminator SHALL 将可行性判定标准定义为：至少 6 个 Effect_Switch 的线性分类准确率达到 75% 以上
6. IF 某个 Effect_Switch 的开启和关闭 embedding 的 cosine similarity 超过 0.99，THEN THE Discriminator SHALL 在报告中标记该效果器为"表征无法区分"，并建议检查该效果器在 Base_Patch 下是否产生可听差异

### 需求 5：Vital 预设 JSON 解析与序列化（Phase 0-1）

**用户故事：** 作为研究人员，我希望能可靠地解析和生成 Vital 预设文件，以便在整个实验流程中正确操作参数。

#### 验收标准

1. WHEN 提供一个 .vital 文件路径时，THE Preset_Parser SHALL 将其解析为结构化的 Python 对象，包含 settings 字典和 modulations 列表
2. THE Preset_Serializer SHALL 将结构化的 Python 预设对象序列化为符合 Vital 格式的 JSON 文件
3. FOR ALL 有效的 Vital 预设文件，解析后再序列化再解析 SHALL 产生与原始解析结果等价的对象（round-trip 属性）
4. THE Preset_Parser SHALL 验证解析结果中包含所有已知的 Effect_Switch 参数键（共 9 个）
5. IF 输入文件不是有效的 JSON 或缺少 settings 键，THEN THE Preset_Parser SHALL 返回描述性错误信息，包含具体的格式违规内容

### 需求 6：实验流水线编排（Phase 0）

**用户故事：** 作为研究人员，我希望能一键运行完整的 Phase 0 实验流程，以便快速迭代实验设计。

#### 验收标准

1. THE Pipeline_Orchestrator SHALL 按顺序执行以下步骤：预设生成 → 音频渲染 → embedding 提取 → 区分能力评估
2. WHEN 某个步骤失败时，THE Pipeline_Orchestrator SHALL 停止执行后续步骤，并输出失败步骤的名称和错误详情
3. THE Pipeline_Orchestrator SHALL 支持从指定步骤恢复执行（跳过已完成的步骤），WHEN 中间产物文件已存在时
4. THE Pipeline_Orchestrator SHALL 将所有中间产物（预设文件、音频文件、embedding 文件）和最终报告保存到一个带时间戳的实验目录中
5. WHEN 实验完成后，THE Pipeline_Orchestrator SHALL 输出实验摘要，包含各步骤耗时、最终可行性判定结果和下一步建议

### 需求 7：核心参数回归（Phase 1）

**用户故事：** 作为研究人员，我希望在 Phase 0 验证通过后，能训练一个从 embedding 预测核心合成器参数的回归模型。

#### 验收标准

1. THE Parameter_Regressor SHALL 接受 MuQ embedding 作为输入，输出约 45 个核心参数的预测值，包括：osc_1 核心参数（level、transpose、tune、wave_frame、unison_voices、unison_detune）、filter_1 核心参数（cutoff、resonance、drive、mix、model、style）、env_1 ADSR 参数（attack、decay、sustain、release）、9 个 Effect_Switch 及对应的 dry_wet 参数
2. THE Training_Data_Generator SHALL 通过随机采样参数空间生成训练数据集，每个样本包含参数向量和对应的渲染音频 embedding
3. THE Parameter_Regressor SHALL 使用 MLP（多层感知机）架构，以 MuQ embedding 为输入，以参数向量为回归目标
4. WHEN 训练完成后，THE Parameter_Regressor SHALL 在测试集上报告每个参数的 MAE（平均绝对误差）和整体的多频谱损失
5. THE Parameter_Regressor SHALL 支持将预测的参数向量导出为有效的 Vital 预设文件，以便主观听音验证
6. IF Phase 0 的可行性判定结果为"不可行"，THEN THE Parameter_Regressor SHALL 不被启动，系统 SHALL 输出建议调整技术路线的提示

### 需求 8：训练数据集生成（Phase 1）

**用户故事：** 作为研究人员，我希望能自动生成大规模的参数-音频配对数据集，以便训练参数回归模型。

#### 验收标准

1. THE Training_Data_Generator SHALL 通过在约 45 个核心参数的有效值域内均匀随机采样，生成指定数量的预设配置
2. THE Training_Data_Generator SHALL 为每个采样的预设配置渲染音频并提取 MuQ embedding，形成（参数向量，embedding）配对
3. THE Training_Data_Generator SHALL 将数据集按 80/10/10 的比例划分为训练集、验证集和测试集
4. THE Training_Data_Generator SHALL 将数据集保存为可高效加载的格式（如 HDF5 或 NumPy .npz），包含参数矩阵和 embedding 矩阵
5. WHEN 生成过程中某个样本的渲染或 embedding 提取失败时，THE Training_Data_Generator SHALL 跳过该样本并记录日志，确保最终数据集中无缺失值
6. THE Training_Data_Generator SHALL 记录每个参数的值域范围和采样分布信息，作为数据集元数据保存
