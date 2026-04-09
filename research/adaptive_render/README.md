# ADSR 自适应渲染研究脚本

研究原型脚本，用于验证和演示基于 ADSR 包络参数的自适应渲染方案。

## 脚本说明

### `demo_adaptive.py`

单 preset 自适应渲染诊断工具。加载一个 `.vital` preset，提取 ADSR 参数，
计算自适应时序（note_off、total_duration），打印诊断信息。

```bash
# 仅查看诊断信息
python research/adaptive_render/demo_adaptive.py path/to/preset.vital

# 同时渲染音频
python research/adaptive_render/demo_adaptive.py path/to/preset.vital \
    --render --vst /path/to/Vital.vst3
```

### `compare_fixed_vs_adaptive.py`

固定 2s 模式与自适应模式的对比工具。对同一 preset 输出两种模式的时序参数差异，
并分析 ADSR 包络是否在固定模式下被截断。

```bash
# 仅查看对比信息
python research/adaptive_render/compare_fixed_vs_adaptive.py path/to/preset.vital

# 同时渲染两种模式的音频
python research/adaptive_render/compare_fixed_vs_adaptive.py path/to/preset.vital \
    --render --vst /path/to/Vital.vst3
```

### `test_extreme_adsr.py`

极端 ADSR preset 测试与验证工具。自动构造 3 个极端 ADSR preset（长 attack、
长 release、长 attack+decay+release），计算时序诊断，可选渲染并验证结果。

```bash
# 仅诊断（不需要 VST）
python research/adaptive_render/test_extreme_adsr.py

# 渲染音频并执行验证
python research/adaptive_render/test_extreme_adsr.py --vst /path/to/Vital.vst3
```

验证项目：
- 自适应 WAV 时长与计算的 total_duration 一致（误差 < 10ms）
- 固定模式 WAV 时长为 2.0s
- 自适应模式下 note_off 后 RMS 呈衰减趋势（release 阶段存在性）

输出目录：`research/adaptive_render/output/extreme/`

## 前置条件

- Python 3.10+
- 项目依赖已安装（`pip install -e .` 或等效方式）
- 渲染功能需要：
  - Vital VST3 插件（通过 `--vst` 参数指定路径）
  - `pedalboard` 库
