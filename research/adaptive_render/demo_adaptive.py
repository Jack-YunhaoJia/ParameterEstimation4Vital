#!/usr/bin/env python3
"""
单 preset 自适应渲染演示脚本。

加载一个 .vital preset，提取 ADSR 参数，计算自适应时序，
打印诊断信息。可选渲染音频输出。

用法:
    python research/adaptive_render/demo_adaptive.py path/to/preset.vital
    python research/adaptive_render/demo_adaptive.py path/to/preset.vital --render --vst /path/to/Vital.vst3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 src/ 下的模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.adaptive_timing import AdaptiveTimingCalculator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单 preset ADSR 自适应时序诊断工具"
    )
    parser.add_argument("preset", type=Path, help=".vital preset 文件路径")
    parser.add_argument(
        "--render", action="store_true", help="渲染音频（需要 --vst 参数）"
    )
    parser.add_argument("--vst", type=Path, default=None, help="Vital VST3 插件路径")
    args = parser.parse_args()

    preset_path: Path = args.preset
    if not preset_path.exists():
        print(f"错误: preset 文件不存在: {preset_path}", file=sys.stderr)
        sys.exit(1)

    calc = AdaptiveTimingCalculator()

    # 提取原始 ADSR 值
    adsr_raw = calc.extract_adsr_from_preset(preset_path)
    print("=" * 60)
    print(f"Preset: {preset_path.name}")
    print("=" * 60)

    print("\n--- Raw ADSR Values (Vital JSON) ---")
    for key, val in adsr_raw.items():
        print(f"  {key}: {val:.6f}")

    # 转换为秒数
    attack_sec = calc.power_law_to_seconds(adsr_raw["env_1_attack"])
    decay_sec = calc.power_law_to_seconds(adsr_raw["env_1_decay"])
    release_sec = calc.power_law_to_seconds(adsr_raw["env_1_release"])
    sustain = adsr_raw["env_1_sustain"]

    print("\n--- Converted to Seconds ---")
    print(f"  attack:  {attack_sec:.4f} s")
    print(f"  decay:   {decay_sec:.4f} s")
    print(f"  sustain: {sustain:.4f} (level)")
    print(f"  release: {release_sec:.4f} s")

    # 计算自适应时序
    timing = calc.compute_timing(preset_path)

    print("\n--- Adaptive Timing ---")
    print(f"  note_off:       {timing.note_off:.4f} s")
    print(f"  total_duration: {timing.total_duration:.4f} s")

    print("\n--- Comparison with Fixed 2s ---")
    print(f"  Fixed note_off:    1.9000 s")
    print(f"  Adaptive note_off: {timing.note_off:.4f} s")
    print(f"  Fixed duration:    2.0000 s")
    print(f"  Adaptive duration: {timing.total_duration:.4f} s")
    print("=" * 60)

    # 可选渲染
    if args.render:
        if args.vst is None:
            print("错误: 渲染需要 --vst 参数指定 Vital VST3 路径", file=sys.stderr)
            sys.exit(1)

        from src.audio_renderer import AudioRenderer, RenderConfig

        config = RenderConfig()
        config.duration_sec = timing.total_duration

        renderer = AudioRenderer(vital_vst_path=args.vst, config=config)

        output_path = preset_path.with_suffix(".adaptive.wav")
        note_off_time = timing.note_off

        print(f"\n渲染中... (duration={timing.total_duration:.2f}s, note_off={note_off_time:.2f}s)")
        success = renderer.render_preset(
            preset_path, output_path, note_off_time=note_off_time
        )
        if success:
            print(f"渲染成功: {output_path}")
        else:
            print("渲染失败", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
