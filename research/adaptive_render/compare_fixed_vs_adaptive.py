#!/usr/bin/env python3
"""
固定 2s vs 自适应渲染对比脚本。

对同一 .vital preset 分别计算固定 2s 模式和自适应模式的时序参数，
输出对比信息。可选渲染两种模式的音频。

用法:
    python research/adaptive_render/compare_fixed_vs_adaptive.py path/to/preset.vital
    python research/adaptive_render/compare_fixed_vs_adaptive.py path/to/preset.vital --render --vst /path/to/Vital.vst3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便导入 src/ 下的模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.adaptive_timing import AdaptiveTimingCalculator


# 固定模式常量
FIXED_DURATION = 2.0
FIXED_NOTE_OFF = FIXED_DURATION - 0.1  # 1.9s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="固定 2s vs 自适应渲染对比工具"
    )
    parser.add_argument("preset", type=Path, help=".vital preset 文件路径")
    parser.add_argument(
        "--render", action="store_true", help="渲染两种模式的音频（需要 --vst 参数）"
    )
    parser.add_argument("--vst", type=Path, default=None, help="Vital VST3 插件路径")
    args = parser.parse_args()

    preset_path: Path = args.preset
    if not preset_path.exists():
        print(f"错误: preset 文件不存在: {preset_path}", file=sys.stderr)
        sys.exit(1)

    calc = AdaptiveTimingCalculator()

    # 提取 ADSR 参数
    adsr_raw = calc.extract_adsr_from_preset(preset_path)
    timing = calc.compute_timing(preset_path)

    attack_sec = calc.power_law_to_seconds(adsr_raw["env_1_attack"])
    decay_sec = calc.power_law_to_seconds(adsr_raw["env_1_decay"])
    release_sec = calc.power_law_to_seconds(adsr_raw["env_1_release"])

    print("=" * 60)
    print(f"Preset: {preset_path.name}")
    print("=" * 60)

    print("\n--- ADSR Parameters ---")
    print(f"  attack:  {attack_sec:.4f} s  (raw={adsr_raw['env_1_attack']:.6f})")
    print(f"  decay:   {decay_sec:.4f} s  (raw={adsr_raw['env_1_decay']:.6f})")
    print(f"  sustain: {adsr_raw['env_1_sustain']:.4f}  (level)")
    print(f"  release: {release_sec:.4f} s  (raw={adsr_raw['env_1_release']:.6f})")

    # 对比表
    print("\n--- Timing Comparison ---")
    print(f"  {'Parameter':<20} {'Fixed 2s':>12} {'Adaptive':>12} {'Diff':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    note_off_diff = timing.note_off - FIXED_NOTE_OFF
    print(f"  {'note_off (s)':<20} {FIXED_NOTE_OFF:>12.4f} {timing.note_off:>12.4f} {note_off_diff:>+12.4f}")

    dur_diff = timing.total_duration - FIXED_DURATION
    print(f"  {'total_duration (s)':<20} {FIXED_DURATION:>12.4f} {timing.total_duration:>12.4f} {dur_diff:>+12.4f}")

    # 音频长度差异（采样数 @ 44100Hz）
    fixed_samples = int(FIXED_DURATION * 44100)
    adaptive_samples = int(timing.total_duration * 44100)
    sample_diff = adaptive_samples - fixed_samples
    print(f"  {'audio samples':>20} {fixed_samples:>12d} {adaptive_samples:>12d} {sample_diff:>+12d}")

    # 分析
    print("\n--- Analysis ---")
    if attack_sec > FIXED_NOTE_OFF:
        print(f"  ⚠ Attack ({attack_sec:.2f}s) > fixed note_off ({FIXED_NOTE_OFF}s): "
              "attack truncated in fixed mode!")
    if attack_sec + decay_sec > FIXED_NOTE_OFF:
        print(f"  ⚠ Attack+Decay ({attack_sec + decay_sec:.2f}s) > fixed note_off ({FIXED_NOTE_OFF}s): "
              "decay truncated in fixed mode!")
    if release_sec > 0.1:
        print(f"  ⚠ Release ({release_sec:.2f}s) > fixed release window (0.1s): "
              "release truncated in fixed mode!")
    if dur_diff < 0.01 and dur_diff > -0.01:
        print("  ✓ Adaptive timing is similar to fixed 2s for this preset.")
    elif dur_diff > 0:
        print(f"  → Adaptive mode adds {dur_diff:.2f}s to capture full ADSR envelope.")
    else:
        print(f"  → Adaptive mode is {abs(dur_diff):.2f}s shorter (minimal ADSR).")

    print("=" * 60)

    # 可选渲染
    if args.render:
        if args.vst is None:
            print("错误: 渲染需要 --vst 参数指定 Vital VST3 路径", file=sys.stderr)
            sys.exit(1)

        from src.audio_renderer import AudioRenderer, RenderConfig

        output_dir = preset_path.parent
        stem = preset_path.stem

        # 固定模式渲染
        print("\n--- Rendering Fixed Mode ---")
        config_fixed = RenderConfig()
        config_fixed.duration_sec = FIXED_DURATION
        renderer_fixed = AudioRenderer(vital_vst_path=args.vst, config=config_fixed)

        out_fixed = output_dir / f"{stem}_fixed.wav"
        success_fixed = renderer_fixed.render_preset(preset_path, out_fixed)
        print(f"  Fixed:    {'✓' if success_fixed else '✗'} {out_fixed}")

        # 自适应模式渲染
        print("\n--- Rendering Adaptive Mode ---")
        config_adaptive = RenderConfig()
        config_adaptive.duration_sec = timing.total_duration
        renderer_adaptive = AudioRenderer(vital_vst_path=args.vst, config=config_adaptive)

        out_adaptive = output_dir / f"{stem}_adaptive.wav"
        success_adaptive = renderer_adaptive.render_preset(
            preset_path, out_adaptive, note_off_time=timing.note_off
        )
        print(f"  Adaptive: {'✓' if success_adaptive else '✗'} {out_adaptive}")

        if success_fixed and success_adaptive:
            print(f"\n  Fixed WAV:    {out_fixed}")
            print(f"  Adaptive WAV: {out_adaptive}")
            print("  可用音频播放器对比两个文件的 ADSR 包络差异。")


if __name__ == "__main__":
    main()
