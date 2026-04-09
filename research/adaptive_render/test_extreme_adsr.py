#!/usr/bin/env python3
"""
极端 ADSR preset 测试脚本。

构造 3 个极端 ADSR 测试 preset，分别用固定 2s 和自适应模式渲染，
验证渲染结果的正确性。

极端 preset:
  1. 长 attack: env_1_attack=0.55 (~2.92s), 其余默认
  2. 长 release: env_1_release=0.55 (~2.92s), 其余默认
  3. 长 attack + decay + release: 各 raw=0.45

用法:
    # 仅诊断（不渲染音频）
    python research/adaptive_render/test_extreme_adsr.py

    # 渲染音频（需要 Vital VST3）
    python research/adaptive_render/test_extreme_adsr.py --vst /path/to/Vital.vst3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import copy

import numpy as np

from src.adaptive_timing import AdaptiveTimingCalculator
from src.preset_generator import PresetGenerator
from src.preset_parser import PresetParser

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
FIXED_DURATION = 2.0
FIXED_NOTE_OFF = FIXED_DURATION - 0.1  # 1.9s
SAMPLE_RATE = 44100
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "extreme"

# ---------------------------------------------------------------------------
# 极端 ADSR 定义
# ---------------------------------------------------------------------------

@dataclass
class ExtremePresetDef:
    """极端 preset 定义。"""
    name: str
    description: str
    env_1_attack: float | None = None
    env_1_decay: float | None = None
    env_1_sustain: float | None = None
    env_1_release: float | None = None


# ADSR 默认值（与 AdaptiveTimingCalculator.DEFAULTS 一致）
_ADSR_DEFAULTS = {
    "env_1_attack": 0.0,
    "env_1_decay": 0.0,
    "env_1_sustain": 1.0,
    "env_1_release": 0.0,
}

EXTREME_PRESETS: list[ExtremePresetDef] = [
    ExtremePresetDef(
        name="long_attack",
        description="长 attack (raw=0.55, ~2.92s), 其余默认",
        env_1_attack=0.55,
    ),
    ExtremePresetDef(
        name="long_release",
        description="长 release (raw=0.55, ~2.92s), 其余默认",
        env_1_release=0.55,
    ),
    ExtremePresetDef(
        name="long_adr",
        description="长 attack + decay + release (各 raw=0.45)",
        env_1_attack=0.45,
        env_1_decay=0.45,
        env_1_release=0.45,
    ),
]


# ---------------------------------------------------------------------------
# Preset 生成
# ---------------------------------------------------------------------------

def generate_extreme_presets(output_dir: Path) -> list[tuple[ExtremePresetDef, Path]]:
    """生成极端 ADSR preset 文件。

    Returns:
        (定义, preset 路径) 列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = PresetParser()
    generator = PresetGenerator(parser=parser)

    results: list[tuple[ExtremePresetDef, Path]] = []

    for preset_def in EXTREME_PRESETS:
        # 从 base patch 开始
        preset = generator.create_base_patch()
        settings = copy.deepcopy(preset.settings)

        # 先将所有 ADSR 参数设为合理默认值（raw 值 [0,1] 范围）
        for key, default_val in _ADSR_DEFAULTS.items():
            settings[key] = default_val

        # 覆盖极端值
        if preset_def.env_1_attack is not None:
            settings["env_1_attack"] = preset_def.env_1_attack
        if preset_def.env_1_decay is not None:
            settings["env_1_decay"] = preset_def.env_1_decay
        if preset_def.env_1_sustain is not None:
            settings["env_1_sustain"] = preset_def.env_1_sustain
        if preset_def.env_1_release is not None:
            settings["env_1_release"] = preset_def.env_1_release

        preset.settings = settings
        preset.extra["preset_name"] = f"extreme_{preset_def.name}"

        # 序列化
        filepath = output_dir / f"extreme_{preset_def.name}.vital"
        parser.serialize(preset, filepath)
        results.append((preset_def, filepath))
        print(f"  生成: {filepath.name}")

    return results


# ---------------------------------------------------------------------------
# 诊断输出
# ---------------------------------------------------------------------------

def print_timing_diagnostics(
    preset_def: ExtremePresetDef,
    preset_path: Path,
    calc: AdaptiveTimingCalculator,
) -> dict:
    """打印单个 preset 的时序诊断信息。

    Returns:
        诊断数据字典
    """
    adsr_raw = calc.extract_adsr_from_preset(preset_path)
    timing = calc.compute_timing(preset_path)

    attack_sec = calc.power_law_to_seconds(adsr_raw["env_1_attack"])
    decay_sec = calc.power_law_to_seconds(adsr_raw["env_1_decay"])
    release_sec = calc.power_law_to_seconds(adsr_raw["env_1_release"])
    sustain = adsr_raw["env_1_sustain"]

    print(f"\n{'=' * 60}")
    print(f"Preset: {preset_def.name} — {preset_def.description}")
    print(f"{'=' * 60}")

    print("\n  ADSR Parameters:")
    print(f"    attack:  {attack_sec:.4f} s  (raw={adsr_raw['env_1_attack']:.4f})")
    print(f"    decay:   {decay_sec:.4f} s  (raw={adsr_raw['env_1_decay']:.4f})")
    print(f"    sustain: {sustain:.4f}  (level)")
    print(f"    release: {release_sec:.4f} s  (raw={adsr_raw['env_1_release']:.4f})")

    print(f"\n  {'Mode':<12} {'note_off':>10} {'duration':>10} {'samples':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Fixed 2s':<12} {FIXED_NOTE_OFF:>10.4f} {FIXED_DURATION:>10.4f} {int(FIXED_DURATION * SAMPLE_RATE):>10d}")
    print(f"  {'Adaptive':<12} {timing.note_off:>10.4f} {timing.total_duration:>10.4f} {int(timing.total_duration * SAMPLE_RATE):>10d}")

    # 问题分析
    issues: list[str] = []
    if attack_sec > FIXED_NOTE_OFF:
        issues.append(f"attack ({attack_sec:.2f}s) > fixed note_off ({FIXED_NOTE_OFF}s): attack 被截断")
    if attack_sec + decay_sec > FIXED_NOTE_OFF:
        issues.append(f"attack+decay ({attack_sec + decay_sec:.2f}s) > fixed note_off: decay 被截断")
    if release_sec > 0.1:
        issues.append(f"release ({release_sec:.2f}s) > fixed release window (0.1s): release 被截断")

    if issues:
        print("\n  ⚠ 固定模式问题:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  ✓ 固定模式无明显截断问题")

    return {
        "name": preset_def.name,
        "attack_sec": attack_sec,
        "decay_sec": decay_sec,
        "sustain": sustain,
        "release_sec": release_sec,
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def render_preset_both_modes(
    preset_path: Path,
    preset_name: str,
    timing,
    vst_path: Path,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    """用固定和自适应两种模式渲染 preset。

    Returns:
        (fixed_wav_path, adaptive_wav_path)，渲染失败时为 None
    """
    from src.audio_renderer import AudioRenderer, RenderConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    # 固定模式
    config_fixed = RenderConfig()
    config_fixed.duration_sec = FIXED_DURATION
    renderer = AudioRenderer(vital_vst_path=vst_path, config=config_fixed)

    fixed_path = output_dir / f"{preset_name}_fixed.wav"
    ok_fixed = renderer.render_preset(preset_path, fixed_path)
    if ok_fixed:
        print(f"    ✓ Fixed:    {fixed_path.name}")
    else:
        print(f"    ✗ Fixed:    渲染失败")
        fixed_path = None

    # 自适应模式
    config_adaptive = RenderConfig()
    config_adaptive.duration_sec = timing.total_duration
    renderer_adaptive = AudioRenderer(vital_vst_path=vst_path, config=config_adaptive)

    adaptive_path = output_dir / f"{preset_name}_adaptive.wav"
    ok_adaptive = renderer_adaptive.render_preset(
        preset_path,
        adaptive_path,
        note_off_time=timing.note_off,
    )
    if ok_adaptive:
        print(f"    ✓ Adaptive: {adaptive_path.name}")
    else:
        print(f"    ✗ Adaptive: 渲染失败")
        adaptive_path = None

    return (fixed_path if ok_fixed else None, adaptive_path if ok_adaptive else None)


# ---------------------------------------------------------------------------
# 验证 (Task 10.2)
# ---------------------------------------------------------------------------

def get_wav_duration(wav_path: Path) -> float:
    """读取 WAV 文件时长（秒）。"""
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


def load_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """读取 WAV 为单声道 numpy 数组。

    Returns:
        (audio_array, sample_rate)
    """
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # 转换为 float
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.int16

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float64)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # 归一化到 [-1, 1]
    max_val = float(np.iinfo(dtype).max)
    audio = audio / max_val

    return audio, rate


def compute_rms_envelope(audio: np.ndarray, sr: int, hop_sec: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """计算 RMS 包络。

    Args:
        audio: 单声道音频
        sr: 采样率
        hop_sec: 帧步长（秒）

    Returns:
        (times, rms_values)
    """
    hop = int(sr * hop_sec)
    window = hop * 2  # 窗口 = 2 * hop
    n_frames = max(1, (len(audio) - window) // hop + 1)

    times = np.zeros(n_frames)
    rms = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        end = min(start + window, len(audio))
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0
        times[i] = start / sr

    return times, rms


def verify_wav_duration(wav_path: Path, expected_sec: float, label: str, tolerance: float = 0.01) -> bool:
    """验证 WAV 时长是否匹配预期值。"""
    actual = get_wav_duration(wav_path)
    diff = abs(actual - expected_sec)
    ok = diff <= tolerance
    status = "✓" if ok else "✗"
    print(f"    {status} {label} duration: {actual:.4f}s (expected {expected_sec:.4f}s, diff={diff:.4f}s)")
    return ok


def verify_release_decay(
    wav_path: Path,
    note_off_sec: float,
    total_duration_sec: float,
    label: str,
) -> bool:
    """验证 note_off 之后 RMS 呈衰减趋势（release 阶段存在性）。

    策略：比较 note_off 附近的 RMS 与音频末尾的 RMS，
    末尾 RMS 应小于 note_off 附近的 RMS。
    """
    audio, sr = load_wav_mono(wav_path)
    times, rms = compute_rms_envelope(audio, sr)

    if len(rms) < 3:
        print(f"    ? {label}: 音频太短，无法分析 RMS 包络")
        return False

    # 找到 note_off 附近的 RMS
    note_off_idx = np.searchsorted(times, note_off_sec)
    note_off_idx = min(note_off_idx, len(rms) - 1)

    # note_off 附近的 RMS（取前后几帧的最大值）
    start_idx = max(0, note_off_idx - 2)
    end_idx = min(len(rms), note_off_idx + 3)
    rms_at_note_off = np.max(rms[start_idx:end_idx])

    # 音频末尾的 RMS（最后几帧）
    tail_start = max(0, len(rms) - 3)
    rms_at_tail = np.mean(rms[tail_start:])

    # release 阶段存在性：末尾 RMS 应小于 note_off 附近
    if rms_at_note_off < 1e-8:
        print(f"    ? {label}: note_off 附近 RMS 接近零，无法判断 release 衰减")
        return False

    decay_ratio = rms_at_tail / rms_at_note_off if rms_at_note_off > 0 else 1.0
    ok = decay_ratio < 0.9  # 末尾至少衰减 10%
    status = "✓" if ok else "✗"
    print(f"    {status} {label} release decay: RMS@note_off={rms_at_note_off:.6f}, "
          f"RMS@tail={rms_at_tail:.6f}, ratio={decay_ratio:.4f}")
    return ok


def run_verification(
    preset_name: str,
    timing,
    fixed_wav: Path | None,
    adaptive_wav: Path | None,
) -> dict:
    """对单个 preset 的渲染结果执行验证。

    Returns:
        验证结果字典
    """
    results: dict = {"name": preset_name, "checks": []}

    print(f"\n  验证 {preset_name}:")

    if adaptive_wav and adaptive_wav.exists():
        # 检查自适应 WAV 时长
        ok = verify_wav_duration(
            adaptive_wav, timing.total_duration, "Adaptive", tolerance=0.01
        )
        results["checks"].append(("adaptive_duration", ok))

        # 检查 release 衰减趋势
        ok = verify_release_decay(
            adaptive_wav, timing.note_off, timing.total_duration, "Adaptive"
        )
        results["checks"].append(("adaptive_release_decay", ok))
    else:
        print("    - Adaptive WAV 不存在，跳过验证")

    if fixed_wav and fixed_wav.exists():
        # 检查固定模式 WAV 时长
        ok = verify_wav_duration(fixed_wav, FIXED_DURATION, "Fixed", tolerance=0.01)
        results["checks"].append(("fixed_duration", ok))
    else:
        print("    - Fixed WAV 不存在，跳过验证")

    return results


# ---------------------------------------------------------------------------
# 诊断报告
# ---------------------------------------------------------------------------

def print_diagnostic_report(diagnostics: list[dict]) -> None:
    """打印汇总诊断报告。"""
    print(f"\n{'=' * 60}")
    print("诊断报告汇总")
    print(f"{'=' * 60}")

    print(f"\n  {'Preset':<16} {'A(s)':>8} {'D(s)':>8} {'R(s)':>8} "
          f"{'note_off':>10} {'duration':>10} {'fixed_ok':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for d in diagnostics:
        t = d["timing"]
        # 固定模式是否能完整捕获 ADSR
        fixed_ok = (
            d["attack_sec"] + d["decay_sec"] <= FIXED_NOTE_OFF
            and d["release_sec"] <= 0.1
        )
        print(f"  {d['name']:<16} {d['attack_sec']:>8.3f} {d['decay_sec']:>8.3f} "
              f"{d['release_sec']:>8.3f} {t.note_off:>10.3f} "
              f"{t.total_duration:>10.3f} {'✓' if fixed_ok else '✗':>10}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="极端 ADSR preset 测试与验证工具"
    )
    parser.add_argument(
        "--vst", type=Path, default=None,
        help="Vital VST3 插件路径（提供时渲染音频并执行验证）",
    )
    args = parser.parse_args()

    calc = AdaptiveTimingCalculator()

    # 1. 生成极端 preset
    print("生成极端 ADSR preset...")
    presets = generate_extreme_presets(OUTPUT_DIR)

    # 2. 时序诊断
    diagnostics: list[dict] = []
    for preset_def, preset_path in presets:
        diag = print_timing_diagnostics(preset_def, preset_path, calc)
        diagnostics.append(diag)

    # 3. 汇总诊断报告
    print_diagnostic_report(diagnostics)

    # 4. 渲染（可选）
    if args.vst is not None:
        vst_path: Path = args.vst
        if not vst_path.exists():
            print(f"错误: VST3 路径不存在: {vst_path}", file=sys.stderr)
            sys.exit(1)

        print("渲染极端 preset（固定 + 自适应）...")
        verification_results: list[dict] = []

        for preset_def, preset_path in presets:
            diag = next(d for d in diagnostics if d["name"] == preset_def.name)
            timing = diag["timing"]

            print(f"\n  渲染 {preset_def.name}:")
            fixed_wav, adaptive_wav = render_preset_both_modes(
                preset_path, preset_def.name, timing, vst_path, OUTPUT_DIR,
            )

            # 5. 验证渲染结果 (Task 10.2)
            vr = run_verification(preset_def.name, timing, fixed_wav, adaptive_wav)
            verification_results.append(vr)

        # 6. 验证汇总
        print(f"\n{'=' * 60}")
        print("验证结果汇总")
        print(f"{'=' * 60}")
        total_checks = 0
        passed_checks = 0
        for vr in verification_results:
            for check_name, ok in vr["checks"]:
                total_checks += 1
                if ok:
                    passed_checks += 1
        print(f"  通过: {passed_checks}/{total_checks}")
        if passed_checks == total_checks:
            print("  ✓ 所有验证通过")
        else:
            print("  ✗ 部分验证未通过，请检查上方详细输出")
    else:
        print("提示: 使用 --vst /path/to/Vital.vst3 渲染音频并执行验证")

    print("\n完成。")


if __name__ == "__main__":
    main()
