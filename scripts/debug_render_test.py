#!/usr/bin/env python3
"""
调试脚本：验证不同 MIDI 条件是否产生不同音高的音频。
同时检查无效音频（静音/削波）的比例。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.audio_renderer import AudioRenderer, RenderConfig
from src.preset_generator import PresetGenerator
from src.preset_parser import PresetParser
from src.smart_sampler import SmartSampler
from src.training_data import CORE_PARAMS

VITAL_VST_PATH = Path("/Library/Audio/Plug-Ins/VST3/Vital.vst3")
OUTPUT_DIR = Path("experiments/debug_render")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_fundamental_freq(audio: np.ndarray, sr: int) -> float:
    """用自相关法估算基频。"""
    if audio.ndim == 2:
        audio = audio[0]
    # 去直流
    audio = audio - np.mean(audio)
    if np.max(np.abs(audio)) < 1e-6:
        return 0.0
    # 自相关
    n = len(audio)
    corr = np.correlate(audio, audio, mode="full")
    corr = corr[n - 1:]  # 只取正半部分
    corr = corr / corr[0]  # 归一化

    # 找第一个过零点后的峰值
    min_lag = int(sr / 2000)  # 最高 2000 Hz
    max_lag = int(sr / 20)    # 最低 20 Hz
    if max_lag >= len(corr):
        max_lag = len(corr) - 1

    # 找过零点
    d = np.diff(corr[:max_lag + 1])
    # 找第一个从负变正的点（过零后上升）
    start = min_lag
    for i in range(min_lag, max_lag):
        if corr[i] < 0:
            start = i
            break

    # 从 start 开始找峰值
    peak_lag = start
    peak_val = -1
    for i in range(start, max_lag):
        if corr[i] > peak_val:
            peak_val = corr[i]
            peak_lag = i

    if peak_lag == 0 or peak_val < 0.1:
        return 0.0
    return sr / peak_lag


def rms_db(audio: np.ndarray) -> float:
    """计算 RMS dB。"""
    if audio.ndim == 2:
        audio = audio[0]
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def main():
    print("=" * 60)
    print("  调试渲染测试")
    print("=" * 60)

    # 生成 5 个不同的预设
    sampler = SmartSampler(seed=42)
    params = sampler.sample(5, strategy="lhs_stratified")

    parser = PresetParser()
    generator = PresetGenerator(parser)

    # MIDI 条件
    conditions = [
        (48, 80, "C3_v80"),
        (60, 80, "C4_v80"),
        (72, 80, "C5_v80"),
        (48, 120, "C3_v120"),
        (60, 120, "C4_v120"),
        (72, 120, "C5_v120"),
    ]

    # 创建渲染器
    print("\n加载 Vital VST3...")
    renderer = AudioRenderer(VITAL_VST_PATH, RenderConfig())

    silent_count = 0
    total_count = 0

    for pi in range(5):
        # 生成预设
        preset = generator.create_base_patch()
        for col, (pname, _, _) in enumerate(CORE_PARAMS):
            if pname in preset.settings:
                preset.settings[pname] = float(params[pi, col])

        preset_path = OUTPUT_DIR / f"test_preset_{pi}.vital"
        parser.serialize(preset, preset_path)

        print(f"\n--- 预设 {pi} ---")
        # 打印一些关键参数
        print(f"  osc_1_level: {preset.settings.get('osc_1_level', 'N/A')}")
        print(f"  osc_1_on: {preset.settings.get('osc_1_on', 'N/A')}")
        print(f"  filter_1_cutoff: {preset.settings.get('filter_1_cutoff', 'N/A')}")
        print(f"  env_1_attack: {preset.settings.get('env_1_attack', 'N/A')}")
        print(f"  env_1_sustain: {preset.settings.get('env_1_sustain', 'N/A')}")
        print(f"  volume: {preset.settings.get('volume', 'N/A')}")

        freqs = {}
        for midi_note, velocity, label in conditions:
            renderer._config.midi_note = midi_note
            renderer._config.velocity = velocity
            renderer._config.duration_sec = 2.0

            audio_path = OUTPUT_DIR / f"test_{pi}_{label}.wav"
            success = renderer.render_preset(preset_path, audio_path)
            total_count += 1

            if not success:
                print(f"  {label}: 渲染失败")
                continue

            # 读取音频分析
            import soundfile as sf
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]

            db = rms_db(audio)
            freq = estimate_fundamental_freq(audio, sr)
            freqs[label] = freq

            is_silent = db < -60
            if is_silent:
                silent_count += 1

            print(f"  {label}: RMS={db:.1f} dB, F0≈{freq:.1f} Hz, "
                  f"peak={np.max(np.abs(audio)):.4f}"
                  f"{' ⚠️ SILENT' if is_silent else ''}")

        # 检查不同八度的频率比
        if "C3_v80" in freqs and "C4_v80" in freqs and freqs["C3_v80"] > 0 and freqs["C4_v80"] > 0:
            ratio_34 = freqs["C4_v80"] / freqs["C3_v80"]
            print(f"  C4/C3 频率比: {ratio_34:.2f} (应≈2.0)")
        if "C4_v80" in freqs and "C5_v80" in freqs and freqs["C4_v80"] > 0 and freqs["C5_v80"] > 0:
            ratio_45 = freqs["C5_v80"] / freqs["C4_v80"]
            print(f"  C5/C4 频率比: {ratio_45:.2f} (应≈2.0)")

    print(f"\n总结: {silent_count}/{total_count} 个样本为静音 ({100*silent_count/total_count:.1f}%)")


if __name__ == "__main__":
    main()
