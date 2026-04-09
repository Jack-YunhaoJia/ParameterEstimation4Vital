#!/usr/bin/env python3
"""
10 preset 批量渲染测试脚本。

使用 SmartSampler 生成 10 个随机 preset（seed=12345），
用自适应模式 + MultiConditionRenderer 渲染全部 6 条件（共 60 个 WAV），
并输出时序统计、验证报告和 CSV 报告。

用法:
    # 仅时序诊断（不渲染音频）
    python research/adaptive_render/test_batch_10.py

    # 渲染音频并执行完整验证（需要 Vital VST3）
    python research/adaptive_render/test_batch_10.py --vst /path/to/Vital.vst3
"""

from __future__ import annotations

import argparse
import csv
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.adaptive_timing import AdaptiveConfig, AdaptiveTiming, AdaptiveTimingCalculator
from src.multi_condition_renderer import DEFAULT_CONDITIONS
from src.preset_generator import PresetGenerator
from src.preset_parser import PresetParser
from src.smart_sampler import SmartSampler
from src.training_data import CORE_PARAMS

# ADSR parameter names that need seconds→raw conversion for Vital JSON
_ADSR_TIME_PARAMS = {"env_1_attack", "env_1_decay", "env_1_release"}

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SEED = 12345
N_PRESETS = 10
SAMPLE_RATE = 44100
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "batch_10"
CONDITION_LABELS = [c.label for c in DEFAULT_CONDITIONS]
SILENT_THRESHOLD_DB = -60.0  # RMS below this is considered silent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wav_info(wav_path: Path) -> tuple[float, float]:
    """Read WAV file and return (duration_sec, rms_db).

    Returns:
        (duration_sec, rms_db) — rms_db is -inf for silent files.
    """
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    duration_sec = n_frames / rate

    # Convert to float audio
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.int16

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    max_val = float(np.iinfo(dtype).max)
    audio = audio / max_val

    rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0
    rms_db = 20.0 * np.log10(rms) if rms > 0 else float("-inf")

    return duration_sec, rms_db


# ---------------------------------------------------------------------------
# Preset generation
# ---------------------------------------------------------------------------

@dataclass
class PresetInfo:
    """Info for a single generated preset."""
    preset_id: str
    preset_path: Path
    param_vector: np.ndarray
    timing: AdaptiveTiming
    attack_sec: float
    decay_sec: float
    release_sec: float


def generate_presets(output_dir: Path) -> list[PresetInfo]:
    """Sample 10 parameter vectors and create .vital preset files.

    Returns:
        List of PresetInfo for each generated preset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    presets_dir = output_dir / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample 10 parameter vectors
    sampler = SmartSampler(seed=SEED)
    param_matrix = sampler.sample_lhs(N_PRESETS)  # (10, 45)

    # 2. Create .vital files
    parser = PresetParser()
    generator = PresetGenerator(parser=parser)
    calc = AdaptiveTimingCalculator()

    presets: list[PresetInfo] = []

    for i in range(N_PRESETS):
        preset_id = f"batch_{i:03d}"
        param_vec = param_matrix[i]

        # Create preset from base patch, override with sampled params
        # CORE_PARAMS samples ADSR time params in seconds [0, 4].
        # Vital JSON stores raw values [0, 1], so convert back.
        preset = generator.create_base_patch()
        for col, (name, _, _) in enumerate(CORE_PARAMS):
            value = float(param_vec[col])
            if name in _ADSR_TIME_PARAMS:
                value = AdaptiveTimingCalculator.seconds_to_power_law(value)
            preset.settings[name] = value
        preset.extra["preset_name"] = preset_id

        # Serialize
        preset_path = presets_dir / f"{preset_id}.vital"
        parser.serialize(preset, preset_path)

        # Compute adaptive timing
        timing = calc.compute_timing(preset_path)

        presets.append(PresetInfo(
            preset_id=preset_id,
            preset_path=preset_path,
            param_vector=param_vec,
            timing=timing,
            attack_sec=timing.attack_sec,
            decay_sec=timing.decay_sec,
            release_sec=timing.release_sec,
        ))

    return presets


# ---------------------------------------------------------------------------
# Timing diagnostics (always runs, no VST needed)
# ---------------------------------------------------------------------------

def print_timing_diagnostics(presets: list[PresetInfo]) -> None:
    """Print adaptive timing statistics for all 10 presets."""
    note_offs = [p.timing.note_off for p in presets]
    durations = [p.timing.total_duration for p in presets]

    print(f"\n{'=' * 72}")
    print("自适应时序诊断 — 10 preset 批量")
    print(f"{'=' * 72}")

    # Per-preset table
    print(f"\n  {'ID':<12} {'A(s)':>8} {'D(s)':>8} {'R(s)':>8} "
          f"{'note_off':>10} {'duration':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for p in presets:
        print(f"  {p.preset_id:<12} {p.attack_sec:>8.3f} {p.decay_sec:>8.3f} "
              f"{p.release_sec:>8.3f} {p.timing.note_off:>10.4f} "
              f"{p.timing.total_duration:>10.4f}")

    # Summary statistics
    print(f"\n  统计:")
    print(f"    note_off  范围: [{min(note_offs):.4f}, {max(note_offs):.4f}] s")
    print(f"    duration  范围: [{min(durations):.4f}, {max(durations):.4f}] s")
    print(f"    最短渲染时长: {min(durations):.4f} s")
    print(f"    最长渲染时长: {max(durations):.4f} s")
    print()


# ---------------------------------------------------------------------------
# Rendering (requires --vst)
# ---------------------------------------------------------------------------

def render_all(presets: list[PresetInfo], vst_path: Path, output_dir: Path) -> None:
    """Render all 10 presets × 6 conditions using MultiConditionRenderer."""
    from src.audio_renderer import AudioRenderer, RenderConfig
    from src.multi_condition_renderer import MultiConditionRenderer

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    calc = AdaptiveTimingCalculator()
    config = RenderConfig()
    renderer = AudioRenderer(vital_vst_path=vst_path, config=config)
    mcr = MultiConditionRenderer(
        renderer=renderer,
        timing_calculator=calc,
    )

    print("渲染 10 preset × 6 条件...")
    for p in presets:
        result = mcr.render_preset(
            preset_path=p.preset_path,
            output_dir=audio_dir,
            preset_id=p.preset_id,
        )
        n_ok = len(result.condition_results)
        n_fail = len(result.failed_conditions)
        print(f"  {p.preset_id}: 成功={n_ok}, 失败={n_fail}")

    print("渲染完成。\n")


# ---------------------------------------------------------------------------
# Verification & statistics (requires rendered WAVs)
# ---------------------------------------------------------------------------

@dataclass
class WavRecord:
    """Single WAV file record for CSV report."""
    preset_id: str
    condition: str
    attack_sec: float
    decay_sec: float
    release_sec: float
    note_off: float
    total_duration: float
    wav_duration: float = 0.0
    rms_db: float = float("-inf")
    exists: bool = False
    silent: bool = False


def verify_and_report(presets: list[PresetInfo], output_dir: Path) -> None:
    """Verify rendered WAVs and output statistics + CSV report."""
    audio_dir = output_dir / "audio"
    records: list[WavRecord] = []
    missing_files: list[str] = []
    empty_files: list[str] = []
    silent_files: list[str] = []
    all_wav_durations: list[float] = []

    # Collect per-preset consistency data: preset_id -> set of (note_off, duration)
    consistency: dict[str, set[tuple[float, float]]] = {}

    for p in presets:
        consistency[p.preset_id] = set()
        for label in CONDITION_LABELS:
            filename = f"{p.preset_id}_{label}.wav"
            wav_path = audio_dir / filename

            rec = WavRecord(
                preset_id=p.preset_id,
                condition=label,
                attack_sec=p.attack_sec,
                decay_sec=p.decay_sec,
                release_sec=p.release_sec,
                note_off=p.timing.note_off,
                total_duration=p.timing.total_duration,
            )

            if not wav_path.exists():
                missing_files.append(filename)
                records.append(rec)
                continue

            if wav_path.stat().st_size == 0:
                empty_files.append(filename)
                rec.exists = True
                records.append(rec)
                continue

            rec.exists = True
            wav_dur, rms_db = get_wav_info(wav_path)
            rec.wav_duration = wav_dur
            rec.rms_db = rms_db
            all_wav_durations.append(wav_dur)

            if rms_db < SILENT_THRESHOLD_DB:
                rec.silent = True
                silent_files.append(filename)

            # Track consistency
            consistency[p.preset_id].add((p.timing.note_off, p.timing.total_duration))

            records.append(rec)

    # --- Print verification results ---
    total_expected = N_PRESETS * len(CONDITION_LABELS)
    n_existing = sum(1 for r in records if r.exists)
    n_non_empty = sum(1 for r in records if r.exists and r.wav_duration > 0)

    print(f"{'=' * 72}")
    print("验证结果")
    print(f"{'=' * 72}")

    # File existence
    print(f"\n  文件存在性: {n_existing}/{total_expected} 存在")
    if missing_files:
        print(f"  ⚠ 缺失文件 ({len(missing_files)}):")
        for f in missing_files[:10]:
            print(f"    - {f}")
    else:
        print("  ✓ 所有 60 个 WAV 文件存在")

    # Empty files
    if empty_files:
        print(f"  ⚠ 空文件 ({len(empty_files)}):")
        for f in empty_files[:10]:
            print(f"    - {f}")
    else:
        print("  ✓ 无空文件")

    # Consistency: same preset's 6 conditions use same note_off and total_duration
    inconsistent = [pid for pid, vals in consistency.items() if len(vals) > 1]
    if inconsistent:
        print(f"  ✗ 时序不一致的 preset ({len(inconsistent)}):")
        for pid in inconsistent:
            print(f"    - {pid}: {consistency[pid]}")
    else:
        print("  ✓ 同一 preset 的 6 个条件使用相同 note_off 和 total_duration")

    # Silent files
    if silent_files:
        print(f"  ⚠ 静音文件 (RMS < {SILENT_THRESHOLD_DB} dB): {len(silent_files)}")
        for f in silent_files[:10]:
            print(f"    - {f}")
    else:
        print("  ✓ 无静音文件")

    # WAV duration statistics
    if all_wav_durations:
        print(f"\n  WAV 时长统计:")
        print(f"    范围: [{min(all_wav_durations):.4f}, {max(all_wav_durations):.4f}] s")
        print(f"    均值: {np.mean(all_wav_durations):.4f} s")

    # RMS statistics
    valid_rms = [r.rms_db for r in records if r.exists and r.rms_db > float("-inf")]
    if valid_rms:
        print(f"\n  RMS 统计:")
        print(f"    范围: [{min(valid_rms):.2f}, {max(valid_rms):.2f}] dB")
        print(f"    均值: {np.mean(valid_rms):.2f} dB")

    # --- Write CSV report ---
    csv_path = output_dir / "batch_10_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "preset_id", "condition", "attack_sec", "decay_sec", "release_sec",
            "note_off", "total_duration", "wav_duration", "rms_db",
        ])
        for rec in records:
            writer.writerow([
                rec.preset_id,
                rec.condition,
                f"{rec.attack_sec:.6f}",
                f"{rec.decay_sec:.6f}",
                f"{rec.release_sec:.6f}",
                f"{rec.note_off:.6f}",
                f"{rec.total_duration:.6f}",
                f"{rec.wav_duration:.6f}" if rec.exists else "",
                f"{rec.rms_db:.2f}" if rec.rms_db > float("-inf") else "",
            ])

    print(f"\n  CSV 报告: {csv_path}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="10 preset 批量渲染测试与验证工具"
    )
    parser.add_argument(
        "--vst", type=Path, default=None,
        help="Vital VST3 插件路径（提供时渲染音频并执行完整验证）",
    )
    args = parser.parse_args()

    # 1. Generate 10 presets
    print("生成 10 个随机 preset (seed=12345)...")
    presets = generate_presets(OUTPUT_DIR)
    print(f"  生成完成: {len(presets)} 个 preset\n")

    # 2. Timing diagnostics (always runs)
    print_timing_diagnostics(presets)

    # 3. Render (optional, requires VST)
    if args.vst is not None:
        vst_path: Path = args.vst
        if not vst_path.exists():
            print(f"错误: VST3 路径不存在: {vst_path}", file=sys.stderr)
            sys.exit(1)

        render_all(presets, vst_path, OUTPUT_DIR)

        # 4. Verify and report
        verify_and_report(presets, OUTPUT_DIR)
    else:
        # Even without VST, write a timing-only CSV
        csv_path = OUTPUT_DIR / "batch_10_timing.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "preset_id", "attack_sec", "decay_sec", "release_sec",
                "note_off", "total_duration",
            ])
            for p in presets:
                writer.writerow([
                    p.preset_id,
                    f"{p.attack_sec:.6f}",
                    f"{p.decay_sec:.6f}",
                    f"{p.release_sec:.6f}",
                    f"{p.timing.note_off:.6f}",
                    f"{p.timing.total_duration:.6f}",
                ])
        print(f"  时序 CSV: {csv_path}")
        print("\n提示: 使用 --vst /path/to/Vital.vst3 渲染音频并执行完整验证")

    print("\n完成。")


if __name__ == "__main__":
    main()
