#!/usr/bin/env python3
"""
固定 2s vs 自适应渲染 A/B 对比脚本。

使用 SmartSampler (seed=99999) 生成 50 个 preset，分别用固定 2s 和自适应模式
计算时序参数并（可选）渲染音频，输出对比报告。

模式:
  - 时序对比（默认）: 比较固定 2s 与自适应时序参数，预测截断问题
  - 渲染对比（--vst）: 渲染两种模式音频，比较 RMS、静音率、过滤率
  - Embedding 对比（--embedding）: 比较两种模式的 embedding 余弦相似度

用法:
    # 仅时序对比（无需 VST）
    python research/adaptive_render/ab_compare.py

    # 渲染对比（需要 Vital VST3）
    python research/adaptive_render/ab_compare.py --vst /path/to/Vital.vst3

    # 渲染 + embedding 对比
    python research/adaptive_render/ab_compare.py --vst /path/to/Vital.vst3 --embedding
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.adaptive_timing import AdaptiveConfig, AdaptiveTiming, AdaptiveTimingCalculator
from src.preset_generator import PresetGenerator
from src.preset_parser import PresetParser
from src.smart_sampler import SmartSampler
from src.training_data import CORE_PARAMS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 99999
N_PRESETS = 50
SAMPLE_RATE = 44100
FIXED_DURATION = 2.0
FIXED_NOTE_OFF = FIXED_DURATION - 0.1  # 1.9s
SILENT_THRESHOLD_DB = -60.0
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "ab_compare"

# ADSR time params that need seconds→raw conversion for Vital JSON
_ADSR_TIME_PARAMS = {"env_1_attack", "env_1_decay", "env_1_release"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PresetTimingInfo:
    """Timing info for a single preset under both modes."""
    preset_id: str
    preset_path: Path
    param_vector: np.ndarray
    # ADSR in seconds (from CORE_PARAMS sampling space)
    attack_sec: float
    decay_sec: float
    sustain: float
    release_sec: float
    # Adaptive timing
    adaptive_note_off: float
    adaptive_duration: float
    # Fixed timing
    fixed_note_off: float = FIXED_NOTE_OFF
    fixed_duration: float = FIXED_DURATION
    # Truncation predictions
    attack_truncated: bool = False
    decay_truncated: bool = False
    release_truncated: bool = False


@dataclass
class RenderResult:
    """Render result for a single preset under one mode."""
    preset_id: str
    mode: str  # "fixed" or "adaptive"
    wav_path: Path | None = None
    wav_duration: float = 0.0
    rms_db: float = float("-inf")
    is_silent: bool = False


@dataclass
class ABComparisonReport:
    """Full A/B comparison report."""
    n_presets: int = 0
    # Timing comparison
    n_attack_truncated: int = 0
    n_decay_truncated: int = 0
    n_release_truncated: int = 0
    n_any_truncated: int = 0
    adaptive_duration_stats: dict = field(default_factory=dict)
    fixed_duration_stats: dict = field(default_factory=dict)
    # Render comparison (populated only with --vst)
    fixed_silent_count: int = 0
    adaptive_silent_count: int = 0
    fixed_filter_rate: float = 0.0
    adaptive_filter_rate: float = 0.0
    # Embedding comparison (populated only with --embedding)
    cosine_similarities: list[float] = field(default_factory=list)
    cosine_sim_stats: dict = field(default_factory=dict)
    # Conclusion
    conclusion: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wav_info(wav_path: Path) -> tuple[float, float]:
    """Read WAV file and return (duration_sec, rms_db)."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    duration_sec = n_frames / rate
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Preset generation (same 50 presets as mini production, seed=99999)
# ---------------------------------------------------------------------------

def generate_presets(output_dir: Path) -> list[PresetTimingInfo]:
    """Sample 50 parameter vectors and create .vital preset files.

    Returns list of PresetTimingInfo with timing computed for both modes.
    """
    presets_dir = output_dir / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)

    sampler = SmartSampler(seed=SEED)
    param_matrix = sampler.sample_lhs(N_PRESETS)  # (50, 45)

    parser = PresetParser()
    generator = PresetGenerator(parser=parser)
    calc = AdaptiveTimingCalculator()

    presets: list[PresetTimingInfo] = []

    for i in range(N_PRESETS):
        preset_id = f"ab_{i:03d}"
        param_vec = param_matrix[i]

        # Create preset from base patch, override with sampled params
        preset = generator.create_base_patch()
        for col, (name, _, _) in enumerate(CORE_PARAMS):
            value = float(param_vec[col])
            if name in _ADSR_TIME_PARAMS:
                value = AdaptiveTimingCalculator.seconds_to_power_law(value)
            preset.settings[name] = value
        preset.extra["preset_name"] = preset_id

        preset_path = presets_dir / f"{preset_id}.vital"
        parser.serialize(preset, preset_path)

        # Compute adaptive timing
        timing = calc.compute_timing(preset_path)

        # Get ADSR in seconds from CORE_PARAMS space
        attack_sec = timing.attack_sec
        decay_sec = timing.decay_sec
        release_sec = timing.release_sec

        # Predict truncation under fixed 2s mode
        attack_truncated = attack_sec > FIXED_NOTE_OFF
        decay_truncated = (attack_sec + decay_sec) > FIXED_NOTE_OFF
        release_truncated = release_sec > (FIXED_DURATION - FIXED_NOTE_OFF)

        presets.append(PresetTimingInfo(
            preset_id=preset_id,
            preset_path=preset_path,
            param_vector=param_vec,
            attack_sec=attack_sec,
            decay_sec=decay_sec,
            sustain=timing.sustain,
            release_sec=release_sec,
            adaptive_note_off=timing.note_off,
            adaptive_duration=timing.total_duration,
            attack_truncated=attack_truncated,
            decay_truncated=decay_truncated,
            release_truncated=release_truncated,
        ))

    return presets


# ---------------------------------------------------------------------------
# Timing comparison (always runs, no VST needed)
# ---------------------------------------------------------------------------

def run_timing_comparison(presets: list[PresetTimingInfo]) -> ABComparisonReport:
    """Compare fixed 2s vs adaptive timing parameters for all presets."""
    report = ABComparisonReport(n_presets=len(presets))

    adaptive_durations = [p.adaptive_duration for p in presets]
    adaptive_note_offs = [p.adaptive_note_off for p in presets]

    # Truncation analysis
    for p in presets:
        if p.attack_truncated:
            report.n_attack_truncated += 1
        if p.decay_truncated:
            report.n_decay_truncated += 1
        if p.release_truncated:
            report.n_release_truncated += 1
        if p.attack_truncated or p.decay_truncated or p.release_truncated:
            report.n_any_truncated += 1

    # Duration statistics
    report.adaptive_duration_stats = {
        "min": float(np.min(adaptive_durations)),
        "max": float(np.max(adaptive_durations)),
        "mean": float(np.mean(adaptive_durations)),
        "std": float(np.std(adaptive_durations)),
        "median": float(np.median(adaptive_durations)),
        "note_off_min": float(np.min(adaptive_note_offs)),
        "note_off_max": float(np.max(adaptive_note_offs)),
        "note_off_mean": float(np.mean(adaptive_note_offs)),
    }
    report.fixed_duration_stats = {
        "duration": FIXED_DURATION,
        "note_off": FIXED_NOTE_OFF,
    }

    return report


def print_timing_report(presets: list[PresetTimingInfo], report: ABComparisonReport) -> None:
    """Print timing comparison report to stdout."""
    print(f"\n{'=' * 76}")
    print("  固定 2s vs 自适应渲染 A/B 时序对比")
    print(f"  {N_PRESETS} presets, seed={SEED}")
    print(f"{'=' * 76}")

    # Per-preset table (show first 20 + any truncated ones)
    print(f"\n  {'ID':<10} {'A(s)':>7} {'D(s)':>7} {'R(s)':>7} "
          f"{'fix_noff':>9} {'adp_noff':>9} {'fix_dur':>8} {'adp_dur':>8} {'trunc':>6}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*6}")

    for p in presets:
        trunc_flags = ""
        if p.attack_truncated:
            trunc_flags += "A"
        if p.decay_truncated:
            trunc_flags += "D"
        if p.release_truncated:
            trunc_flags += "R"
        if not trunc_flags:
            trunc_flags = "-"

        print(f"  {p.preset_id:<10} {p.attack_sec:>7.3f} {p.decay_sec:>7.3f} "
              f"{p.release_sec:>7.3f} {FIXED_NOTE_OFF:>9.3f} "
              f"{p.adaptive_note_off:>9.3f} {FIXED_DURATION:>8.3f} "
              f"{p.adaptive_duration:>8.3f} {trunc_flags:>6}")

    # Summary
    print(f"\n  --- 截断预测（固定 2s 模式下的 ADSR 信息丢失）---")
    print(f"    Attack 被截断:  {report.n_attack_truncated}/{N_PRESETS} "
          f"({report.n_attack_truncated / N_PRESETS * 100:.1f}%)")
    print(f"    Decay 被截断:   {report.n_decay_truncated}/{N_PRESETS} "
          f"({report.n_decay_truncated / N_PRESETS * 100:.1f}%)")
    print(f"    Release 被截断: {report.n_release_truncated}/{N_PRESETS} "
          f"({report.n_release_truncated / N_PRESETS * 100:.1f}%)")
    print(f"    任一阶段截断:   {report.n_any_truncated}/{N_PRESETS} "
          f"({report.n_any_truncated / N_PRESETS * 100:.1f}%)")

    stats = report.adaptive_duration_stats
    print(f"\n  --- 自适应时序统计 ---")
    print(f"    duration 范围: [{stats['min']:.3f}, {stats['max']:.3f}] s")
    print(f"    duration 均值: {stats['mean']:.3f} s (std={stats['std']:.3f})")
    print(f"    duration 中位数: {stats['median']:.3f} s")
    print(f"    note_off 范围: [{stats['note_off_min']:.3f}, {stats['note_off_max']:.3f}] s")
    print(f"    note_off 均值: {stats['note_off_mean']:.3f} s")
    print()


# ---------------------------------------------------------------------------
# Rendering (requires --vst)
# ---------------------------------------------------------------------------

def render_both_modes(
    presets: list[PresetTimingInfo],
    vst_path: Path,
    output_dir: Path,
) -> tuple[list[RenderResult], list[RenderResult]]:
    """Render all presets in both fixed and adaptive modes.

    Uses a single MIDI condition (C4, velocity=100) for comparison.

    Returns:
        (fixed_results, adaptive_results) lists.
    """
    from src.audio_renderer import AudioRenderer, RenderConfig

    fixed_dir = output_dir / "audio_fixed"
    adaptive_dir = output_dir / "audio_adaptive"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    fixed_results: list[RenderResult] = []
    adaptive_results: list[RenderResult] = []

    calc = AdaptiveTimingCalculator()

    for p in presets:
        # --- Fixed mode ---
        config_fixed = RenderConfig()
        config_fixed.duration_sec = FIXED_DURATION
        config_fixed.midi_note = 60  # C4
        config_fixed.velocity = 100
        renderer_fixed = AudioRenderer(vital_vst_path=vst_path, config=config_fixed)

        fixed_wav = fixed_dir / f"{p.preset_id}_fixed.wav"
        success_fixed = renderer_fixed.render_preset(p.preset_path, fixed_wav)

        fr = RenderResult(preset_id=p.preset_id, mode="fixed")
        if success_fixed and fixed_wav.exists() and fixed_wav.stat().st_size > 0:
            fr.wav_path = fixed_wav
            fr.wav_duration, fr.rms_db = get_wav_info(fixed_wav)
            fr.is_silent = fr.rms_db < SILENT_THRESHOLD_DB
        fixed_results.append(fr)

        # --- Adaptive mode ---
        timing = calc.compute_timing(p.preset_path)
        config_adaptive = RenderConfig()
        config_adaptive.duration_sec = timing.total_duration
        config_adaptive.midi_note = 60
        config_adaptive.velocity = 100
        renderer_adaptive = AudioRenderer(vital_vst_path=vst_path, config=config_adaptive)

        adaptive_wav = adaptive_dir / f"{p.preset_id}_adaptive.wav"
        success_adaptive = renderer_adaptive.render_preset(
            p.preset_path, adaptive_wav, note_off_time=timing.note_off,
        )

        ar = RenderResult(preset_id=p.preset_id, mode="adaptive")
        if success_adaptive and adaptive_wav.exists() and adaptive_wav.stat().st_size > 0:
            ar.wav_path = adaptive_wav
            ar.wav_duration, ar.rms_db = get_wav_info(adaptive_wav)
            ar.is_silent = ar.rms_db < SILENT_THRESHOLD_DB
        adaptive_results.append(ar)

        status = (
            f"  {p.preset_id}: fixed={'✓' if success_fixed else '✗'} "
            f"adaptive={'✓' if success_adaptive else '✗'}"
        )
        print(status)

    return fixed_results, adaptive_results


def analyze_render_results(
    report: ABComparisonReport,
    fixed_results: list[RenderResult],
    adaptive_results: list[RenderResult],
) -> None:
    """Populate report with render comparison statistics."""
    report.fixed_silent_count = sum(1 for r in fixed_results if r.is_silent)
    report.adaptive_silent_count = sum(1 for r in adaptive_results if r.is_silent)

    n_fixed_valid = sum(1 for r in fixed_results if r.wav_path is not None)
    n_adaptive_valid = sum(1 for r in adaptive_results if r.wav_path is not None)

    report.fixed_filter_rate = (
        report.fixed_silent_count / n_fixed_valid * 100 if n_fixed_valid > 0 else 0.0
    )
    report.adaptive_filter_rate = (
        report.adaptive_silent_count / n_adaptive_valid * 100 if n_adaptive_valid > 0 else 0.0
    )


def print_render_report(
    report: ABComparisonReport,
    fixed_results: list[RenderResult],
    adaptive_results: list[RenderResult],
) -> None:
    """Print render comparison report."""
    print(f"\n  --- 渲染对比结果 ---")

    # Per-preset RMS comparison
    print(f"\n  {'ID':<10} {'fix_rms':>10} {'adp_rms':>10} {'fix_silent':>11} {'adp_silent':>11}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*11} {'-'*11}")

    for fr, ar in zip(fixed_results, adaptive_results):
        fix_rms_str = f"{fr.rms_db:.2f}" if fr.rms_db > float("-inf") else "N/A"
        adp_rms_str = f"{ar.rms_db:.2f}" if ar.rms_db > float("-inf") else "N/A"
        print(f"  {fr.preset_id:<10} {fix_rms_str:>10} {adp_rms_str:>10} "
              f"{'YES' if fr.is_silent else 'no':>11} "
              f"{'YES' if ar.is_silent else 'no':>11}")

    # Summary
    print(f"\n  静音样本数:")
    print(f"    固定 2s:  {report.fixed_silent_count}/{N_PRESETS} "
          f"({report.fixed_filter_rate:.1f}%)")
    print(f"    自适应:   {report.adaptive_silent_count}/{N_PRESETS} "
          f"({report.adaptive_filter_rate:.1f}%)")

    diff = report.fixed_silent_count - report.adaptive_silent_count
    if diff > 0:
        print(f"    ✓ 自适应模式减少了 {diff} 个静音样本")
    elif diff == 0:
        print(f"    → 两种模式静音样本数相同")
    else:
        print(f"    ⚠ 自适应模式多了 {abs(diff)} 个静音样本")

    # Verify: adaptive silence rate <= fixed silence rate
    if report.adaptive_filter_rate <= report.fixed_filter_rate:
        print(f"    ✓ 验证通过: 自适应静音过滤率 ({report.adaptive_filter_rate:.1f}%) "
              f"<= 固定模式 ({report.fixed_filter_rate:.1f}%)")
    else:
        print(f"    ⚠ 验证失败: 自适应静音过滤率 ({report.adaptive_filter_rate:.1f}%) "
              f"> 固定模式 ({report.fixed_filter_rate:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Embedding comparison (requires --embedding)
# ---------------------------------------------------------------------------

def compute_embedding_similarities(
    report: ABComparisonReport,
    fixed_results: list[RenderResult],
    adaptive_results: list[RenderResult],
) -> None:
    """Compute cosine similarity between fixed and adaptive embeddings.

    Requires the embedding extractor (MuQ model).
    """
    from src.embedding_extractor import EmbeddingExtractor

    extractor = EmbeddingExtractor()
    similarities: list[float] = []

    for fr, ar in zip(fixed_results, adaptive_results):
        if fr.wav_path is None or ar.wav_path is None:
            continue
        if fr.is_silent or ar.is_silent:
            continue

        try:
            emb_fixed = extractor.extract(fr.wav_path)
            emb_adaptive = extractor.extract(ar.wav_path)
            sim = cosine_similarity(emb_fixed, emb_adaptive)
            similarities.append(sim)
        except Exception as e:
            print(f"  ⚠ Embedding 提取失败 ({fr.preset_id}): {e}")

    report.cosine_similarities = similarities
    if similarities:
        sims = np.array(similarities)
        report.cosine_sim_stats = {
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "median": float(np.median(sims)),
            "n_compared": len(similarities),
            "n_low_similarity": int(np.sum(sims < 0.9)),
            "n_very_different": int(np.sum(sims < 0.8)),
        }


def print_embedding_report(report: ABComparisonReport) -> None:
    """Print embedding comparison report."""
    print(f"\n  --- Embedding 余弦相似度对比 ---")

    if not report.cosine_similarities:
        print(f"    无有效 embedding 对比数据")
        return

    stats = report.cosine_sim_stats
    print(f"    对比样本数: {stats['n_compared']}")
    print(f"    相似度范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"    相似度均值: {stats['mean']:.4f} (std={stats['std']:.4f})")
    print(f"    相似度中位数: {stats['median']:.4f}")
    print(f"    低相似度 (<0.9): {stats['n_low_similarity']}")
    print(f"    差异显著 (<0.8): {stats['n_very_different']}")

    if stats["mean"] < 0.95:
        print(f"    → 两种模式的 embedding 存在显著差异，"
              f"自适应模式捕获了不同的音频特征")
    else:
        print(f"    → 两种模式的 embedding 高度相似，"
              f"ADSR 差异对 embedding 影响较小")
    print()


# ---------------------------------------------------------------------------
# Analysis & conclusion (Task 13.2)
# ---------------------------------------------------------------------------

def generate_conclusion(
    report: ABComparisonReport,
    has_render: bool,
    has_embedding: bool,
) -> str:
    """Generate conclusive summary of the A/B comparison."""
    lines: list[str] = []

    # Timing-based analysis (always available)
    pct_truncated = report.n_any_truncated / report.n_presets * 100
    lines.append(f"时序分析: {report.n_any_truncated}/{report.n_presets} "
                 f"({pct_truncated:.0f}%) 的 preset 在固定 2s 模式下存在 ADSR 截断。")

    if report.n_release_truncated > 0:
        lines.append(f"  - {report.n_release_truncated} 个 preset 的 release 阶段被截断"
                     f"（固定模式仅有 0.1s release 窗口）。")
    if report.n_attack_truncated > 0:
        lines.append(f"  - {report.n_attack_truncated} 个 preset 的 attack 阶段被截断。")

    stats = report.adaptive_duration_stats
    lines.append(f"自适应模式渲染时长: {stats['mean']:.2f}s 均值 "
                 f"(范围 [{stats['min']:.2f}, {stats['max']:.2f}]s)。")

    # Render-based analysis
    if has_render:
        lines.append("")
        diff = report.fixed_silent_count - report.adaptive_silent_count
        if diff > 0:
            lines.append(f"渲染验证: 自适应模式减少了 {diff} 个静音样本 "
                         f"(固定={report.fixed_silent_count}, "
                         f"自适应={report.adaptive_silent_count})。")
        elif diff == 0:
            lines.append(f"渲染验证: 两种模式静音样本数相同 "
                         f"({report.fixed_silent_count})。")
        else:
            lines.append(f"渲染验证: ⚠ 自适应模式静音样本反而增加了 {abs(diff)} 个。")

        if report.adaptive_filter_rate <= report.fixed_filter_rate:
            lines.append(f"✓ 自适应静音过滤率 ({report.adaptive_filter_rate:.1f}%) "
                         f"<= 固定模式 ({report.fixed_filter_rate:.1f}%)。")
        else:
            lines.append(f"⚠ 自适应静音过滤率 ({report.adaptive_filter_rate:.1f}%) "
                         f"> 固定模式 ({report.fixed_filter_rate:.1f}%)。")

    # Embedding-based analysis
    if has_embedding and report.cosine_similarities:
        lines.append("")
        stats_e = report.cosine_sim_stats
        lines.append(f"Embedding 对比: 均值余弦相似度 = {stats_e['mean']:.4f}, "
                     f"{stats_e['n_low_similarity']} 个样本相似度 < 0.9。")
        if stats_e["mean"] < 0.95:
            lines.append("→ 自适应模式捕获了更多 ADSR 包络信息，"
                         "embedding 表征存在有意义的差异。")
        else:
            lines.append("→ 两种模式的 embedding 高度相似，"
                         "ADSR 差异主要体现在时域而非频域特征。")

    # Overall conclusion
    lines.append("")
    if pct_truncated > 20:
        lines.append("结论: 自适应渲染模式显著改善了 ADSR 包络的信息保真度。"
                     f"固定 2s 模式下 {pct_truncated:.0f}% 的 preset 存在截断，"
                     "自适应模式通过动态调整渲染时长避免了这些信息丢失。")
    elif pct_truncated > 5:
        lines.append("结论: 自适应渲染模式对部分 preset 有改善效果。"
                     f"固定 2s 模式下 {pct_truncated:.0f}% 的 preset 存在截断。")
    else:
        lines.append("结论: 在当前采样分布下，固定 2s 模式的截断问题不严重，"
                     "但自适应模式仍能为极端 ADSR 参数提供更好的覆盖。")

    conclusion = "\n".join(lines)
    report.conclusion = conclusion
    return conclusion


# ---------------------------------------------------------------------------
# Output: CSV + JSON report
# ---------------------------------------------------------------------------

def save_reports(
    presets: list[PresetTimingInfo],
    report: ABComparisonReport,
    output_dir: Path,
    fixed_results: list[RenderResult] | None = None,
    adaptive_results: list[RenderResult] | None = None,
) -> None:
    """Save CSV and JSON reports to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV: per-preset timing comparison ---
    csv_path = output_dir / "ab_timing_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "preset_id", "attack_sec", "decay_sec", "sustain", "release_sec",
            "fixed_note_off", "adaptive_note_off",
            "fixed_duration", "adaptive_duration",
            "attack_truncated", "decay_truncated", "release_truncated",
        ]
        if fixed_results is not None:
            header.extend(["fixed_rms_db", "fixed_silent",
                           "adaptive_rms_db", "adaptive_silent"])
        writer.writerow(header)

        for i, p in enumerate(presets):
            row = [
                p.preset_id,
                f"{p.attack_sec:.6f}",
                f"{p.decay_sec:.6f}",
                f"{p.sustain:.6f}",
                f"{p.release_sec:.6f}",
                f"{FIXED_NOTE_OFF:.4f}",
                f"{p.adaptive_note_off:.4f}",
                f"{FIXED_DURATION:.4f}",
                f"{p.adaptive_duration:.4f}",
                str(p.attack_truncated),
                str(p.decay_truncated),
                str(p.release_truncated),
            ]
            if fixed_results is not None and adaptive_results is not None:
                fr = fixed_results[i]
                ar = adaptive_results[i]
                row.extend([
                    f"{fr.rms_db:.2f}" if fr.rms_db > float("-inf") else "",
                    str(fr.is_silent),
                    f"{ar.rms_db:.2f}" if ar.rms_db > float("-inf") else "",
                    str(ar.is_silent),
                ])
            writer.writerow(row)

    print(f"  CSV 报告: {csv_path}")

    # --- JSON: full report ---
    json_path = output_dir / "ab_comparison_report.json"
    report_dict = {
        "seed": SEED,
        "n_presets": report.n_presets,
        "fixed_mode": {
            "duration": FIXED_DURATION,
            "note_off": FIXED_NOTE_OFF,
        },
        "truncation_analysis": {
            "n_attack_truncated": report.n_attack_truncated,
            "n_decay_truncated": report.n_decay_truncated,
            "n_release_truncated": report.n_release_truncated,
            "n_any_truncated": report.n_any_truncated,
            "pct_any_truncated": report.n_any_truncated / report.n_presets * 100,
        },
        "adaptive_duration_stats": report.adaptive_duration_stats,
    }

    if fixed_results is not None:
        report_dict["render_comparison"] = {
            "fixed_silent_count": report.fixed_silent_count,
            "adaptive_silent_count": report.adaptive_silent_count,
            "fixed_filter_rate_pct": report.fixed_filter_rate,
            "adaptive_filter_rate_pct": report.adaptive_filter_rate,
            "silence_reduction": report.fixed_silent_count - report.adaptive_silent_count,
        }

    if report.cosine_similarities:
        report_dict["embedding_comparison"] = report.cosine_sim_stats

    report_dict["conclusion"] = report.conclusion

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"  JSON 报告: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="固定 2s vs 自适应渲染 A/B 对比工具"
    )
    parser.add_argument(
        "--vst", type=Path, default=None,
        help="Vital VST3 插件路径（提供时渲染音频并对比 RMS/静音率）",
    )
    parser.add_argument(
        "--embedding", action="store_true",
        help="计算 embedding 余弦相似度（需要 --vst 和 embedding 模型）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"输出目录（默认: {OUTPUT_DIR}）",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    if args.embedding and args.vst is None:
        print("错误: --embedding 需要同时指定 --vst", file=sys.stderr)
        sys.exit(1)

    if args.vst is not None and not args.vst.exists():
        print(f"错误: VST3 路径不存在: {args.vst}", file=sys.stderr)
        sys.exit(1)

    # 1. Generate 50 presets (same seed as mini production)
    print(f"生成 {N_PRESETS} 个 preset (seed={SEED})...")
    presets = generate_presets(output_dir)
    print(f"  生成完成: {len(presets)} 个 preset\n")

    # 2. Timing comparison (always runs)
    report = run_timing_comparison(presets)
    print_timing_report(presets, report)

    # 3. Render comparison (optional)
    fixed_results: list[RenderResult] | None = None
    adaptive_results: list[RenderResult] | None = None

    if args.vst is not None:
        print("渲染两种模式的音频...")
        fixed_results, adaptive_results = render_both_modes(
            presets, args.vst, output_dir,
        )
        analyze_render_results(report, fixed_results, adaptive_results)
        print_render_report(report, fixed_results, adaptive_results)

    # 4. Embedding comparison (optional)
    if args.embedding and fixed_results is not None and adaptive_results is not None:
        print("计算 embedding 余弦相似度...")
        compute_embedding_similarities(report, fixed_results, adaptive_results)
        print_embedding_report(report)

    # 5. Generate conclusion (Task 13.2)
    conclusion = generate_conclusion(
        report,
        has_render=fixed_results is not None,
        has_embedding=bool(report.cosine_similarities),
    )
    print(f"\n{'=' * 76}")
    print("  结论")
    print(f"{'=' * 76}")
    print(f"\n{conclusion}")
    print(f"\n{'=' * 76}")

    # 6. Save reports
    save_reports(presets, report, output_dir, fixed_results, adaptive_results)

    if args.vst is None:
        print("\n提示: 使用 --vst /path/to/Vital.vst3 渲染音频并执行完整对比")
        print("      使用 --vst ... --embedding 同时对比 embedding 差异")

    print("\n完成。")


if __name__ == "__main__":
    main()
