#!/usr/bin/env python3
"""
50 preset 端到端自适应渲染流水线测试脚本。

调用 ParallelProducer 执行完整生产流程：
  参数采样 → preset 生成 → 自适应渲染 → 音频过滤 → embedding 提取 → 质量验证

生产完成后自动执行诊断验证：
  - checkpoint 自适应时序参数记录
  - WAV 文件时长分布（应不再全是 2.0s）
  - 过滤率统计（与固定 2s 模式历史对比）
  - embedding 提取成功率
  - production_summary.json 完整性

用法:
    python research/adaptive_render/run_mini_production.py --vst /path/to/Vital.vst3
    python research/adaptive_render/run_mini_production.py --vst /path/to/Vital.vst3 --config configs/test_adaptive_50.yaml
    python research/adaptive_render/run_mini_production.py --vst /path/to/Vital.vst3 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import wave
from pathlib import Path

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/test_adaptive_50.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "mini_production"


# ---------------------------------------------------------------------------
# Config loading (reuses pattern from scripts/run_production.py)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base_config if specified."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_path = config.get("base_config")
    if base_path:
        base_full = Path(config_path).parent.parent / base_path
        if not base_full.exists():
            base_full = _PROJECT_ROOT / base_path
        if base_full.exists():
            with open(base_full, "r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
            merged = {**base, **config}
            merged.pop("base_config", None)
            return merged
    return config


def build_production_config(config: dict):
    """Build ProductionConfig from YAML dict."""
    from src.parallel_producer import ProductionConfig

    prod = config.get("production", {})
    parallel = config.get("parallel", {})
    sampling = config.get("sampling", {})
    ar = config.get("adaptive_rendering", {})

    return ProductionConfig(
        target_samples=prod.get("target_samples", 300),
        n_workers=parallel.get("n_workers", 4),
        embedding_batch_size=parallel.get("embedding_batch_size", 32),
        embedding_device=parallel.get("embedding_device", "mps"),
        checkpoint_interval=parallel.get("checkpoint_interval", 50),
        sampling_strategy=sampling.get("strategy", "lhs_stratified"),
        seed=sampling.get("seed", 99999),
        filter_margin=prod.get("filter_margin", 0.02),
        n_conditions=len(config.get("multi_condition", {}).get("conditions", [])) or 6,
        adaptive_rendering=ar.get("enabled", True),
        sustain_margin=ar.get("sustain_margin", 0.2),
        tail_margin=ar.get("tail_margin", 0.1),
        target_length_sec=ar.get("target_length_sec"),
        max_duration_sec=ar.get("max_duration_sec", 30.0),
    )


# ---------------------------------------------------------------------------
# Production execution
# ---------------------------------------------------------------------------

def run_production(vst_path: Path, config_path: str, output_dir: Path,
                   resume: bool = False) -> dict | None:
    """Run the full production pipeline via ParallelProducer.

    Returns:
        Production summary dict, or None on failure.
    """
    from src.audio_preprocessor import AudioPreprocessor, PreprocessConfig
    from src.checkpoint_manager import CheckpointManager
    from src.distribution_analyzer import DistributionAnalyzer
    from src.parallel_producer import ParallelProducer
    from src.quality_validator import QualityValidator
    from src.smart_sampler import SmartSampler

    config = load_config(config_path)
    prod_config = build_production_config(config)

    logger.info(
        "生产配置: target=%d, workers=%d, adaptive=%s, seed=%d",
        prod_config.target_samples,
        prod_config.n_workers,
        prod_config.adaptive_rendering,
        prod_config.seed,
    )

    # Preprocessing config
    pp_cfg = config.get("preprocessing", {})
    preprocessor = AudioPreprocessor(PreprocessConfig(
        silence_threshold_db=pp_cfg.get("silence_threshold_db", -60.0),
        clipping_threshold=pp_cfg.get("clipping_threshold", 0.99),
        clipping_ratio_limit=pp_cfg.get("clipping_ratio_limit", 0.10),
        target_peak_db=pp_cfg.get("target_peak_db", -1.0),
        target_sample_rate=pp_cfg.get("target_sample_rate", 16000),
        tail_silence_threshold_db=pp_cfg.get("tail_silence_threshold_db", -50.0),
        min_duration_sec=pp_cfg.get("min_duration_sec", 0.5),
    ))

    sampler = SmartSampler(seed=prod_config.seed)

    q_cfg = config.get("quality", {})
    validator = QualityValidator(
        silence_threshold_db=pp_cfg.get("silence_threshold_db", -60.0),
        clipping_ratio_limit=pp_cfg.get("clipping_ratio_limit", 0.10),
        spectral_entropy_threshold=q_cfg.get("spectral_entropy_threshold", 0.05),
        near_duplicate_threshold=q_cfg.get("near_duplicate_threshold", 0.999),
        pca_collapse_threshold=q_cfg.get("pca_collapse_threshold", 0.95),
    )

    analyzer = DistributionAnalyzer(
        diversity_threshold=config.get("distribution", {}).get("diversity_threshold", 0.95),
    )

    checkpoint_manager = CheckpointManager(output_dir)

    producer = ParallelProducer(
        vital_vst_path=vst_path,
        config=prod_config,
        preprocessor=preprocessor,
        sampler=sampler,
        validator=validator,
        analyzer=analyzer,
        checkpoint_manager=checkpoint_manager,
    )

    # Resource estimate
    n_presets = producer.n_presets
    n_conditions = prod_config.n_conditions
    estimate = producer.estimate_resources(n_presets, n_conditions)
    print(f"\n  预设数量: {estimate['n_presets']:,}")
    print(f"  总样本数: {estimate['n_samples']:,}")
    print(f"  预计渲染时间: {estimate['render_time_sec']:.0f} 秒")

    # Run
    summary = producer.produce(output_dir, resume=resume)
    return {
        "total_presets": summary.total_presets,
        "total_samples": summary.total_samples,
        "valid_samples": summary.valid_samples,
        "filtered_samples": summary.filtered_samples,
        "failed_samples": summary.failed_samples,
        "total_time_sec": summary.total_time_sec,
        "filter_reasons": dict(summary.filter_reasons) if summary.filter_reasons else {},
    }


# ---------------------------------------------------------------------------
# Post-production verification (Task 12.3)
# ---------------------------------------------------------------------------

def get_wav_duration(wav_path: Path) -> float | None:
    """Read WAV file duration in seconds. Returns None on error."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def verify_checkpoint_adaptive_timing(output_dir: Path) -> dict:
    """Verify checkpoint records adaptive timing params per preset.

    Returns:
        Dict with verification results.
    """
    checkpoint_path = output_dir / "checkpoint.json"
    result = {
        "checkpoint_exists": checkpoint_path.exists(),
        "has_adaptive_timing": False,
        "presets_with_timing": 0,
        "presets_without_timing": 0,
        "sample_timings": [],
    }

    if not checkpoint_path.exists():
        return result

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    # Check if any sample records contain adaptive timing info
    # The checkpoint stores sample statuses; adaptive timing is per-preset
    # and recorded in the render results
    statuses = checkpoint.get("statuses", checkpoint.get("samples", []))
    if isinstance(statuses, list):
        for s in statuses[:5]:
            if isinstance(s, dict):
                result["sample_timings"].append({
                    k: v for k, v in s.items()
                    if k in ("sample_id", "status", "note_off", "total_duration",
                             "adaptive_timing")
                })
        result["has_adaptive_timing"] = any(
            isinstance(s, dict) and (
                "note_off" in s or "total_duration" in s or "adaptive_timing" in s
            )
            for s in statuses
        )

    return result


def verify_wav_duration_distribution(output_dir: Path) -> dict:
    """Verify WAV file duration distribution (should not all be 2.0s).

    Returns:
        Dict with duration statistics.
    """
    audio_dir = output_dir / "audio"
    if not audio_dir.exists():
        return {"error": "audio directory not found", "n_files": 0}

    wav_files = sorted(audio_dir.glob("*.wav"))
    durations: list[float] = []
    for wf in wav_files:
        dur = get_wav_duration(wf)
        if dur is not None:
            durations.append(dur)

    if not durations:
        return {"n_files": 0, "error": "no valid WAV files"}

    durations_arr = np.array(durations)
    # Count how many are exactly 2.0s (within tolerance)
    n_fixed_2s = int(np.sum(np.abs(durations_arr - 2.0) < 0.01))
    unique_durations = len(set(round(d, 3) for d in durations))

    # Duration histogram bins
    bins = [0, 1, 2, 3, 5, 10, 15, 20, 30, 60]
    hist, _ = np.histogram(durations_arr, bins=bins)
    histogram = {f"{bins[i]}-{bins[i+1]}s": int(hist[i]) for i in range(len(hist))}

    return {
        "n_files": len(durations),
        "min_duration": float(np.min(durations_arr)),
        "max_duration": float(np.max(durations_arr)),
        "mean_duration": float(np.mean(durations_arr)),
        "std_duration": float(np.std(durations_arr)),
        "n_fixed_2s": n_fixed_2s,
        "pct_fixed_2s": n_fixed_2s / len(durations) * 100,
        "unique_durations": unique_durations,
        "all_same_duration": unique_durations <= 1,
        "histogram": histogram,
    }


def verify_filter_rate(production_summary: dict | None) -> dict:
    """Compare filter rate with fixed 2s mode historical rate.

    Historical reference: fixed 2s mode typically has ~5-15% filter rate
    due to silence from long-attack presets.
    """
    if production_summary is None:
        return {"error": "no production summary available"}

    total = production_summary.get("total_samples", 0)
    filtered = production_summary.get("filtered_samples", 0)
    valid = production_summary.get("valid_samples", 0)
    failed = production_summary.get("failed_samples", 0)
    reasons = production_summary.get("filter_reasons", {})

    filter_rate = filtered / total * 100 if total > 0 else 0.0
    silence_filtered = reasons.get("silence", 0) + reasons.get("silent", 0)

    return {
        "total_samples": total,
        "valid_samples": valid,
        "filtered_samples": filtered,
        "failed_samples": failed,
        "filter_rate_pct": filter_rate,
        "silence_filtered": silence_filtered,
        "filter_reasons": reasons,
        "historical_fixed_2s_filter_rate": "~5-15% (reference)",
        "adaptive_improvement": (
            "预期静音过滤率下降（长 attack preset 不再被误判为静音）"
            if filter_rate < 15 else
            "过滤率偏高，需进一步分析"
        ),
    }


def verify_embeddings(output_dir: Path) -> dict:
    """Verify embedding extraction success."""
    # Check for HDF5 or embedding files
    hdf5_files = list(output_dir.glob("*.hdf5")) + list(output_dir.glob("*.h5"))
    embedding_dir = output_dir / "embeddings"

    result = {
        "hdf5_files": [f.name for f in hdf5_files],
        "embedding_dir_exists": embedding_dir.exists(),
        "n_embedding_files": 0,
    }

    if embedding_dir.exists():
        emb_files = list(embedding_dir.glob("*.npy")) + list(embedding_dir.glob("*.npz"))
        result["n_embedding_files"] = len(emb_files)

    # Check checkpoint for embedding status
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        statuses = checkpoint.get("statuses", checkpoint.get("samples", []))
        if isinstance(statuses, list):
            n_with_embedding = sum(
                1 for s in statuses
                if isinstance(s, dict) and s.get("status") in ("embedded", "validated", "saved")
            )
            n_rendered = sum(
                1 for s in statuses
                if isinstance(s, dict) and s.get("status") in (
                    "rendered", "preprocessed", "embedded", "validated", "saved",
                    "render_passed",
                )
            )
            result["n_with_embedding"] = n_with_embedding
            result["n_rendered"] = n_rendered
            if n_rendered > 0:
                result["embedding_success_rate_pct"] = n_with_embedding / n_rendered * 100

    return result


def verify_production_summary(output_dir: Path) -> dict:
    """Verify production_summary.json completeness."""
    summary_path = output_dir / "production_summary.json"
    result = {
        "summary_exists": summary_path.exists(),
        "fields_present": [],
        "fields_missing": [],
    }

    expected_fields = [
        "total_presets", "total_samples", "valid_samples",
        "filtered_samples", "failed_samples", "total_time_sec",
    ]

    if not summary_path.exists():
        result["fields_missing"] = expected_fields
        return result

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    for field in expected_fields:
        if field in summary:
            result["fields_present"].append(field)
        else:
            result["fields_missing"].append(field)

    result["summary_data"] = {k: summary.get(k) for k in expected_fields}
    result["complete"] = len(result["fields_missing"]) == 0

    return result


def run_verification(output_dir: Path, production_summary: dict | None) -> dict:
    """Run all post-production verifications and return combined report."""
    print(f"\n{'=' * 72}")
    print("端到端流水线诊断报告")
    print(f"{'=' * 72}")

    report = {}

    # 1. Checkpoint adaptive timing
    print("\n--- 1. Checkpoint 自适应时序参数 ---")
    ckpt_result = verify_checkpoint_adaptive_timing(output_dir)
    report["checkpoint_timing"] = ckpt_result
    if ckpt_result["checkpoint_exists"]:
        print(f"  ✓ checkpoint.json 存在")
        if ckpt_result["has_adaptive_timing"]:
            print(f"  ✓ 包含自适应时序参数")
        else:
            print(f"  ⚠ 未检测到自适应时序参数（可能存储在其他位置）")
        if ckpt_result["sample_timings"]:
            print(f"  样本示例 (前 {len(ckpt_result['sample_timings'])} 条):")
            for t in ckpt_result["sample_timings"]:
                print(f"    {t}")
    else:
        print(f"  ✗ checkpoint.json 不存在")

    # 2. WAV duration distribution
    print("\n--- 2. WAV 文件时长分布 ---")
    dur_result = verify_wav_duration_distribution(output_dir)
    report["duration_distribution"] = dur_result
    if dur_result.get("n_files", 0) > 0:
        print(f"  WAV 文件数: {dur_result['n_files']}")
        print(f"  时长范围: [{dur_result['min_duration']:.3f}, {dur_result['max_duration']:.3f}] s")
        print(f"  均值: {dur_result['mean_duration']:.3f} s, 标准差: {dur_result['std_duration']:.3f} s")
        print(f"  固定 2.0s 文件数: {dur_result['n_fixed_2s']} ({dur_result['pct_fixed_2s']:.1f}%)")
        print(f"  唯一时长数: {dur_result['unique_durations']}")
        if dur_result["all_same_duration"]:
            print(f"  ✗ 所有文件时长相同 — 自适应渲染可能未生效！")
        else:
            print(f"  ✓ 时长分布多样化 — 自适应渲染生效")
        print(f"  时长直方图:")
        for bucket, count in dur_result.get("histogram", {}).items():
            bar = "█" * count
            print(f"    {bucket:>8}: {count:>4} {bar}")
    else:
        print(f"  ✗ 无有效 WAV 文件")

    # 3. Filter rate
    print("\n--- 3. 过滤率统计 ---")
    filter_result = verify_filter_rate(production_summary)
    report["filter_rate"] = filter_result
    if "error" not in filter_result:
        print(f"  总样本: {filter_result['total_samples']}")
        print(f"  有效样本: {filter_result['valid_samples']}")
        print(f"  过滤样本: {filter_result['filtered_samples']}")
        print(f"  失败样本: {filter_result['failed_samples']}")
        print(f"  过滤率: {filter_result['filter_rate_pct']:.1f}%")
        print(f"  静音过滤: {filter_result['silence_filtered']}")
        print(f"  历史固定 2s 过滤率: {filter_result['historical_fixed_2s_filter_rate']}")
        print(f"  评估: {filter_result['adaptive_improvement']}")
        if filter_result.get("filter_reasons"):
            print(f"  过滤原因明细:")
            for reason, count in filter_result["filter_reasons"].items():
                print(f"    {reason}: {count}")
    else:
        print(f"  ⚠ {filter_result['error']}")

    # 4. Embedding extraction
    print("\n--- 4. Embedding 提取验证 ---")
    emb_result = verify_embeddings(output_dir)
    report["embeddings"] = emb_result
    if emb_result.get("n_with_embedding", 0) > 0:
        print(f"  已渲染样本: {emb_result.get('n_rendered', '?')}")
        print(f"  已提取 embedding: {emb_result['n_with_embedding']}")
        rate = emb_result.get("embedding_success_rate_pct", 0)
        print(f"  提取成功率: {rate:.1f}%")
        if rate >= 95:
            print(f"  ✓ embedding 提取成功率良好")
        else:
            print(f"  ⚠ embedding 提取成功率偏低")
    else:
        print(f"  HDF5 文件: {emb_result['hdf5_files']}")
        print(f"  Embedding 目录: {'存在' if emb_result['embedding_dir_exists'] else '不存在'}")
        print(f"  ⚠ 未检测到 embedding 状态记录")

    # 5. Production summary
    print("\n--- 5. production_summary.json 完整性 ---")
    summary_result = verify_production_summary(output_dir)
    report["production_summary"] = summary_result
    if summary_result["summary_exists"]:
        print(f"  ✓ production_summary.json 存在")
        if summary_result.get("complete"):
            print(f"  ✓ 所有必需字段完整")
        else:
            print(f"  ⚠ 缺失字段: {summary_result['fields_missing']}")
        if summary_result.get("summary_data"):
            for k, v in summary_result["summary_data"].items():
                print(f"    {k}: {v}")
    else:
        print(f"  ⚠ production_summary.json 不存在（可能尚未生成）")

    # Save report
    report_path = output_dir / "diagnostic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  诊断报告已保存: {report_path}")

    print(f"\n{'=' * 72}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "mini_production.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="50 preset 端到端自适应渲染流水线测试"
    )
    parser.add_argument(
        "--vst", type=Path, required=True,
        help="Vital VST3 插件路径",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help=f"配置文件路径（默认: {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"输出目录（默认: {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从断点恢复生产",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="仅运行验证（跳过生产，对已有输出执行诊断）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    setup_logging(output_dir)

    if not args.verify_only:
        # Validate VST path
        if not args.vst.exists():
            print(f"错误: VST3 路径不存在: {args.vst}", file=sys.stderr)
            return 1

        print(f"\n{'=' * 72}")
        print("  50 preset 端到端自适应渲染流水线测试")
        print(f"{'=' * 72}")
        print(f"  配置: {args.config}")
        print(f"  输出: {output_dir}")
        print(f"  VST:  {args.vst}")
        print(f"  恢复: {args.resume}")
        print()

        # Run production
        try:
            summary = run_production(
                vst_path=args.vst,
                config_path=args.config,
                output_dir=output_dir,
                resume=args.resume,
            )
        except KeyboardInterrupt:
            print("\n⚠️ 生产已中断。使用 --resume 从断点恢复。")
            return 130
        except Exception as e:
            logger.error("生产失败: %s", e, exc_info=True)
            print(f"\n❌ 生产失败: {e}")
            # Still run verification on partial results
            summary = None

        # Print production summary
        if summary:
            print(f"\n  生产完成:")
            print(f"    总预设: {summary['total_presets']}")
            print(f"    总样本: {summary['total_samples']}")
            print(f"    有效样本: {summary['valid_samples']}")
            print(f"    过滤样本: {summary['filtered_samples']}")
            print(f"    耗时: {summary['total_time_sec']:.1f}s")
    else:
        print(f"仅验证模式 — 跳过生产，对 {output_dir} 执行诊断")
        summary = None

    # Run verification
    run_verification(output_dir, summary)

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
