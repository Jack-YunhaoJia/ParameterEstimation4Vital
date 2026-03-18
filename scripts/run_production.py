#!/usr/bin/env python3
"""
数据生产流水线入口脚本。

读取 production.yaml 配置 → 实例化所有组件 → 输出资源估算 → 执行生产流水线

Usage:
    python3 scripts/run_production.py --output-dir experiments/production_run
    python3 scripts/run_production.py --output-dir experiments/production_run --resume
    python3 scripts/run_production.py --output-dir experiments/production_run --target-samples 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.audio_preprocessor import AudioPreprocessor, PreprocessConfig
from src.audio_renderer import AudioRenderer, RenderConfig
from src.checkpoint_manager import CheckpointManager
from src.distribution_analyzer import DistributionAnalyzer
from src.multi_condition_renderer import MultiConditionRenderer, MidiCondition
from src.parallel_producer import ParallelProducer, ProductionConfig
from src.quality_validator import QualityValidator
from src.smart_sampler import SmartSampler

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="运行数据生产流水线，生成大规模合成器参数-音频数据集。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径。",
    )
    parser.add_argument(
        "--vst-path",
        type=str,
        default=None,
        help="Vital VST3 插件路径（覆盖配置文件）。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/production.yaml",
        help="生产配置文件路径（默认: configs/production.yaml）。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从断点恢复生产。",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=None,
        help="目标样本数（覆盖配置文件）。",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="渲染并行度（覆盖配置文件）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Embedding 提取设备（覆盖配置文件）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（覆盖配置文件）。",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        choices=[
            "sampling", "rendering", "preprocessing",
            "embedding", "validation", "saving", "analysis",
        ],
        help="从指定阶段恢复执行（跳过该阶段之前的所有已完成阶段）。",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保留历史检查点文件（使用 --no-keep-checkpoints 禁用）。",
    )
    parser.add_argument(
        "--resample-workers",
        type=int,
        default=4,
        help="重采样并行线程数（默认: 4）。",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="跳过资源估算确认，直接开始生产。",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file, merging with base_config if specified."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 如果指定了 base_config，先加载基础配置再合并
    base_path = config.get("base_config")
    if base_path:
        base_full = Path(config_path).parent.parent / base_path
        if base_full.exists():
            with open(base_full, "r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
            # 浅合并：production 配置覆盖 base
            merged = {**base, **config}
            merged.pop("base_config", None)
            return merged
    return config


def setup_logging(output_dir: Path) -> None:
    """Configure logging with console and file handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        output_dir / "production.log", encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)


def build_production_config(config: dict, args: argparse.Namespace) -> ProductionConfig:
    """Build ProductionConfig from YAML config with CLI overrides."""
    prod = config.get("production", {})
    parallel = config.get("parallel", {})
    sampling = config.get("sampling", {})

    return ProductionConfig(
        target_samples=args.target_samples or prod.get("target_samples", 100_000),
        n_workers=args.n_workers or parallel.get("n_workers", 11),
        embedding_batch_size=parallel.get("embedding_batch_size", 32),
        embedding_device=args.device or parallel.get("embedding_device", "mps"),
        checkpoint_interval=parallel.get("checkpoint_interval", 100),
        sampling_strategy=sampling.get("strategy", "lhs_stratified"),
        seed=args.seed if args.seed is not None else sampling.get("seed", 42),
        filter_margin=prod.get("filter_margin", 0.02),
        n_conditions=len(config.get("multi_condition", {}).get("conditions", [])) or 6,
        resample_workers=args.resample_workers,
    )


def print_resource_estimate(estimate: dict) -> None:
    """Print resource estimate in a human-readable format."""
    print("\n" + "=" * 60)
    print("  数据生产资源估算")
    print("=" * 60)
    print(f"  预设数量:       {estimate['n_presets']:,}")
    print(f"  总样本数:       {estimate['n_samples']:,}")
    print(f"  预计有效样本:   {estimate['n_valid_estimate']:,}")
    print(f"  WAV 存储:       {estimate['wav_size_gb']:.2f} GB")
    print(f"  HDF5 大小:      {estimate['hdf5_size_mb']:.2f} MB")
    print(f"  渲染时间:       {estimate['render_time_sec']:.0f} 秒")
    print(f"  Embedding 时间: {estimate['embed_time_sec']:.0f} 秒")
    print(f"  预计总耗时:     {estimate['estimated_hours']:.2f} 小时")
    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for production pipeline.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    setup_logging(output_dir)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("已加载配置: %s", args.config)

        # Resolve VST path
        vst_path = args.vst_path or config.get("vital", {}).get("vst3_path", "")
        if not vst_path:
            print("❌ 未指定 Vital VST3 路径。请使用 --vst-path 或在配置文件中设置。")
            return 1

        # Build production config
        prod_config = build_production_config(config, args)
        logger.info(
            "生产配置: target=%d, workers=%d, device=%s, strategy=%s",
            prod_config.target_samples,
            prod_config.n_workers,
            prod_config.embedding_device,
            prod_config.sampling_strategy,
        )

        # Instantiate components
        preprocess_cfg = config.get("preprocessing", {})
        preprocessor = AudioPreprocessor(PreprocessConfig(
            silence_threshold_db=preprocess_cfg.get("silence_threshold_db", -60.0),
            clipping_threshold=preprocess_cfg.get("clipping_threshold", 0.99),
            clipping_ratio_limit=preprocess_cfg.get("clipping_ratio_limit", 0.10),
            target_peak_db=preprocess_cfg.get("target_peak_db", -1.0),
            target_sample_rate=preprocess_cfg.get("target_sample_rate", 16000),
            tail_silence_threshold_db=preprocess_cfg.get("tail_silence_threshold_db", -50.0),
            min_duration_sec=preprocess_cfg.get("min_duration_sec", 0.5),
        ))

        sampler = SmartSampler(seed=prod_config.seed)

        quality_cfg = config.get("quality", {})
        validator = QualityValidator(
            silence_threshold_db=preprocess_cfg.get("silence_threshold_db", -60.0),
            clipping_ratio_limit=preprocess_cfg.get("clipping_ratio_limit", 0.10),
            spectral_entropy_threshold=quality_cfg.get("spectral_entropy_threshold", 0.05),
            near_duplicate_threshold=quality_cfg.get("near_duplicate_threshold", 0.999),
            pca_collapse_threshold=quality_cfg.get("pca_collapse_threshold", 0.95),
        )

        dist_cfg = config.get("distribution", {})
        analyzer = DistributionAnalyzer(
            diversity_threshold=dist_cfg.get("diversity_threshold", 0.95),
        )

        # Create CheckpointManager
        checkpoint_manager = CheckpointManager(
            output_dir,
            keep_checkpoints=args.keep_checkpoints,
        )

        # Create ParallelProducer
        producer = ParallelProducer(
            vital_vst_path=Path(vst_path),
            config=prod_config,
            preprocessor=preprocessor,
            sampler=sampler,
            validator=validator,
            analyzer=analyzer,
            checkpoint_manager=checkpoint_manager,
        )

        # Print resource estimate and wait for confirmation
        estimate = producer.estimate_resources(
            producer.n_presets, prod_config.n_conditions
        )
        print_resource_estimate(estimate)

        if not args.yes:
            response = input("是否开始生产？(y/N): ").strip().lower()
            if response not in ("y", "yes"):
                print("已取消。")
                return 0

        # Run production
        logger.info("开始数据生产...")
        summary = producer.produce(
            output_dir,
            resume=args.resume,
            resume_from=args.resume_from,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("  生产完成")
        print("=" * 60)
        print(f"  总预设数:   {summary.total_presets:,}")
        print(f"  总样本数:   {summary.total_samples:,}")
        print(f"  有效样本:   {summary.valid_samples:,}")
        print(f"  过滤样本:   {summary.filtered_samples:,}")
        print(f"  失败样本:   {summary.failed_samples:,}")
        print(f"  总耗时:     {summary.total_time_sec:.1f} 秒 ({summary.total_time_sec / 3600:.2f} 小时)")
        if summary.filter_reasons:
            print("  过滤原因:")
            for reason, count in summary.filter_reasons.items():
                print(f"    {reason}: {count}")
        print("=" * 60)
        print(f"\n✅ 生产完成。结果保存至 {output_dir}")
        return 0

    except KeyboardInterrupt:
        logger.info("用户中断。")
        print("\n⚠️ 生产已中断。使用 --resume 从断点恢复。")
        return 130
    except Exception as e:
        logger.error("致命错误: %s", e, exc_info=True)
        print(f"\n❌ 生产失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
