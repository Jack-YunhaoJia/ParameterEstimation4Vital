#!/usr/bin/env python3
"""
Pilot 运行脚本 — 使用 9 个真实 preset 完成端到端验证。

配置：
  - corpus_dir = presets/（workspace-relative）
  - run_mode = "pilot"
  - pilot_max_base_presets = 9
  - pilot_max_variants_per_base = 4（保守 variant cap）

需要环境变量：
  - VITAL_VST_PATH: Vital VST3 插件路径

Usage:
    python scripts/run_pilot.py
    python scripts/run_pilot.py --output-dir experiments/pilot_run
    python scripts/run_pilot.py --no-resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preset_dataset_producer import PresetDatasetProducer, ProducerConfig

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the pilot run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pilot 运行：使用 9 个真实 preset 完成端到端验证。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/pilot_run",
        help="输出目录路径（默认: experiments/pilot_run）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用 checkpoint resume，从头开始",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pilot production pipeline."""
    setup_logging()
    args = parse_args(argv)

    # --- Require VITAL_VST_PATH ---
    vital_vst_path = os.environ.get("VITAL_VST_PATH")
    if not vital_vst_path:
        logger.error(
            "环境变量 VITAL_VST_PATH 未设置。"
            "请设置为 Vital VST3 插件路径，例如：\n"
            "  export VITAL_VST_PATH=/Library/Audio/Plug-Ins/VST3/Vital.vst3"
        )
        return 1

    vst_path = Path(vital_vst_path)
    if not vst_path.exists():
        logger.error("VITAL_VST_PATH 指向的路径不存在: %s", vst_path)
        return 1

    # --- Build pilot config ---
    corpus_dir = _PROJECT_ROOT / "presets"
    output_dir = Path(args.output_dir)

    config = ProducerConfig(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        vital_vst_path=vst_path,
        run_mode="pilot",
        pilot_max_base_presets=9,
        pilot_max_variants_per_base=4,
        resume=not args.no_resume,
        seed=42,
    )

    logger.info("=" * 60)
    logger.info("Pilot Run Configuration")
    logger.info("=" * 60)
    logger.info("  corpus_dir:                  %s", config.corpus_dir)
    logger.info("  output_dir:                  %s", config.output_dir)
    logger.info("  run_mode:                    %s", config.run_mode)
    logger.info("  pilot_max_base_presets:       %d", config.pilot_max_base_presets)
    logger.info("  pilot_max_variants_per_base:  %d", config.pilot_max_variants_per_base)
    logger.info("  resume:                      %s", config.resume)
    logger.info("  vital_vst_path:              %s", config.vital_vst_path)
    logger.info("=" * 60)

    # --- Run production ---
    producer = PresetDatasetProducer(config)
    summary = producer.produce()

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Pilot Production Summary")
    print("=" * 60)
    summary_dict = asdict(summary)
    print(json.dumps(summary_dict, indent=2, ensure_ascii=False))
    print("=" * 60)

    logger.info("Pilot run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
