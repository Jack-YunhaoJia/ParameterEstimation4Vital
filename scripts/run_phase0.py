#!/usr/bin/env python3
"""
Phase 0 可行性验证实验入口脚本。

一键运行完整的 Phase 0 流水线：
预设生成 → 音频渲染 → MuQ embedding 提取 → 效果器区分能力评估

Usage:
    python3 scripts/run_phase0.py
    python3 scripts/run_phase0.py --output-dir experiments --device cpu
    python3 scripts/run_phase0.py --start-from render_audio --config configs/default.yaml
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

from src.audio_renderer import AudioRenderer, RenderConfig
from src.discriminator import Discriminator
from src.embedding_extractor import EmbeddingExtractor
from src.pipeline import PipelineOrchestrator, PipelineStep
from src.preset_generator import PresetGenerator
from src.preset_parser import PresetParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 0 feasibility experiment for Vital synth parameter estimation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for experiment outputs (default: from config, 'experiments').",
    )
    parser.add_argument(
        "--vst-path",
        type=str,
        default=None,
        help="Path to Vital VST3 plugin (default: from config).",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        choices=["generate_presets", "render_audio", "extract_embeddings", "evaluate"],
        help="Resume pipeline from a specific step (skip earlier completed steps).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for MuQ model inference (default: cpu).",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: Config file does not exist.
        yaml.YAMLError: Config file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def setup_logging(experiment_dir: Path | None = None) -> None:
    """Configure logging with console and optional file handlers.

    Args:
        experiment_dir: If provided, also log to experiment_dir/experiment.log.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler (added later once experiment dir is known)
    if experiment_dir is not None:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            experiment_dir / "experiment.log", encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)


def print_summary(result) -> None:
    """Print a human-readable experiment summary to the console."""
    print("\n" + "=" * 60)
    print("  Phase 0 Experiment Summary")
    print("=" * 60)
    print(f"  Experiment directory: {result.experiment_dir}")

    if result.step_timings:
        print("\n  Step Timings:")
        total_time = 0.0
        for step_name, elapsed in result.step_timings.items():
            print(f"    {step_name}: {elapsed:.2f}s")
            total_time += elapsed
        print(f"    {'total':>20s}: {total_time:.2f}s")

    if result.error:
        print(f"\n  ❌ Error: {result.error}")
        print("  Next step: Fix the error and resume with --start-from.")
    elif result.feasibility is not None:
        feasibility = result.feasibility
        status = "✅ FEASIBLE" if feasibility.is_feasible else "❌ NOT FEASIBLE"
        print(f"\n  Result: {status}")
        print(f"  Pass count: {feasibility.pass_count}/{len(feasibility.results)} effects")

        print("\n  Per-effect results:")
        for r in feasibility.results:
            marker = "✓" if r.is_distinguishable else "✗"
            warning = " ⚠ too similar" if r.is_too_similar else ""
            print(
                f"    [{marker}] {r.effect_name:20s}  "
                f"cos_sim={r.cosine_similarity:.4f}  "
                f"accuracy={r.classification_accuracy:.4f}{warning}"
            )

        print(f"\n  Recommendation: {feasibility.recommendation}")

        if feasibility.is_feasible:
            print("\n  ➡ Next step: Proceed to Phase 1 — parameter regression training.")
            print(f"  Run Phase 1: python3 scripts/run_phase1.py --phase0-report {result.experiment_dir}/report.json")
        else:
            print(
                "\n  ➡ Next step: Investigate alternative audio representations "
                "or verify effect configurations."
            )
    else:
        print("\n  Pipeline completed without evaluation results.")

    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for Phase 0 experiment.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    args = parse_args(argv)

    # Set up initial console logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Loaded configuration from %s", args.config)

        # Resolve parameters (CLI overrides > config > defaults)
        output_dir = args.output_dir or config.get("experiment", {}).get("output_base", "experiments")
        vst_path = args.vst_path or config.get("vital", {}).get("vst3_path", "")
        device = args.device or config.get("muq", {}).get("device", "cpu")

        # Resolve start_from step
        start_from = None
        if args.start_from:
            start_from = PipelineStep(args.start_from)

        logger.info("Output directory: %s", output_dir)
        logger.info("VST path: %s", vst_path)
        logger.info("Device: %s", device)
        if start_from:
            logger.info("Resuming from step: %s", start_from.value)

        # Instantiate components
        logger.info("Initializing components...")

        parser = PresetParser()

        generator = PresetGenerator(parser, base_patch_template=None)

        render_config = RenderConfig(
            midi_note=config.get("midi", {}).get("note", 60),
            velocity=config.get("midi", {}).get("velocity", 100),
            duration_sec=config.get("midi", {}).get("duration_sec", 2.0),
            sample_rate=config.get("audio", {}).get("sample_rate", 44100),
            timeout_sec=config.get("render", {}).get("timeout_sec", 30.0),
        )
        renderer = AudioRenderer(vital_vst_path=Path(vst_path), config=render_config)

        extractor = EmbeddingExtractor(
            model_path=config.get("muq", {}).get("model_repo"),
            device=device,
            target_sample_rate=config.get("muq", {}).get("target_sample_rate", 16000),
        )

        discriminator = Discriminator()

        # Create orchestrator and run
        orchestrator = PipelineOrchestrator(
            generator=generator,
            renderer=renderer,
            extractor=extractor,
            discriminator=discriminator,
        )

        logger.info("Starting Phase 0 pipeline...")
        result = orchestrator.run(output_base=Path(output_dir), start_from=start_from)

        # Print summary to console
        print_summary(result)

        # Save config snapshot to experiment directory
        config_snapshot_path = result.experiment_dir / "config.yaml"
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info("Saved config snapshot to %s", config_snapshot_path)

        if result.error:
            return 1
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
