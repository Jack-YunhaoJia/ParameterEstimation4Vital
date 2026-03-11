#!/usr/bin/env python3
"""
Phase 1 参数回归训练入口脚本。

读取 Phase 0 报告 → 检查可行性 → 生成训练数据 → 训练回归模型 → 输出评估报告

Usage:
    python3 scripts/run_phase1.py --phase0-report experiments/20250715_143022/report.json
    python3 scripts/run_phase1.py --phase0-report path/to/report.json --epochs 200 --lr 0.0005
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.audio_renderer import AudioRenderer, RenderConfig
from src.embedding_extractor import EmbeddingExtractor
from src.parameter_regressor import (
    FeasibilityGateError,
    ParameterRegressor,
    RegressionMetrics,
    check_phase0_feasibility,
    evaluate_model,
    train_model,
)
from src.preset_generator import PresetGenerator
from src.preset_parser import PresetParser
from src.training_data import (
    CORE_PARAMS,
    NUM_PARAMS,
    TrainingDataGenerator,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 parameter regression training for Vital synth parameter estimation.",
    )
    parser.add_argument(
        "--phase0-report",
        type=str,
        required=True,
        help="Path to Phase 0 report.json (required).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/phase1",
        help="Output directory for Phase 1 results (default: experiments/phase1).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of training samples to generate (default: 10000).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--vst-path",
        type=str,
        default=None,
        help="Path to Vital VST3 plugin (default: from config).",
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
        help="Device for model training and inference (default: cpu).",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def setup_logging(output_dir: Path | None = None) -> None:
    """Configure logging with console and optional file handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            output_dir / "phase1.log", encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)


def print_evaluation_summary(metrics: RegressionMetrics) -> None:
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 60)
    print("  Phase 1 Evaluation Summary")
    print("=" * 60)

    print(f"\n  Overall MAE: {metrics.overall_mae:.6f}")

    if metrics.per_param_mae:
        print("\n  Per-parameter MAE:")
        for name, mae in sorted(metrics.per_param_mae.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:30s}  MAE={mae:.6f}")

    if metrics.spectral_loss > 0:
        print(f"\n  Spectral Loss: {metrics.spectral_loss:.6f}")

    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for Phase 1 training.

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    setup_logging(output_dir)

    try:
        # Step 1: Check Phase 0 feasibility gate
        report_path = Path(args.phase0_report)
        logger.info("Checking Phase 0 feasibility: %s", report_path)

        try:
            check_phase0_feasibility(report_path)
        except FeasibilityGateError as e:
            print(f"\n❌ Phase 1 blocked: {e}")
            print(
                "\n💡 Suggestion: Adjust your technical approach — consider "
                "alternative audio representations, different embedding models, "
                "or verify that effect configurations produce audible differences."
            )
            return 1

        logger.info("Phase 0 feasibility check passed. Starting Phase 1...")

        # Load configuration
        config = load_config(args.config)
        logger.info("Loaded configuration from %s", args.config)

        # Resolve parameters (CLI overrides > config > defaults)
        vst_path = args.vst_path or config.get("vital", {}).get("vst3_path", "")
        n_samples = args.n_samples
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        device = args.device

        logger.info("Parameters: n_samples=%d, epochs=%d, batch_size=%d, lr=%f, device=%s",
                     n_samples, epochs, batch_size, lr, device)

        # Step 2: Instantiate components
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

        # Step 3: Generate training data
        logger.info("Generating training dataset with %d samples...", n_samples)
        data_gen = TrainingDataGenerator(generator, renderer, extractor)
        metadata = data_gen.generate_dataset(n_samples, output_dir)

        if metadata.total_samples == 0:
            logger.error("No training data generated. Aborting.")
            print("\n❌ No training data could be generated. Check VST and MuQ setup.")
            return 1

        logger.info("Dataset generated: %d samples (%d failed)",
                     metadata.total_samples, metadata.failed_samples)

        # Step 4: Load HDF5 dataset and create DataLoaders
        hdf5_path = output_dir / "dataset.h5"
        logger.info("Loading dataset from %s", hdf5_path)
        dataset = TrainingDataGenerator.load_hdf5(hdf5_path)

        train_ds = TensorDataset(
            torch.from_numpy(dataset["train_embeddings"]),
            torch.from_numpy(dataset["train_params"]),
        )
        val_ds = TensorDataset(
            torch.from_numpy(dataset["val_embeddings"]),
            torch.from_numpy(dataset["val_params"]),
        )
        test_ds = TensorDataset(
            torch.from_numpy(dataset["test_embeddings"]),
            torch.from_numpy(dataset["test_params"]),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Step 5: Create and train model
        embedding_dim = dataset["train_embeddings"].shape[1]
        model = ParameterRegressor(input_dim=embedding_dim, output_dim=NUM_PARAMS)
        logger.info("Model: input_dim=%d, output_dim=%d", embedding_dim, NUM_PARAMS)

        logger.info("Training for %d epochs...", epochs)
        history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, device=device,
        )

        # Step 6: Evaluate model
        logger.info("Evaluating model on test set...")
        param_names = [name for name, _, _ in CORE_PARAMS]
        metrics = evaluate_model(model, test_loader, param_names=param_names, device=device)

        # Step 7: Print evaluation summary
        print_evaluation_summary(metrics)

        # Step 8: Save model checkpoint and evaluation report
        checkpoint_path = output_dir / "model_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": embedding_dim,
            "output_dim": NUM_PARAMS,
            "epochs": epochs,
            "lr": lr,
            "history": history,
        }, checkpoint_path)
        logger.info("Saved model checkpoint to %s", checkpoint_path)

        eval_report = {
            "overall_mae": metrics.overall_mae,
            "spectral_loss": metrics.spectral_loss,
            "per_param_mae": metrics.per_param_mae,
            "training": {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "n_samples": metadata.total_samples,
                "failed_samples": metadata.failed_samples,
            },
        }
        eval_report_path = output_dir / "evaluation_report.json"
        eval_report_path.write_text(
            json.dumps(eval_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved evaluation report to %s", eval_report_path)

        print(f"✅ Phase 1 complete. Results saved to {output_dir}")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
