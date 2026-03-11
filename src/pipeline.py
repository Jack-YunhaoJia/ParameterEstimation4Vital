"""
流水线编排模块。

按顺序执行 Phase 0 实验流程：预设生成 → 音频渲染 → embedding 提取 → 区分能力评估。
支持失败停止、从指定步骤恢复、带时间戳的实验目录管理。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.audio_renderer import AudioRenderer
from src.discriminator import Discriminator, FeasibilityReport
from src.embedding_extractor import EmbeddingExtractor
from src.preset_generator import PresetGenerator

logger = logging.getLogger(__name__)


class PipelineStep(Enum):
    """Phase 0 流水线步骤。"""

    GENERATE_PRESETS = "generate_presets"
    RENDER_AUDIO = "render_audio"
    EXTRACT_EMBEDDINGS = "extract_embeddings"
    EVALUATE = "evaluate"


# Ordered list for iteration
_STEP_ORDER = [
    PipelineStep.GENERATE_PRESETS,
    PipelineStep.RENDER_AUDIO,
    PipelineStep.EXTRACT_EMBEDDINGS,
    PipelineStep.EVALUATE,
]


@dataclass
class PipelineResult:
    """流水线执行结果。

    Attributes:
        experiment_dir: 实验目录路径
        step_timings: 各步骤耗时（秒）
        feasibility: 可行性报告（评估步骤完成后）
        error: 错误信息（某步骤失败时）
    """

    experiment_dir: Path
    step_timings: dict[str, float] = field(default_factory=dict)
    feasibility: FeasibilityReport | None = None
    error: str | None = None


class PipelineOrchestrator:
    """Phase 0 实验流水线编排器。

    协调 PresetGenerator、AudioRenderer、EmbeddingExtractor、Discriminator
    按顺序执行完整的可行性验证实验。支持失败停止和从指定步骤恢复。
    """

    def __init__(
        self,
        generator: PresetGenerator,
        renderer: AudioRenderer,
        extractor: EmbeddingExtractor,
        discriminator: Discriminator,
    ) -> None:
        self._generator = generator
        self._renderer = renderer
        self._extractor = extractor
        self._discriminator = discriminator

    def run(
        self, output_base: Path, start_from: PipelineStep | None = None
    ) -> PipelineResult:
        """Execute complete Phase 0 pipeline.

        Creates a timestamped experiment directory under output_base and
        runs each step in order. Supports resuming from a specified step
        when intermediate product files already exist.

        Args:
            output_base: Base directory for experiments (e.g. "experiments")
            start_from: Resume from this step, skipping earlier completed steps.
                If None, runs all steps from the beginning.

        Returns:
            PipelineResult with experiment directory, timings, feasibility, and error info.
        """
        output_base = Path(output_base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = output_base / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories for intermediate products
        presets_dir = experiment_dir / "presets"
        audio_dir = experiment_dir / "audio"
        embeddings_dir = experiment_dir / "embeddings"

        result = PipelineResult(experiment_dir=experiment_dir)

        # Determine which steps to execute
        steps_to_run = self._resolve_steps(start_from)

        logger.info(
            "Starting pipeline in %s (steps: %s)",
            experiment_dir,
            [s.value for s in steps_to_run],
        )

        # Step dispatch table
        step_handlers = {
            PipelineStep.GENERATE_PRESETS: lambda: self._step_generate_presets(
                presets_dir
            ),
            PipelineStep.RENDER_AUDIO: lambda: self._step_render_audio(
                presets_dir, audio_dir
            ),
            PipelineStep.EXTRACT_EMBEDDINGS: lambda: self._step_extract_embeddings(
                audio_dir, embeddings_dir
            ),
            PipelineStep.EVALUATE: lambda: self._step_evaluate(embeddings_dir),
        }

        feasibility: FeasibilityReport | None = None

        for step in steps_to_run:
            # Check if we can skip this step (resume logic)
            if self._can_skip_step(step, presets_dir, audio_dir, embeddings_dir):
                logger.info("Skipping step '%s' — intermediate products exist", step.value)
                continue

            logger.info("Executing step: %s", step.value)
            t_start = time.time()

            try:
                step_result = step_handlers[step]()
                elapsed = time.time() - t_start
                result.step_timings[step.value] = elapsed
                logger.info("Step '%s' completed in %.2fs", step.value, elapsed)

                # Capture feasibility from evaluate step
                if step == PipelineStep.EVALUATE and step_result is not None:
                    feasibility = step_result
            except Exception as exc:
                elapsed = time.time() - t_start
                result.step_timings[step.value] = elapsed
                error_msg = f"Step '{step.value}' failed: {exc}"
                result.error = error_msg
                logger.error(error_msg)
                break

        result.feasibility = feasibility

        # Save report.json
        self._save_report(experiment_dir, result)

        # Log summary
        self._log_summary(result)

        return result

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_generate_presets(self, presets_dir: Path) -> None:
        """Generate all effect variant presets."""
        paths = self._generator.generate_all_variants(presets_dir)
        logger.info("Generated %d preset files", len(paths))

    def _step_render_audio(self, presets_dir: Path, audio_dir: Path) -> None:
        """Render all presets to audio."""
        summary = self._renderer.render_batch(presets_dir, audio_dir)
        logger.info(
            "Render summary: %d success, %d failed",
            summary.success_count,
            summary.failure_count,
        )
        if summary.failure_count > 0:
            logger.warning("Failed renders: %s", summary.failed_files)

    def _step_extract_embeddings(
        self, audio_dir: Path, embeddings_dir: Path
    ) -> None:
        """Extract embeddings from rendered audio."""
        emb_result = self._extractor.extract_batch(audio_dir)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        output_path = embeddings_dir / "all_embeddings.npz"
        self._extractor.save(emb_result, output_path)
        logger.info(
            "Extracted %d embeddings, saved to %s",
            len(emb_result.embeddings),
            output_path,
        )

    def _step_evaluate(self, embeddings_dir: Path) -> FeasibilityReport:
        """Evaluate discrimination ability using saved embeddings."""
        emb_path = embeddings_dir / "all_embeddings.npz"
        emb_result = EmbeddingExtractor.load(emb_path)
        report = self._discriminator.evaluate_all(emb_result)
        return report

    # ------------------------------------------------------------------
    # Resume / skip logic
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_steps(
        start_from: PipelineStep | None,
    ) -> list[PipelineStep]:
        """Return the ordered list of steps to run, starting from start_from."""
        if start_from is None:
            return list(_STEP_ORDER)
        start_idx = _STEP_ORDER.index(start_from)
        return list(_STEP_ORDER[start_idx:])

    @staticmethod
    def _can_skip_step(
        step: PipelineStep,
        presets_dir: Path,
        audio_dir: Path,
        embeddings_dir: Path,
    ) -> bool:
        """Check if a step's intermediate products already exist.

        Only skips when the step is NOT in the explicitly requested range
        (handled by _resolve_steps). This method checks file existence for
        the resume scenario where start_from is set and earlier products exist.
        """
        if step == PipelineStep.GENERATE_PRESETS:
            return presets_dir.exists() and any(presets_dir.glob("*.vital"))
        if step == PipelineStep.RENDER_AUDIO:
            return audio_dir.exists() and any(audio_dir.glob("*.wav"))
        if step == PipelineStep.EXTRACT_EMBEDDINGS:
            return (embeddings_dir / "all_embeddings.npz").exists()
        # EVALUATE step is never skipped — always re-run evaluation
        return False

    # ------------------------------------------------------------------
    # Report & summary
    # ------------------------------------------------------------------

    @staticmethod
    def _save_report(experiment_dir: Path, result: PipelineResult) -> None:
        """Save experiment report as JSON."""
        report_data: dict = {
            "experiment_dir": str(result.experiment_dir),
            "step_timings": result.step_timings,
            "error": result.error,
        }

        if result.feasibility is not None:
            report_data["feasibility"] = {
                "is_feasible": result.feasibility.is_feasible,
                "pass_count": result.feasibility.pass_count,
                "recommendation": result.feasibility.recommendation,
                "effects": [
                    {
                        "effect_name": r.effect_name,
                        "cosine_similarity": r.cosine_similarity,
                        "classification_accuracy": r.classification_accuracy,
                        "is_distinguishable": r.is_distinguishable,
                        "is_too_similar": r.is_too_similar,
                    }
                    for r in result.feasibility.results
                ],
            }

        report_path = experiment_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved report to %s", report_path)

    @staticmethod
    def _log_summary(result: PipelineResult) -> None:
        """Log experiment summary with timings and next-step suggestions."""
        logger.info("=" * 60)
        logger.info("Experiment Summary")
        logger.info("  Directory: %s", result.experiment_dir)

        if result.step_timings:
            logger.info("  Step Timings:")
            for step_name, elapsed in result.step_timings.items():
                logger.info("    %s: %.2fs", step_name, elapsed)

        if result.error:
            logger.info("  Error: %s", result.error)
            logger.info("  Next step: Fix the error and resume from the failed step.")
        elif result.feasibility is not None:
            logger.info(
                "  Feasibility: %s (%d/%d effects pass)",
                "PASS" if result.feasibility.is_feasible else "FAIL",
                result.feasibility.pass_count,
                len(result.feasibility.results),
            )
            logger.info("  Recommendation: %s", result.feasibility.recommendation)
            if result.feasibility.is_feasible:
                logger.info("  Next step: Proceed to Phase 1 — parameter regression training.")
            else:
                logger.info(
                    "  Next step: Investigate alternative audio representations "
                    "or verify effect configurations."
                )
        else:
            logger.info("  Pipeline completed without evaluation results.")

        logger.info("=" * 60)
