"""
PipelineOrchestrator 模块单元测试。

使用 mock 对象替代 PresetGenerator、AudioRenderer、EmbeddingExtractor、Discriminator，
测试流水线编排逻辑：步骤顺序、失败停止、恢复执行、报告保存等。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.audio_renderer import RenderSummary
from src.discriminator import (
    EffectDiscriminationResult,
    FeasibilityReport,
)
from src.embedding_extractor import EmbeddingResult
from src.pipeline import (
    PipelineOrchestrator,
    PipelineResult,
    PipelineStep,
    _STEP_ORDER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate_all_variants.return_value = [Path("p1.vital"), Path("p2.vital")]
    return gen


@pytest.fixture
def mock_renderer():
    renderer = MagicMock()
    renderer.render_batch.return_value = RenderSummary(
        success_count=2, failure_count=0, failed_files=[]
    )
    return renderer


@pytest.fixture
def mock_extractor():
    extractor = MagicMock()
    emb_result = EmbeddingResult(
        embeddings={"file1.wav": np.zeros(32), "file2.wav": np.ones(32)},
        dimension=32,
    )
    extractor.extract_batch.return_value = emb_result
    # Mock the class-level load method
    extractor.load = MagicMock(return_value=emb_result)
    return extractor


@pytest.fixture
def mock_discriminator():
    disc = MagicMock()
    report = FeasibilityReport(
        results=[
            EffectDiscriminationResult("chorus_on", 0.8, 0.9, True, False),
        ],
        pass_count=7,
        is_feasible=True,
        recommendation="Proceed to Phase 1.",
    )
    disc.evaluate_all.return_value = report
    return disc


@pytest.fixture
def orchestrator(mock_generator, mock_renderer, mock_extractor, mock_discriminator):
    return PipelineOrchestrator(
        generator=mock_generator,
        renderer=mock_renderer,
        extractor=mock_extractor,
        discriminator=mock_discriminator,
    )


# ---------------------------------------------------------------------------
# 1. PipelineStep enum values
# ---------------------------------------------------------------------------

class TestPipelineStep:
    def test_enum_values(self):
        assert PipelineStep.GENERATE_PRESETS.value == "generate_presets"
        assert PipelineStep.RENDER_AUDIO.value == "render_audio"
        assert PipelineStep.EXTRACT_EMBEDDINGS.value == "extract_embeddings"
        assert PipelineStep.EVALUATE.value == "evaluate"

    def test_step_count(self):
        assert len(PipelineStep) == 4

    def test_step_order(self):
        assert _STEP_ORDER == [
            PipelineStep.GENERATE_PRESETS,
            PipelineStep.RENDER_AUDIO,
            PipelineStep.EXTRACT_EMBEDDINGS,
            PipelineStep.EVALUATE,
        ]


# ---------------------------------------------------------------------------
# 2. PipelineResult dataclass creation
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_creation_minimal(self):
        result = PipelineResult(experiment_dir=Path("exp/20250101_000000"))
        assert result.experiment_dir == Path("exp/20250101_000000")
        assert result.step_timings == {}
        assert result.feasibility is None
        assert result.error is None

    def test_creation_with_all_fields(self):
        report = FeasibilityReport(is_feasible=True, pass_count=7)
        result = PipelineResult(
            experiment_dir=Path("exp/test"),
            step_timings={"generate_presets": 1.5},
            feasibility=report,
            error=None,
        )
        assert result.step_timings["generate_presets"] == 1.5
        assert result.feasibility.is_feasible is True


# ---------------------------------------------------------------------------
# 3. run() executes all 4 steps in order
# ---------------------------------------------------------------------------

class TestRunStepOrder:
    def test_all_steps_called_in_order(
        self, orchestrator, mock_generator, mock_renderer,
        mock_extractor, mock_discriminator, tmp_path
    ):
        """Verify all 4 components are called in the correct order."""
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orchestrator.run(tmp_path)

        mock_generator.generate_all_variants.assert_called_once()
        mock_renderer.render_batch.assert_called_once()
        mock_extractor.extract_batch.assert_called_once()
        mock_discriminator.evaluate_all.assert_called_once()
        assert result.error is None

    def test_call_order_verified(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        """Verify the exact call order using a shared call tracker."""
        call_order = []

        mock_generator.generate_all_variants.side_effect = lambda d: (
            call_order.append("generate"), [Path("p.vital")]
        )[1]
        mock_renderer.render_batch.side_effect = lambda p, a: (
            call_order.append("render"),
            RenderSummary(success_count=1, failure_count=0, failed_files=[]),
        )[1]
        mock_extractor.extract_batch.side_effect = lambda d: (
            call_order.append("extract"),
            EmbeddingResult(embeddings={"f.wav": np.zeros(32)}, dimension=32),
        )[1]
        mock_discriminator.evaluate_all.side_effect = lambda e: (
            call_order.append("evaluate"),
            FeasibilityReport(is_feasible=True, pass_count=7),
        )[1]

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load.return_value = EmbeddingResult(
                embeddings={"f.wav": np.zeros(32)}, dimension=32
            )
            orch.run(tmp_path)

        assert call_order == ["generate", "render", "extract", "evaluate"]


# ---------------------------------------------------------------------------
# 4. run() creates timestamped experiment directory
# ---------------------------------------------------------------------------

class TestExperimentDirectory:
    def test_creates_timestamped_dir(self, orchestrator, mock_extractor, tmp_path):
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orchestrator.run(tmp_path)

        assert result.experiment_dir.exists()
        assert result.experiment_dir.parent == tmp_path
        # Directory name should be a timestamp like 20250715_143022
        dir_name = result.experiment_dir.name
        assert len(dir_name) == 15  # YYYYMMDD_HHMMSS
        assert dir_name[8] == "_"


# ---------------------------------------------------------------------------
# 5. run() stops on step failure and records error
# ---------------------------------------------------------------------------

class TestFailureStop:
    def test_stops_on_generate_failure(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_generator.generate_all_variants.side_effect = RuntimeError("disk full")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        result = orch.run(tmp_path)

        assert result.error is not None
        assert "generate_presets" in result.error
        assert "disk full" in result.error
        # Subsequent steps should NOT be called
        mock_renderer.render_batch.assert_not_called()
        mock_extractor.extract_batch.assert_not_called()
        mock_discriminator.evaluate_all.assert_not_called()

    def test_stops_on_render_failure(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_renderer.render_batch.side_effect = RuntimeError("VST crash")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        result = orch.run(tmp_path)

        assert result.error is not None
        assert "render_audio" in result.error
        mock_generator.generate_all_variants.assert_called_once()
        mock_extractor.extract_batch.assert_not_called()
        mock_discriminator.evaluate_all.assert_not_called()

    def test_stops_on_extract_failure(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_extractor.extract_batch.side_effect = RuntimeError("model error")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        result = orch.run(tmp_path)

        assert result.error is not None
        assert "extract_embeddings" in result.error
        mock_discriminator.evaluate_all.assert_not_called()

    def test_stops_on_evaluate_failure(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_discriminator.evaluate_all.side_effect = RuntimeError("eval error")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load.return_value = EmbeddingResult(
                embeddings={"f.wav": np.zeros(32)}, dimension=32
            )
            result = orch.run(tmp_path)

        assert result.error is not None
        assert "evaluate" in result.error


# ---------------------------------------------------------------------------
# 6. run() returns correct step timings
# ---------------------------------------------------------------------------

class TestStepTimings:
    def test_all_steps_timed(self, orchestrator, mock_extractor, tmp_path):
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orchestrator.run(tmp_path)

        assert "generate_presets" in result.step_timings
        assert "render_audio" in result.step_timings
        assert "extract_embeddings" in result.step_timings
        assert "evaluate" in result.step_timings
        for timing in result.step_timings.values():
            assert timing >= 0.0

    def test_failed_step_still_timed(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_renderer.render_batch.side_effect = RuntimeError("fail")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        result = orch.run(tmp_path)

        assert "generate_presets" in result.step_timings
        assert "render_audio" in result.step_timings
        assert result.step_timings["render_audio"] >= 0.0
        # Steps after failure should not be timed
        assert "extract_embeddings" not in result.step_timings


# ---------------------------------------------------------------------------
# 7. run() with start_from skips earlier steps
# ---------------------------------------------------------------------------

class TestResumeFromStep:
    def test_start_from_render_skips_generate(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        """Starting from RENDER_AUDIO should skip GENERATE_PRESETS."""
        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )

        # Create presets dir with a .vital file so render can proceed
        exp_dir = tmp_path
        # We need to pre-create the experiment dir structure
        # The orchestrator creates its own timestamped dir, so we need
        # to ensure the presets dir inside it has files for the skip check

        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load.return_value = EmbeddingResult(
                embeddings={"f.wav": np.zeros(32)}, dimension=32
            )
            result = orch.run(tmp_path, start_from=PipelineStep.RENDER_AUDIO)

        # generate should NOT be called (not in steps_to_run)
        mock_generator.generate_all_variants.assert_not_called()
        # render should be called
        mock_renderer.render_batch.assert_called_once()

    def test_start_from_evaluate_skips_first_three(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        """Starting from EVALUATE should skip first 3 steps."""
        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )

        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load.return_value = EmbeddingResult(
                embeddings={"f.wav": np.zeros(32)}, dimension=32
            )
            result = orch.run(tmp_path, start_from=PipelineStep.EVALUATE)

        mock_generator.generate_all_variants.assert_not_called()
        mock_renderer.render_batch.assert_not_called()
        mock_extractor.extract_batch.assert_not_called()
        mock_discriminator.evaluate_all.assert_called_once()

    def test_start_from_extract_skips_generate_and_render(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )

        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load.return_value = EmbeddingResult(
                embeddings={"f.wav": np.zeros(32)}, dimension=32
            )
            result = orch.run(tmp_path, start_from=PipelineStep.EXTRACT_EMBEDDINGS)

        mock_generator.generate_all_variants.assert_not_called()
        mock_renderer.render_batch.assert_not_called()
        mock_extractor.extract_batch.assert_called_once()
        mock_discriminator.evaluate_all.assert_called_once()


# ---------------------------------------------------------------------------
# 8. run() saves report.json
# ---------------------------------------------------------------------------

class TestReportSaving:
    def test_report_json_created(self, orchestrator, mock_extractor, tmp_path):
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orchestrator.run(tmp_path)

        report_path = result.experiment_dir / "report.json"
        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)

        assert "step_timings" in data
        assert "experiment_dir" in data

    def test_report_contains_feasibility(self, orchestrator, mock_extractor, tmp_path):
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orchestrator.run(tmp_path)

        report_path = result.experiment_dir / "report.json"
        with open(report_path) as f:
            data = json.load(f)

        assert "feasibility" in data
        assert data["feasibility"]["is_feasible"] is True
        assert data["feasibility"]["pass_count"] == 7

    def test_report_contains_error_on_failure(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_generator.generate_all_variants.side_effect = RuntimeError("boom")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        result = orch.run(tmp_path)

        report_path = result.experiment_dir / "report.json"
        with open(report_path) as f:
            data = json.load(f)

        assert data["error"] is not None
        assert "boom" in data["error"]


# ---------------------------------------------------------------------------
# 9. run() returns feasibility report from Discriminator
# ---------------------------------------------------------------------------

class TestFeasibilityReturn:
    def test_returns_feasibility_report(self, orchestrator, mock_extractor, tmp_path):
        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orchestrator.run(tmp_path)

        assert result.feasibility is not None
        assert result.feasibility.is_feasible is True
        assert result.feasibility.pass_count == 7

    def test_no_feasibility_on_early_failure(
        self, mock_generator, mock_renderer, mock_extractor,
        mock_discriminator, tmp_path
    ):
        mock_generator.generate_all_variants.side_effect = RuntimeError("fail")

        orch = PipelineOrchestrator(
            mock_generator, mock_renderer, mock_extractor, mock_discriminator
        )
        result = orch.run(tmp_path)

        assert result.feasibility is None


# ---------------------------------------------------------------------------
# 10. run() handles empty preset directory gracefully
# ---------------------------------------------------------------------------

class TestEmptyPresetDir:
    def test_empty_presets_propagates_renderer_behavior(
        self, mock_renderer, mock_extractor, mock_discriminator, tmp_path
    ):
        """When generator produces no files, renderer gets empty dir."""
        gen = MagicMock()
        gen.generate_all_variants.return_value = []

        # Renderer returns empty summary for empty dir
        mock_renderer.render_batch.return_value = RenderSummary(
            success_count=0, failure_count=0, failed_files=[]
        )

        orch = PipelineOrchestrator(
            gen, mock_renderer, mock_extractor, mock_discriminator
        )

        with patch("src.pipeline.EmbeddingExtractor") as mock_ee_cls:
            mock_ee_cls.load = mock_extractor.load
            result = orch.run(tmp_path)

        # Pipeline should still complete without error
        gen.generate_all_variants.assert_called_once()
        mock_renderer.render_batch.assert_called_once()
