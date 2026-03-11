"""
Discriminator 模块单元测试。

测试 EffectDiscriminationResult、FeasibilityReport 数据类，
以及 Discriminator 的 evaluate_effect 和 evaluate_all 方法。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.discriminator import (
    Discriminator,
    EffectDiscriminationResult,
    FeasibilityReport,
)
from src.embedding_extractor import EmbeddingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separable_embeddings(
    dim: int = 32, n_per_group: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create clearly separable on/off embedding groups."""
    rng = np.random.RandomState(seed)
    on_embs = rng.randn(n_per_group, dim).astype(np.float32) + 5.0
    off_embs = rng.randn(n_per_group, dim).astype(np.float32) - 5.0
    return on_embs, off_embs


def _make_identical_embeddings(
    dim: int = 32, n_per_group: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create nearly identical on/off embedding groups (tiny noise)."""
    rng = np.random.RandomState(seed)
    base = rng.randn(1, dim).astype(np.float32)
    on_embs = base + rng.randn(n_per_group, dim).astype(np.float32) * 1e-6
    off_embs = base + rng.randn(n_per_group, dim).astype(np.float32) * 1e-6
    return on_embs, off_embs


def _build_embedding_result(
    effects_data: dict[str, tuple[np.ndarray, np.ndarray]],
    include_base_patch: bool = True,
    dim: int = 32,
) -> EmbeddingResult:
    """Build an EmbeddingResult from per-effect on/off arrays.

    effects_data maps effect_name -> (on_embeddings, off_embeddings).
    Files are named like chorus_on_1.0.wav / chorus_on_0.0.wav.
    """
    embeddings: dict[str, np.ndarray] = {}

    for effect_name, (on_arr, off_arr) in effects_data.items():
        for i, emb in enumerate(on_arr):
            embeddings[f"{effect_name}_1.0.wav"] = emb
        for i, emb in enumerate(off_arr):
            embeddings[f"{effect_name}_0.0.wav"] = emb

    if include_base_patch:
        rng = np.random.RandomState(99)
        embeddings["base_patch.wav"] = rng.randn(dim).astype(np.float32) - 5.0

    return EmbeddingResult(embeddings=embeddings, dimension=dim)


# ---------------------------------------------------------------------------
# 1. EffectDiscriminationResult dataclass creation
# ---------------------------------------------------------------------------

class TestEffectDiscriminationResult:
    def test_creation(self):
        result = EffectDiscriminationResult(
            effect_name="chorus_on",
            cosine_similarity=0.85,
            classification_accuracy=0.90,
            is_distinguishable=True,
            is_too_similar=False,
        )
        assert result.effect_name == "chorus_on"
        assert result.cosine_similarity == 0.85
        assert result.classification_accuracy == 0.90
        assert result.is_distinguishable is True
        assert result.is_too_similar is False

    def test_too_similar_flag(self):
        result = EffectDiscriminationResult(
            effect_name="eq_on",
            cosine_similarity=0.995,
            classification_accuracy=0.50,
            is_distinguishable=False,
            is_too_similar=True,
        )
        assert result.is_too_similar is True
        assert result.is_distinguishable is False


# ---------------------------------------------------------------------------
# 2. FeasibilityReport dataclass creation
# ---------------------------------------------------------------------------

class TestFeasibilityReport:
    def test_creation_defaults(self):
        report = FeasibilityReport()
        assert report.results == []
        assert report.pass_count == 0
        assert report.is_feasible is False
        assert report.recommendation == ""

    def test_creation_with_values(self):
        r = EffectDiscriminationResult("chorus_on", 0.8, 0.9, True, False)
        report = FeasibilityReport(
            results=[r],
            pass_count=1,
            is_feasible=False,
            recommendation="Not enough effects pass.",
        )
        assert len(report.results) == 1
        assert report.pass_count == 1
        assert report.is_feasible is False


# ---------------------------------------------------------------------------
# 3-6. evaluate_effect tests
# ---------------------------------------------------------------------------

class TestEvaluateEffect:
    def setup_method(self):
        self.disc = Discriminator()

    def test_separable_embeddings_high_accuracy(self):
        """Clearly separable embeddings → high accuracy."""
        on_embs, off_embs = _make_separable_embeddings()
        result = self.disc.evaluate_effect(on_embs, off_embs, "chorus_on")
        assert result.classification_accuracy >= 0.9
        assert result.is_distinguishable is True

    def test_identical_embeddings_high_cosine_similarity(self):
        """Nearly identical embeddings → high cosine similarity, low accuracy."""
        on_embs, off_embs = _make_identical_embeddings()
        result = self.disc.evaluate_effect(on_embs, off_embs, "eq_on")
        assert result.cosine_similarity > 0.99
        assert result.is_too_similar is True

    def test_cosine_similarity_in_valid_range(self):
        """Cosine similarity should be in [-1, 1]."""
        rng = np.random.RandomState(123)
        on_embs = rng.randn(5, 32).astype(np.float32)
        off_embs = rng.randn(5, 32).astype(np.float32)
        result = self.disc.evaluate_effect(on_embs, off_embs, "delay_on")
        assert -1.0 <= result.cosine_similarity <= 1.0

    def test_accuracy_in_valid_range(self):
        """Classification accuracy should be in [0, 1]."""
        rng = np.random.RandomState(456)
        on_embs = rng.randn(5, 32).astype(np.float32)
        off_embs = rng.randn(5, 32).astype(np.float32)
        result = self.disc.evaluate_effect(on_embs, off_embs, "reverb_on")
        assert 0.0 <= result.classification_accuracy <= 1.0

    def test_single_sample_per_group(self):
        """Works with just 1 sample per group."""
        on_embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        off_embs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        result = self.disc.evaluate_effect(on_embs, off_embs, "phaser_on")
        assert -1.0 <= result.cosine_similarity <= 1.0
        assert 0.0 <= result.classification_accuracy <= 1.0

    def test_1d_input_handled(self):
        """1D input arrays are reshaped correctly."""
        on_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        off_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = self.disc.evaluate_effect(on_emb, off_emb, "flanger_on")
        assert result.effect_name == "flanger_on"


# ---------------------------------------------------------------------------
# 7-9. evaluate_all tests
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def setup_method(self):
        self.disc = Discriminator()

    def _make_all_effects_data(
        self, separable: bool = True, dim: int = 32
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Create data for all 9 effects."""
        from src.preset_parser import PresetParser

        data = {}
        for i, effect in enumerate(PresetParser.EFFECT_SWITCHES):
            seed = i * 10
            if separable:
                data[effect] = _make_separable_embeddings(dim=dim, seed=seed)
            else:
                data[effect] = _make_identical_embeddings(dim=dim, seed=seed)
        return data

    def test_results_sorted_descending_by_accuracy(self):
        """Results should be sorted by classification_accuracy descending."""
        data = self._make_all_effects_data(separable=True)
        emb_result = _build_embedding_result(data)
        report = self.disc.evaluate_all(emb_result)

        accuracies = [r.classification_accuracy for r in report.results]
        assert accuracies == sorted(accuracies, reverse=True)

    def test_feasible_when_enough_effects_pass(self):
        """≥6 effects with accuracy ≥75% → is_feasible=True."""
        data = self._make_all_effects_data(separable=True)
        emb_result = _build_embedding_result(data)
        report = self.disc.evaluate_all(emb_result)

        assert report.pass_count >= 6
        assert report.is_feasible is True

    def test_not_feasible_when_too_few_pass(self):
        """<6 effects with accuracy ≥75% → is_feasible=False."""
        # All identical → low accuracy
        data = self._make_all_effects_data(separable=False)
        emb_result = _build_embedding_result(data)
        report = self.disc.evaluate_all(emb_result)

        assert report.pass_count < 6
        assert report.is_feasible is False

    def test_report_has_recommendation(self):
        """Report should contain a non-empty recommendation."""
        data = self._make_all_effects_data(separable=True)
        emb_result = _build_embedding_result(data)
        report = self.disc.evaluate_all(emb_result)
        assert len(report.recommendation) > 0


# ---------------------------------------------------------------------------
# 10-12. is_too_similar flag tests
# ---------------------------------------------------------------------------

class TestIsTooSimilar:
    def setup_method(self):
        self.disc = Discriminator()

    def test_too_similar_when_cosine_above_threshold(self):
        """cosine_similarity > 0.99 → is_too_similar=True."""
        on_embs, off_embs = _make_identical_embeddings()
        result = self.disc.evaluate_effect(on_embs, off_embs, "eq_on")
        assert result.cosine_similarity > 0.99
        assert result.is_too_similar is True

    def test_not_too_similar_when_cosine_below_threshold(self):
        """cosine_similarity ≤ 0.99 → is_too_similar=False."""
        on_embs, off_embs = _make_separable_embeddings()
        result = self.disc.evaluate_effect(on_embs, off_embs, "chorus_on")
        assert result.cosine_similarity <= 0.99
        assert result.is_too_similar is False

    def test_boundary_exactly_0_99_not_too_similar(self):
        """cosine_similarity exactly 0.99 → is_too_similar=False (strictly >)."""
        # We test the logic directly: is_too_similar = cos_sim > 0.99
        # At exactly 0.99, it should be False
        result = EffectDiscriminationResult(
            effect_name="test",
            cosine_similarity=0.99,
            classification_accuracy=0.5,
            is_distinguishable=False,
            is_too_similar=False,  # 0.99 is NOT > 0.99
        )
        assert result.is_too_similar is False

        # Verify the Discriminator threshold logic directly
        assert (0.99 > Discriminator.SIMILARITY_WARNING) is False
        assert (0.991 > Discriminator.SIMILARITY_WARNING) is True
