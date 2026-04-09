"""
SmartSampler 单元测试。

验证 LHS 采样、分层采样、离散参数约束、种子可复现性和报告生成。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.smart_sampler import (
    SmartSampler,
    SamplingReport,
    _FILTER_MODEL_INDEX,
    _FILTER_STYLE_INDEX,
)
from src.training_data import (
    CORE_PARAMS,
    EFFECT_SWITCH_INDICES,
    EFFECT_SWITCH_NAMES,
    NUM_PARAMS,
)


class TestSmartSamplerInit:
    """Test SmartSampler initialization."""

    def test_default_seed(self):
        sampler = SmartSampler()
        assert sampler.seed == 42

    def test_custom_seed(self):
        sampler = SmartSampler(seed=123)
        assert sampler.seed == 123


class TestSampleLHS:
    """Test LHS sampling method."""

    def test_output_shape(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(100)
        assert params.shape == (100, NUM_PARAMS)
        assert params.dtype == np.float32

    def test_params_within_range(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(200)
        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            assert params[:, col].min() >= lo, f"{name} below min"
            assert params[:, col].max() <= hi, f"{name} above max"

    def test_effect_switches_binary(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(200)
        for idx in EFFECT_SWITCH_INDICES:
            unique_vals = set(np.unique(params[:, idx]))
            assert unique_vals <= {0.0, 1.0}, f"Switch at index {idx} has non-binary values"

    def test_filter_model_discrete(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(500)
        model_vals = set(np.unique(params[:, _FILTER_MODEL_INDEX]))
        assert model_vals <= {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}

    def test_filter_style_discrete(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(500)
        style_vals = set(np.unique(params[:, _FILTER_STYLE_INDEX]))
        assert style_vals <= {0.0, 1.0, 2.0, 3.0}

    def test_n_equals_1(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(1)
        assert params.shape == (1, NUM_PARAMS)


class TestSampleStratifiedSwitches:
    """Test stratified switch sampling method."""

    def test_output_shape(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_stratified_switches(512)
        assert params.shape == (512, NUM_PARAMS)

    def test_effect_switches_binary(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_stratified_switches(512)
        for idx in EFFECT_SWITCH_INDICES:
            unique_vals = set(np.unique(params[:, idx]))
            assert unique_vals <= {0.0, 1.0}

    def test_layer_proportions(self):
        """Each layer's sample count should be proportional to C(9, k)."""
        n = 1024
        sampler = SmartSampler(seed=42)
        params = sampler.sample_stratified_switches(n)

        switch_matrix = params[:, EFFECT_SWITCH_INDICES]
        active_counts = switch_matrix.sum(axis=1).astype(int)

        num_switches = len(EFFECT_SWITCH_INDICES)
        total_comb = sum(math.comb(num_switches, k) for k in range(num_switches + 1))

        for k in range(num_switches + 1):
            expected = round(n * math.comb(num_switches, k) / total_comb)
            actual = int((active_counts == k).sum())
            # Allow ±1 deviation due to rounding
            assert abs(actual - expected) <= 1, (
                f"Layer k={k}: expected ~{expected}, got {actual}"
            )

    def test_params_within_range(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_stratified_switches(200)
        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            assert params[:, col].min() >= lo, f"{name} below min"
            assert params[:, col].max() <= hi, f"{name} above max"


class TestSampleUnified:
    """Test unified sample() entry point."""

    def test_lhs_strategy(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample(100, strategy="lhs")
        assert params.shape == (100, NUM_PARAMS)

    def test_lhs_stratified_strategy(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample(100, strategy="lhs_stratified")
        assert params.shape == (100, NUM_PARAMS)

    def test_default_strategy(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample(100)
        assert params.shape == (100, NUM_PARAMS)

    def test_invalid_strategy_raises(self):
        sampler = SmartSampler(seed=42)
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            sampler.sample(100, strategy="invalid")


class TestSeedReproducibility:
    """Test that same seed produces identical results."""

    def test_lhs_reproducible(self):
        s1 = SmartSampler(seed=99)
        s2 = SmartSampler(seed=99)
        p1 = s1.sample_lhs(50)
        p2 = s2.sample_lhs(50)
        np.testing.assert_array_equal(p1, p2)

    def test_stratified_reproducible(self):
        s1 = SmartSampler(seed=99)
        s2 = SmartSampler(seed=99)
        p1 = s1.sample_stratified_switches(50)
        p2 = s2.sample_stratified_switches(50)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        s1 = SmartSampler(seed=1)
        s2 = SmartSampler(seed=2)
        p1 = s1.sample_lhs(50)
        p2 = s2.sample_lhs(50)
        assert not np.array_equal(p1, p2)


class TestGenerateReport:
    """Test sampling report generation."""

    def test_report_structure(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(200)
        report = sampler.generate_report(params)

        assert isinstance(report, SamplingReport)
        assert report.n_samples == 200
        assert report.seed == 42
        assert len(report.per_param_ks_statistic) == NUM_PARAMS
        assert len(report.per_param_ks_pvalue) == NUM_PARAMS

    def test_ks_values_in_range(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(200)
        report = sampler.generate_report(params)

        for name in report.per_param_ks_statistic:
            assert 0.0 <= report.per_param_ks_statistic[name] <= 1.0
            assert 0.0 <= report.per_param_ks_pvalue[name] <= 1.0

    def test_effect_switch_distribution(self):
        sampler = SmartSampler(seed=42)
        params = sampler.sample_stratified_switches(512)
        report = sampler.generate_report(params)

        total = sum(report.effect_switch_distribution.values())
        assert total == 512

    def test_lhs_marginal_uniformity(self):
        """LHS samples should pass KS test for continuous params."""
        sampler = SmartSampler(seed=42)
        params = sampler.sample_lhs(500)
        report = sampler.generate_report(params)

        # Continuous params should have high p-values (uniform distribution)
        for name, lo, hi in CORE_PARAMS:
            if name not in EFFECT_SWITCH_NAMES and name not in ("filter_1_model", "filter_1_style"):
                pvalue = report.per_param_ks_pvalue[name]
                assert pvalue > 0.01, (
                    f"Continuous param {name} failed KS test: p={pvalue:.4f}"
                )
