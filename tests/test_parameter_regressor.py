"""
ParameterRegressor 单元测试。

测试 RegressionMetrics 数据类、MLP 模型前向传播、输出范围约束、
export_preset 反归一化、Phase 0 门控、训练循环和评估逻辑。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.parameter_regressor import (
    FeasibilityGateError,
    ParameterRegressor,
    RegressionMetrics,
    check_phase0_feasibility,
    evaluate_model,
    train_model,
)
from src.preset_parser import PresetParser
from src.training_data import CORE_PARAMS, NUM_PARAMS


# ---------------------------------------------------------------------------
# 1. RegressionMetrics dataclass creation
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_creation_defaults(self):
        metrics = RegressionMetrics()
        assert metrics.per_param_mae == {}
        assert metrics.overall_mae == 0.0
        assert metrics.spectral_loss == 0.0

    def test_creation_with_values(self):
        per_param = {"osc_1_level": 0.05, "filter_1_cutoff": 0.12}
        metrics = RegressionMetrics(
            per_param_mae=per_param,
            overall_mae=0.085,
            spectral_loss=0.42,
        )
        assert metrics.per_param_mae == per_param
        assert metrics.overall_mae == 0.085
        assert metrics.spectral_loss == 0.42


# ---------------------------------------------------------------------------
# 2. Model forward pass: input (batch, 1024) → output (batch, 45)
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 45)

    def test_single_sample(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        x = torch.randn(1, 1024)
        out = model(x)
        assert out.shape == (1, 45)


# ---------------------------------------------------------------------------
# 3. Output values in [0, 1] range (sigmoid)
# ---------------------------------------------------------------------------

class TestOutputRange:
    def test_output_in_unit_range(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        model.eval()
        x = torch.randn(16, 1024)
        with torch.no_grad():
            out = model(x)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_extreme_inputs(self):
        """Large magnitude inputs should still produce [0, 1] outputs."""
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        model.eval()
        x_large = torch.randn(4, 1024) * 100.0
        with torch.no_grad():
            out = model(x_large)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)


# ---------------------------------------------------------------------------
# 4. Different batch sizes work correctly
# ---------------------------------------------------------------------------

class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 16, 32, 64])
    def test_various_batch_sizes(self, batch_size):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        model.eval()
        x = torch.randn(batch_size, 1024)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 45)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)


# ---------------------------------------------------------------------------
# 5 & 6. export_preset produces valid VitalPreset with correct denormalization
# ---------------------------------------------------------------------------

class TestExportPreset:
    def test_produces_valid_preset(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        parser = PresetParser()

        # All zeros → should map to min values
        params = torch.zeros(45)
        preset = model.export_preset(params, parser)

        assert hasattr(preset, "settings")
        assert hasattr(preset, "modulations")
        assert hasattr(preset, "extra")
        assert isinstance(preset.settings, dict)

    def test_denormalization_zeros(self):
        """Normalized 0.0 should map to min value for each param."""
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        parser = PresetParser()

        params = torch.zeros(45)
        preset = model.export_preset(params, parser)

        for i, (name, lo, hi) in enumerate(CORE_PARAMS):
            assert name in preset.settings
            assert abs(preset.settings[name] - lo) < 1e-6, (
                f"{name}: expected {lo}, got {preset.settings[name]}"
            )

    def test_denormalization_ones(self):
        """Normalized 1.0 should map to max value for each param."""
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        parser = PresetParser()

        params = torch.ones(45)
        preset = model.export_preset(params, parser)

        for i, (name, lo, hi) in enumerate(CORE_PARAMS):
            assert name in preset.settings
            assert abs(preset.settings[name] - hi) < 1e-6, (
                f"{name}: expected {hi}, got {preset.settings[name]}"
            )

    def test_denormalization_midpoint(self):
        """Normalized 0.5 should map to midpoint of each param range."""
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        parser = PresetParser()

        params = torch.full((45,), 0.5)
        preset = model.export_preset(params, parser)

        for i, (name, lo, hi) in enumerate(CORE_PARAMS):
            expected = lo + 0.5 * (hi - lo)
            assert abs(preset.settings[name] - expected) < 1e-5, (
                f"{name}: expected {expected}, got {preset.settings[name]}"
            )

    def test_spot_check_specific_params(self):
        """Spot check a few specific parameter denormalizations."""
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        parser = PresetParser()

        # Set specific normalized values
        params = torch.zeros(45)
        # osc_1_transpose: range [-48, 48], set normalized to 0.75
        # Expected: -48 + 0.75 * 96 = -48 + 72 = 24.0
        params[1] = 0.75
        # filter_1_cutoff: range [8, 136], set normalized to 0.5
        # Expected: 8 + 0.5 * 128 = 72.0
        params[6] = 0.5

        preset = model.export_preset(params, parser)

        assert abs(preset.settings["osc_1_transpose"] - 24.0) < 1e-5
        assert abs(preset.settings["filter_1_cutoff"] - 72.0) < 1e-5

    def test_preset_extra_fields(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        parser = PresetParser()
        params = torch.zeros(45)
        preset = model.export_preset(params, parser)

        assert preset.extra["author"] == "ParameterRegressor"
        assert preset.extra["preset_name"] == "Predicted Preset"


# ---------------------------------------------------------------------------
# 7. Phase 0 gate check logic
# ---------------------------------------------------------------------------

class TestPhase0Gate:
    def test_feasible_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps({
                "is_feasible": True,
                "recommendation": "Proceed to Phase 1",
            }))
            assert check_phase0_feasibility(report_path) is True

    def test_not_feasible_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps({
                "is_feasible": False,
                "recommendation": "Adjust approach",
            }))
            with pytest.raises(FeasibilityGateError, match="NOT feasible"):
                check_phase0_feasibility(report_path)

    def test_missing_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "nonexistent.json"
            with pytest.raises(FeasibilityGateError, match="not found"):
                check_phase0_feasibility(report_path)

    def test_invalid_json_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text("not valid json {{{")
            with pytest.raises(FeasibilityGateError, match="Cannot read"):
                check_phase0_feasibility(report_path)

    def test_missing_is_feasible_key(self):
        """Report without is_feasible key should be treated as not feasible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps({"some_other_key": True}))
            with pytest.raises(FeasibilityGateError, match="NOT feasible"):
                check_phase0_feasibility(report_path)


# ---------------------------------------------------------------------------
# 8. Training loop runs without error (1 epoch, small data)
# ---------------------------------------------------------------------------

class TestTrainModel:
    def _make_data_loader(self, n_samples=32, batch_size=8):
        """Create a small DataLoader with random data."""
        embeddings = torch.randn(n_samples, 1024)
        # Targets in [0, 1] since model outputs sigmoid
        params = torch.rand(n_samples, 45)
        dataset = TensorDataset(embeddings, params)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_training_runs_1_epoch(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        train_loader = self._make_data_loader(n_samples=32, batch_size=8)

        history = train_model(
            model, train_loader, epochs=1, lr=1e-3, device="cpu"
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_training_with_validation(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        train_loader = self._make_data_loader(n_samples=32, batch_size=8)
        val_loader = self._make_data_loader(n_samples=8, batch_size=4)

        history = train_model(
            model, train_loader, val_loader=val_loader,
            epochs=2, lr=1e-3, device="cpu",
        )

        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert all(v > 0 for v in history["val_loss"])

    def test_training_loss_decreases(self):
        """With enough epochs, training loss should generally decrease."""
        torch.manual_seed(42)
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        train_loader = self._make_data_loader(n_samples=64, batch_size=16)

        history = train_model(
            model, train_loader, epochs=10, lr=1e-3, device="cpu"
        )

        # First loss should be higher than last (generally)
        assert history["train_loss"][0] > history["train_loss"][-1]


# ---------------------------------------------------------------------------
# 9. Evaluation returns valid RegressionMetrics
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def _make_data_loader(self, n_samples=16, batch_size=8):
        embeddings = torch.randn(n_samples, 1024)
        params = torch.rand(n_samples, 45)
        dataset = TensorDataset(embeddings, params)
        return DataLoader(dataset, batch_size=batch_size)

    def test_returns_regression_metrics(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        test_loader = self._make_data_loader()

        metrics = evaluate_model(model, test_loader, device="cpu")

        assert isinstance(metrics, RegressionMetrics)
        assert len(metrics.per_param_mae) == NUM_PARAMS
        assert metrics.overall_mae >= 0.0
        assert metrics.spectral_loss >= 0.0

    def test_per_param_mae_keys(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        test_loader = self._make_data_loader()
        param_names = [name for name, _, _ in CORE_PARAMS]

        metrics = evaluate_model(
            model, test_loader, param_names=param_names, device="cpu"
        )

        for name in param_names:
            assert name in metrics.per_param_mae
            assert metrics.per_param_mae[name] >= 0.0

    def test_empty_loader(self):
        model = ParameterRegressor(input_dim=1024, output_dim=45)
        empty_loader = DataLoader(
            TensorDataset(torch.empty(0, 1024), torch.empty(0, 45)),
            batch_size=1,
        )

        metrics = evaluate_model(model, empty_loader, device="cpu")
        assert metrics.overall_mae == 0.0
        assert metrics.per_param_mae == {}

    def test_perfect_prediction_has_zero_mae(self):
        """If model predicts exactly the target, MAE should be ~0."""
        model = ParameterRegressor(input_dim=1024, output_dim=45)

        # Create data where we know the model output
        model.eval()
        embeddings = torch.randn(8, 1024)
        with torch.no_grad():
            targets = model(embeddings)  # Use model's own output as target

        dataset = TensorDataset(embeddings, targets)
        loader = DataLoader(dataset, batch_size=8)

        metrics = evaluate_model(model, loader, device="cpu")
        assert metrics.overall_mae < 1e-5
