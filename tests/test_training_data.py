"""
TrainingDataGenerator 单元测试。

测试核心参数定义、参数采样、数据集划分、HDF5 存储和 generate_dataset 流程。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from src.training_data import (
    CORE_PARAMS,
    EFFECT_SWITCH_INDICES,
    EFFECT_SWITCH_NAMES,
    NUM_PARAMS,
    DatasetMetadata,
    TrainingDataGenerator,
    split_dataset,
)


# ---------------------------------------------------------------------------
# Test DatasetMetadata dataclass
# ---------------------------------------------------------------------------

class TestDatasetMetadata:
    def test_creation_defaults(self):
        meta = DatasetMetadata()
        assert meta.param_ranges == {}
        assert meta.param_names == []
        assert meta.total_samples == 0
        assert meta.split_ratio == (0.8, 0.1, 0.1)
        assert meta.failed_samples == 0

    def test_creation_with_values(self):
        ranges = {"osc_1_level": (0.0, 1.0)}
        meta = DatasetMetadata(
            param_ranges=ranges,
            param_names=["osc_1_level"],
            total_samples=100,
            split_ratio=(0.7, 0.15, 0.15),
            failed_samples=5,
        )
        assert meta.param_ranges == ranges
        assert meta.param_names == ["osc_1_level"]
        assert meta.total_samples == 100
        assert meta.split_ratio == (0.7, 0.15, 0.15)
        assert meta.failed_samples == 5


# ---------------------------------------------------------------------------
# Test CORE_PARAMS definition
# ---------------------------------------------------------------------------

class TestCoreParams:
    def test_has_45_entries(self):
        assert len(CORE_PARAMS) == 45
        assert NUM_PARAMS == 45

    def test_all_ranges_valid(self):
        """Every parameter must have min < max."""
        for name, lo, hi in CORE_PARAMS:
            assert lo < hi, f"{name}: min={lo} >= max={hi}"

    def test_unique_names(self):
        names = [name for name, _, _ in CORE_PARAMS]
        assert len(names) == len(set(names)), "Duplicate parameter names"

    def test_effect_switches_present(self):
        param_names = {name for name, _, _ in CORE_PARAMS}
        for sw in EFFECT_SWITCH_NAMES:
            assert sw in param_names, f"Missing effect switch: {sw}"

    def test_effect_switch_indices_correct(self):
        for idx in EFFECT_SWITCH_INDICES:
            name, _, _ = CORE_PARAMS[idx]
            assert name in EFFECT_SWITCH_NAMES


# ---------------------------------------------------------------------------
# Test sample_parameters
# ---------------------------------------------------------------------------

class TestSampleParameters:
    def setup_method(self):
        # Create a TrainingDataGenerator with mocked dependencies
        self.generator = MagicMock()
        self.renderer = MagicMock()
        self.extractor = MagicMock()
        self.tdg = TrainingDataGenerator(
            self.generator, self.renderer, self.extractor
        )

    def test_returns_correct_shape(self):
        result = self.tdg.sample_parameters(10)
        assert result.shape == (10, 45)

    def test_returns_correct_shape_single(self):
        result = self.tdg.sample_parameters(1)
        assert result.shape == (1, 45)

    def test_values_within_ranges(self):
        np.random.seed(42)
        result = self.tdg.sample_parameters(500)
        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            col_vals = result[:, col]
            assert np.all(col_vals >= lo), (
                f"{name}: min value {col_vals.min()} < {lo}"
            )
            assert np.all(col_vals <= hi), (
                f"{name}: max value {col_vals.max()} > {hi}"
            )

    def test_effect_switches_are_binary(self):
        np.random.seed(42)
        result = self.tdg.sample_parameters(200)
        for idx in EFFECT_SWITCH_INDICES:
            name, _, _ = CORE_PARAMS[idx]
            unique_vals = set(np.unique(result[:, idx]))
            assert unique_vals.issubset({0.0, 1.0}), (
                f"{name} has non-binary values: {unique_vals}"
            )

    def test_dtype_is_float32(self):
        result = self.tdg.sample_parameters(5)
        assert result.dtype == np.float32



# ---------------------------------------------------------------------------
# Test split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_basic_split_100(self):
        n_train, n_val, n_test = split_dataset(100)
        assert n_train + n_val + n_test == 100
        assert n_train == 80
        assert n_val == 10
        assert n_test == 10

    def test_basic_split_1000(self):
        n_train, n_val, n_test = split_dataset(1000)
        assert n_train + n_val + n_test == 1000

    def test_small_n(self):
        n_train, n_val, n_test = split_dataset(3)
        assert n_train + n_val + n_test == 3

    def test_n_equals_1(self):
        n_train, n_val, n_test = split_dataset(1)
        assert n_train + n_val + n_test == 1

    def test_proportions_close_to_target(self):
        """Each split should be within 1 sample of ideal proportion."""
        for n in [10, 50, 99, 101, 500, 1000]:
            n_train, n_val, n_test = split_dataset(n)
            assert n_train + n_val + n_test == n
            assert abs(n_train - round(n * 0.8)) <= 1
            assert abs(n_val - round(n * 0.1)) <= 1
            assert abs(n_test - round(n * 0.1)) <= 1


# ---------------------------------------------------------------------------
# Test HDF5 save/load round-trip
# ---------------------------------------------------------------------------

class TestHDF5RoundTrip:
    def setup_method(self):
        self.generator = MagicMock()
        self.renderer = MagicMock()
        self.extractor = MagicMock()
        self.tdg = TrainingDataGenerator(
            self.generator, self.renderer, self.extractor
        )

    def test_save_load_roundtrip(self):
        np.random.seed(42)
        n_train, n_val, n_test = 80, 10, 10
        embed_dim = 1024

        train_p = np.random.randn(n_train, 45).astype(np.float32)
        train_e = np.random.randn(n_train, embed_dim).astype(np.float32)
        val_p = np.random.randn(n_val, 45).astype(np.float32)
        val_e = np.random.randn(n_val, embed_dim).astype(np.float32)
        test_p = np.random.randn(n_test, 45).astype(np.float32)
        test_e = np.random.randn(n_test, embed_dim).astype(np.float32)

        metadata = DatasetMetadata(
            param_ranges={name: (lo, hi) for name, lo, hi in CORE_PARAMS},
            param_names=[name for name, _, _ in CORE_PARAMS],
            total_samples=100,
            split_ratio=(0.8, 0.1, 0.1),
            failed_samples=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dataset.h5"
            self.tdg.save_hdf5(
                path, train_p, train_e, val_p, val_e, test_p, test_e, metadata
            )

            assert path.exists()

            loaded = self.tdg.load_hdf5(path)

            np.testing.assert_array_almost_equal(
                loaded["train_params"], train_p
            )
            np.testing.assert_array_almost_equal(
                loaded["train_embeddings"], train_e
            )
            np.testing.assert_array_almost_equal(
                loaded["val_params"], val_p
            )
            np.testing.assert_array_almost_equal(
                loaded["val_embeddings"], val_e
            )
            np.testing.assert_array_almost_equal(
                loaded["test_params"], test_p
            )
            np.testing.assert_array_almost_equal(
                loaded["test_embeddings"], test_e
            )

            assert loaded["param_names"] == metadata.param_names
            assert loaded["param_ranges"].shape == (45, 2)
            assert loaded["generation_log"]["total_samples"] == 100
            assert loaded["generation_log"]["failed_samples"] == 5

    def test_hdf5_structure(self):
        """Verify the HDF5 file has the expected group/dataset structure."""
        metadata = DatasetMetadata(
            param_ranges={name: (lo, hi) for name, lo, hi in CORE_PARAMS},
            param_names=[name for name, _, _ in CORE_PARAMS],
            total_samples=10,
            split_ratio=(0.8, 0.1, 0.1),
            failed_samples=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "structure.h5"
            self.tdg.save_hdf5(
                path,
                np.zeros((8, 45), dtype=np.float32),
                np.zeros((8, 1024), dtype=np.float32),
                np.zeros((1, 45), dtype=np.float32),
                np.zeros((1, 1024), dtype=np.float32),
                np.zeros((1, 45), dtype=np.float32),
                np.zeros((1, 1024), dtype=np.float32),
                metadata,
            )

            with h5py.File(path, "r") as f:
                assert "train" in f
                assert "val" in f
                assert "test" in f
                assert "metadata" in f
                assert "params" in f["train"]
                assert "embeddings" in f["train"]
                assert "param_names" in f["metadata"]
                assert "param_ranges" in f["metadata"]
                assert "generation_log" in f["metadata"]



# ---------------------------------------------------------------------------
# Test generate_dataset with mocked renderer/extractor
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def _make_tdg(self, render_success=True, embed_dim=1024):
        """Create a TrainingDataGenerator with mocked dependencies."""
        from src.preset_parser import PresetParser, VitalPreset

        parser = MagicMock(spec=PresetParser)
        parser.serialize = MagicMock()

        generator = MagicMock()
        base_preset = VitalPreset(
            settings={name: 0.0 for name, _, _ in CORE_PARAMS},
            modulations=[],
            extra={},
        )
        generator.create_base_patch.return_value = base_preset
        generator._parser = parser

        renderer = MagicMock()
        renderer.render_preset.return_value = render_success

        extractor = MagicMock()
        extractor.extract.return_value = np.random.randn(embed_dim).astype(
            np.float32
        )

        return TrainingDataGenerator(generator, renderer, extractor)

    def test_generates_dataset_successfully(self):
        np.random.seed(42)
        tdg = self._make_tdg(render_success=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = tdg.generate_dataset(20, Path(tmpdir))

            assert meta.total_samples == 20
            assert meta.failed_samples == 0
            assert len(meta.param_names) == 45
            assert len(meta.param_ranges) == 45

            # Check HDF5 file was created
            hdf5_path = Path(tmpdir) / "dataset.h5"
            assert hdf5_path.exists()

            # Check metadata JSON was created
            meta_json_path = Path(tmpdir) / "metadata.json"
            assert meta_json_path.exists()

            # Verify HDF5 contents
            loaded = tdg.load_hdf5(hdf5_path)
            total = (
                loaded["train_params"].shape[0]
                + loaded["val_params"].shape[0]
                + loaded["test_params"].shape[0]
            )
            assert total == 20

    def test_handles_all_failures_gracefully(self):
        np.random.seed(42)
        tdg = self._make_tdg(render_success=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = tdg.generate_dataset(5, Path(tmpdir))

            assert meta.total_samples == 0
            assert meta.failed_samples == 5

    def test_handles_partial_failures(self):
        np.random.seed(42)
        tdg = self._make_tdg(render_success=True)

        # Make every other render fail
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return call_count[0] % 2 == 0  # fail odd, succeed even

        tdg._renderer.render_preset.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = tdg.generate_dataset(10, Path(tmpdir))

            # Half should succeed, half should fail
            assert meta.total_samples == 5
            assert meta.failed_samples == 5

            hdf5_path = Path(tmpdir) / "dataset.h5"
            assert hdf5_path.exists()

    def test_no_missing_values_in_output(self):
        """Ensure the final dataset has no NaN or Inf values."""
        np.random.seed(42)
        tdg = self._make_tdg(render_success=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tdg.generate_dataset(20, Path(tmpdir))

            loaded = tdg.load_hdf5(Path(tmpdir) / "dataset.h5")
            for key in [
                "train_params", "train_embeddings",
                "val_params", "val_embeddings",
                "test_params", "test_embeddings",
            ]:
                assert np.all(np.isfinite(loaded[key])), (
                    f"{key} contains non-finite values"
                )

    def test_metadata_contains_all_params(self):
        np.random.seed(42)
        tdg = self._make_tdg(render_success=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            meta = tdg.generate_dataset(10, Path(tmpdir))

            expected_names = [name for name, _, _ in CORE_PARAMS]
            assert meta.param_names == expected_names

            for name, lo, hi in CORE_PARAMS:
                assert name in meta.param_ranges
                assert meta.param_ranges[name] == (lo, hi)
