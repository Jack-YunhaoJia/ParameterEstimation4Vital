# Feature: preset-corpus-pipeline, Task 14.10
"""
Tests for PresetDatasetProducer interrupt/resume drill.

Validates checkpoint/resume logic at the unit level:
- Checkpoint round-trip: save → load produces identical state
- Resume skips completed presets: completed_base_preset_ids are respected
- Split assignment consistency: preserved across resume
- No duplicate sample_id, base_preset_id in same split, or split_local_index
- Final HDF5 can finalize correctly after resume and splits don't leak

Validates: Requirements 9.4, 9.5, 9.6
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.preset_dataset_producer import (
    EMBEDDING_DIM,
    PresetDatasetProducer,
    ProducerConfig,
    ProductionCheckpoint,
)
from src.preset_schema_extractor import CorpusSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, resume: bool = True) -> ProducerConfig:
    """Create a minimal ProducerConfig pointing at tmp_path."""
    return ProducerConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=tmp_path / "output",
        vital_vst_path=tmp_path / "fake.vst3",
        run_mode="pilot",
        resume=resume,
        seed=42,
    )


def _make_minimal_schema() -> CorpusSchema:
    """Create a minimal CorpusSchema with 3 parameters for testing."""
    param_names = ["osc_1_level", "osc_1_on", "osc_1_destination"]
    return CorpusSchema(
        param_names=param_names,
        param_types={
            "osc_1_level": "continuous",
            "osc_1_on": "binary",
            "osc_1_destination": "categorical",
        },
        param_value_encoding={
            "osc_1_level": "identity",
            "osc_1_on": "identity",
            "osc_1_destination": "category_id",
        },
        default_values={
            "osc_1_level": 0.5,
            "osc_1_on": 1.0,
            "osc_1_destination": 0.0,
        },
        corpus_min={
            "osc_1_level": 0.0,
            "osc_1_on": 0.0,
            "osc_1_destination": 0.0,
        },
        corpus_max={
            "osc_1_level": 1.0,
            "osc_1_on": 1.0,
            "osc_1_destination": 3.0,
        },
        presence_ratio={
            "osc_1_level": 1.0,
            "osc_1_on": 1.0,
            "osc_1_destination": 0.8,
        },
        category_values={
            "osc_1_destination": ["0", "1", "2", "3"],
        },
    )


def _write_checkpoint(output_dir: Path, checkpoint: ProductionCheckpoint) -> None:
    """Write a checkpoint JSON file to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_path = output_dir / "checkpoint.json"
    data = {
        "run_mode": checkpoint.run_mode,
        "selected_base_preset_ids": checkpoint.selected_base_preset_ids,
        "completed_base_preset_ids": checkpoint.completed_base_preset_ids,
        "split_assignment": checkpoint.split_assignment,
        "schema_path": checkpoint.schema_path,
        "shard_manifest_path": checkpoint.shard_manifest_path,
        "summary_so_far": checkpoint.summary_so_far,
        "last_completed_stage": checkpoint.last_completed_stage,
    }
    cp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _populate_shard(shard_dir: Path, n_samples: int, D: int, id_prefix: str = "preset") -> None:
    """Write a single shard npz file with n_samples records and unique IDs."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = len(list(shard_dir.glob("shard_*.npz")))
    arrays = {
        "param_values": np.random.randn(n_samples, D).astype(np.float32),
        "present_mask": np.ones((n_samples, D), dtype=np.uint8),
        "embeddings": np.random.randn(n_samples, EMBEDDING_DIM).astype(np.float32),
        "base_preset_id": np.array(
            [f"{id_prefix}_{i}" for i in range(n_samples)], dtype=object
        ),
        "variant_id": np.arange(n_samples, dtype=np.int32),
        "route_mask_json": np.array(
            ['{"mask_vector":[1],"masked_edge_names":[]}'] * n_samples, dtype=object
        ),
        "sample_id": np.array(
            [f"sample_{id_prefix}_{i}" for i in range(n_samples)], dtype=object
        ),
        "midi_note": np.full(n_samples, 60, dtype=np.int32),
        "midi_velocity": np.full(n_samples, 100, dtype=np.int32),
        "midi_duration": np.full(n_samples, 2.0, dtype=np.float32),
        "rms_db": np.full(n_samples, -20.0, dtype=np.float32),
    }
    np.savez(shard_dir / f"shard_{shard_idx:04d}.npz", **arrays)


# ---------------------------------------------------------------------------
# Test: Checkpoint round-trip (save → load produces identical state)
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    """Validates: Requirements 9.5"""

    def test_checkpoint_save_then_load_produces_identical_state(self, tmp_path: Path):
        """Save a checkpoint, then load it via _load_or_init_checkpoint.
        The loaded checkpoint should have identical fields."""
        config = _make_config(tmp_path, resume=True)
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        all_ids = ["alpha", "bravo", "charlie", "delta", "echo"]
        completed = ["alpha", "bravo", "charlie"]
        split_map = {
            "alpha": "train", "bravo": "train", "charlie": "val",
            "delta": "val", "echo": "test",
        }

        original_cp = ProductionCheckpoint(
            run_mode="pilot",
            selected_base_preset_ids=all_ids,
            completed_base_preset_ids=completed,
            split_assignment=split_map,
            schema_path=str(output_dir / "corpus_schema.json"),
            shard_manifest_path=str(output_dir / "shard_manifest.json"),
            summary_so_far={"total_variants": 42},
            last_completed_stage="processing",
        )

        # Save via the producer's _save_checkpoint
        producer = PresetDatasetProducer(config)
        producer._checkpoint = original_cp
        producer._save_checkpoint()

        # Load via _load_or_init_checkpoint
        loaded_cp = producer._load_or_init_checkpoint(
            base_ids=all_ids,
            id_to_split=split_map,
            schema_path=str(output_dir / "corpus_schema.json"),
        )

        assert loaded_cp.run_mode == original_cp.run_mode
        assert loaded_cp.selected_base_preset_ids == original_cp.selected_base_preset_ids
        assert loaded_cp.completed_base_preset_ids == original_cp.completed_base_preset_ids
        assert loaded_cp.split_assignment == original_cp.split_assignment
        assert loaded_cp.schema_path == original_cp.schema_path

    def test_checkpoint_file_missing_creates_fresh(self, tmp_path: Path):
        """When no checkpoint file exists, _load_or_init_checkpoint creates a fresh one."""
        config = _make_config(tmp_path, resume=True)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        producer = PresetDatasetProducer(config)
        all_ids = ["a", "b", "c"]
        split_map = {"a": "train", "b": "val", "c": "test"}

        cp = producer._load_or_init_checkpoint(
            base_ids=all_ids,
            id_to_split=split_map,
            schema_path="/tmp/schema.json",
        )

        assert cp.completed_base_preset_ids == []
        assert cp.selected_base_preset_ids == all_ids
        assert cp.run_mode == "pilot"

    def test_resume_false_ignores_existing_checkpoint(self, tmp_path: Path):
        """When resume=False, existing checkpoint file is ignored."""
        config = _make_config(tmp_path, resume=False)
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write a checkpoint with some completed IDs
        _write_checkpoint(output_dir, ProductionCheckpoint(
            run_mode="pilot",
            selected_base_preset_ids=["x", "y", "z"],
            completed_base_preset_ids=["x", "y"],
            split_assignment={"x": "train", "y": "val", "z": "test"},
            schema_path="/tmp/schema.json",
            shard_manifest_path="",
            summary_so_far={},
            last_completed_stage="processing",
        ))

        producer = PresetDatasetProducer(config)
        cp = producer._load_or_init_checkpoint(
            base_ids=["x", "y", "z"],
            id_to_split={"x": "train", "y": "val", "z": "test"},
            schema_path="/tmp/schema.json",
        )

        # Should start fresh since resume=False
        assert cp.completed_base_preset_ids == []


# ---------------------------------------------------------------------------
# Test: Resume skips completed presets
# ---------------------------------------------------------------------------

class TestResumeSkipsCompleted:
    """Validates: Requirements 9.5, 9.6"""

    def test_completed_base_preset_ids_loaded_correctly(self, tmp_path: Path):
        """A checkpoint with 3/5 completed presets should load those 3 as completed."""
        config = _make_config(tmp_path, resume=True)
        output_dir = config.output_dir

        all_ids = ["p1", "p2", "p3", "p4", "p5"]
        completed = ["p1", "p2", "p3"]
        split_map = {pid: "train" for pid in all_ids}

        _write_checkpoint(output_dir, ProductionCheckpoint(
            run_mode="pilot",
            selected_base_preset_ids=all_ids,
            completed_base_preset_ids=completed,
            split_assignment=split_map,
            schema_path="schema.json",
            shard_manifest_path="",
            summary_so_far={},
            last_completed_stage="processing",
        ))

        producer = PresetDatasetProducer(config)
        cp = producer._load_or_init_checkpoint(
            base_ids=all_ids,
            id_to_split=split_map,
            schema_path="schema.json",
        )

        assert set(cp.completed_base_preset_ids) == {"p1", "p2", "p3"}
        # Remaining presets to process
        remaining = [pid for pid in all_ids if pid not in set(cp.completed_base_preset_ids)]
        assert remaining == ["p4", "p5"]

    def test_no_duplicate_production_after_resume(self, tmp_path: Path):
        """Simulating the produce() loop logic: completed presets are skipped,
        no base_preset_id is processed twice."""
        config = _make_config(tmp_path, resume=True)
        output_dir = config.output_dir

        all_ids = ["a", "b", "c", "d", "e"]
        completed = ["a", "b"]
        split_map = {"a": "train", "b": "train", "c": "val", "d": "val", "e": "test"}

        _write_checkpoint(output_dir, ProductionCheckpoint(
            run_mode="pilot",
            selected_base_preset_ids=all_ids,
            completed_base_preset_ids=completed,
            split_assignment=split_map,
            schema_path="schema.json",
            shard_manifest_path="",
            summary_so_far={},
            last_completed_stage="processing",
        ))

        producer = PresetDatasetProducer(config)
        cp = producer._load_or_init_checkpoint(
            base_ids=all_ids,
            id_to_split=split_map,
            schema_path="schema.json",
        )

        # Simulate the produce() loop: skip completed, track processed
        completed_set = set(cp.completed_base_preset_ids)
        processed_ids: list[str] = []

        for base_id in all_ids:
            if base_id in completed_set:
                continue
            processed_ids.append(base_id)
            cp.completed_base_preset_ids.append(base_id)

        # Only the remaining presets should have been processed
        assert processed_ids == ["c", "d", "e"]
        # No duplicates in the final completed list
        assert len(cp.completed_base_preset_ids) == len(set(cp.completed_base_preset_ids))
        assert set(cp.completed_base_preset_ids) == set(all_ids)


# ---------------------------------------------------------------------------
# Test: Split assignment consistency across resume
# ---------------------------------------------------------------------------

class TestSplitAssignmentConsistency:
    """Validates: Requirements 9.4, 9.5"""

    def test_split_assignment_preserved_across_resume(self, tmp_path: Path):
        """Split assignments saved in checkpoint should be identical after load."""
        config = _make_config(tmp_path, resume=True)
        output_dir = config.output_dir

        all_ids = ["x1", "x2", "x3", "x4", "x5"]
        split_map = {"x1": "train", "x2": "train", "x3": "val", "x4": "test", "x5": "test"}

        _write_checkpoint(output_dir, ProductionCheckpoint(
            run_mode="pilot",
            selected_base_preset_ids=all_ids,
            completed_base_preset_ids=["x1"],
            split_assignment=split_map,
            schema_path="schema.json",
            shard_manifest_path="",
            summary_so_far={},
            last_completed_stage="processing",
        ))

        producer = PresetDatasetProducer(config)
        cp = producer._load_or_init_checkpoint(
            base_ids=all_ids,
            id_to_split=split_map,
            schema_path="schema.json",
        )

        # The split assignment from the checkpoint should match exactly
        assert cp.split_assignment == split_map

    def test_deterministic_split_produces_same_result(self, tmp_path: Path):
        """_split_by_base_preset with the same seed produces identical splits."""
        config = _make_config(tmp_path)
        producer = PresetDatasetProducer(config)

        ids = [f"preset_{i}" for i in range(20)]
        split1 = producer._split_by_base_preset(ids)
        split2 = producer._split_by_base_preset(ids)

        assert split1 == split2


# ---------------------------------------------------------------------------
# Test: No duplicate IDs in finalized HDF5 after resume
# ---------------------------------------------------------------------------

class TestNoDuplicateIdsAfterResume:
    """Validates: Requirements 9.4, 9.5, 9.6"""

    def test_sample_id_unique_across_all_splits(self, tmp_path: Path):
        """After simulating interrupt + resume with separate shards,
        the finalized HDF5 should have globally unique sample_ids."""
        output_dir = tmp_path / "output"
        schema = _make_minimal_schema()
        D = len(schema.param_names)

        # Simulate phase 1 (before interrupt): write shard for train
        train_shard_dir = output_dir / "shards" / "train"
        train_shard_dir.mkdir(parents=True, exist_ok=True)
        phase1_arrays = {
            "param_values": np.random.randn(3, D).astype(np.float32),
            "present_mask": np.ones((3, D), dtype=np.uint8),
            "embeddings": np.random.randn(3, EMBEDDING_DIM).astype(np.float32),
            "base_preset_id": np.array(["p1", "p1", "p2"], dtype=object),
            "variant_id": np.array([0, 1, 0], dtype=np.int32),
            "route_mask_json": np.array(
                ['{"mask_vector":[1],"masked_edge_names":[]}'] * 3, dtype=object
            ),
            "sample_id": np.array(["sid_001", "sid_002", "sid_003"], dtype=object),
            "midi_note": np.full(3, 60, dtype=np.int32),
            "midi_velocity": np.full(3, 100, dtype=np.int32),
            "midi_duration": np.full(3, 2.0, dtype=np.float32),
            "rms_db": np.full(3, -20.0, dtype=np.float32),
        }
        np.savez(train_shard_dir / "shard_0000.npz", **phase1_arrays)

        # Simulate phase 2 (after resume): write another shard for train
        phase2_arrays = {
            "param_values": np.random.randn(2, D).astype(np.float32),
            "present_mask": np.ones((2, D), dtype=np.uint8),
            "embeddings": np.random.randn(2, EMBEDDING_DIM).astype(np.float32),
            "base_preset_id": np.array(["p3", "p3"], dtype=object),
            "variant_id": np.array([0, 1], dtype=np.int32),
            "route_mask_json": np.array(
                ['{"mask_vector":[1],"masked_edge_names":[]}'] * 2, dtype=object
            ),
            "sample_id": np.array(["sid_004", "sid_005"], dtype=object),
            "midi_note": np.full(2, 60, dtype=np.int32),
            "midi_velocity": np.full(2, 100, dtype=np.int32),
            "midi_duration": np.full(2, 2.0, dtype=np.float32),
            "rms_db": np.full(2, -20.0, dtype=np.float32),
        }
        np.savez(train_shard_dir / "shard_0001.npz", **phase2_arrays)

        # Write minimal shards for val and test
        _populate_shard(output_dir / "shards" / "val", 2, D, id_prefix="val")
        _populate_shard(output_dir / "shards" / "test", 2, D, id_prefix="test")

        # Finalize
        config = _make_config(tmp_path)
        config.output_dir = output_dir
        producer = PresetDatasetProducer(config)
        producer._schema = schema
        producer._split_assignment = {
            "train": ["p1", "p2", "p3"],
            "val": ["val_0", "val_1"],
            "test": ["test_0", "test_1"],
        }
        producer._checkpoint = None
        producer._finalize_hdf5()

        hdf5_path = output_dir / "preset_corpus_dataset.h5"
        assert hdf5_path.exists()

        # Verify uniqueness
        all_sample_ids: list[str] = []
        with h5py.File(hdf5_path, "r") as hf:
            for split_name in ("train", "val", "test"):
                if split_name in hf:
                    sids = hf[split_name]["sample_id"][:]
                    decoded = [
                        s.decode("utf-8") if isinstance(s, bytes) else str(s)
                        for s in sids
                    ]
                    all_sample_ids.extend(decoded)

        assert len(all_sample_ids) == len(set(all_sample_ids)), (
            f"Duplicate sample_ids found: {len(all_sample_ids)} total, "
            f"{len(set(all_sample_ids))} unique"
        )

    def test_split_local_index_unique_within_each_split(self, tmp_path: Path):
        """split_local_index should be unique within each split (0..N-1)."""
        output_dir = tmp_path / "output"
        schema = _make_minimal_schema()
        D = len(schema.param_names)

        # Write two shards for train to simulate resume
        train_dir = output_dir / "shards" / "train"
        _populate_shard(train_dir, 3, D, id_prefix="batch1")
        _populate_shard(train_dir, 4, D, id_prefix="batch2")

        # Minimal val/test
        _populate_shard(output_dir / "shards" / "val", 2, D, id_prefix="val")
        _populate_shard(output_dir / "shards" / "test", 1, D, id_prefix="test")

        config = _make_config(tmp_path)
        config.output_dir = output_dir
        producer = PresetDatasetProducer(config)
        producer._schema = schema
        producer._split_assignment = {"train": [], "val": [], "test": []}
        producer._checkpoint = None
        producer._finalize_hdf5()

        hdf5_path = output_dir / "preset_corpus_dataset.h5"
        with h5py.File(hdf5_path, "r") as hf:
            for split_name in ("train", "val", "test"):
                if split_name in hf:
                    idx = hf[split_name]["split_local_index"][:]
                    n = len(idx)
                    if n > 0:
                        expected = np.arange(n, dtype=np.int32)
                        np.testing.assert_array_equal(
                            idx, expected,
                            err_msg=f"split_local_index in '{split_name}' is not 0..{n-1}",
                        )


# ---------------------------------------------------------------------------
# Test: HDF5 finalize after resume — splits don't leak
# ---------------------------------------------------------------------------

class TestHdf5FinalizeAfterResume:
    """Validates: Requirements 9.4, 9.5, 9.6"""

    def test_finalize_after_resume_produces_valid_hdf5(self, tmp_path: Path):
        """Simulate a two-phase production (interrupt + resume) and verify
        the finalized HDF5 has correct structure and no split leakage."""
        output_dir = tmp_path / "output"
        schema = _make_minimal_schema()
        D = len(schema.param_names)

        # Phase 1 shards: train gets p1, p2; val gets p3
        train_dir = output_dir / "shards" / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            train_dir / "shard_0000.npz",
            param_values=np.random.randn(2, D).astype(np.float32),
            present_mask=np.ones((2, D), dtype=np.uint8),
            embeddings=np.random.randn(2, EMBEDDING_DIM).astype(np.float32),
            base_preset_id=np.array(["p1", "p2"], dtype=object),
            variant_id=np.array([0, 0], dtype=np.int32),
            route_mask_json=np.array(['{}', '{}'], dtype=object),
            sample_id=np.array(["s1", "s2"], dtype=object),
            midi_note=np.array([60, 60], dtype=np.int32),
            midi_velocity=np.array([100, 100], dtype=np.int32),
            midi_duration=np.array([2.0, 2.0], dtype=np.float32),
            rms_db=np.array([-20.0, -20.0], dtype=np.float32),
        )

        val_dir = output_dir / "shards" / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            val_dir / "shard_0000.npz",
            param_values=np.random.randn(1, D).astype(np.float32),
            present_mask=np.ones((1, D), dtype=np.uint8),
            embeddings=np.random.randn(1, EMBEDDING_DIM).astype(np.float32),
            base_preset_id=np.array(["p3"], dtype=object),
            variant_id=np.array([0], dtype=np.int32),
            route_mask_json=np.array(['{}'], dtype=object),
            sample_id=np.array(["s3"], dtype=object),
            midi_note=np.array([60], dtype=np.int32),
            midi_velocity=np.array([100], dtype=np.int32),
            midi_duration=np.array([2.0], dtype=np.float32),
            rms_db=np.array([-20.0], dtype=np.float32),
        )

        # Phase 2 shards (after resume): train gets p4; test gets p5
        np.savez(
            train_dir / "shard_0001.npz",
            param_values=np.random.randn(1, D).astype(np.float32),
            present_mask=np.ones((1, D), dtype=np.uint8),
            embeddings=np.random.randn(1, EMBEDDING_DIM).astype(np.float32),
            base_preset_id=np.array(["p4"], dtype=object),
            variant_id=np.array([0], dtype=np.int32),
            route_mask_json=np.array(['{}'], dtype=object),
            sample_id=np.array(["s4"], dtype=object),
            midi_note=np.array([60], dtype=np.int32),
            midi_velocity=np.array([100], dtype=np.int32),
            midi_duration=np.array([2.0], dtype=np.float32),
            rms_db=np.array([-20.0], dtype=np.float32),
        )

        test_dir = output_dir / "shards" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            test_dir / "shard_0000.npz",
            param_values=np.random.randn(1, D).astype(np.float32),
            present_mask=np.ones((1, D), dtype=np.uint8),
            embeddings=np.random.randn(1, EMBEDDING_DIM).astype(np.float32),
            base_preset_id=np.array(["p5"], dtype=object),
            variant_id=np.array([0], dtype=np.int32),
            route_mask_json=np.array(['{}'], dtype=object),
            sample_id=np.array(["s5"], dtype=object),
            midi_note=np.array([60], dtype=np.int32),
            midi_velocity=np.array([100], dtype=np.int32),
            midi_duration=np.array([2.0], dtype=np.float32),
            rms_db=np.array([-20.0], dtype=np.float32),
        )

        # Finalize
        config = _make_config(tmp_path)
        config.output_dir = output_dir
        producer = PresetDatasetProducer(config)
        producer._schema = schema
        producer._split_assignment = {
            "train": ["p1", "p2", "p4"],
            "val": ["p3"],
            "test": ["p5"],
        }
        producer._checkpoint = ProductionCheckpoint(
            run_mode="pilot",
            selected_base_preset_ids=["p1", "p2", "p3", "p4", "p5"],
            completed_base_preset_ids=["p1", "p2", "p3", "p4", "p5"],
            split_assignment={"p1": "train", "p2": "train", "p3": "val", "p4": "train", "p5": "test"},
            schema_path="schema.json",
            shard_manifest_path="",
            summary_so_far={},
            last_completed_stage="done",
        )
        producer._finalize_hdf5()

        hdf5_path = output_dir / "preset_corpus_dataset.h5"
        assert hdf5_path.exists()

        with h5py.File(hdf5_path, "r") as hf:
            # All three splits should exist
            for split_name in ("train", "val", "test"):
                assert split_name in hf, f"Missing split: {split_name}"

            # Train should have 3 samples (p1, p2, p4)
            assert hf["train"]["param_values"].shape[0] == 3
            # Val should have 1 sample (p3)
            assert hf["val"]["param_values"].shape[0] == 1
            # Test should have 1 sample (p5)
            assert hf["test"]["param_values"].shape[0] == 1

            # Verify no base_preset_id appears in multiple splits (no leakage)
            all_bids_by_split: dict[str, set[str]] = {}
            for split_name in ("train", "val", "test"):
                bids = hf[split_name]["base_preset_id"][:]
                decoded = {
                    b.decode("utf-8") if isinstance(b, bytes) else str(b)
                    for b in bids
                }
                all_bids_by_split[split_name] = decoded

            train_bids = all_bids_by_split["train"]
            val_bids = all_bids_by_split["val"]
            test_bids = all_bids_by_split["test"]

            assert train_bids.isdisjoint(val_bids), (
                f"Leakage between train and val: {train_bids & val_bids}"
            )
            assert train_bids.isdisjoint(test_bids), (
                f"Leakage between train and test: {train_bids & test_bids}"
            )
            assert val_bids.isdisjoint(test_bids), (
                f"Leakage between val and test: {val_bids & test_bids}"
            )

            # Verify sample_ids are globally unique
            all_sids: list[str] = []
            for split_name in ("train", "val", "test"):
                sids = hf[split_name]["sample_id"][:]
                all_sids.extend(
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in sids
                )
            assert len(all_sids) == len(set(all_sids)), "Duplicate sample_ids across splits"

            # Verify metadata group exists
            assert "metadata" in hf

    def test_corrupted_checkpoint_prevents_resume(self, tmp_path: Path):
        """A corrupted checkpoint file should cause a fresh start (not crash)."""
        config = _make_config(tmp_path, resume=True)
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write corrupted checkpoint
        cp_path = output_dir / "checkpoint.json"
        cp_path.write_text("{ this is not valid json !!!", encoding="utf-8")

        producer = PresetDatasetProducer(config)
        cp = producer._load_or_init_checkpoint(
            base_ids=["a", "b"],
            id_to_split={"a": "train", "b": "test"},
            schema_path="schema.json",
        )

        # Should fall back to fresh checkpoint
        assert cp.completed_base_preset_ids == []
        assert cp.selected_base_preset_ids == ["a", "b"]
