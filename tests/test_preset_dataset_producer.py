# Feature: preset-corpus-pipeline, Properties 20, 21, 24
"""
Tests for PresetDatasetProducer: split leakage prevention, HDF5 field
completeness, and factory-only mode filtering.

Property 20: Base_Preset_Group_Split prevents leakage
Property 21: HDF5 dataset contains all required fields
Property 24: Factory_Only_Mode filters non-factory presets

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 10.7
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.preset_dataset_producer import (
    EMBEDDING_DIM,
    PresetDatasetProducer,
    ProducerConfig,
)
from src.preset_parser import VitalPreset
from src.preset_schema_extractor import CorpusSchema
from src.wavetable_catalog import WavetableCatalog


# ---------------------------------------------------------------------------
# Property 20: Base_Preset_Group_Split prevents leakage
# ---------------------------------------------------------------------------


@given(
    base_ids=st.lists(
        st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
            min_size=1,
            max_size=20,
        ),
        min_size=1,
        max_size=200,
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_20_base_preset_group_split_prevents_leakage(
    base_ids: list[str],
):
    """Property 20: Base_Preset_Group_Split prevents leakage.

    **Validates: Requirements 9.4**

    For any dataset split produced by PresetDatasetProducer, for any two
    samples sharing the same base_preset_id, both samples should be in the
    same split (train, val, or test).
    """
    # Use dummy paths — _split_by_base_preset doesn't touch the filesystem
    dummy = Path("/tmp/dummy_split_test")
    config = ProducerConfig(
        corpus_dir=dummy / "corpus",
        output_dir=dummy / "output",
        vital_vst_path=dummy / "fake.vst3",
    )
    producer = PresetDatasetProducer(config)
    split_result = producer._split_by_base_preset(base_ids)

    # Build reverse mapping: id -> split_name
    id_to_split: dict[str, str] = {}
    for split_name, ids in split_result.items():
        for bid in ids:
            # Each ID must appear in exactly one split
            assert bid not in id_to_split, (
                f"base_preset_id '{bid}' appears in multiple splits: "
                f"'{id_to_split[bid]}' and '{split_name}'"
            )
            id_to_split[bid] = split_name

    # Every unique ID from the input must be assigned to exactly one split
    unique_input_ids = set(base_ids)
    assert set(id_to_split.keys()) == unique_input_ids

    # The three splits must be disjoint sets
    train_set = set(split_result.get("train", []))
    val_set = set(split_result.get("val", []))
    test_set = set(split_result.get("test", []))
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # Union must cover all unique IDs
    assert train_set | val_set | test_set == unique_input_ids



# ---------------------------------------------------------------------------
# Property 21: HDF5 dataset contains all required fields
# ---------------------------------------------------------------------------

# Expected datasets in each split group
EXPECTED_SPLIT_DATASETS = {
    "param_values",
    "present_mask",
    "embeddings",
    "base_preset_id",
    "variant_id",
    "route_mask_json",
    "sample_id",
    "midi_note",
    "midi_velocity",
    "midi_duration",
    "rms_db",
    "split_local_index",
}

# Expected datasets in the metadata group
EXPECTED_METADATA_DATASETS = {
    "param_names",
    "param_types",
    "param_value_encoding",
    "default_values",
    "corpus_min",
    "corpus_max",
    "presence_ratio",
    "category_values_json",
    "route_edge_schema_json",
    "corpus_schema_json",
    "run_mode",
    "checkpoint_manifest_json",
}


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


def _populate_shard(shard_dir: Path, n_samples: int, D: int) -> None:
    """Write a single shard npz file with n_samples records."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "param_values": np.random.randn(n_samples, D).astype(np.float32),
        "present_mask": np.ones((n_samples, D), dtype=np.uint8),
        "embeddings": np.random.randn(n_samples, EMBEDDING_DIM).astype(np.float32),
        "base_preset_id": np.array([f"preset_{i}" for i in range(n_samples)], dtype=object),
        "variant_id": np.arange(n_samples, dtype=np.int32),
        "route_mask_json": np.array(['{"mask_vector":[1],"masked_edge_names":[]}'] * n_samples, dtype=object),
        "sample_id": np.array([f"sample_{i}" for i in range(n_samples)], dtype=object),
        "midi_note": np.full(n_samples, 60, dtype=np.int32),
        "midi_velocity": np.full(n_samples, 100, dtype=np.int32),
        "midi_duration": np.full(n_samples, 2.0, dtype=np.float32),
        "rms_db": np.full(n_samples, -20.0, dtype=np.float32),
    }
    np.savez(shard_dir / "shard_0000.npz", **arrays)


def test_property_21_hdf5_dataset_contains_all_required_fields(tmp_path: Path):
    """Property 21: HDF5 dataset contains all required fields.

    **Validates: Requirements 9.1, 9.2, 9.3**

    For any HDF5 file produced by PresetDatasetProducer, each split group
    (train/val/test) should contain all required datasets, and the metadata
    group should contain all required fields.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = _make_minimal_schema()
    D = len(schema.param_names)

    # Populate shard files for each split
    for split_name in ("train", "val", "test"):
        _populate_shard(output_dir / "shards" / split_name, n_samples=3, D=D)

    # Create a producer and inject the schema + minimal state
    config = ProducerConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=output_dir,
        vital_vst_path=tmp_path / "fake.vst3",
        run_mode="pilot",
    )
    producer = PresetDatasetProducer(config)
    producer._schema = schema
    producer._split_assignment = {
        "train": ["preset_0"],
        "val": ["preset_1"],
        "test": ["preset_2"],
    }
    producer._checkpoint = None

    # Call _finalize_hdf5 directly
    producer._finalize_hdf5()

    hdf5_path = output_dir / "preset_corpus_dataset.h5"
    assert hdf5_path.exists(), "HDF5 file was not created"

    with h5py.File(hdf5_path, "r") as hf:
        # Verify split groups exist and contain all required datasets
        for split_name in ("train", "val", "test"):
            assert split_name in hf, f"Missing split group: {split_name}"
            grp = hf[split_name]
            actual_datasets = set(grp.keys())
            missing = EXPECTED_SPLIT_DATASETS - actual_datasets
            assert not missing, (
                f"Split '{split_name}' missing datasets: {missing}"
            )

        # Verify metadata group exists and contains all required datasets
        assert "metadata" in hf, "Missing metadata group"
        meta = hf["metadata"]
        actual_meta = set(meta.keys())
        missing_meta = EXPECTED_METADATA_DATASETS - actual_meta
        assert not missing_meta, (
            f"Metadata group missing datasets: {missing_meta}"
        )

        # Verify param_values shape matches schema dimension
        for split_name in ("train", "val", "test"):
            pv = hf[split_name]["param_values"]
            assert pv.shape[1] == D, (
                f"param_values dim mismatch in {split_name}: "
                f"expected {D}, got {pv.shape[1]}"
            )

        # Verify embeddings dimension
        for split_name in ("train", "val", "test"):
            emb = hf[split_name]["embeddings"]
            assert emb.shape[1] == EMBEDDING_DIM, (
                f"embeddings dim mismatch in {split_name}: "
                f"expected {EMBEDDING_DIM}, got {emb.shape[1]}"
            )

        # Verify run_mode is stored correctly
        run_mode_val = meta["run_mode"][()].decode("utf-8") if isinstance(
            meta["run_mode"][()], bytes
        ) else str(meta["run_mode"][()])
        assert run_mode_val == "pilot"


def test_property_21_empty_splits_still_have_all_fields(tmp_path: Path):
    """Property 21 edge case: empty splits still contain all required dataset keys.

    **Validates: Requirements 9.1, 9.2, 9.3**
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = _make_minimal_schema()

    # No shard files — all splits will be empty
    config = ProducerConfig(
        corpus_dir=tmp_path / "corpus",
        output_dir=output_dir,
        vital_vst_path=tmp_path / "fake.vst3",
        run_mode="pilot",
    )
    producer = PresetDatasetProducer(config)
    producer._schema = schema
    producer._split_assignment = {"train": [], "val": [], "test": []}
    producer._checkpoint = None

    producer._finalize_hdf5()

    hdf5_path = output_dir / "preset_corpus_dataset.h5"
    assert hdf5_path.exists()

    with h5py.File(hdf5_path, "r") as hf:
        for split_name in ("train", "val", "test"):
            assert split_name in hf
            grp = hf[split_name]
            actual_datasets = set(grp.keys())
            missing = EXPECTED_SPLIT_DATASETS - actual_datasets
            assert not missing, (
                f"Empty split '{split_name}' missing datasets: {missing}"
            )
            # All datasets should have 0 rows
            assert hf[split_name]["param_values"].shape[0] == 0

        assert "metadata" in hf
        actual_meta = set(hf["metadata"].keys())
        missing_meta = EXPECTED_METADATA_DATASETS - actual_meta
        assert not missing_meta, (
            f"Metadata group missing datasets: {missing_meta}"
        )



# ---------------------------------------------------------------------------
# Property 24: Factory_Only_Mode filters non-factory presets
# ---------------------------------------------------------------------------


def test_property_24_factory_only_mode_filters_non_factory_presets():
    """Property 24: Factory_Only_Mode filters non-factory presets.

    **Validates: Requirements 10.7**

    For any dataset produced with factory_only_mode=True, no sample should
    have custom/unknown wavetable source type. Only factory presets remain.
    """
    catalog = WavetableCatalog()

    # Create presets with different wavetable types
    # Preset with factory wavetables only (Init is built-in factory)
    factory_preset = VitalPreset(
        settings={"osc_1_on": 1, "osc_1_level": 0.5},
        modulations=[],
        extra={"wavetables": [{"name": "Init"}, {"name": "Init"}]},
    )

    # Preset with an embedded wavetable (name not in filesystem or built-in)
    custom_preset = VitalPreset(
        settings={"osc_1_on": 1, "osc_1_level": 0.7},
        modulations=[],
        extra={"wavetables": [{"name": "My Custom Wave"}]},
    )

    # Preset with an unnamed wavetable (embedded)
    unknown_preset = VitalPreset(
        settings={"osc_1_on": 1, "osc_1_level": 0.3},
        modulations=[],
        extra={"wavetables": [{"name": ""}]},
    )

    # Preset with mixed: one factory, one embedded
    mixed_preset = VitalPreset(
        settings={"osc_1_on": 1, "osc_1_level": 0.6},
        modulations=[],
        extra={"wavetables": [{"name": "Init"}, {"name": "User Pad"}]},
    )

    # Preset with no wavetables at all (should pass factory filter)
    no_wt_preset = VitalPreset(
        settings={"osc_1_on": 1, "osc_1_level": 0.4},
        modulations=[],
        extra={},
    )

    all_presets = [
        ("factory_only", factory_preset),
        ("custom_wt", custom_preset),
        ("unknown_wt", unknown_preset),
        ("mixed_wt", mixed_preset),
        ("no_wt", no_wt_preset),
    ]

    # Simulate the factory_only_mode filter logic from PresetDatasetProducer.produce()
    filtered: list[tuple[str, VitalPreset]] = []
    filter_reasons: dict[str, int] = {}

    for base_id, preset in all_presets:
        wt_map = catalog.resolve_oscillator_wavetables(preset)
        has_non_factory = any(
            e is not None and e.source_type != "factory"
            for e in wt_map.values()
        )
        if has_non_factory:
            filter_reasons["non_factory_wavetable"] = (
                filter_reasons.get("non_factory_wavetable", 0) + 1
            )
        else:
            filtered.append((base_id, preset))

    # Verify: custom, unknown, and mixed presets should be filtered out
    filtered_ids = {bid for bid, _ in filtered}
    assert "custom_wt" not in filtered_ids, "Custom wavetable preset should be filtered"
    assert "unknown_wt" not in filtered_ids, "Unknown wavetable preset should be filtered"
    assert "mixed_wt" not in filtered_ids, "Mixed wavetable preset should be filtered"

    # Verify: factory-only and no-wavetable presets should remain
    assert "factory_only" in filtered_ids, "Factory-only preset should remain"
    assert "no_wt" in filtered_ids, "No-wavetable preset should remain"

    # Verify: all remaining presets have only factory or None wavetable entries
    for base_id, preset in filtered:
        wt_map = catalog.resolve_oscillator_wavetables(preset)
        for osc_key, entry in wt_map.items():
            if entry is not None:
                assert entry.source_type == "factory", (
                    f"Preset '{base_id}' osc '{osc_key}' has non-factory "
                    f"wavetable source_type='{entry.source_type}' after filtering"
                )

    # Verify filter_reasons count (custom, unknown/embedded, mixed = 3 filtered)
    assert filter_reasons.get("non_factory_wavetable", 0) == 3, (
        f"Expected 3 filtered presets, got {filter_reasons.get('non_factory_wavetable', 0)}"
    )
