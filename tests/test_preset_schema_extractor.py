# Feature: preset-corpus-pipeline, Property 7
"""
Property-based tests for PresetSchemaExtractor.

Property 7: Schema key union and type classification
- param_names is a superset of every preset's settings keys
- *_on / *_bypass / legato → binary
- *_destination / *_model / *_style / *_type → categorical

Validates: Requirements 4.1, 4.2
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from src.preset_schema_extractor import PresetSchemaExtractor
from tests.conftest import vital_presets


@given(presets=st.lists(vital_presets(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_7_schema_key_union_and_type_classification(
    presets,
):
    """Property 7: Schema key union and type classification.

    **Validates: Requirements 4.1, 4.2**

    For any list of VitalPreset objects:
    1. CorpusSchema.param_names is a superset of every preset's settings keys.
    2. Any param matching *_on, *_bypass, or exactly 'legato' is classified as binary.
    3. Any param matching *_destination, *_model, *_style, *_type is classified as categorical.
    """
    extractor = PresetSchemaExtractor()
    schema = extractor.extract(presets)

    param_names_set = set(schema.param_names)

    # --- 4.1: param_names is a superset of every preset's settings keys ---
    for preset in presets:
        assert set(preset.settings.keys()).issubset(param_names_set), (
            f"param_names missing keys from a preset: "
            f"{set(preset.settings.keys()) - param_names_set}"
        )

    # --- 4.2: type classification correctness ---
    binary_patterns = [
        re.compile(r".*_on$"),
        re.compile(r".*_bypass$"),
        re.compile(r"^legato$"),
    ]
    categorical_patterns = [
        re.compile(r".*_destination$"),
        re.compile(r".*_model$"),
        re.compile(r".*_style$"),
        re.compile(r".*_type$"),
    ]

    for name in schema.param_names:
        ptype = schema.param_types[name]

        is_binary_match = any(p.match(name) for p in binary_patterns)
        is_categorical_match = any(p.match(name) for p in categorical_patterns)

        if is_binary_match:
            assert ptype == "binary", (
                f"Expected '{name}' to be binary, got '{ptype}'"
            )
        elif is_categorical_match:
            assert ptype == "categorical", (
                f"Expected '{name}' to be categorical, got '{ptype}'"
            )


# Feature: preset-corpus-pipeline, Property 8, 10, 11


@given(presets=st.lists(vital_presets(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_8_schema_statistics_consistent_with_corpus(presets):
    """Property 8: Schema statistics are consistent with corpus data.

    **Validates: Requirements 4.3**

    For any corpus and resulting CorpusSchema, for each param p:
    - corpus_min[p] <= corpus_max[p]
    - 0.0 <= presence_ratio[p] <= 1.0
    """
    extractor = PresetSchemaExtractor()
    schema = extractor.extract(presets)

    for p in schema.param_names:
        assert schema.corpus_min[p] <= schema.corpus_max[p], (
            f"corpus_min[{p}]={schema.corpus_min[p]} > "
            f"corpus_max[{p}]={schema.corpus_max[p]}"
        )
        assert 0.0 <= schema.presence_ratio[p] <= 1.0, (
            f"presence_ratio[{p}]={schema.presence_ratio[p]} not in [0, 1]"
        )


@given(presets=st.lists(vital_presets(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_10_schema_json_round_trip(presets):
    """Property 10: Schema JSON round-trip.

    **Validates: Requirements 4.5**

    For any CorpusSchema, saving to JSON then loading from JSON should produce
    an identical CorpusSchema (same param_names order, same param_types, same
    default_values, same corpus_min, same corpus_max, same presence_ratio,
    same category_values).
    """
    import tempfile
    from pathlib import Path

    extractor = PresetSchemaExtractor()
    original = extractor.extract(presets)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "schema.json"
        extractor.save_schema(original, json_path)
        loaded = extractor.load_schema(json_path)

        assert original.param_names == loaded.param_names, (
            "param_names order differs after round-trip"
        )
        assert original.param_types == loaded.param_types, (
            "param_types differs after round-trip"
        )
        assert original.default_values == loaded.default_values, (
            "default_values differs after round-trip"
        )
        assert original.corpus_min == loaded.corpus_min, (
            "corpus_min differs after round-trip"
        )
        assert original.corpus_max == loaded.corpus_max, (
            "corpus_max differs after round-trip"
        )
        assert original.presence_ratio == loaded.presence_ratio, (
            "presence_ratio differs after round-trip"
        )
        assert original.category_values == loaded.category_values, (
            "category_values differs after round-trip"
        )


@given(presets=st.lists(vital_presets(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_11_schema_ordering_is_deterministic(presets):
    """Property 11: Schema ordering is deterministic.

    **Validates: Requirements 4.7**

    For any list of VitalPreset objects, calling extract() twice with the same
    input should produce CorpusSchema objects with identical param_names ordering.
    """
    extractor = PresetSchemaExtractor()
    schema_1 = extractor.extract(presets)
    schema_2 = extractor.extract(presets)

    assert schema_1.param_names == schema_2.param_names, (
        "param_names ordering differs between two extract() calls on same input"
    )


# Feature: preset-corpus-pipeline, Property 9, Property 25

import json
import logging
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from src.preset_schema_extractor import PresetSchemaExtractor
from tests.conftest import vital_presets


@given(presets=st.lists(vital_presets(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_9_schema_includes_modulation_and_wavetable_fields(presets):
    """Property 9: Schema includes modulation and wavetable fields.

    **Validates: Requirements 4.4, 10.5**

    For any CorpusSchema, param_names should contain:
    - modulation_k_source, modulation_k_destination, modulation_k_amount,
      modulation_k_bypass, modulation_k_bipolar, modulation_k_power,
      modulation_k_stereo for k in 1..64
    - osc_i_wavetable_id, osc_i_wavetable_source_type for i in 1..3
    """
    extractor = PresetSchemaExtractor()
    schema = extractor.extract(presets)

    param_names_set = set(schema.param_names)

    # Check modulation slot fields for k=1..64
    mod_suffixes = ["source", "destination", "amount", "bypass", "bipolar", "power", "stereo"]
    for k in range(1, 65):
        for suffix in mod_suffixes:
            field_name = f"modulation_{k}_{suffix}"
            assert field_name in param_names_set, (
                f"Missing modulation field: {field_name}"
            )

    # Check wavetable fields for i=1..3
    for i in range(1, 4):
        wt_id = f"osc_{i}_wavetable_id"
        wt_src = f"osc_{i}_wavetable_source_type"
        assert wt_id in param_names_set, f"Missing wavetable field: {wt_id}"
        assert wt_src in param_names_set, f"Missing wavetable field: {wt_src}"


@given(presets=st.lists(vital_presets(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_property_25_inventory_cross_reference_completeness(presets):
    """Property 25: Inventory cross-reference completeness.

    **Validates: Requirements 4.6**

    For any corpus and available vital_param_inventory.json, the set of keys
    logged as "not in inventory" should equal exactly the set of corpus keys
    not present in the inventory's continuous_params list.
    """
    inventory_path = Path("vital_param_inventory.json")
    if not inventory_path.exists():
        return  # skip if inventory not available

    # Load inventory to compute expected "not in inventory" keys
    with open(inventory_path, encoding="utf-8") as f:
        inventory_data = json.load(f)
    inventory_set = set(inventory_data.get("continuous_params", []))

    # Compute corpus keys union
    corpus_keys: set[str] = set()
    for p in presets:
        corpus_keys.update(p.settings.keys())

    expected_not_in_inventory = corpus_keys - inventory_set

    # Capture log records using a handler instead of caplog fixture
    log_records: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(record)

    handler = _CaptureHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("src.preset_schema_extractor")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        extractor = PresetSchemaExtractor(inventory_path=inventory_path)
        extractor.extract(presets)

        # Parse logged "not in inventory" keys
        logged_keys: set[str] = set()
        for record in log_records:
            if "Corpus key not in inventory:" in record.getMessage():
                key = record.getMessage().split("Corpus key not in inventory:")[-1].strip()
                logged_keys.add(key)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)

    assert logged_keys == expected_not_in_inventory, (
        f"Logged 'not in inventory' keys mismatch.\n"
        f"  Logged:   {sorted(logged_keys)}\n"
        f"  Expected: {sorted(expected_not_in_inventory)}"
    )
