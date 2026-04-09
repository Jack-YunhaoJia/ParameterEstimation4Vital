# Feature: preset-corpus-pipeline, Property 22
"""
Property-based tests for WavetableCatalog resolution and classification.

Property 22: Wavetable resolution and classification correctness

Validates: Requirements 10.1, 10.3, 10.4
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck

from src.preset_parser import VitalPreset
from src.wavetable_catalog import WavetableCatalog, WavetableEntry, FACTORY_WAVETABLE_NAMES
from tests.conftest import vital_presets

VALID_SOURCE_TYPES = {"factory", "third_party", "embedded"}


@given(preset=vital_presets())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_22_wavetable_resolution_and_classification_correctness(
    preset: VitalPreset,
):
    """Property 22: Wavetable resolution and classification correctness.

    **Validates: Requirements 10.1, 10.3, 10.4**

    For any VitalPreset processed by WavetableCatalog.resolve_oscillator_wavetables():
    if the wavetable binding is resolvable, the returned entry should have a valid
    wavetable_id and source_type in {"factory", "custom", "unknown"}; if the binding
    is unresolvable, the source_type should be "unknown".
    """
    catalog = WavetableCatalog()
    result = catalog.resolve_oscillator_wavetables(preset)

    # Result must contain exactly the three oscillator keys
    assert set(result.keys()) == {"osc_1", "osc_2", "osc_3"}

    wavetables = preset.extra.get("wavetables", [])

    for idx, osc_key in enumerate(("osc_1", "osc_2", "osc_3")):
        entry = result[osc_key]

        if idx >= len(wavetables):
            # No wavetable data for this oscillator slot
            assert entry is None
        elif not isinstance(wavetables[idx], dict):
            # Non-dict wavetable data is unresolvable
            assert entry is None
        else:
            # Wavetable data exists and is a dict — entry must be returned
            wt_data = wavetables[idx]
            assert isinstance(entry, WavetableEntry)

            # wavetable_id must be a non-negative integer
            assert isinstance(entry.wavetable_id, int)
            assert entry.wavetable_id >= 0

            # source_type must be one of the valid types
            assert entry.source_type in VALID_SOURCE_TYPES

            # Classification correctness check
            name = wt_data.get("name")
            if not isinstance(name, str) or not name.strip():
                # No valid name → embedded
                assert entry.source_type == "embedded"
            elif name in FACTORY_WAVETABLE_NAMES:
                assert entry.source_type == "factory"
            else:
                # Without filesystem scanning, non-factory names → embedded
                assert entry.source_type in {"third_party", "embedded"}
