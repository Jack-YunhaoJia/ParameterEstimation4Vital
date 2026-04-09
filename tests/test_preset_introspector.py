# Feature: preset-corpus-pipeline, Property 2
"""
Property-based tests for PresetIntrospector.

Property 2: Introspection partitions settings and counts are consistent.
Validates: Requirements 2.1, 2.2, 2.3
"""

from __future__ import annotations

from hypothesis import given, settings

from src.preset_introspector import PresetIntrospector
from src.preset_parser import PresetParser, VitalPreset

from tests.conftest import vital_presets


@given(preset=vital_presets())
@settings(max_examples=100)
def test_introspection_partitions_and_counts_consistent(preset):
    """**Validates: Requirements 2.1, 2.2, 2.3**

    For any VitalPreset, the IntrospectionReport should satisfy:
    (a) applied_count + skipped_count + unsupported_count == total_settings_count
    (b) the three param lists are pairwise disjoint and their union equals the settings keys
    (c) active_modulation_count equals the number of modulation slots where both
        source and destination are non-empty strings
    (d) wavetable_count equals len(preset.extra.get("wavetables", []))
    """
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    report = introspector.introspect(preset, preset_path="<generated>")

    # (a) Count partition: applied + skipped + unsupported == total
    assert report.applied_count + report.skipped_count + report.unsupported_count == report.total_settings_count
    assert report.total_settings_count == len(preset.settings)

    # (b) Param lists are pairwise disjoint and their union equals settings keys
    applied_set = set(report.applied_params)
    skipped_set = set(report.skipped_params)
    unsupported_set = set(report.unsupported_params)

    # Pairwise disjoint
    assert applied_set.isdisjoint(skipped_set), "applied and skipped overlap"
    assert applied_set.isdisjoint(unsupported_set), "applied and unsupported overlap"
    assert skipped_set.isdisjoint(unsupported_set), "skipped and unsupported overlap"

    # Union equals settings keys
    union = applied_set | skipped_set | unsupported_set
    assert union == set(preset.settings.keys()), (
        f"Union of param lists != settings keys. "
        f"Missing: {set(preset.settings.keys()) - union}, "
        f"Extra: {union - set(preset.settings.keys())}"
    )

    # List lengths match counts
    assert len(report.applied_params) == report.applied_count
    assert len(report.skipped_params) == report.skipped_count
    assert len(report.unsupported_params) == report.unsupported_count

    # (c) active_modulation_count equals slots where both source and destination are non-empty
    expected_active_mod_count = sum(
        1
        for mod in preset.modulations
        if mod.get("source", "") and mod.get("destination", "")
    )
    assert report.active_modulation_count == expected_active_mod_count

    # (d) wavetable_count equals len(preset.extra.get("wavetables", []))
    expected_wt_count = len(preset.extra.get("wavetables", []))
    assert report.wavetable_count == expected_wt_count


# ---------------------------------------------------------------------------
# Unit tests for PresetIntrospector
# Validates: Requirements 2.1, 2.4
# ---------------------------------------------------------------------------


def test_empty_settings_preset():
    """VitalPreset with empty settings dict should produce all-zero counts.

    Validates: Requirements 2.1, 2.4
    """
    preset = VitalPreset(settings={}, modulations=[], extra={})
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    report = introspector.introspect(preset, preset_path="<empty>")

    assert report.total_settings_count == 0
    assert report.applied_count == 0
    assert report.skipped_count == 0
    assert report.unsupported_count == 0
    assert report.applied_params == []
    assert report.skipped_params == []
    assert report.unsupported_params == []
    assert report.active_modulation_count == 0
    assert report.wavetable_count == 0


def test_all_applied_preset():
    """VitalPreset with only known int/float params that have pedalboard mappings.

    All params should be classified as applied; skipped and unsupported should be 0.

    Validates: Requirements 2.1, 2.4
    """
    settings = {
        "osc_1_level": 0.5,
        "filter_1_cutoff": 60.0,
        "volume": 0.8,
        "osc_1_tune": 0.0,
        "reverb_decay_time": 3.5,
    }
    preset = VitalPreset(settings=settings, modulations=[], extra={})
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    report = introspector.introspect(preset, preset_path="<all_applied>")

    assert report.total_settings_count == 5
    assert report.applied_count == 5
    assert report.skipped_count == 0
    assert report.unsupported_count == 0
    assert set(report.applied_params) == set(settings.keys())


def test_unsupported_non_numeric_values():
    """VitalPreset with string/list values in settings should classify them as unsupported.

    Validates: Requirements 2.1, 2.4
    """
    settings = {
        "osc_1_level": 0.5,          # int/float with mapping → applied
        "some_string_param": "hello", # string → unsupported
        "some_list_param": [1, 2, 3], # list → unsupported
    }
    preset = VitalPreset(settings=settings, modulations=[], extra={})
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    report = introspector.introspect(preset, preset_path="<unsupported>")

    assert report.total_settings_count == 3
    assert report.applied_count == 1
    assert report.unsupported_count == 2
    assert "osc_1_level" in report.applied_params
    assert "some_string_param" in report.unsupported_params
    assert "some_list_param" in report.unsupported_params


def test_empty_modulations():
    """VitalPreset with modulations=[] should have active_modulation_count==0.

    Validates: Requirements 2.1, 2.4
    """
    settings = {"volume": 0.8}
    preset = VitalPreset(settings=settings, modulations=[], extra={})
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    report = introspector.introspect(preset, preset_path="<no_mods>")

    assert report.active_modulation_count == 0
    assert report.active_modulations == []
