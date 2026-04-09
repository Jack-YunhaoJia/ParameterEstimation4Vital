# Feature: preset-corpus-pipeline, Property 3, Property 18
"""
Property-based tests for MutatedPresetWriter and PresetParser round-trip.

Property 3: PresetParser round-trip preserves all data
Property 18: Written file names match expected pattern

Validates: Requirements 2.5, 7.3, 7.4, 11.1, 11.2, 11.3
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.preset_parser import PresetParser, VitalPreset
from src.mutated_preset_writer import MutatedPresetWriter
from src.route_mask_augmenter import MaskedVariant, MaskMetadata, RouteMask, RouteMaskAugmenter
from tests.conftest import vital_presets


@given(preset=vital_presets())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_3_preset_parser_round_trip_preserves_all_data(
    preset: VitalPreset,
):
    """Property 3: PresetParser round-trip preserves all data.

    **Validates: Requirements 2.5, 7.4, 11.1, 11.2, 11.3**

    For any valid VitalPreset (with arbitrary settings, modulations, and
    wavetables), parse(serialize(preset)) should produce a VitalPreset with
    identical settings dict, identical modulations list, and identical
    wavetable data in extra.
    """
    parser = PresetParser()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "round_trip_test.vital"

        # Serialize the preset
        parser.serialize(preset, filepath)

        # Parse it back
        parsed = parser.parse(filepath)

    # Settings must be identical
    assert parsed.settings == preset.settings, (
        f"Settings mismatch after round-trip.\n"
        f"Original keys: {sorted(preset.settings.keys())}\n"
        f"Parsed keys: {sorted(parsed.settings.keys())}"
    )

    # Modulations must be identical
    assert parsed.modulations == preset.modulations, (
        f"Modulations mismatch after round-trip.\n"
        f"Original count: {len(preset.modulations)}\n"
        f"Parsed count: {len(parsed.modulations)}"
    )

    # Wavetable data in extra must be identical
    original_wt = preset.extra.get("wavetables")
    parsed_wt = parsed.extra.get("wavetables")
    assert parsed_wt == original_wt, (
        f"Wavetable data mismatch after round-trip.\n"
        f"Original: {original_wt}\n"
        f"Parsed: {parsed_wt}"
    )


# Strategy for generating base_preset_id strings (safe for filenames)
_base_preset_ids = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
    min_size=1,
    max_size=30,
)

# Strategy for generating variant_ids
_variant_ids = st.integers(min_value=0, max_value=9999)


@given(
    preset=vital_presets(),
    base_preset_id=_base_preset_ids,
    variant_id=_variant_ids,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_18_written_file_names_match_expected_pattern(
    preset: VitalPreset,
    base_preset_id: str,
    variant_id: int,
):
    """Property 18: Written file names match expected pattern.

    **Validates: Requirements 7.3**

    For any MaskedVariant written by MutatedPresetWriter.write(), the output
    file name should match the pattern {base_preset_id}_mask_{variant_id:04d}.vital.
    """
    parser = PresetParser()
    writer = MutatedPresetWriter(parser=parser)

    # Build a MaskedVariant with the given preset and metadata
    variant = MaskedVariant(
        preset=preset,
        route_mask=RouteMask(mask_vector=[], masked_edge_names=[]),
        metadata=MaskMetadata(
            base_preset_id=base_preset_id,
            variant_id=variant_id,
            num_masked_edges=0,
            total_active_edges=0,
            maskable_edge_names=[],
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write the variant
        output_path = writer.write(variant, output_dir)

        # Verify the file was created
        assert output_path.exists(), f"Output file not created: {output_path}"

        # Verify the filename matches the expected pattern
        expected_filename = f"{base_preset_id}_mask_{variant_id:04d}.vital"
        assert output_path.name == expected_filename, (
            f"Filename mismatch: expected '{expected_filename}', "
            f"got '{output_path.name}'"
        )

        # Also verify via regex pattern
        pattern = re.compile(r"^.+_mask_\d{4}\.vital$")
        assert pattern.match(output_path.name), (
            f"Filename '{output_path.name}' does not match pattern "
            r"'{base_preset_id}_mask_{variant_id:04d}.vital'"
        )

        # Verify the file is in the expected output directory
        assert output_path.parent == output_dir, (
            f"Output file not in expected directory: "
            f"{output_path.parent} != {output_dir}"
        )


# Feature: preset-corpus-pipeline, Property 23

from src.route_graph_builder import RouteGraphBuilder


# Continuous oscillator parameters that must be preserved when the oscillator
# is not masked off.  Pattern: osc_{i}_{suffix} for i in 1..3.
_OSC_CONTINUOUS_SUFFIXES: list[str] = [
    "wave_frame",
    "frame_spread",
    "spectral_morph_amount",
    "spectral_morph_spread",
    "spectral_morph_type",
    "unison_voices",
    "unison_detune",
    "unison_blend",
    "spectral_unison",
]

# On-key for each oscillator index
_OSC_ON_KEYS = {1: "osc_1_on", 2: "osc_2_on", 3: "osc_3_on"}


def _osc_continuous_keys(osc_index: int) -> list[str]:
    """Return the list of continuous parameter keys for a given oscillator."""
    return [f"osc_{osc_index}_{suffix}" for suffix in _OSC_CONTINUOUS_SUFFIXES]


def _osc_was_masked_off(variant: MaskedVariant) -> set[int]:
    """Return the set of oscillator indices (1-3) that were masked off.

    An oscillator is masked off when its signal edge appears in
    masked_edge_names (i.e. the mask disabled it via set_on_to_0).
    """
    masked: set[int] = set()
    for name in variant.route_mask.masked_edge_names:
        for i in (1, 2, 3):
            if name.startswith(f"signal:osc_{i}->"):
                masked.add(i)
    return masked


@given(preset=vital_presets())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_23_oscillator_continuous_params_preserved_through_pipeline(
    preset: VitalPreset,
):
    """Property 23: Oscillator continuous parameters preserved through pipeline.

    **Validates: Requirements 10.6**

    For any preset processed through the augmentation pipeline, the continuous
    oscillator parameters (osc_i_wave_frame, osc_i_frame_spread,
    osc_i_spectral_morph_*, osc_i_unison_*) in the mutated preset should be
    identical to the original preset's values, unless the oscillator itself was
    masked off.
    """
    graph_builder = RouteGraphBuilder()
    augmenter = RouteMaskAugmenter(graph_builder=graph_builder)

    variants = augmenter.augment(preset, base_preset_id="prop23_test")

    for variant in variants:
        masked_oscs = _osc_was_masked_off(variant)

        for osc_i in (1, 2, 3):
            if osc_i in masked_oscs:
                # Oscillator was masked off — continuous params may differ
                continue

            for key in _osc_continuous_keys(osc_i):
                original_val = preset.settings.get(key)
                mutated_val = variant.preset.settings.get(key)
                assert mutated_val == original_val, (
                    f"Oscillator {osc_i} continuous param '{key}' changed "
                    f"through pipeline (variant_id={variant.metadata.variant_id}).\n"
                    f"Original: {original_val}\n"
                    f"Mutated:  {mutated_val}\n"
                    f"Masked oscillators: {masked_oscs}\n"
                    f"Masked edges: {variant.route_mask.masked_edge_names}"
                )
