# Feature: preset-corpus-pipeline, Property 14
"""
Property-based tests for RouteMaskAugmenter.

Property 14: Variant 0 is always the original unmasked preset

Validates: Requirements 6.1
"""

from __future__ import annotations

from hypothesis import given, settings

from src.preset_parser import VitalPreset
from src.route_graph_builder import RouteGraphBuilder
from src.route_mask_augmenter import RouteMaskAugmenter
from tests.conftest import vital_presets


@given(preset=vital_presets())
@settings(max_examples=100)
def test_property_14_variant_0_is_always_original_unmasked_preset(preset: VitalPreset):
    """Property 14: Variant 0 is always the original unmasked preset.

    **Validates: Requirements 6.1**

    For any base preset, the first element of RouteMaskAugmenter.augment()
    should have variant_id == 0, mask_vector all 1s, and preset.settings
    identical to the original preset's settings.
    """
    builder = RouteGraphBuilder()
    augmenter = RouteMaskAugmenter(graph_builder=builder)

    variants = augmenter.augment(preset, base_preset_id="test_preset")

    # Must have at least variant_0
    assert len(variants) >= 1, "augment() must return at least variant_0"

    variant_0 = variants[0]

    # variant_id must be 0
    assert variant_0.metadata.variant_id == 0, (
        f"First variant's variant_id={variant_0.metadata.variant_id}, expected 0"
    )

    # mask_vector must be all 1s
    assert all(b == 1 for b in variant_0.route_mask.mask_vector), (
        f"Variant 0 mask_vector should be all 1s, got {variant_0.route_mask.mask_vector}"
    )

    # No edges should be masked
    assert variant_0.route_mask.masked_edge_names == [], (
        f"Variant 0 should have no masked edges, got {variant_0.route_mask.masked_edge_names}"
    )

    # Settings must be identical to the original preset
    assert variant_0.preset.settings == preset.settings, (
        "Variant 0 settings should be identical to the original preset's settings"
    )

    # Modulations must be identical to the original preset
    assert variant_0.preset.modulations == preset.modulations, (
        "Variant 0 modulations should be identical to the original preset's modulations"
    )

    # Extra must be identical to the original preset
    assert variant_0.preset.extra == preset.extra, (
        "Variant 0 extra should be identical to the original preset's extra"
    )

    # num_masked_edges must be 0
    assert variant_0.metadata.num_masked_edges == 0, (
        f"Variant 0 num_masked_edges={variant_0.metadata.num_masked_edges}, expected 0"
    )


# Feature: preset-corpus-pipeline, Property 15
"""
Property 15: Variant count respects edge count bounds

Validates: Requirements 6.2, 6.3
"""


@given(preset=vital_presets())
@settings(max_examples=100)
def test_property_15_variant_count_respects_edge_count_bounds(preset: VitalPreset):
    """Property 15: Variant count respects edge count bounds.

    **Validates: Requirements 6.2, 6.3**

    For any preset with K active maskable edges:
    - If K <= 8: total variant count (including variant 0) <= min(2^K, 1 + effective_max)
    - If K > 8: variant count (excluding variant 0) <= effective_max

    Uses default augmenter settings (min_variants=16, max_variants=64) and run_mode="full".
    """
    builder = RouteGraphBuilder()
    augmenter = RouteMaskAugmenter(graph_builder=builder, min_variants=16, max_variants=64)

    # Build the route graph to determine K
    graph = builder.build(preset)
    maskable_edges = [e for e in graph.edges if e.is_maskable and e.is_active]
    K = len(maskable_edges)

    # Run augmentation in full mode
    variants = augmenter.augment(preset, base_preset_id="test_preset", run_mode="full")

    # effective_max for full mode = max_variants = 64
    effective_max = 64

    if K <= 8:
        upper_bound = min(2 ** K, 1 + effective_max)
        assert len(variants) <= upper_bound, (
            f"K={K}, expected total variants <= min(2^{K}={2**K}, 1+{effective_max}={1+effective_max})"
            f" = {upper_bound}, but got {len(variants)}"
        )
    else:
        # K > 8: variant count excluding variant_0 should not exceed effective_max
        non_zero_variants = len(variants) - 1  # subtract variant_0
        assert non_zero_variants <= effective_max, (
            f"K={K}, expected non-zero variants <= {effective_max}, "
            f"but got {non_zero_variants}"
        )


# Feature: preset-corpus-pipeline, Property 16
"""
Property 16: All masks retain at least one sound-producing path

Validates: Requirements 6.4
"""


@given(preset=vital_presets())
@settings(max_examples=100)
def test_property_16_all_masks_retain_at_least_one_sound_producing_path(preset: VitalPreset):
    """Property 16: All masks retain at least one sound-producing path.

    **Validates: Requirements 6.4**

    For any MaskedVariant produced by augment(), the mutated preset's route
    graph should still contain at least one active path from an oscillator or
    sample source to the audio output. We verify this by rebuilding the route
    graph from the mutated preset and checking source-to-output reachability.
    """
    builder = RouteGraphBuilder()
    augmenter = RouteMaskAugmenter(graph_builder=builder)

    variants = augmenter.augment(preset, base_preset_id="test_preset")

    # Sound sources and their on-keys
    _SOUND_SOURCES = {"osc_1", "osc_2", "osc_3", "sample"}
    _SOURCE_ON_KEYS = {
        "osc_1": "osc_1_on",
        "osc_2": "osc_2_on",
        "osc_3": "osc_3_on",
        "sample": "sample_on",
    }
    _SOURCE_DEST_KEYS = {
        "osc_1": "osc_1_destination",
        "osc_2": "osc_2_destination",
        "osc_3": "osc_3_destination",
        "sample": "sample_destination",
    }
    _FILTER_ON_KEYS = {
        "filter_1": "filter_1_on",
        "filter_2": "filter_2_on",
        "filter_fx": "filter_fx_on",
    }

    def _is_on(value) -> bool:
        if isinstance(value, (int, float)):
            return float(value) == 1.0
        return False

    def has_sound_path(p: VitalPreset) -> bool:
        """Check if the preset has at least one source-to-output reachable path."""
        s = p.settings
        for source in _SOUND_SOURCES:
            on_key = _SOURCE_ON_KEYS[source]
            dest_key = _SOURCE_DEST_KEYS[source]

            if not _is_on(s.get(on_key, 0)):
                continue

            dest_value = s.get(dest_key, 0)
            dest_label = RouteGraphBuilder._resolve_source_destination(source, dest_value)

            if dest_label == "direct":
                return True

            if dest_label == "filter_1+filter_2":
                filter_targets = ["filter_1", "filter_2"]
            else:
                filter_targets = [dest_label]

            for filt in filter_targets:
                filt_on_key = _FILTER_ON_KEYS.get(filt)
                if filt_on_key is None:
                    continue
                if _is_on(s.get(filt_on_key, 0)):
                    return True

        return False

    # Check that the original preset has at least one sound path
    # (if it doesn't, augmenter should only return variant_0)
    original_has_path = has_sound_path(preset)

    for variant in variants:
        if variant.metadata.variant_id == 0:
            # variant_0 is the original — skip reachability check
            # (it may or may not have a sound path depending on the random preset)
            continue

        # For all non-zero variants, the augmenter guarantees a sound path
        assert has_sound_path(variant.preset), (
            f"Variant {variant.metadata.variant_id} has no sound-producing path. "
            f"mask_vector={variant.route_mask.mask_vector}, "
            f"masked_edges={variant.route_mask.masked_edge_names}"
        )


# Feature: preset-corpus-pipeline, Property 17
"""
Property 17: Mask mutation correctly modifies preset parameters

Validates: Requirements 6.5
"""


@given(preset=vital_presets())
@settings(max_examples=100)
def test_property_17_mask_mutation_correctly_modifies_preset_parameters(preset: VitalPreset):
    """Property 17: Mask mutation correctly modifies preset parameters.

    **Validates: Requirements 6.5**

    For any MaskedVariant where mask_vector[i] == 0:
    - If the edge has mutation_rule == "set_on_to_0": the mutated preset's
      corresponding *_on parameter should be 0
    - If the edge has mutation_rule == "set_bypass_to_1": the corresponding
      modulation slot's bypass should be 1
    """
    builder = RouteGraphBuilder()
    augmenter = RouteMaskAugmenter(graph_builder=builder)

    variants = augmenter.augment(preset, base_preset_id="test_preset")

    # Build the route graph to get the maskable edges in order
    graph = builder.build(preset)
    maskable_edges = [e for e in graph.edges if e.is_maskable and e.is_active]

    # On-key mapping for signal edges
    _ON_KEY_MAP = {
        "osc_1": "osc_1_on",
        "osc_2": "osc_2_on",
        "osc_3": "osc_3_on",
        "sample": "sample_on",
        "filter_1": "filter_1_on",
        "filter_2": "filter_2_on",
        "filter_fx": "filter_fx_on",
    }

    for variant in variants:
        if variant.metadata.variant_id == 0:
            # variant_0 is unmasked — no mutations expected
            continue

        mask = variant.route_mask.mask_vector
        assert len(mask) == len(maskable_edges), (
            f"Variant {variant.metadata.variant_id}: mask length {len(mask)} "
            f"!= maskable edges count {len(maskable_edges)}"
        )

        for i, edge in enumerate(maskable_edges):
            if mask[i] == 0:
                if edge.mutation_rule == "set_on_to_0":
                    # Signal edge: the *_on key should be 0 in the mutated preset
                    on_key = _ON_KEY_MAP.get(edge.source)
                    if on_key and on_key in variant.preset.settings:
                        assert variant.preset.settings[on_key] == 0, (
                            f"Variant {variant.metadata.variant_id}, edge {i} "
                            f"({edge.source}->{edge.destination}): "
                            f"expected {on_key}=0, got {variant.preset.settings[on_key]}"
                        )

                elif edge.mutation_rule == "set_bypass_to_1":
                    # Modulation edge: find matching slot and check bypass == 1
                    found_slot = False
                    for mod in variant.preset.modulations:
                        if (mod.get("source") == edge.source
                                and mod.get("destination") == edge.destination):
                            found_slot = True
                            assert mod.get("bypass") == 1, (
                                f"Variant {variant.metadata.variant_id}, edge {i} "
                                f"({edge.source}->{edge.destination}): "
                                f"expected bypass=1, got {mod.get('bypass')}"
                            )
                            break
                    assert found_slot, (
                        f"Variant {variant.metadata.variant_id}, edge {i} "
                        f"({edge.source}->{edge.destination}): "
                        f"no matching modulation slot found in mutated preset"
                    )
