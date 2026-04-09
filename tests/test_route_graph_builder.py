# Feature: preset-corpus-pipeline, Property 12, Property 13
"""
Property-based tests for RouteGraphBuilder.

Property 12: Route edge extraction matches preset state
Property 13: RouteGraph structural invariants

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.preset_parser import VitalPreset
from src.route_graph_builder import RouteGraphBuilder
from tests.conftest import vital_presets


@given(preset=vital_presets())
@settings(max_examples=100)
def test_property_12_route_edge_extraction_matches_preset_state(preset: VitalPreset):
    """Property 12: Route edge extraction matches preset state.

    **Validates: Requirements 5.1, 5.2**

    For any VitalPreset:
    1. The RouteGraph contains a signal edge for osc_i (i in {1,2,3}) with
       is_active == (preset.settings.get("osc_i_on", 0) == 1).
    2. The RouteGraph contains a modulation edge for each modulation slot
       where both source and destination are non-empty strings.
    """
    builder = RouteGraphBuilder()
    graph = builder.build(preset)

    # --- 5.1: osc_i signal edges match osc_i_on state ---
    for i in range(1, 4):
        osc_key = f"osc_{i}_on"
        expected_active = float(preset.settings.get(osc_key, 0)) == 1.0

        osc_edges = [
            e for e in graph.edges
            if e.edge_type == "signal" and e.source == f"osc_{i}"
        ]
        assert len(osc_edges) == 1, (
            f"Expected exactly 1 signal edge for osc_{i}, got {len(osc_edges)}"
        )
        assert osc_edges[0].is_active == expected_active, (
            f"osc_{i} edge is_active={osc_edges[0].is_active}, "
            f"expected {expected_active} (osc_{i}_on={preset.settings.get(osc_key, 0)})"
        )

    # --- 5.2: modulation edges match active modulation slots ---
    expected_mod_count = sum(
        1 for mod in preset.modulations
        if isinstance(mod.get("source", ""), str) and mod.get("source", "") != ""
        and isinstance(mod.get("destination", ""), str) and mod.get("destination", "") != ""
    )
    actual_mod_edges = [
        e for e in graph.edges if e.edge_type == "modulation"
    ]
    assert len(actual_mod_edges) == expected_mod_count, (
        f"Expected {expected_mod_count} modulation edges, got {len(actual_mod_edges)}"
    )


@given(preset=vital_presets())
@settings(max_examples=100)
def test_property_13_route_graph_structural_invariants(preset: VitalPreset):
    """Property 13: RouteGraph structural invariants.

    **Validates: Requirements 5.3, 5.4, 5.5**

    For any RouteGraph:
    (a) Every edge has edge_type in {"signal", "modulation"}.
    (b) total_active_edges equals the count of edges where is_active == True.
    (c) total_active_maskable_edges equals the count of edges where
        is_active == True and is_maskable == True.
    (d) Edges with source "effect_chain" or "stereo_routing" have
        is_maskable == False.
    """
    builder = RouteGraphBuilder()
    graph = builder.build(preset)

    # (a) edge_type validity
    for edge in graph.edges:
        assert edge.edge_type in {"signal", "modulation"}, (
            f"Invalid edge_type '{edge.edge_type}' for edge {edge.source} → {edge.destination}"
        )

    # (b) total_active_edges count
    expected_active = sum(1 for e in graph.edges if e.is_active)
    assert graph.total_active_edges == expected_active, (
        f"total_active_edges={graph.total_active_edges}, "
        f"expected {expected_active}"
    )

    # (c) total_active_maskable_edges count
    expected_active_maskable = sum(
        1 for e in graph.edges if e.is_active and e.is_maskable
    )
    assert graph.total_active_maskable_edges == expected_active_maskable, (
        f"total_active_maskable_edges={graph.total_active_maskable_edges}, "
        f"expected {expected_active_maskable}"
    )

    # (d) effect_chain and stereo_routing edges are not maskable
    for edge in graph.edges:
        if edge.source in ("effect_chain", "stereo_routing"):
            assert edge.is_maskable is False, (
                f"Edge with source '{edge.source}' should have is_maskable=False, "
                f"got is_maskable={edge.is_maskable}"
            )
