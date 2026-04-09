# Feature: preset-corpus-pipeline, Property 4
"""
Property-based tests for PresetRenderAudit.

Property 4: Audit report counts are internally consistent.
Validates: Requirements 3.5, 3.6
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.preset_introspector import PresetIntrospector
from src.preset_parser import PresetParser, VitalPreset
from src.preset_render_audit import PresetRenderAudit, RenderAuditConfig

from tests.conftest import vital_presets


# ---------------------------------------------------------------------------
# Mock renderer backend for testing without a real VST
# ---------------------------------------------------------------------------


@dataclass
class MockRenderResult:
    """Mock render result conforming to RenderResultLike protocol."""

    success: bool
    audio: np.ndarray | None
    error: str | None


class MockRendererBackend:
    """Mock renderer that succeeds or fails based on preset index.

    Args:
        fail_indices: Set of call indices (0-based) that should fail.
    """

    def __init__(self, fail_indices: set[int] | None = None) -> None:
        self._fail_indices = set(fail_indices or set())
        self._call_count = 0

    def render_single(
        self, preset_path: Path, output_path: Path
    ) -> MockRenderResult:
        idx = self._call_count
        self._call_count += 1
        if idx in self._fail_indices:
            return MockRenderResult(
                success=False, audio=None, error="mock render failure"
            )
        audio = np.random.default_rng(idx).standard_normal(44100).astype(
            np.float32
        ) * 0.1
        return MockRenderResult(success=True, audio=audio, error=None)


# ---------------------------------------------------------------------------
# Helper: write a VitalPreset to a .vital file on disk
# ---------------------------------------------------------------------------


def _write_preset_file(preset: VitalPreset, path: Path) -> None:
    """Serialize a VitalPreset to a .vital JSON file."""
    parser = PresetParser()
    parser.serialize(preset, path)


# ---------------------------------------------------------------------------
# Hypothesis strategy: generate a batch of presets with a fail-index set
# ---------------------------------------------------------------------------


@st.composite
def preset_batches(draw: st.DrawFn):
    """Generate a list of VitalPresets together with a set of fail indices.

    Returns (presets, fail_indices) where fail_indices ⊆ {0..len(presets)-1}.
    """
    presets = draw(
        st.lists(vital_presets(), min_size=0, max_size=8)
    )
    n = len(presets)
    if n == 0:
        return presets, set()
    fail_indices = draw(
        st.frozensets(st.integers(min_value=0, max_value=n - 1))
    )
    return presets, set(fail_indices)


# ---------------------------------------------------------------------------
# Property 4: Audit report counts are internally consistent
# ---------------------------------------------------------------------------


@given(data=preset_batches())
@settings(max_examples=100)
def test_audit_report_counts_internally_consistent(data, tmp_path_factory):
    """**Validates: Requirements 3.5, 3.6**

    For any batch of preset paths (including paths that cause render failures),
    the RenderAuditReport should satisfy:
    (a) len(details) == total_presets
    (b) render_success_count + render_failure_count == total_presets
    (c) modulation_gap_count equals the count of details where has_modulation_gap == True
    (d) wavetable_gap_count equals the count of details where has_wavetable_gap == True
    """
    presets, fail_indices = data

    # Write presets to temporary .vital files
    tmp_dir = tmp_path_factory.mktemp("audit_batch")
    preset_paths: list[Path] = []
    for i, preset in enumerate(presets):
        p = tmp_dir / f"preset_{i}.vital"
        _write_preset_file(preset, p)
        preset_paths.append(p)

    # Build audit components with mock renderer
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    renderer = MockRendererBackend(fail_indices=fail_indices)
    config = RenderAuditConfig()
    audit = PresetRenderAudit(introspector, renderer, config)

    # Run audit
    report = audit.audit_batch(preset_paths)

    # (a) len(details) == total_presets
    assert len(report.details) == report.total_presets
    assert report.total_presets == len(preset_paths)

    # (b) render_success_count + render_failure_count == total_presets
    assert report.render_success_count + report.render_failure_count == report.total_presets

    # (c) modulation_gap_count == count of details with has_modulation_gap == True
    expected_mod_gap = sum(1 for d in report.details if d.has_modulation_gap)
    assert report.modulation_gap_count == expected_mod_gap

    # (d) wavetable_gap_count == count of details with has_wavetable_gap == True
    expected_wt_gap = sum(1 for d in report.details if d.has_wavetable_gap)
    assert report.wavetable_gap_count == expected_wt_gap


# ---------------------------------------------------------------------------
# Property 5: Gap flagging matches introspection results
# ---------------------------------------------------------------------------


@given(data=preset_batches())
@settings(max_examples=100)
def test_gap_flagging_matches_introspection(data, tmp_path_factory):
    """**Validates: Requirements 3.3, 3.4**

    For any preset where IntrospectionReport.active_modulation_count > 0
    and the renderer skips modulations (which the mock renderer always does),
    the audit detail should have has_modulation_gap == True.

    Similarly, for any preset where IntrospectionReport.wavetable_count > 0
    and the renderer does not inject wavetables, the audit detail should
    have has_wavetable_gap == True.
    """
    presets, fail_indices = data

    # Write presets to temporary .vital files
    tmp_dir = tmp_path_factory.mktemp("gap_flag_batch")
    preset_paths: list[Path] = []
    for i, preset in enumerate(presets):
        p = tmp_dir / f"preset_{i}.vital"
        _write_preset_file(preset, p)
        preset_paths.append(p)

    # Build audit components with mock renderer (always skips modulations/wavetables)
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    renderer = MockRendererBackend(fail_indices=fail_indices)
    config = RenderAuditConfig()
    audit = PresetRenderAudit(introspector, renderer, config)

    # Run audit
    report = audit.audit_batch(preset_paths)

    # For each detail, independently introspect the preset and verify gap flags
    for i, detail in enumerate(report.details):
        if detail.introspection is None:
            # Parse/introspect failed — gaps default to False, which is acceptable
            continue

        intro = detail.introspection

        # Property 5a: modulation gap flag
        # The mock renderer always skips modulations, so any preset with
        # active_modulation_count > 0 should have has_modulation_gap == True
        if intro.active_modulation_count > 0:
            assert detail.has_modulation_gap, (
                f"Preset {i}: active_modulation_count={intro.active_modulation_count} "
                f"but has_modulation_gap is False"
            )
        else:
            assert not detail.has_modulation_gap, (
                f"Preset {i}: active_modulation_count=0 "
                f"but has_modulation_gap is True"
            )

        # Property 5b: wavetable gap flag
        # The mock renderer always skips wavetables, so any preset with
        # wavetable_count > 0 should have has_wavetable_gap == True
        if intro.wavetable_count > 0:
            assert detail.has_wavetable_gap, (
                f"Preset {i}: wavetable_count={intro.wavetable_count} "
                f"but has_wavetable_gap is False"
            )
        else:
            assert not detail.has_wavetable_gap, (
                f"Preset {i}: wavetable_count=0 "
                f"but has_wavetable_gap is True"
            )


# ---------------------------------------------------------------------------
# Property 6: Backend recommendation follows threshold rules
# ---------------------------------------------------------------------------


@st.composite
def audit_report_inputs(draw: st.DrawFn):
    """Generate inputs for constructing a RenderAuditReport with known gap rates.

    Returns (presets_with_active_modulations, modulation_gap_count,
             presets_with_wavetables, wavetable_gap_count,
             modulation_gap_threshold, wavetable_gap_threshold,
             override_not_ready).
    """
    # Generate counts ensuring gap_count <= base_count
    presets_with_active_modulations = draw(
        st.integers(min_value=0, max_value=100)
    )
    modulation_gap_count = draw(
        st.integers(
            min_value=0,
            max_value=presets_with_active_modulations,
        )
    )

    presets_with_wavetables = draw(st.integers(min_value=0, max_value=100))
    wavetable_gap_count = draw(
        st.integers(min_value=0, max_value=presets_with_wavetables)
    )

    # Thresholds: positive floats in (0, 1]
    modulation_gap_threshold = draw(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False)
    )
    wavetable_gap_threshold = draw(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False)
    )

    override_not_ready = draw(st.booleans())

    return (
        presets_with_active_modulations,
        modulation_gap_count,
        presets_with_wavetables,
        wavetable_gap_count,
        modulation_gap_threshold,
        wavetable_gap_threshold,
        override_not_ready,
    )


@given(data=audit_report_inputs())
@settings(max_examples=200)
def test_backend_recommendation_follows_threshold_rules(data):
    """**Validates: Requirements 3.7**

    For any RenderAuditReport:
    - modulation_gap_rate = modulation_gap_count / presets_with_active_modulations
      (when > 0, else 0.0)
    - wavetable_gap_rate = wavetable_gap_count / presets_with_wavetables
      (when > 0, else 0.0)
    - If override_not_ready is True → always "current_backend_acceptable"
    - If either gap rate exceeds its configured threshold →
      "backend_replacement_recommended"
    - Otherwise → "current_backend_acceptable"
    """
    (
        presets_with_active_modulations,
        modulation_gap_count,
        presets_with_wavetables,
        wavetable_gap_count,
        modulation_gap_threshold,
        wavetable_gap_threshold,
        override_not_ready,
    ) = data

    # Compute expected gap rates
    mod_gap_rate = (
        modulation_gap_count / presets_with_active_modulations
        if presets_with_active_modulations > 0
        else 0.0
    )
    wt_gap_rate = (
        wavetable_gap_count / presets_with_wavetables
        if presets_with_wavetables > 0
        else 0.0
    )

    # Use the actual _compute_recommendation method via a minimal PresetRenderAudit
    config = RenderAuditConfig(
        modulation_gap_threshold=modulation_gap_threshold,
        wavetable_gap_threshold=wavetable_gap_threshold,
        override_not_ready=override_not_ready,
    )

    # We only need the _compute_recommendation method, so create a minimal audit
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    renderer = MockRendererBackend()
    audit = PresetRenderAudit(introspector, renderer, config)

    recommendation = audit._compute_recommendation(mod_gap_rate, wt_gap_rate)

    # Compute expected recommendation independently
    if override_not_ready:
        expected = "current_backend_acceptable"
    elif mod_gap_rate > modulation_gap_threshold:
        expected = "backend_replacement_recommended"
    elif wt_gap_rate > wavetable_gap_threshold:
        expected = "backend_replacement_recommended"
    else:
        expected = "current_backend_acceptable"

    assert recommendation == expected, (
        f"mod_gap_rate={mod_gap_rate:.4f} (threshold={modulation_gap_threshold:.4f}), "
        f"wt_gap_rate={wt_gap_rate:.4f} (threshold={wavetable_gap_threshold:.4f}), "
        f"override={override_not_ready}: "
        f"got '{recommendation}', expected '{expected}'"
    )
