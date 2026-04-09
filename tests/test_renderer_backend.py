# Feature: preset-corpus-pipeline, Property 19
"""
Property-based tests for RendererBackend.

Property 19: Renderer failure returns failure status without crash.
Validates: Requirements 8.3
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


from src.renderer_backend import RenderResult, RendererBackend
from src.audio_renderer import RenderConfig


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Exception types that a renderer might raise
EXCEPTION_TYPES = [RuntimeError, TimeoutError, OSError, ValueError, IOError]


@st.composite
def render_exceptions(draw: st.DrawFn):
    """Generate a random exception type and message for simulating render failures."""
    exc_type = draw(st.sampled_from(EXCEPTION_TYPES))
    message = draw(
        st.text(
            alphabet=st.sampled_from(
                "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:.!?"
            ),
            min_size=1,
            max_size=80,
        )
    )
    return exc_type, message


@st.composite
def preset_paths(draw: st.DrawFn) -> Path:
    """Generate a random preset file path."""
    name = draw(
        st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
            min_size=1,
            max_size=30,
        )
    )
    return Path(f"/tmp/presets/{name}.vital")


@st.composite
def output_paths(draw: st.DrawFn) -> Path:
    """Generate a random output file path."""
    name = draw(
        st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-"),
            min_size=1,
            max_size=30,
        )
    )
    return Path(f"/tmp/output/{name}.wav")


# ---------------------------------------------------------------------------
# Property 19: Renderer failure returns failure status without crash
# ---------------------------------------------------------------------------


@given(
    exc_data=render_exceptions(),
    preset_path=preset_paths(),
    output_path=output_paths(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_renderer_failure_returns_failure_status_without_crash(
    exc_data, preset_path, output_path
):
    """**Validates: Requirements 8.3**

    Property 19: For any preset path that causes a render exception or timeout,
    RendererBackend.render_single() should return a RenderResult with
    success == False and a non-None error string, without raising an exception.
    """
    exc_type, exc_message = exc_data

    # Create a RendererBackend with mocked internals to avoid needing a real VST
    config = RenderConfig()

    with patch.object(RendererBackend, "__init__", lambda self, *a, **kw: None):
        backend = RendererBackend.__new__(RendererBackend)
        backend._config = config

        # Mock the internal renderer to raise the generated exception
        mock_renderer = MagicMock()
        mock_renderer.render_preset.side_effect = exc_type(exc_message)
        backend._renderer = mock_renderer

        # Call render_single — it must NOT raise
        result = backend.render_single(preset_path, output_path)

    # Verify the contract: failure status, no crash
    assert isinstance(result, RenderResult)
    assert result.success is False
    assert result.error is not None
    assert isinstance(result.error, str)
    assert len(result.error) > 0
    assert result.audio_path is None
    assert result.audio is None
    assert result.midi_note == config.midi_note
    assert result.midi_velocity == config.velocity


@given(
    preset_path=preset_paths(),
    output_path=output_paths(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_renderer_returning_false_yields_failure_result(
    preset_path, output_path
):
    """**Validates: Requirements 8.3**

    When the underlying renderer returns False (non-exception failure),
    render_single should still return RenderResult(success=False) with
    a non-None error string.
    """
    config = RenderConfig()

    with patch.object(RendererBackend, "__init__", lambda self, *a, **kw: None):
        backend = RendererBackend.__new__(RendererBackend)
        backend._config = config

        mock_renderer = MagicMock()
        mock_renderer.render_preset.return_value = False
        backend._renderer = mock_renderer

        result = backend.render_single(preset_path, output_path)

    assert isinstance(result, RenderResult)
    assert result.success is False
    assert result.error is not None
    assert isinstance(result.error, str)
    assert len(result.error) > 0
    assert result.audio_path is None
    assert result.audio is None
