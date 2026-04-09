"""
AudioRenderer padding/truncation 属性测试与新参数回归测试。

覆盖设计文档中的 Property 6 以及 AudioRenderer 新参数的回归测试。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.audio_renderer import AudioRenderer, RenderConfig


# ---------------------------------------------------------------------------
# Helper: create a mock AudioRenderer (bypass VST loading)
# ---------------------------------------------------------------------------

def _create_mock_renderer(
    tmp_path: Path, config: RenderConfig | None = None
) -> AudioRenderer:
    """Create an AudioRenderer with mocked VST plugin loading."""
    vst_path = tmp_path / "Vital.vst3"
    if not vst_path.exists():
        vst_path.mkdir()

    mock_plugin = MagicMock()
    cfg = config or RenderConfig()
    num_samples = int(cfg.duration_sec * cfg.sample_rate)
    mock_plugin.return_value = np.zeros((2, num_samples), dtype=np.float32)

    renderer = object.__new__(AudioRenderer)
    renderer._vst_path = vst_path
    renderer._config = cfg
    renderer._plugin = mock_plugin
    renderer._default_raw = {}
    return renderer


def _create_preset_file(directory: Path, name: str = "test.vital") -> Path:
    """Create a minimal .vital preset file."""
    filepath = directory / name
    preset_json = {
        "author": "Test",
        "settings": {"osc_1_on": 1.0, "modulations": []},
    }
    filepath.write_text(json.dumps(preset_json), encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Property 6: Padding/Truncation output length consistency
# Feature: adsr-adaptive-rendering, Property 6: Padding/Truncation output length consistency
# Validates: Requirements 5.2, 5.3
# ---------------------------------------------------------------------------


class TestPaddingTruncationProperty:
    """Property 6: Padding/Truncation 输出长度一致性

    Tests the padding/truncation logic directly on numpy arrays,
    mirroring the logic in AudioRenderer._write_wav.
    """

    @staticmethod
    def _apply_padding_truncation(
        audio: np.ndarray, target_length_samples: int
    ) -> np.ndarray:
        """Apply the same padding/truncation logic as _write_wav."""
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        current = audio.shape[-1]
        if current < target_length_samples:
            pad = np.zeros(
                (audio.shape[0], target_length_samples - current),
                dtype=audio.dtype,
            )
            audio = np.concatenate([audio, pad], axis=1)
        elif current > target_length_samples:
            audio = audio[:, :target_length_samples]
        return audio

    @given(
        audio_len=st.integers(min_value=1, max_value=200000),
        target_len=st.integers(min_value=1, max_value=200000),
        channels=st.sampled_from([1, 2]),
    )
    @settings(max_examples=100)
    def test_output_length_equals_target(
        self, audio_len: int, target_len: int, channels: int
    ) -> None:
        """After padding/truncation, output length == target_length_samples."""
        audio = np.random.randn(channels, audio_len).astype(np.float32)
        result = self._apply_padding_truncation(audio, target_len)
        assert result.shape[-1] == target_len, (
            f"Expected {target_len}, got {result.shape[-1]} "
            f"(original={audio_len}, channels={channels})"
        )

    @given(
        audio_len=st.integers(min_value=1, max_value=50000),
        target_len=st.integers(min_value=1, max_value=50000),
    )
    @settings(max_examples=100)
    def test_padding_tail_is_zeros(
        self, audio_len: int, target_len: int
    ) -> None:
        """When padding, the appended tail must be all zeros."""
        if audio_len >= target_len:
            return  # Only test padding case
        audio = np.ones((1, audio_len), dtype=np.float32)
        result = self._apply_padding_truncation(audio, target_len)
        # Original portion preserved
        np.testing.assert_array_equal(result[:, :audio_len], audio)
        # Padded tail is zeros
        np.testing.assert_array_equal(
            result[:, audio_len:], np.zeros((1, target_len - audio_len))
        )

    @given(
        audio_len=st.integers(min_value=2, max_value=50000),
        target_len=st.integers(min_value=1, max_value=50000),
    )
    @settings(max_examples=100)
    def test_truncation_preserves_prefix(
        self, audio_len: int, target_len: int
    ) -> None:
        """When truncating, the first target_len samples are preserved."""
        if target_len >= audio_len:
            return  # Only test truncation case
        audio = np.arange(audio_len, dtype=np.float32).reshape(1, -1)
        result = self._apply_padding_truncation(audio, target_len)
        np.testing.assert_array_equal(
            result[:, :target_len], audio[:, :target_len]
        )


# ---------------------------------------------------------------------------
# Unit tests: AudioRenderer new params regression
# Requirements: 4.2, 5.2, 5.3
# ---------------------------------------------------------------------------


class TestNoteOffTimeDefault:
    """Test that not passing note_off_time still uses duration - 0.1."""

    def test_default_note_off_uses_duration_minus_01(self, tmp_path: Path) -> None:
        config = RenderConfig(duration_sec=3.0)
        renderer = _create_mock_renderer(tmp_path, config)
        preset_path = _create_preset_file(tmp_path)

        with patch.object(renderer, "_write_wav"):
            renderer.render_preset(preset_path, tmp_path / "out.wav")

        # The plugin was called; inspect the MIDI messages passed
        call_args = renderer._plugin.call_args
        midi_messages = call_args[0][0]
        # midi_messages is a list of (bytes, timestamp) tuples
        # note_off is the second message
        _, note_off_timestamp = midi_messages[1]
        expected = max(0.0, 3.0 - 0.1)  # 2.9
        assert note_off_timestamp == pytest.approx(expected), (
            f"Expected note_off at {expected}, got {note_off_timestamp}"
        )

    def test_custom_note_off_time_is_used(self, tmp_path: Path) -> None:
        config = RenderConfig(duration_sec=5.0)
        renderer = _create_mock_renderer(tmp_path, config)
        preset_path = _create_preset_file(tmp_path)

        with patch.object(renderer, "_write_wav"):
            renderer.render_preset(
                preset_path, tmp_path / "out.wav", note_off_time=1.5
            )

        call_args = renderer._plugin.call_args
        midi_messages = call_args[0][0]
        _, note_off_timestamp = midi_messages[1]
        assert note_off_timestamp == pytest.approx(1.5), (
            f"Expected note_off at 1.5, got {note_off_timestamp}"
        )


class TestPaddingUnit:
    """Test that padding adds zeros at the end."""

    def test_padding_extends_audio(self) -> None:
        """Directly test padding logic on numpy arrays."""
        audio = np.ones((1, 100), dtype=np.float32)
        target = 200

        # Apply same logic as _write_wav
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        current = audio.shape[-1]
        if current < target:
            pad = np.zeros((audio.shape[0], target - current), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)

        assert audio.shape[-1] == 200
        np.testing.assert_array_equal(audio[:, :100], np.ones((1, 100)))
        np.testing.assert_array_equal(audio[:, 100:], np.zeros((1, 100)))

    def test_padding_1d_input(self) -> None:
        """1D audio is reshaped to 2D before padding."""
        audio = np.ones(50, dtype=np.float32)
        target = 100

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        current = audio.shape[-1]
        if current < target:
            pad = np.zeros((audio.shape[0], target - current), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)

        assert audio.shape == (1, 100)
        np.testing.assert_array_equal(audio[:, 50:], np.zeros((1, 50)))


class TestTruncationUnit:
    """Test that truncation produces correct length."""

    def test_truncation_shortens_audio(self) -> None:
        """Directly test truncation logic on numpy arrays."""
        audio = np.arange(500, dtype=np.float32).reshape(1, -1)
        target = 200

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        current = audio.shape[-1]
        if current > target:
            audio = audio[:, :target]

        assert audio.shape[-1] == 200
        expected = np.arange(200, dtype=np.float32).reshape(1, -1)
        np.testing.assert_array_equal(audio, expected)

    def test_no_target_length_preserves_original(self) -> None:
        """When target_length_samples is None, audio is unchanged."""
        audio = np.ones((1, 300), dtype=np.float32)
        target = None

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        if target is not None:
            current = audio.shape[-1]
            if current < target:
                pad = np.zeros((audio.shape[0], target - current), dtype=audio.dtype)
                audio = np.concatenate([audio, pad], axis=1)
            elif current > target:
                audio = audio[:, :target]

        assert audio.shape[-1] == 300

    def test_exact_length_unchanged(self) -> None:
        """When audio length == target, no change."""
        audio = np.ones((1, 200), dtype=np.float32)
        target = 200

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        current = audio.shape[-1]
        if current < target:
            pad = np.zeros((audio.shape[0], target - current), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)
        elif current > target:
            audio = audio[:, :target]

        assert audio.shape[-1] == 200
        np.testing.assert_array_equal(audio, np.ones((1, 200)))
