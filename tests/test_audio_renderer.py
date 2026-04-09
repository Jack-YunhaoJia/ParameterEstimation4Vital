"""
AudioRenderer 单元测试。

使用 mock 对象替代实际的 Vital VST3 插件，验证：
- RenderConfig 和 RenderSummary 数据类
- render_preset 成功/失败行为
- render_batch 批量渲染、超时处理、错误恢复
- 输出文件名映射（.vital → .wav）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.audio_renderer import (
    AudioRenderer,
    RenderConfig,
    RenderSummary,
    RenderTimeoutError,
)


# ---------------------------------------------------------------------------
# RenderConfig tests
# ---------------------------------------------------------------------------

class TestRenderConfig:
    def test_default_values(self):
        config = RenderConfig()
        assert config.midi_note == 60  # C4
        assert config.velocity == 100
        assert config.duration_sec == 2.0
        assert config.sample_rate == 44100
        assert config.timeout_sec == 30.0

    def test_custom_values(self):
        config = RenderConfig(
            midi_note=72, velocity=80, duration_sec=3.0,
            sample_rate=48000, timeout_sec=10.0,
        )
        assert config.midi_note == 72
        assert config.velocity == 80
        assert config.duration_sec == 3.0
        assert config.sample_rate == 48000
        assert config.timeout_sec == 10.0


# ---------------------------------------------------------------------------
# RenderSummary tests
# ---------------------------------------------------------------------------

class TestRenderSummary:
    def test_default_values(self):
        summary = RenderSummary()
        assert summary.success_count == 0
        assert summary.failure_count == 0
        assert summary.failed_files == []

    def test_creation_with_counts(self):
        summary = RenderSummary(
            success_count=5, failure_count=2,
            failed_files=["a.vital", "b.vital"],
        )
        assert summary.success_count == 5
        assert summary.failure_count == 2
        assert len(summary.failed_files) == 2

    def test_failed_files_length_matches_failure_count(self):
        summary = RenderSummary(
            success_count=3, failure_count=2,
            failed_files=["x.vital", "y.vital"],
        )
        assert len(summary.failed_files) == summary.failure_count

    def test_success_plus_failure_equals_total(self):
        summary = RenderSummary(
            success_count=7, failure_count=3,
            failed_files=["a.vital", "b.vital", "c.vital"],
        )
        assert summary.success_count + summary.failure_count == 10

    def test_no_shared_state_between_instances(self):
        s1 = RenderSummary()
        s2 = RenderSummary()
        s1.failed_files.append("test.vital")
        assert s2.failed_files == []


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
    # Mock plugin __call__ to return stereo silence
    sample_rate = (config or RenderConfig()).sample_rate
    duration = (config or RenderConfig()).duration_sec
    num_samples = int(duration * sample_rate)
    mock_plugin.return_value = np.zeros((2, num_samples), dtype=np.float32)

    # Directly construct, bypassing __init__'s plugin loading
    renderer = object.__new__(AudioRenderer)
    renderer._vst_path = vst_path
    renderer._config = config or RenderConfig()
    renderer._plugin = mock_plugin
    renderer._default_raw = {}
    return renderer


def _create_preset_file(directory: Path, name: str) -> Path:
    """Create a minimal .vital preset file."""
    filepath = directory / name
    preset_json = {
        "author": "Test",
        "settings": {
            "osc_1_on": 1.0,
            "modulations": [],
        },
    }
    filepath.write_text(json.dumps(preset_json), encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# AudioRenderer.__init__ tests
# ---------------------------------------------------------------------------

class TestAudioRendererInit:
    def test_raises_file_not_found_for_missing_vst(self, tmp_path: Path):
        fake_path = tmp_path / "nonexistent.vst3"
        with pytest.raises(FileNotFoundError, match="not found"):
            AudioRenderer(fake_path)

    def test_raises_runtime_error_on_load_failure(self, tmp_path: Path):
        vst_path = tmp_path / "Vital.vst3"
        vst_path.mkdir()
        # Create a mock pedalboard module with load_plugin that raises
        mock_pedalboard = MagicMock()
        mock_pedalboard.load_plugin.side_effect = Exception("load failed")
        with patch.dict("sys.modules", {"pedalboard": mock_pedalboard}):
            with pytest.raises(RuntimeError, match="Failed to load"):
                AudioRenderer(vst_path)

    def test_successful_init_with_mock(self, tmp_path: Path):
        vst_path = tmp_path / "Vital.vst3"
        vst_path.mkdir()
        mock_plugin = MagicMock()
        mock_pedalboard = MagicMock()
        mock_pedalboard.load_plugin.return_value = mock_plugin
        with patch.dict("sys.modules", {"pedalboard": mock_pedalboard}):
            renderer = AudioRenderer(vst_path)
        assert renderer._plugin is mock_plugin


# ---------------------------------------------------------------------------
# render_preset tests
# ---------------------------------------------------------------------------

class TestRenderPreset:
    def test_returns_true_on_success(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_path = _create_preset_file(tmp_path, "test.vital")
        output_path = tmp_path / "output" / "test.wav"

        # Mock _write_wav to avoid actual file I/O
        with patch.object(renderer, "_write_wav"):
            result = renderer.render_preset(preset_path, output_path)
        assert result is True

    def test_returns_false_on_failure(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        # Non-existent preset file
        preset_path = tmp_path / "nonexistent.vital"
        output_path = tmp_path / "output.wav"

        result = renderer.render_preset(preset_path, output_path)
        assert result is False

    def test_returns_false_when_plugin_raises(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        renderer._plugin.side_effect = RuntimeError("VST crash")
        preset_path = _create_preset_file(tmp_path, "crash.vital")
        output_path = tmp_path / "crash.wav"

        result = renderer.render_preset(preset_path, output_path)
        assert result is False


# ---------------------------------------------------------------------------
# Output filename mapping tests
# ---------------------------------------------------------------------------

class TestFilenameMapping:
    def test_vital_to_wav_mapping(self, tmp_path: Path):
        """Output filename should be same base name with .wav extension."""
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        _create_preset_file(preset_dir, "chorus_on_1.0.vital")

        with patch.object(renderer, "render_preset", return_value=True) as mock_render:
            renderer.render_batch(preset_dir, output_dir)

        # Check the output_path argument
        call_args = mock_render.call_args
        output_path = call_args[0][1]
        assert output_path.name == "chorus_on_1.0.wav"
        assert output_path.suffix == ".wav"

    def test_multiple_files_mapping(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        names = ["base_patch.vital", "reverb_on_1.0.vital", "delay_on_0.0.vital"]
        for name in names:
            _create_preset_file(preset_dir, name)

        with patch.object(renderer, "render_preset", return_value=True) as mock_render:
            renderer.render_batch(preset_dir, output_dir)

        output_names = [call[0][1].name for call in mock_render.call_args_list]
        assert "base_patch.wav" in output_names
        assert "reverb_on_1.0.wav" in output_names
        assert "delay_on_0.0.wav" in output_names


# ---------------------------------------------------------------------------
# render_batch tests
# ---------------------------------------------------------------------------

class TestRenderBatch:
    def test_processes_all_vital_files(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        for i in range(5):
            _create_preset_file(preset_dir, f"preset_{i}.vital")

        with patch.object(renderer, "render_preset", return_value=True):
            summary = renderer.render_batch(preset_dir, output_dir)

        assert summary.success_count == 5
        assert summary.failure_count == 0
        assert summary.failed_files == []

    def test_returns_correct_summary_with_failures(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        for i in range(4):
            _create_preset_file(preset_dir, f"preset_{i}.vital")

        # Simulate: first 2 succeed, last 2 fail
        with patch.object(
            renderer, "render_preset",
            side_effect=[True, True, False, False],
        ):
            summary = renderer.render_batch(preset_dir, output_dir)

        assert summary.success_count == 2
        assert summary.failure_count == 2
        assert len(summary.failed_files) == 2
        assert summary.success_count + summary.failure_count == 4

    def test_continues_after_individual_failures(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        for i in range(3):
            _create_preset_file(preset_dir, f"preset_{i}.vital")

        # First fails, rest succeed
        with patch.object(
            renderer, "render_preset",
            side_effect=[False, True, True],
        ) as mock_render:
            summary = renderer.render_batch(preset_dir, output_dir)

        # All 3 were attempted
        assert mock_render.call_count == 3
        assert summary.success_count == 2
        assert summary.failure_count == 1

    def test_empty_directory_returns_zero_summary(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        summary = renderer.render_batch(preset_dir, output_dir)
        assert summary.success_count == 0
        assert summary.failure_count == 0
        assert summary.failed_files == []

    def test_creates_output_directory(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "new_output_dir"

        _create_preset_file(preset_dir, "test.vital")

        with patch.object(renderer, "render_preset", return_value=True):
            renderer.render_batch(preset_dir, output_dir)

        assert output_dir.exists()

    def test_ignores_non_vital_files(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        _create_preset_file(preset_dir, "good.vital")
        (preset_dir / "readme.txt").write_text("not a preset")
        (preset_dir / "data.json").write_text("{}")

        with patch.object(renderer, "render_preset", return_value=True) as mock_render:
            summary = renderer.render_batch(preset_dir, output_dir)

        assert mock_render.call_count == 1
        assert summary.success_count == 1

    def test_failed_files_contains_correct_names(self, tmp_path: Path):
        renderer = _create_mock_renderer(tmp_path)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        _create_preset_file(preset_dir, "good.vital")
        _create_preset_file(preset_dir, "bad.vital")

        with patch.object(
            renderer, "render_preset",
            side_effect=[False, True],
        ):
            summary = renderer.render_batch(preset_dir, output_dir)

        # Files are sorted, so "bad.vital" comes first
        assert "bad.vital" in summary.failed_files
        assert "good.vital" not in summary.failed_files


# ---------------------------------------------------------------------------
# Timeout handling tests
# ---------------------------------------------------------------------------

class TestRenderTimeout:
    def test_timeout_returns_false(self, tmp_path: Path):
        config = RenderConfig(timeout_sec=0.5)
        renderer = _create_mock_renderer(tmp_path, config)
        preset_path = _create_preset_file(tmp_path, "slow.vital")
        output_path = tmp_path / "slow.wav"

        # Mock render_preset to sleep longer than timeout
        def slow_render(*args, **kwargs):
            time.sleep(2.0)
            return True

        with patch.object(renderer, "render_preset", side_effect=slow_render):
            result = renderer._render_with_timeout(preset_path, output_path)

        assert result is False

    def test_timeout_in_batch_skips_and_continues(self, tmp_path: Path):
        config = RenderConfig(timeout_sec=0.3)
        renderer = _create_mock_renderer(tmp_path, config)
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        output_dir = tmp_path / "audio"

        _create_preset_file(preset_dir, "fast.vital")
        _create_preset_file(preset_dir, "slow.vital")

        call_count = [0]

        def mock_render(preset_path, output_path):
            call_count[0] += 1
            if "slow" in str(preset_path):
                time.sleep(2.0)
                return True
            return True

        with patch.object(renderer, "render_preset", side_effect=mock_render):
            summary = renderer.render_batch(preset_dir, output_dir)

        # slow.vital should have timed out
        assert summary.failure_count >= 1
        assert "slow.vital" in summary.failed_files
