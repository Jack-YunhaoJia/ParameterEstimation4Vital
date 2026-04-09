"""
MultiConditionRenderer 单元测试。

使用 mock 对象替代实际的 AudioRenderer（需要 VST 插件），验证：
- DEFAULT_CONDITIONS 默认条件配置
- MultiConditionRenderer 初始化（默认和自定义条件）
- render_preset 全部成功场景
- render_preset 部分失败场景（容错性）
- 输出文件命名格式
- failed_conditions 记录
- 成功数 + 失败数 = 总条件数
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from src.multi_condition_renderer import (
    DEFAULT_CONDITIONS,
    MidiCondition,
    MultiConditionRenderer,
    MultiConditionResult,
)


# ---------------------------------------------------------------------------
# 辅助函数：创建 mock AudioRenderer
# ---------------------------------------------------------------------------

def _create_mock_audio_renderer(
    render_success: bool = True,
    fail_on_labels: list[str] | None = None,
) -> MagicMock:
    """创建 mock AudioRenderer 实例。

    Args:
        render_success: 默认渲染是否成功
        fail_on_labels: 指定哪些条件标签会导致渲染失败（抛出异常）
    """
    fail_on_labels = fail_on_labels or []

    @dataclass
    class MockConfig:
        """模拟 RenderConfig。"""
        midi_note: int = 60
        velocity: int = 100
        duration_sec: float = 2.0
        sample_rate: int = 44100
        timeout_sec: float = 30.0

    mock_renderer = MagicMock()
    mock_renderer._config = MockConfig()

    # 记录每次调用时的条件参数，用于验证
    call_records = []

    def mock_render_preset(preset_path, output_path, note_off_time=None, target_length_samples=None):
        """模拟渲染，根据当前配置判断是否失败。"""
        current_note = mock_renderer._config.midi_note
        current_velocity = mock_renderer._config.velocity
        current_duration = mock_renderer._config.duration_sec

        call_records.append({
            "note": current_note,
            "velocity": current_velocity,
            "duration": current_duration,
            "output_path": output_path,
        })

        # 检查当前条件是否在失败列表中
        for cond in DEFAULT_CONDITIONS:
            if (cond.note == current_note
                    and cond.velocity == current_velocity
                    and cond.label in fail_on_labels):
                raise RuntimeError(f"模拟渲染失败: {cond.label}")

        return render_success

    mock_renderer.render_preset = mock_render_preset
    mock_renderer._call_records = call_records

    return mock_renderer


# ---------------------------------------------------------------------------
# DEFAULT_CONDITIONS 测试
# ---------------------------------------------------------------------------

class TestDefaultConditions:
    """测试默认条件配置。"""

    def test_has_6_conditions(self):
        """默认条件列表应包含 6 个条件。"""
        assert len(DEFAULT_CONDITIONS) == 6

    def test_c3_v80_values(self):
        """C3_v80 条件：note=48, velocity=80, duration=2.0。"""
        c = DEFAULT_CONDITIONS[0]
        assert c.note == 48
        assert c.velocity == 80
        assert c.duration_sec == 2.0
        assert c.label == "C3_v80"

    def test_c3_v120_values(self):
        """C3_v120 条件：note=48, velocity=120, duration=2.0。"""
        c = DEFAULT_CONDITIONS[1]
        assert c.note == 48
        assert c.velocity == 120
        assert c.duration_sec == 2.0
        assert c.label == "C3_v120"

    def test_c4_v80_values(self):
        """C4_v80 条件：note=60, velocity=80, duration=2.0。"""
        c = DEFAULT_CONDITIONS[2]
        assert c.note == 60
        assert c.velocity == 80
        assert c.duration_sec == 2.0
        assert c.label == "C4_v80"

    def test_c4_v120_values(self):
        """C4_v120 条件：note=60, velocity=120, duration=2.0。"""
        c = DEFAULT_CONDITIONS[3]
        assert c.note == 60
        assert c.velocity == 120
        assert c.duration_sec == 2.0
        assert c.label == "C4_v120"

    def test_c5_v80_values(self):
        """C5_v80 条件：note=72, velocity=80, duration=2.0。"""
        c = DEFAULT_CONDITIONS[4]
        assert c.note == 72
        assert c.velocity == 80
        assert c.duration_sec == 2.0
        assert c.label == "C5_v80"

    def test_c5_v120_values(self):
        """C5_v120 条件：note=72, velocity=120, duration=2.0。"""
        c = DEFAULT_CONDITIONS[5]
        assert c.note == 72
        assert c.velocity == 120
        assert c.duration_sec == 2.0
        assert c.label == "C5_v120"

    def test_all_durations_are_2_seconds(self):
        """所有默认条件的时长均为 2.0 秒。"""
        for c in DEFAULT_CONDITIONS:
            assert c.duration_sec == 2.0

    def test_three_pitches_two_velocities(self):
        """3 个音高 × 2 个力度 = 6 个条件。"""
        notes = {c.note for c in DEFAULT_CONDITIONS}
        velocities = {c.velocity for c in DEFAULT_CONDITIONS}
        assert notes == {48, 60, 72}
        assert velocities == {80, 120}


# ---------------------------------------------------------------------------
# MultiConditionRenderer 初始化测试
# ---------------------------------------------------------------------------

class TestMultiConditionRendererInit:
    """测试 MultiConditionRenderer 初始化。"""

    def test_default_conditions(self):
        """conditions=None 时使用 DEFAULT_CONDITIONS。"""
        mock_renderer = _create_mock_audio_renderer()
        mcr = MultiConditionRenderer(mock_renderer)
        assert mcr.conditions is DEFAULT_CONDITIONS
        assert len(mcr.conditions) == 6

    def test_custom_conditions(self):
        """传入自定义条件列表。"""
        custom = [
            MidiCondition(note=60, velocity=100, duration_sec=1.0, label="test"),
        ]
        mock_renderer = _create_mock_audio_renderer()
        mcr = MultiConditionRenderer(mock_renderer, conditions=custom)
        assert mcr.conditions is custom
        assert len(mcr.conditions) == 1

    def test_empty_conditions_list(self):
        """传入空条件列表（不使用默认）。"""
        mock_renderer = _create_mock_audio_renderer()
        mcr = MultiConditionRenderer(mock_renderer, conditions=[])
        assert mcr.conditions == []


# ---------------------------------------------------------------------------
# render_preset 全部成功测试
# ---------------------------------------------------------------------------

class TestRenderPresetAllSuccess:
    """测试所有条件渲染成功的场景。"""

    def test_all_conditions_succeed(self, tmp_path: Path):
        """所有条件渲染成功时，condition_results 包含所有条件。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "preset_001")

        assert result.preset_id == "preset_001"
        assert len(result.condition_results) == 6
        assert len(result.failed_conditions) == 0

    def test_condition_results_keys(self, tmp_path: Path):
        """成功条件的键应为条件标签。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        expected_labels = {"C3_v80", "C3_v120", "C4_v80", "C4_v120", "C5_v80", "C5_v120"}
        assert set(result.condition_results.keys()) == expected_labels

    def test_creates_output_directory(self, tmp_path: Path):
        """输出目录不存在时应自动创建。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "nested" / "output"

        mcr.render_preset(preset_path, output_dir, "p1")
        assert output_dir.exists()


# ---------------------------------------------------------------------------
# 输出文件命名格式测试
# ---------------------------------------------------------------------------

class TestOutputFileNaming:
    """测试输出文件命名格式 {preset_id}_{condition_label}.wav。"""

    def test_file_naming_format(self, tmp_path: Path):
        """输出文件名应为 {preset_id}_{condition_label}.wav。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "preset_001")

        for label, path in result.condition_results.items():
            expected_name = f"preset_001_{label}.wav"
            assert path.name == expected_name

    def test_file_paths_in_output_dir(self, tmp_path: Path):
        """所有输出文件路径应在 output_dir 下。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        for path in result.condition_results.values():
            assert path.parent == output_dir

    def test_custom_preset_id_in_filename(self, tmp_path: Path):
        """自定义 preset_id 应正确出现在文件名中。"""
        custom_conditions = [
            MidiCondition(note=60, velocity=100, duration_sec=1.0, label="test_cond"),
        ]
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        mcr = MultiConditionRenderer(mock_renderer, conditions=custom_conditions)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "my_preset_42")

        assert "test_cond" in result.condition_results
        assert result.condition_results["test_cond"].name == "my_preset_42_test_cond.wav"


# ---------------------------------------------------------------------------
# render_preset 部分失败测试（容错性）
# ---------------------------------------------------------------------------

class TestRenderPresetPartialFailure:
    """测试部分条件渲染失败的容错行为。"""

    def test_some_conditions_fail_by_exception(self, tmp_path: Path):
        """部分条件抛出异常时，其他条件仍然成功。"""
        mock_renderer = _create_mock_audio_renderer(
            fail_on_labels=["C3_v80", "C5_v120"]
        )
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        # 2 个失败，4 个成功
        assert len(result.failed_conditions) == 2
        assert len(result.condition_results) == 4
        assert "C3_v80" in result.failed_conditions
        assert "C5_v120" in result.failed_conditions

    def test_failed_conditions_recorded(self, tmp_path: Path):
        """失败条件应记录到 failed_conditions 列表。"""
        mock_renderer = _create_mock_audio_renderer(
            fail_on_labels=["C4_v80"]
        )
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        assert "C4_v80" in result.failed_conditions
        assert "C4_v80" not in result.condition_results

    def test_success_plus_failure_equals_total(self, tmp_path: Path):
        """成功条件数 + 失败条件数 = 总条件数。"""
        mock_renderer = _create_mock_audio_renderer(
            fail_on_labels=["C3_v120", "C5_v80"]
        )
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        total = len(result.condition_results) + len(result.failed_conditions)
        assert total == len(DEFAULT_CONDITIONS)

    def test_render_returns_false_counts_as_failure(self, tmp_path: Path):
        """render_preset 返回 False 时应计为失败。"""
        mock_renderer = _create_mock_audio_renderer(render_success=False)
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        assert len(result.condition_results) == 0
        assert len(result.failed_conditions) == 6

    def test_all_conditions_fail(self, tmp_path: Path):
        """所有条件都失败时，condition_results 为空。"""
        all_labels = [c.label for c in DEFAULT_CONDITIONS]
        mock_renderer = _create_mock_audio_renderer(
            fail_on_labels=all_labels
        )
        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        result = mcr.render_preset(preset_path, output_dir, "p1")

        assert len(result.condition_results) == 0
        assert len(result.failed_conditions) == 6


# ---------------------------------------------------------------------------
# 配置恢复测试
# ---------------------------------------------------------------------------

class TestConfigRestoration:
    """测试渲染完成后原始配置是否恢复。"""

    def test_config_restored_after_success(self, tmp_path: Path):
        """渲染成功后，renderer 的配置应恢复为原始值。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        original_note = mock_renderer._config.midi_note
        original_velocity = mock_renderer._config.velocity
        original_duration = mock_renderer._config.duration_sec

        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        assert mock_renderer._config.midi_note == original_note
        assert mock_renderer._config.velocity == original_velocity
        assert mock_renderer._config.duration_sec == original_duration

    def test_config_restored_after_failure(self, tmp_path: Path):
        """渲染失败后，renderer 的配置也应恢复为原始值。"""
        mock_renderer = _create_mock_audio_renderer(
            fail_on_labels=["C3_v80"]
        )
        original_note = mock_renderer._config.midi_note
        original_velocity = mock_renderer._config.velocity
        original_duration = mock_renderer._config.duration_sec

        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        assert mock_renderer._config.midi_note == original_note
        assert mock_renderer._config.velocity == original_velocity
        assert mock_renderer._config.duration_sec == original_duration


# ---------------------------------------------------------------------------
# 自适应模式集成测试
# ---------------------------------------------------------------------------

class TestAdaptiveModeIntegration:
    """测试 MultiConditionRenderer 自适应模式集成。"""

    def test_adaptive_mode_calls_compute_timing(self, tmp_path: Path):
        """自适应模式下，timing_calculator.compute_timing 应被调用。"""
        from unittest.mock import MagicMock as Mock
        from src.adaptive_timing import AdaptiveConfig, AdaptiveTiming, AdaptiveTimingCalculator

        mock_renderer = _create_mock_audio_renderer(render_success=True)

        mock_calculator = Mock(spec=AdaptiveTimingCalculator)
        mock_calculator._config = AdaptiveConfig(target_length_sec=None)
        mock_calculator.compute_timing.return_value = AdaptiveTiming(
            attack_sec=0.5,
            decay_sec=0.3,
            sustain=0.8,
            release_sec=1.0,
            note_off=1.0,
            total_duration=2.1,
        )

        mcr = MultiConditionRenderer(mock_renderer, timing_calculator=mock_calculator)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        mock_calculator.compute_timing.assert_called_once_with(preset_path)

    def test_adaptive_mode_sets_duration_from_timing(self, tmp_path: Path):
        """自适应模式下，renderer 的 duration_sec 应设为 timing.total_duration。"""
        from unittest.mock import MagicMock as Mock
        from src.adaptive_timing import AdaptiveConfig, AdaptiveTiming, AdaptiveTimingCalculator

        mock_renderer = _create_mock_audio_renderer(render_success=True)
        durations_used = []

        original_render = mock_renderer.render_preset

        def tracking_render(preset_path, output_path, note_off_time=None, target_length_samples=None):
            durations_used.append(mock_renderer._config.duration_sec)
            return original_render(preset_path, output_path, note_off_time=note_off_time, target_length_samples=target_length_samples)

        mock_renderer.render_preset = tracking_render

        mock_calculator = Mock(spec=AdaptiveTimingCalculator)
        mock_calculator._config = AdaptiveConfig(target_length_sec=None)
        mock_calculator.compute_timing.return_value = AdaptiveTiming(
            attack_sec=0.5,
            decay_sec=0.3,
            sustain=0.8,
            release_sec=1.0,
            note_off=1.0,
            total_duration=5.5,
        )

        mcr = MultiConditionRenderer(mock_renderer, timing_calculator=mock_calculator)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        # All 6 conditions should use total_duration=5.5
        assert all(d == 5.5 for d in durations_used)
        assert len(durations_used) == 6

    def test_adaptive_mode_passes_note_off_and_target(self, tmp_path: Path):
        """自适应模式下，note_off_time 和 target_length_samples 应正确传递。"""
        from unittest.mock import MagicMock as Mock
        from src.adaptive_timing import AdaptiveConfig, AdaptiveTiming, AdaptiveTimingCalculator

        mock_renderer = _create_mock_audio_renderer(render_success=True)
        render_kwargs_list = []

        def capturing_render(preset_path, output_path, note_off_time=None, target_length_samples=None):
            render_kwargs_list.append({
                "note_off_time": note_off_time,
                "target_length_samples": target_length_samples,
            })
            return True

        mock_renderer.render_preset = capturing_render

        mock_calculator = Mock(spec=AdaptiveTimingCalculator)
        mock_calculator._config = AdaptiveConfig(target_length_sec=4.0)
        mock_calculator.compute_timing.return_value = AdaptiveTiming(
            attack_sec=0.5,
            decay_sec=0.3,
            sustain=0.8,
            release_sec=1.0,
            note_off=1.0,
            total_duration=2.1,
        )

        mcr = MultiConditionRenderer(mock_renderer, timing_calculator=mock_calculator)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        expected_target_samples = int(4.0 * 44100)
        for kwargs in render_kwargs_list:
            assert kwargs["note_off_time"] == 1.0
            assert kwargs["target_length_samples"] == expected_target_samples

    def test_adaptive_mode_no_target_length(self, tmp_path: Path):
        """target_length_sec=None 时，target_length_samples 应为 None。"""
        from unittest.mock import MagicMock as Mock
        from src.adaptive_timing import AdaptiveConfig, AdaptiveTiming, AdaptiveTimingCalculator

        mock_renderer = _create_mock_audio_renderer(render_success=True)
        render_kwargs_list = []

        def capturing_render(preset_path, output_path, note_off_time=None, target_length_samples=None):
            render_kwargs_list.append({
                "note_off_time": note_off_time,
                "target_length_samples": target_length_samples,
            })
            return True

        mock_renderer.render_preset = capturing_render

        mock_calculator = Mock(spec=AdaptiveTimingCalculator)
        mock_calculator._config = AdaptiveConfig(target_length_sec=None)
        mock_calculator.compute_timing.return_value = AdaptiveTiming(
            attack_sec=0.5,
            decay_sec=0.3,
            sustain=0.8,
            release_sec=1.0,
            note_off=1.0,
            total_duration=2.1,
        )

        mcr = MultiConditionRenderer(mock_renderer, timing_calculator=mock_calculator)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        for kwargs in render_kwargs_list:
            assert kwargs["note_off_time"] == 1.0
            assert kwargs["target_length_samples"] is None


class TestFixedModeRegression:
    """固定模式回归测试：无 timing_calculator 时行为与原有一致。"""

    def test_fixed_mode_uses_condition_duration(self, tmp_path: Path):
        """固定模式下，duration_sec 应来自 condition.duration_sec。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        durations_used = []

        original_render = mock_renderer.render_preset

        def tracking_render(preset_path, output_path, note_off_time=None, target_length_samples=None):
            durations_used.append(mock_renderer._config.duration_sec)
            return original_render(preset_path, output_path, note_off_time=note_off_time, target_length_samples=target_length_samples)

        mock_renderer.render_preset = tracking_render

        mcr = MultiConditionRenderer(mock_renderer)  # No timing_calculator

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        # All default conditions have duration_sec=2.0
        assert all(d == 2.0 for d in durations_used)

    def test_fixed_mode_no_note_off_or_target(self, tmp_path: Path):
        """固定模式下，note_off_time 和 target_length_samples 应为 None。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        render_kwargs_list = []

        def capturing_render(preset_path, output_path, note_off_time=None, target_length_samples=None):
            render_kwargs_list.append({
                "note_off_time": note_off_time,
                "target_length_samples": target_length_samples,
            })
            return True

        mock_renderer.render_preset = capturing_render

        mcr = MultiConditionRenderer(mock_renderer)  # No timing_calculator

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        for kwargs in render_kwargs_list:
            assert kwargs["note_off_time"] is None
            assert kwargs["target_length_samples"] is None

    def test_fixed_mode_config_restored(self, tmp_path: Path):
        """固定模式下，渲染后配置应恢复。"""
        mock_renderer = _create_mock_audio_renderer(render_success=True)
        original_note = mock_renderer._config.midi_note
        original_velocity = mock_renderer._config.velocity
        original_duration = mock_renderer._config.duration_sec

        mcr = MultiConditionRenderer(mock_renderer)

        preset_path = tmp_path / "test.vital"
        preset_path.write_text("{}")
        output_dir = tmp_path / "output"

        mcr.render_preset(preset_path, output_dir, "p1")

        assert mock_renderer._config.midi_note == original_note
        assert mock_renderer._config.velocity == original_velocity
        assert mock_renderer._config.duration_sec == original_duration
