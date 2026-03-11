"""
多条件渲染模块。

对同一预设使用多个 MIDI 条件（音高、力度、时长）渲染，
生成多个音频样本以增加数据多样性。
内部复用现有 AudioRenderer 实例。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MidiCondition:
    """MIDI 渲染条件。

    Attributes:
        note: MIDI 音符编号
        velocity: 力度（0-127）
        duration_sec: 渲染时长（秒）
        label: 条件标签（如 "C3_v80"）
    """

    note: int
    velocity: int
    duration_sec: float
    label: str


@dataclass
class MultiConditionResult:
    """多条件渲染结果。

    Attributes:
        preset_id: 预设标识
        condition_results: 条件标签 → 音频文件路径（成功的条件）
        failed_conditions: 失败的条件标签列表
    """

    preset_id: str
    condition_results: dict[str, Path] = field(default_factory=dict)
    failed_conditions: list[str] = field(default_factory=list)


# 默认 6 条件配置：3 音高（C3/C4/C5）× 2 力度（80/120）
DEFAULT_CONDITIONS: list[MidiCondition] = [
    MidiCondition(note=48, velocity=80, duration_sec=2.0, label="C3_v80"),
    MidiCondition(note=48, velocity=120, duration_sec=2.0, label="C3_v120"),
    MidiCondition(note=60, velocity=80, duration_sec=2.0, label="C4_v80"),
    MidiCondition(note=60, velocity=120, duration_sec=2.0, label="C4_v120"),
    MidiCondition(note=72, velocity=80, duration_sec=2.0, label="C5_v80"),
    MidiCondition(note=72, velocity=120, duration_sec=2.0, label="C5_v120"),
]


class MultiConditionRenderer:
    """多条件渲染器。

    对同一预设使用多个 MIDI 条件渲染，生成多个音频样本。
    内部复用现有 AudioRenderer，通过临时修改 RenderConfig 的
    MIDI 参数实现多条件渲染。
    """

    def __init__(
        self,
        renderer,
        conditions: list[MidiCondition] | None = None,
    ) -> None:
        """初始化多条件渲染器。

        Args:
            renderer: 现有 AudioRenderer 实例
            conditions: MIDI 条件列表，None 时使用 DEFAULT_CONDITIONS
        """
        self._renderer = renderer
        self._conditions = conditions if conditions is not None else DEFAULT_CONDITIONS

    @property
    def conditions(self) -> list[MidiCondition]:
        """当前使用的 MIDI 条件列表。"""
        return self._conditions

    def render_preset(
        self,
        preset_path: Path,
        output_dir: Path,
        preset_id: str,
    ) -> MultiConditionResult:
        """为单个预设渲染所有条件。

        对每个 MIDI 条件，临时修改 renderer 的配置参数，
        渲染音频并保存。单个条件失败不影响其他条件。

        输出文件名格式：{preset_id}_{condition_label}.wav

        Args:
            preset_path: .vital 预设文件路径
            output_dir: 音频输出目录
            preset_id: 预设唯一标识

        Returns:
            MultiConditionResult 包含各条件的渲染结果
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = MultiConditionResult(preset_id=preset_id)

        # 保存原始配置，渲染完成后恢复
        original_note = self._renderer._config.midi_note
        original_velocity = self._renderer._config.velocity
        original_duration = self._renderer._config.duration_sec

        try:
            for condition in self._conditions:
                try:
                    # 临时修改渲染配置
                    self._renderer._config.midi_note = condition.note
                    self._renderer._config.velocity = condition.velocity
                    self._renderer._config.duration_sec = condition.duration_sec

                    # 生成输出文件路径
                    filename = f"{preset_id}_{condition.label}.wav"
                    output_path = output_dir / filename

                    # 调用 AudioRenderer 渲染
                    success = self._renderer.render_preset(preset_path, output_path)

                    if success:
                        result.condition_results[condition.label] = output_path
                        logger.debug(
                            "条件 %s 渲染成功: %s", condition.label, output_path
                        )
                    else:
                        result.failed_conditions.append(condition.label)
                        logger.warning(
                            "条件 %s 渲染失败（返回 False）: preset=%s",
                            condition.label,
                            preset_path,
                        )

                except Exception as e:
                    # 捕获异常，记录失败条件，继续渲染其他条件
                    result.failed_conditions.append(condition.label)
                    logger.error(
                        "条件 %s 渲染异常: preset=%s, error=%s",
                        condition.label,
                        preset_path,
                        e,
                    )
        finally:
            # 恢复原始配置
            self._renderer._config.midi_note = original_note
            self._renderer._config.velocity = original_velocity
            self._renderer._config.duration_sec = original_duration

        logger.info(
            "预设 %s 多条件渲染完成: 成功=%d, 失败=%d",
            preset_id,
            len(result.condition_results),
            len(result.failed_conditions),
        )

        return result
