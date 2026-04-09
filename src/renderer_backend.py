"""
渲染后端抽象层。

封装 AudioRenderer 和 MultiConditionRenderer，提供稳定接口。
所有渲染失败均返回 RenderResult(success=False) 而非抛出异常，
确保流水线不会因单次渲染失败而中断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio_renderer import AudioRenderer, RenderConfig
from src.multi_condition_renderer import (
    MidiCondition,
    MultiConditionRenderer,
)

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    """单次渲染结果。

    Attributes:
        success: 渲染是否成功
        audio_path: 输出 WAV 文件路径（成功时非 None）
        audio: 原始音频数据（成功时从 WAV 文件加载）
        error: 错误信息（失败时非 None）
        midi_note: 渲染使用的 MIDI 音符编号
        midi_velocity: 渲染使用的 MIDI 力度
    """

    success: bool
    audio_path: Path | None
    audio: np.ndarray | None
    error: str | None
    midi_note: int
    midi_velocity: int


class RendererBackend:
    """渲染后端，封装 AudioRenderer 和 MultiConditionRenderer。

    提供稳定接口，支持后续 backend 替换而不影响
    PresetDatasetProducer 等上层逻辑。
    """

    def __init__(
        self,
        vital_vst_path: Path,
        render_config: RenderConfig | None = None,
        conditions: list[MidiCondition] | None = None,
    ) -> None:
        """初始化渲染后端。

        内部创建 AudioRenderer 和 MultiConditionRenderer 实例。

        Args:
            vital_vst_path: Vital VST3 插件路径
            render_config: 渲染配置，None 时使用默认配置
            conditions: 多条件渲染的 MIDI 条件列表，None 时使用默认 6 条件
        """
        self._config = render_config or RenderConfig()
        self._renderer = AudioRenderer(vital_vst_path, config=self._config)
        self._multi_renderer = MultiConditionRenderer(
            self._renderer, conditions=conditions
        )

    @property
    def renderer(self) -> AudioRenderer:
        """底层 AudioRenderer 实例（只读访问）。"""
        return self._renderer

    @property
    def multi_renderer(self) -> MultiConditionRenderer:
        """底层 MultiConditionRenderer 实例（只读访问）。"""
        return self._multi_renderer

    def _load_audio(self, audio_path: Path) -> np.ndarray | None:
        """从 WAV 文件加载原始音频数据。

        Args:
            audio_path: WAV 文件路径

        Returns:
            音频数据数组，加载失败时返回 None
        """
        try:
            audio, _ = sf.read(audio_path, dtype="float32")
            return audio
        except Exception as e:
            logger.warning("无法加载音频文件 %s: %s", audio_path, e)
            return None

    def render_single(
        self, preset_path: Path, output_path: Path
    ) -> RenderResult:
        """单条件渲染。

        使用 AudioRenderer.render_preset() 渲染单个预设。
        捕获所有异常，渲染失败时返回 RenderResult(success=False)。

        Args:
            preset_path: .vital 预设文件路径
            output_path: 输出 WAV 文件路径

        Returns:
            RenderResult 包含渲染状态和音频数据
        """
        midi_note = self._config.midi_note
        midi_velocity = self._config.velocity

        try:
            success = self._renderer.render_preset(preset_path, output_path)

            if success:
                audio = self._load_audio(output_path)
                return RenderResult(
                    success=True,
                    audio_path=output_path,
                    audio=audio,
                    error=None,
                    midi_note=midi_note,
                    midi_velocity=midi_velocity,
                )
            else:
                return RenderResult(
                    success=False,
                    audio_path=None,
                    audio=None,
                    error="render_preset returned False",
                    midi_note=midi_note,
                    midi_velocity=midi_velocity,
                )

        except Exception as e:
            logger.error(
                "渲染失败: preset=%s, error=%s", preset_path, e
            )
            return RenderResult(
                success=False,
                audio_path=None,
                audio=None,
                error=str(e),
                midi_note=midi_note,
                midi_velocity=midi_velocity,
            )

    def render_multi_condition(
        self, preset_path: Path, output_dir: Path, preset_id: str
    ) -> list[RenderResult]:
        """多条件渲染，返回每个条件的结果。

        使用 MultiConditionRenderer.render_preset() 渲染多个 MIDI 条件。
        捕获所有异常，每个条件的失败独立记录。

        Args:
            preset_path: .vital 预设文件路径
            output_dir: 音频输出目录
            preset_id: 预设唯一标识

        Returns:
            RenderResult 列表，每个条件一个结果
        """
        results: list[RenderResult] = []

        try:
            multi_result = self._multi_renderer.render_preset(
                preset_path, output_dir, preset_id
            )

            conditions = self._multi_renderer.conditions

            for condition in conditions:
                if condition.label in multi_result.condition_results:
                    audio_path = multi_result.condition_results[condition.label]
                    audio = self._load_audio(audio_path)
                    results.append(
                        RenderResult(
                            success=True,
                            audio_path=audio_path,
                            audio=audio,
                            error=None,
                            midi_note=condition.note,
                            midi_velocity=condition.velocity,
                        )
                    )
                else:
                    results.append(
                        RenderResult(
                            success=False,
                            audio_path=None,
                            audio=None,
                            error=f"Condition {condition.label} failed",
                            midi_note=condition.note,
                            midi_velocity=condition.velocity,
                        )
                    )

        except Exception as e:
            logger.error(
                "多条件渲染整体失败: preset=%s, error=%s", preset_path, e
            )
            # Return failure results for all conditions
            conditions = self._multi_renderer.conditions
            for condition in conditions:
                results.append(
                    RenderResult(
                        success=False,
                        audio_path=None,
                        audio=None,
                        error=str(e),
                        midi_note=condition.note,
                        midi_velocity=condition.velocity,
                    )
                )

        return results
