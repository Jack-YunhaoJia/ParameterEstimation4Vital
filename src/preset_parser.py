"""
Vital 预设解析与序列化模块。

负责 .vital 预设文件（JSON 格式）的读取、解析、验证和写入。
所有其他组件的基础依赖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PresetParseError(Exception):
    """Vital 预设解析错误，包含文件路径和具体违规描述。"""

    def __init__(self, filepath: str | Path, message: str) -> None:
        self.filepath = str(filepath)
        self.message = message
        super().__init__(f"Error parsing '{self.filepath}': {message}")


@dataclass
class VitalPreset:
    """Vital 预设的结构化表示。

    Attributes:
        settings: 参数键值对字典（772 个连续参数，不含 modulations/wavetables）
        modulations: 调制槽位列表（64 个槽位，嵌套在原始 JSON 的 settings 内部）
        extra: 原始 JSON 中的其他顶层键（如 author, preset_name, wavetables 等）
    """

    settings: dict[str, float | int | str | list]
    modulations: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class PresetParser:
    """Vital 预设文件的解析器和序列化器。"""

    EFFECT_SWITCHES: list[str] = [
        "chorus_on",
        "compressor_on",
        "delay_on",
        "distortion_on",
        "eq_on",
        "flanger_on",
        "phaser_on",
        "reverb_on",
        "filter_fx_on",
    ]

    def parse(self, filepath: Path) -> VitalPreset:
        """解析 .vital 文件为 VitalPreset 对象。

        读取 JSON 文件，从 settings 中提取 modulations 和 wavetables，
        将剩余参数作为 settings，其他顶层键存入 extra。

        Args:
            filepath: .vital 文件路径

        Returns:
            解析后的 VitalPreset 对象

        Raises:
            PresetParseError: JSON 无效或缺少 settings 键
        """
        filepath = Path(filepath)

        # 读取文件
        try:
            text = filepath.read_text(encoding="utf-8")
        except (OSError, IOError) as e:
            raise PresetParseError(filepath, f"Cannot read file: {e}")

        # 解析 JSON
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise PresetParseError(filepath, f"Invalid JSON: {e}")

        if not isinstance(raw, dict):
            raise PresetParseError(filepath, "JSON root must be an object")

        # 验证 settings 键存在
        if "settings" not in raw:
            raise PresetParseError(filepath, "Missing required key: 'settings'")

        raw_settings = raw["settings"]
        if not isinstance(raw_settings, dict):
            raise PresetParseError(
                filepath, "'settings' must be a dictionary"
            )

        # 从 settings 中提取 modulations（嵌套在 settings 内部）
        modulations = raw_settings.pop("modulations", [])
        if not isinstance(modulations, list):
            modulations = []

        # 从 settings 中提取 wavetables（嵌套在 settings 内部）
        wavetables = raw_settings.pop("wavetables", None)

        # 剩余的 settings 就是纯参数键值对
        settings = dict(raw_settings)

        # 构建 extra：原始 JSON 中除 settings 外的所有顶层键 + wavetables
        extra: dict[str, Any] = {}
        for key, value in raw.items():
            if key != "settings":
                extra[key] = value
        if wavetables is not None:
            extra["wavetables"] = wavetables

        return VitalPreset(
            settings=settings,
            modulations=modulations,
            extra=extra,
        )

    def serialize(self, preset: VitalPreset, filepath: Path) -> None:
        """将 VitalPreset 序列化为 .vital JSON 文件。

        重建原始 JSON 结构：将 modulations 和 wavetables 放回 settings 内部，
        其他 extra 键放在顶层。

        Args:
            preset: 要序列化的 VitalPreset 对象
            filepath: 输出文件路径
        """
        filepath = Path(filepath)

        # 重建 settings：参数 + modulations + wavetables
        rebuilt_settings: dict[str, Any] = dict(preset.settings)
        rebuilt_settings["modulations"] = list(preset.modulations)

        # 如果 extra 中有 wavetables，放回 settings 内部
        wavetables = preset.extra.get("wavetables")
        if wavetables is not None:
            rebuilt_settings["wavetables"] = wavetables

        # 构建顶层 JSON 对象
        output: dict[str, Any] = {}
        for key, value in preset.extra.items():
            if key != "wavetables":
                output[key] = value
        output["settings"] = rebuilt_settings

        # 写入文件
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def validate_effect_switches(self, preset: VitalPreset) -> bool:
        """验证预设包含所有 9 个 Effect_Switch 键。

        Args:
            preset: 要验证的 VitalPreset 对象

        Returns:
            True 如果所有 9 个 Effect_Switch 键都存在于 settings 中
        """
        return all(
            switch in preset.settings for switch in self.EFFECT_SWITCHES
        )
