"""
变异预设写入模块。

将掩码变异后的预设写回 `.vital` 文件，保留所有未修改的 modulations 和 wavetables 数据。
使用 PresetParser.serialize() 进行序列化。

Requirements: 7.1, 7.2, 7.3
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.preset_parser import PresetParser
from src.route_mask_augmenter import MaskedVariant

logger = logging.getLogger(__name__)


class MutatedPresetWriter:
    """将掩码变异后的预设写回 `.vital` 文件。

    使用 PresetParser.serialize() 序列化变异预设，
    文件名格式为 {base_preset_id}_mask_{variant_id:04d}.vital。
    保留所有未修改的 modulations 和 wavetables 数据。
    """

    def __init__(self, parser: PresetParser) -> None:
        self._parser = parser

    def write(self, variant: MaskedVariant, output_dir: Path) -> Path:
        """将变异预设写入 .vital 文件。

        Args:
            variant: 掩码变异结果，包含变异后的 VitalPreset 和元数据
            output_dir: 输出目录路径

        Returns:
            写入的文件路径
        """
        output_dir = Path(output_dir)
        filename = (
            f"{variant.metadata.base_preset_id}"
            f"_mask_{variant.metadata.variant_id:04d}.vital"
        )
        filepath = output_dir / filename

        self._parser.serialize(variant.preset, filepath)

        logger.debug(
            "Wrote variant %s/%04d to %s",
            variant.metadata.base_preset_id,
            variant.metadata.variant_id,
            filepath,
        )
        return filepath

    def write_batch(
        self, variants: list[MaskedVariant], output_dir: Path
    ) -> list[Path]:
        """批量写入变异预设。

        Args:
            variants: MaskedVariant 列表
            output_dir: 输出目录路径

        Returns:
            写入的文件路径列表
        """
        return [self.write(v, output_dir) for v in variants]
