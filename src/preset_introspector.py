"""
Vital 预设解析审计模块。

对单个 VitalPreset 执行参数分类审计，统计 applied / skipped / unsupported
参数、active modulations 和 wavetables，输出结构化 IntrospectionReport。

分类逻辑复用 audio_renderer.py 中的 _vital_name_to_pedalboard() 函数。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.audio_renderer import _vital_name_to_pedalboard
from src.preset_parser import PresetParser, VitalPreset

logger = logging.getLogger(__name__)


@dataclass
class IntrospectionReport:
    """单个预设的审计报告。

    Attributes:
        preset_path: 预设文件路径
        total_settings_count: settings 中的总参数数量
        applied_count: 可被 renderer 设置的 int/float 参数数量
        skipped_count: 存在映射但被渲染器跳过的参数数量
        unsupported_count: 无法映射到 pedalboard 参数名的参数数量
        applied_params: applied 参数名列表
        skipped_params: skipped 参数名列表
        unsupported_params: unsupported 参数名列表
        active_modulation_count: active modulation 槽位数量
        active_modulations: active modulation 详情列表
        wavetable_count: wavetable 数量
        wavetable_names: wavetable 名称列表
    """

    preset_path: str
    total_settings_count: int
    applied_count: int
    skipped_count: int
    unsupported_count: int
    applied_params: list[str] = field(default_factory=list)
    skipped_params: list[str] = field(default_factory=list)
    unsupported_params: list[str] = field(default_factory=list)
    active_modulation_count: int = 0
    active_modulations: list[dict] = field(default_factory=list)
    wavetable_count: int = 0
    wavetable_names: list[str] = field(default_factory=list)


class PresetIntrospector:
    """对 VitalPreset 执行参数分类审计。

    分类逻辑：
    - applied: isinstance(value, (int, float)) 且 _vital_name_to_pedalboard(key)
               返回非 None（有 pedalboard 映射）
    - skipped: 当前实现中为空（需要加载 pedalboard 插件获取参数范围才能
               判断 _vital_value_to_raw() 是否返回 None）
    - unsupported: 非 int/float 值，或无 pedalboard 映射
    """

    def __init__(self, parser: PresetParser) -> None:
        """初始化 PresetIntrospector。

        Args:
            parser: PresetParser 实例，用于预设解析
        """
        self._parser = parser

    def introspect(
        self, preset: VitalPreset, preset_path: str = ""
    ) -> IntrospectionReport:
        """对 VitalPreset 执行参数分类审计。

        对每个 settings key 分类为 applied / skipped / unsupported，
        统计 active modulations 和 wavetables。

        Args:
            preset: 要审计的 VitalPreset 对象
            preset_path: 预设文件路径（用于报告）

        Returns:
            结构化的 IntrospectionReport
        """
        applied_params: list[str] = []
        skipped_params: list[str] = []
        unsupported_params: list[str] = []

        for key, value in preset.settings.items():
            # 非 int/float 值 → unsupported
            if not isinstance(value, (int, float)):
                unsupported_params.append(key)
                continue

            # 无 pedalboard 映射 → unsupported
            pb_name = _vital_name_to_pedalboard(key)
            if pb_name is None:
                unsupported_params.append(key)
                continue

            # 有映射的 int/float 参数 → applied
            applied_params.append(key)

        # 统计 active modulations（source 和 destination 均非空）
        active_modulations: list[dict] = []
        for slot_idx, mod in enumerate(preset.modulations):
            source = mod.get("source", "")
            destination = mod.get("destination", "")
            if source and destination:
                active_modulations.append(
                    {
                        "source": source,
                        "destination": destination,
                        "slot": slot_idx,
                    }
                )

        # 统计 wavetables
        wavetables = preset.extra.get("wavetables", [])
        wavetable_names: list[str] = []
        for idx, wt in enumerate(wavetables):
            if isinstance(wt, dict):
                name = wt.get("name", f"wavetable_{idx}")
            else:
                name = f"wavetable_{idx}"
            wavetable_names.append(name)

        total = len(preset.settings)

        report = IntrospectionReport(
            preset_path=preset_path,
            total_settings_count=total,
            applied_count=len(applied_params),
            skipped_count=len(skipped_params),
            unsupported_count=len(unsupported_params),
            applied_params=applied_params,
            skipped_params=skipped_params,
            unsupported_params=unsupported_params,
            active_modulation_count=len(active_modulations),
            active_modulations=active_modulations,
            wavetable_count=len(wavetable_names),
            wavetable_names=wavetable_names,
        )

        logger.info(
            "Introspection for '%s': %d total, %d applied, %d skipped, "
            "%d unsupported, %d active modulations, %d wavetables",
            preset_path,
            total,
            report.applied_count,
            report.skipped_count,
            report.unsupported_count,
            report.active_modulation_count,
            report.wavetable_count,
        )

        return report
