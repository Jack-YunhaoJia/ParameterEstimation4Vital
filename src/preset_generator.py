"""
Vital 预设生成模块。

基于 Base_Patch 模板生成控制变量预设，用于 Phase 0 可行性验证实验。
每个预设仅在一个 Effect_Switch 上与 Base_Patch 不同。
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from src.preset_parser import PresetParser, VitalPreset

logger = logging.getLogger(__name__)


def _load_default_template() -> dict[str, float]:
    """从 vital_param_inventory.json 加载参数名列表，构建全零默认模板。

    对关键参数设置合理的 Vital 初始默认值。
    """
    inventory_path = Path(__file__).parent.parent / "vital_param_inventory.json"
    if inventory_path.exists():
        with open(inventory_path, "r", encoding="utf-8") as f:
            inventory = json.load(f)
        param_names: list[str] = inventory.get("continuous_params", [])
    else:
        param_names = []

    # 以 0.0 初始化所有参数
    template: dict[str, float] = {name: 0.0 for name in param_names}

    # 设置 Vital 合理的默认值（基于 Init Patch 的典型值）
    _vital_defaults: dict[str, float] = {
        # 振荡器 1 - 开启，默认电平
        "osc_1_on": 1.0,
        "osc_1_level": 0.70710678118,  # -3dB, sqrt(0.5)
        "osc_1_midi_track": 1.0,
        "osc_1_random_phase": 1.0,
        "osc_1_smooth_interpolation": 1.0,
        "osc_1_unison_voices": 1.0,
        "osc_1_wave_frame": 0.0,
        "osc_1_transpose": 0.0,
        "osc_1_tune": 0.0,
        "osc_1_pan": 0.0,
        "osc_1_phase": 0.0,
        "osc_1_detune_range": 2.0,
        "osc_1_detune_power": 1.5,
        "osc_1_unison_detune": 10.0,
        "osc_1_unison_blend": 0.8,
        # 振荡器 2/3 - 关闭
        "osc_2_on": 0.0,
        "osc_2_level": 0.70710678118,
        "osc_2_midi_track": 1.0,
        "osc_2_random_phase": 1.0,
        "osc_2_smooth_interpolation": 1.0,
        "osc_2_unison_voices": 1.0,
        "osc_3_on": 0.0,
        "osc_3_level": 0.70710678118,
        "osc_3_midi_track": 1.0,
        "osc_3_random_phase": 1.0,
        "osc_3_smooth_interpolation": 1.0,
        "osc_3_unison_voices": 1.0,
        # 滤波器 1 - 开启，默认低通
        "filter_1_on": 1.0,
        "filter_1_cutoff": 60.0,
        "filter_1_resonance": 0.0,
        "filter_1_drive": 0.0,
        "filter_1_mix": 1.0,
        "filter_1_model": 0.0,
        "filter_1_style": 0.0,
        "filter_1_keytrack": 0.0,
        "filter_1_blend": 0.0,
        # 滤波器 2 - 关闭
        "filter_2_on": 0.0,
        "filter_2_cutoff": 60.0,
        "filter_2_mix": 1.0,
        # 包络 1 (Amp Envelope) - 典型 ADSR
        "env_1_attack": 0.0,
        "env_1_decay": 1.5,
        "env_1_sustain": 1.0,
        "env_1_release": 0.3,
        "env_1_attack_power": 0.0,
        "env_1_decay_power": 0.0,
        "env_1_release_power": 0.0,
        "env_1_delay": 0.0,
        "env_1_hold": 0.0,
        # 包络 2-6 默认
        "env_2_attack": 0.0,
        "env_2_decay": 1.5,
        "env_2_sustain": 1.0,
        "env_2_release": 0.3,
        # 所有效果器 - 关闭
        "chorus_on": 0.0,
        "compressor_on": 0.0,
        "delay_on": 0.0,
        "distortion_on": 0.0,
        "eq_on": 0.0,
        "flanger_on": 0.0,
        "phaser_on": 0.0,
        "reverb_on": 0.0,
        "filter_fx_on": 0.0,
        # 效果器 dry/wet 默认值
        "chorus_dry_wet": 0.5,
        "delay_dry_wet": 0.5,
        "flanger_dry_wet": 0.5,
        "phaser_dry_wet": 0.5,
        "reverb_dry_wet": 0.5,
        "distortion_mix": 0.5,
        "compressor_mix": 1.0,
        "filter_fx_mix": 1.0,
        # 全局
        "beats_per_minute": 2.0,  # Vital 内部 BPM 编码
        "polyphony": 1.0,
        "oversampling": 1.0,
        "volume": 0.70710678118,
        "voice_amplitude": 0.0,
        "voice_tune": 0.0,
        "voice_transpose": 0.0,
        "velocity_track": 0.0,
        "pitch_bend_range": 2.0,
        "stereo_routing": 0.0,
        "stereo_mode": 0.0,
        # LFO 默认
        "lfo_1_frequency": 2.0,
        "lfo_1_sync": 1.0,
        "lfo_1_tempo": 7.0,
        # Sample - 关闭
        "sample_on": 0.0,
    }

    for key, value in _vital_defaults.items():
        if key in template:
            template[key] = value

    return template


class PresetGenerator:
    """基于 Base_Patch 生成控制变量预设。

    为 Phase 0 实验生成仅在单个 Effect_Switch 上不同的预设文件。
    """

    def __init__(
        self,
        parser: PresetParser,
        base_patch_template: dict[str, float] | None = None,
    ) -> None:
        """初始化预设生成器。

        Args:
            parser: PresetParser 实例，用于序列化预设文件
            base_patch_template: 包含所有 772 个参数默认值的字典。
                如果为 None，则使用内置默认模板。
        """
        self._parser = parser
        if base_patch_template is not None:
            self._template = dict(base_patch_template)
        else:
            self._template = _load_default_template()

    def create_base_patch(self) -> VitalPreset:
        """创建 Base_Patch 预设。

        确保 osc_1_on=1.0、filter_1_on=1.0，所有效果器关闭，
        使用默认波表。

        Returns:
            配置好的 VitalPreset 对象
        """
        settings = copy.deepcopy(self._template)

        # 强制设置 Base_Patch 关键参数
        settings["osc_1_on"] = 1.0
        settings["filter_1_on"] = 1.0

        # 确保所有效果器在 Base_Patch 中关闭
        for switch in PresetParser.EFFECT_SWITCHES:
            settings[switch] = 0.0

        return VitalPreset(
            settings=settings,
            modulations=[],
            extra={
                "author": "",
                "comments": "",
                "macro1": "",
                "macro2": "",
                "macro3": "",
                "macro4": "",
                "preset_name": "Base Patch",
                "preset_style": "",
            },
        )

    def create_effect_variant(
        self, effect_name: str, state: float
    ) -> VitalPreset:
        """生成仅指定 Effect_Switch 与 Base_Patch 不同的预设。

        Args:
            effect_name: 效果器开关名（如 "chorus_on"）
            state: 0.0（关）或 1.0（开）

        Returns:
            修改了指定效果器开关的 VitalPreset 对象

        Raises:
            ValueError: 如果 effect_name 不在 EFFECT_SWITCHES 中
        """
        if effect_name not in PresetParser.EFFECT_SWITCHES:
            valid_list = ", ".join(sorted(PresetParser.EFFECT_SWITCHES))
            raise ValueError(
                f"Invalid effect name '{effect_name}'. "
                f"Valid effect switches: [{valid_list}]"
            )

        base = self.create_base_patch()
        variant = VitalPreset(
            settings=copy.deepcopy(base.settings),
            modulations=copy.deepcopy(base.modulations),
            extra=copy.deepcopy(base.extra),
        )
        variant.settings[effect_name] = state
        variant.extra["preset_name"] = f"{effect_name}_{state}"

        return variant

    def generate_all_variants(self, output_dir: Path) -> list[Path]:
        """为 9 个效果器各生成开/关预设，共 18 个文件 + 1 个 base_patch。

        文件命名格式：
        - base_patch.vital
        - {effect_name}_{state}.vital（如 chorus_on_1.0.vital）

        Args:
            output_dir: 输出目录路径

        Returns:
            生成的文件路径列表（19 个文件）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files: list[Path] = []

        # 生成 base_patch
        base = self.create_base_patch()
        base_path = output_dir / "base_patch.vital"
        self._parser.serialize(base, base_path)
        generated_files.append(base_path)
        logger.info("Generated base patch: %s", base_path)

        # 为每个效果器生成开/关变体
        for effect_name in PresetParser.EFFECT_SWITCHES:
            for state in [0.0, 1.0]:
                variant = self.create_effect_variant(effect_name, state)
                filename = f"{effect_name}_{state}.vital"
                filepath = output_dir / filename
                self._parser.serialize(variant, filepath)
                generated_files.append(filepath)
                logger.info(
                    "Generated variant: %s = %s -> %s",
                    effect_name,
                    state,
                    filepath,
                )

        logger.info(
            "Generated %d preset files in %s",
            len(generated_files),
            output_dir,
        )
        return generated_files
