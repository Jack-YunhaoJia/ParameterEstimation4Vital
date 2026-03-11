"""
Vital 音频渲染模块。

通过 pedalboard 加载 Vital VST3 插件，将预设文件渲染为 WAV 音频。
支持单个预设渲染和批量渲染，包含超时处理和错误恢复。

参数设置通过 pedalboard 的 plugin.parameters[name].raw_value 接口，
而非 raw_state（Vital 不支持通过 raw_state 加载 JSON 预设）。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 延迟导入 CORE_PARAMS 范围，用于非线性参数的归一化
_CORE_PARAM_RANGES: dict[str, tuple[float, float]] | None = None


def _get_core_param_ranges() -> dict[str, tuple[float, float]]:
    """获取 CORE_PARAMS 的 (min, max) 范围映射。"""
    global _CORE_PARAM_RANGES
    if _CORE_PARAM_RANGES is None:
        from src.training_data import CORE_PARAMS
        _CORE_PARAM_RANGES = {name: (lo, hi) for name, lo, hi in CORE_PARAMS}
    return _CORE_PARAM_RANGES


# ---------------------------------------------------------------------------
# Vital JSON 参数名 → pedalboard 参数名 映射
# ---------------------------------------------------------------------------
# pedalboard 暴露的 VST3 参数名与 Vital JSON 中的参数名存在系统性差异：
#   osc_X_*      → oscillator_X_*
#   env_X_*      → envelope_X_*
#   *_on         → *_switch
#   *_dry_wet    → *_mix
#   *_keytrack   → *_key_track
#   osc_X_random_phase → oscillator_X_phase_randomization
#   osc_X_spectral_morph_* → oscillator_X_frequency_morph_*
#   filter_1_blend_transpose → filter_1_comb_blend_offset
#   reverb_high_shelf_* → reverb_high_*
#   reverb_low_shelf_*  → reverb_low_*
#   reverb_pre_high_cutoff → reverb_pre_high_cutoff (same)
#   reverb_pre_low_cutoff  → reverb_pre_low_cutoff  (same)

def _vital_name_to_pedalboard(vital_name: str) -> str | None:
    """将 Vital JSON 参数名转换为 pedalboard 参数名。

    Returns:
        pedalboard 参数名，或 None 如果无法映射。
    """
    # 特殊映射表（不规则的命名差异）
    _SPECIAL_MAP: dict[str, str] = {
        # osc_X_on 必须映射到 oscillator_X_switch（不能先展开 osc_ 前缀）
        "osc_1_on": "oscillator_1_switch",
        "osc_2_on": "oscillator_2_switch",
        "osc_3_on": "oscillator_3_switch",
        "osc_1_random_phase": "oscillator_1_phase_randomization",
        "osc_2_random_phase": "oscillator_2_phase_randomization",
        "osc_3_random_phase": "oscillator_3_phase_randomization",
        "osc_1_spectral_morph_amount": "oscillator_1_frequency_morph_amount",
        "osc_1_spectral_morph_spread": "oscillator_1_frequency_morph_spread",
        "osc_1_spectral_morph_type": "oscillator_1_frequency_morph_type",
        "osc_2_spectral_morph_amount": "oscillator_2_frequency_morph_amount",
        "osc_2_spectral_morph_spread": "oscillator_2_frequency_morph_spread",
        "osc_2_spectral_morph_type": "oscillator_2_frequency_morph_type",
        "osc_3_spectral_morph_amount": "oscillator_3_frequency_morph_amount",
        "osc_3_spectral_morph_spread": "oscillator_3_frequency_morph_spread",
        "osc_3_spectral_morph_type": "oscillator_3_frequency_morph_type",
        "filter_1_blend_transpose": "filter_1_comb_blend_offset",
        "filter_2_blend_transpose": "filter_2_comb_blend_offset",
        "filter_fx_blend_transpose": "filter_fx_comb_blend_offset",
        "reverb_high_shelf_cutoff": "reverb_high_cutoff",
        "reverb_high_shelf_gain": "reverb_high_gain",
        "reverb_low_shelf_cutoff": "reverb_low_cutoff",
        "reverb_low_shelf_gain": "reverb_low_gain",
    }

    if vital_name in _SPECIAL_MAP:
        return _SPECIAL_MAP[vital_name]

    name = vital_name

    # osc_X_* → oscillator_X_*
    if name.startswith("osc_"):
        name = "oscillator_" + name[4:]
        return name

    # env_X_* → envelope_X_*
    if name.startswith("env_"):
        name = "envelope_" + name[4:]
        return name

    # *_on → *_switch
    if name.endswith("_on"):
        return name[:-3] + "_switch"

    # *_dry_wet → *_mix
    if name.endswith("_dry_wet"):
        return name[:-8] + "_mix"

    # *_keytrack → *_key_track
    if "keytrack" in name:
        return name.replace("keytrack", "key_track")

    # 其余参数名相同（如 filter_1_cutoff, chorus_feedback 等）
    return name


# ---------------------------------------------------------------------------
# Vital JSON 值 → pedalboard raw_value 映射
# ---------------------------------------------------------------------------
# Vital JSON 参数值和 pedalboard raw_value [0,1] 之间存在三种映射关系：
#
# 1. "直接映射" (304 个参数): Vital JSON 值 == raw_value
#    例: osc_1_level=0.707 → raw=0.707
#    特征: 默认值完全匹配
#
# 2. "线性归一化" (180 个参数): raw = (vital_value - pb_min) / (pb_max - pb_min)
#    例: osc_1_transpose=0 → raw=(0-(-48))/(48-(-48))=0.5
#    特征: 参数有 pb_min/pb_max，且归一化后匹配默认 raw_value
#
# 3. "非线性/未知映射" (188 个参数): 复杂的内部映射
#    例: filter_1_cutoff=60 → raw=0.406 (非线性 MIDI note 映射)
#    例: env_1_decay=1.5 → raw=0.420 (指数映射)
#    策略: 跳过这些参数，保持插件默认值
#
# 4. 特殊参数: volume (dB), unison_detune (sqrt), unison_voices (discrete)

# 需要线性归一化的参数（通过 pattern 匹配，避免维护巨大的集合）
# 这些参数的 Vital JSON 值是物理值，需要 (val - min) / (max - min) 转换
def _is_normalize_param(vital_name: str) -> bool:
    """判断参数是否需要线性归一化（Vital 物理值 → raw_value）。"""
    # 先排除已知非线性参数（优先级高于 pattern 匹配）
    if vital_name in _NONLINEAR_PARAMS:
        return False

    # 基于 pattern 的规则（覆盖 180 个归一化参数）
    patterns = (
        "_transpose",      # osc_X_transpose, sample_transpose, voice_transpose, filter_X_formant_transpose
        "_tune",           # osc_X_tune, sample_tune
        "_pan",            # osc_X_pan, sample_pan
        "_spread",         # osc_X_distortion_spread, osc_X_spectral_morph_spread, filter_X_formant_spread
        "_power",          # env_X_attack_power, osc_X_detune_power, modulation_X_power
        "_stereo",         # lfo_X_stereo
        "_slope",          # portamento_slope
        "_keytrack",       # filter_X_keytrack
        "_detune_range",   # osc_X_detune_range
        "pitch_wheel",     # pitch_wheel
        "velocity_track",  # velocity_track
    )
    if any(vital_name.endswith(p) or vital_name == p for p in patterns):
        return True
    # modulation_X_amount
    if vital_name.startswith("modulation_") and vital_name.endswith("_amount"):
        return True
    # distortion_drive, eq_X_gain, reverb_low_shelf_gain
    if vital_name in ("distortion_drive",):
        return True
    if vital_name.endswith("_gain") and (
        vital_name.startswith("eq_") or vital_name == "reverb_low_shelf_gain"
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# 已知非线性映射参数（跳过设置，保持插件默认值）
# ---------------------------------------------------------------------------
# 这些参数使用 Vital 内部的非线性映射（指数、对数、离散等），
# 无法通过简单的线性变换得到正确的 raw_value。
_NONLINEAR_PARAM_PATTERNS: tuple[str, ...] = (
    # 效果器内部参数（非 dry_wet/mix/on）
    "chorus_delay_", "chorus_feedback", "chorus_frequency", "chorus_mod_depth",
    "chorus_sync", "chorus_tempo", "chorus_voices",
    "compressor_band_gain", "compressor_high_gain", "compressor_low_gain",
    "delay_feedback", "delay_filter_cutoff", "delay_filter_spread",
    "delay_frequency", "delay_sync", "delay_tempo",
    "distortion_filter_cutoff", "distortion_filter_resonance",
    "flanger_center", "flanger_feedback", "flanger_frequency", "flanger_mod_depth",
    "flanger_phase_offset", "flanger_sync", "flanger_tempo",
    "phaser_blend", "phaser_center", "phaser_feedback", "phaser_frequency",
    "phaser_mod_depth", "phaser_phase_offset", "phaser_sync", "phaser_tempo",
    "reverb_chorus_amount", "reverb_chorus_frequency",
    "reverb_high_shelf_cutoff", "reverb_high_shelf_gain",
    "reverb_pre_high_cutoff",
    # EQ 内部参数
    "eq_band_cutoff", "eq_band_resonance",
    "eq_high_cutoff", "eq_high_resonance", "eq_low_cutoff", "eq_low_resonance",
    # Filter 非线性参数（cutoff 已移至已知映射）
    "filter_1_resonance", "filter_1_formant_resonance",
    "filter_1_formant_x", "filter_1_formant_y", "filter_1_blend_transpose",
    "filter_2_cutoff", "filter_2_resonance", "filter_2_formant_resonance",
    "filter_2_formant_x", "filter_2_formant_y", "filter_2_blend_transpose",
    "filter_fx_cutoff", "filter_fx_resonance", "filter_fx_formant_resonance",
    "filter_fx_formant_x", "filter_fx_formant_y", "filter_fx_blend_transpose",
    # LFO 非线性参数
    "lfo_1_frequency", "lfo_1_smooth_mode", "lfo_1_smooth_time", "lfo_1_sync", "lfo_1_tempo",
    "lfo_2_frequency", "lfo_2_smooth_mode", "lfo_2_smooth_time", "lfo_2_sync", "lfo_2_tempo",
    "lfo_3_frequency", "lfo_3_smooth_mode", "lfo_3_smooth_time", "lfo_3_sync", "lfo_3_tempo",
    "lfo_4_frequency", "lfo_4_smooth_mode", "lfo_4_smooth_time", "lfo_4_sync", "lfo_4_tempo",
    "lfo_5_frequency", "lfo_5_smooth_mode", "lfo_5_smooth_time", "lfo_5_sync", "lfo_5_tempo",
    "lfo_6_frequency", "lfo_6_smooth_mode", "lfo_6_smooth_time", "lfo_6_sync", "lfo_6_tempo",
    "lfo_7_frequency", "lfo_7_smooth_mode", "lfo_7_smooth_time", "lfo_7_sync", "lfo_7_tempo",
    "lfo_8_frequency", "lfo_8_smooth_mode", "lfo_8_smooth_time", "lfo_8_sync", "lfo_8_tempo",
    # Envelope 非线性参数（attack/decay/release 已移至 power-law 映射）
    "env_1_decay_power", "env_1_release_power",
    "env_2_attack", "env_2_decay", "env_2_release",
    "env_2_decay_power", "env_2_release_power",
    "env_3_attack", "env_3_decay", "env_3_release", "env_3_sustain",
    "env_3_decay_power", "env_3_release_power",
    "env_4_attack", "env_4_decay", "env_4_release", "env_4_sustain",
    "env_4_decay_power", "env_4_release_power",
    "env_5_attack", "env_5_decay", "env_5_release", "env_5_sustain",
    "env_5_decay_power", "env_5_release_power",
    "env_6_attack", "env_6_decay", "env_6_release", "env_6_sustain",
    "env_6_decay_power", "env_6_release_power",
    # Oscillator 非线性参数
    "osc_1_distortion_amount", "osc_1_distortion_phase", "osc_1_phase",
    "osc_1_smooth_interpolation", "osc_1_spectral_morph_amount",
    "osc_1_spectral_unison", "osc_1_stereo_spread", "osc_1_view_2d",
    "osc_2_destination", "osc_2_detune_power", "osc_2_detune_range",
    "osc_2_distortion_amount", "osc_2_distortion_phase", "osc_2_phase",
    "osc_2_smooth_interpolation", "osc_2_spectral_morph_amount",
    "osc_2_spectral_unison", "osc_2_stereo_spread", "osc_2_view_2d",
    "osc_3_destination", "osc_3_detune_power", "osc_3_detune_range",
    "osc_3_distortion_amount", "osc_3_distortion_phase", "osc_3_phase",
    "osc_3_smooth_interpolation", "osc_3_spectral_morph_amount",
    "osc_3_spectral_unison", "osc_3_stereo_spread", "osc_3_view_2d",
    # 全局非线性参数
    "beats_per_minute", "oversampling", "polyphony", "pitch_bend_range",
    "sample_destination", "sample_level", "sample_loop",
    "stereo_routing", "voice_amplitude", "voice_priority", "voice_tune",
)

# 转为 frozenset 加速查找
_NONLINEAR_PARAMS: frozenset[str] = frozenset(_NONLINEAR_PARAM_PATTERNS)


def _is_nonlinear_param(vital_name: str) -> bool:
    """判断参数是否使用非线性映射（应跳过）。"""
    return vital_name in _NONLINEAR_PARAMS


# ---------------------------------------------------------------------------
# 已知非线性映射的精确转换函数
# ---------------------------------------------------------------------------
# 通过实验验证（pedalboard string_value + 渲染测量）得到的精确映射：
#
# Envelope attack/decay/release:
#   time_seconds = 32 * raw^4  (max_time=32s, power=4)
#   raw = (time_seconds / 32) ^ 0.25
#   验证: raw=0.25 → 0.125s, raw=0.5 → 2.0s, raw=0.75 → 10.125s, raw=1.0 → 32s
#
# Filter cutoff (MIDI note):
#   raw = (midi_note - 8) / 128
#   验证: note=60 → raw=0.40625 (匹配插件默认值)
#
# Frequency params (chorus/delay/flanger/phaser_frequency):
#   使用 Vital 的 kFrequency 映射，暂用 CORE_PARAMS 线性归一化近似

# Envelope 时间参数的 power-law 映射常数
_ENV_MAX_TIME: float = 32.0   # 最大时间（秒）
_ENV_POWER: float = 4.0       # 幂次

# 使用 power-law 映射的 envelope 参数
_ENV_TIME_PARAMS: frozenset[str] = frozenset({
    "env_1_attack", "env_1_decay", "env_1_release",
})

# 使用线性 MIDI note 映射的 filter cutoff 参数
# raw = (midi_note - 8) / 128
_FILTER_CUTOFF_PARAMS: frozenset[str] = frozenset({
    "filter_1_cutoff",
})
_CUTOFF_MIN_NOTE: float = 8.0
_CUTOFF_MAX_NOTE: float = 136.0

# 使用 power-law 映射的 frequency 参数 (time = max * raw^power)
# 通过 string_value 验证: raw=0.5 → 2.0, raw=1.0 → 32.0 (同 envelope)
# 但 frequency 参数的 max 和 power 可能不同，暂用 CORE_PARAMS 线性近似
_FREQ_POWER_PARAMS: frozenset[str] = frozenset({
    "delay_frequency", "chorus_frequency", "flanger_frequency", "phaser_frequency",
})


def _vital_value_to_raw(
    vital_name: str,
    vital_value: float,
    pb_min: float | bool | None,
    pb_max: float | bool | None,
) -> float | None:
    """将 Vital JSON 参数值转换为 pedalboard raw_value [0, 1]。

    映射策略（基于实验验证）：
    1. 特殊参数: volume / unison_detune / unison_voices
    2. Envelope 时间参数: power-law 映射 raw = (time / 32)^0.25
    3. Filter cutoff: 线性 MIDI note 映射 raw = (note - 8) / 128
    4. 归一化参数: (vital_value - pb_min) / (pb_max - pb_min)
    5. 直接映射参数: Vital 值就是 raw_value
    6. CORE_PARAMS 回退: 线性归一化
    7. 未知映射参数: 跳过

    Args:
        vital_name: Vital JSON 参数名
        vital_value: Vital JSON 中的参数值
        pb_min: pedalboard 参数的 min_value
        pb_max: pedalboard 参数的 max_value

    Returns:
        raw_value [0, 1]，或 None 表示跳过此参数
    """
    # 特殊处理：volume（线性增益 → dB → 归一化）
    if vital_name == "volume":
        if pb_min is not None and pb_max is not None:
            pb_min_f = float(pb_min)
            pb_max_f = float(pb_max)
            if vital_value <= 0:
                db_val = pb_min_f
            else:
                db_val = 20.0 * np.log10(vital_value)
                db_val = max(pb_min_f, min(pb_max_f, db_val))
            if pb_max_f > pb_min_f:
                return (db_val - pb_min_f) / (pb_max_f - pb_min_f)
        return None

    # 特殊处理：unison_detune（sqrt 映射）
    if "unison_detune" in vital_name:
        return np.sqrt(max(0.0, min(1.0, vital_value / 100.0)))

    # 特殊处理：unison_voices（离散值，raw = (val - 1) / 15）
    if "unison_voices" in vital_name:
        return max(0.0, min(1.0, (vital_value - 1.0) / 15.0))

    # Envelope 时间参数: power-law 映射
    # time = 32 * raw^4  →  raw = (time / 32)^0.25
    if vital_name in _ENV_TIME_PARAMS:
        if vital_value <= 0:
            return 0.0
        raw = (vital_value / _ENV_MAX_TIME) ** (1.0 / _ENV_POWER)
        return max(0.0, min(1.0, raw))

    # Filter cutoff: 线性 MIDI note 映射
    # raw = (midi_note - 8) / 128
    if vital_name in _FILTER_CUTOFF_PARAMS:
        raw = (vital_value - _CUTOFF_MIN_NOTE) / (_CUTOFF_MAX_NOTE - _CUTOFF_MIN_NOTE)
        return max(0.0, min(1.0, raw))

    # 归一化参数：(vital_value - pb_min) / (pb_max - pb_min)
    if _is_normalize_param(vital_name):
        if pb_min is not None and pb_max is not None:
            pb_min_f = float(pb_min) if not isinstance(pb_min, bool) else 0.0
            pb_max_f = float(pb_max) if not isinstance(pb_max, bool) else 1.0
            if pb_max_f > pb_min_f:
                raw = (vital_value - pb_min_f) / (pb_max_f - pb_min_f)
                return max(0.0, min(1.0, raw))
        return None

    # 直接映射：Vital 值在 [0, 1] 范围内
    if 0.0 <= vital_value <= 1.0:
        if _is_nonlinear_param(vital_name):
            # CORE_PARAM 回退：线性归一化
            core_ranges = _get_core_param_ranges()
            if vital_name in core_ranges:
                lo, hi = core_ranges[vital_name]
                if hi > lo:
                    return max(0.0, min(1.0, (vital_value - lo) / (hi - lo)))
            return None
        return vital_value

    # 超出 [0, 1]：CORE_PARAM 线性归一化回退
    core_ranges = _get_core_param_ranges()
    if vital_name in core_ranges:
        lo, hi = core_ranges[vital_name]
        if hi > lo:
            return max(0.0, min(1.0, (vital_value - lo) / (hi - lo)))

    # 未知参数：跳过
    return None


@dataclass
class RenderConfig:
    """音频渲染配置。

    Attributes:
        midi_note: MIDI 音符编号（60 = C4）
        velocity: MIDI 力度（0-127）
        duration_sec: 渲染时长（秒）
        sample_rate: 采样率（Hz）
        timeout_sec: 单个预设渲染超时时间（秒）
    """

    midi_note: int = 60  # C4
    velocity: int = 100
    duration_sec: float = 2.0
    sample_rate: int = 44100
    timeout_sec: float = 30.0


@dataclass
class RenderSummary:
    """批量渲染结果摘要。

    Attributes:
        success_count: 成功渲染的预设数量
        failure_count: 失败的预设数量
        failed_files: 失败的文件名列表
    """

    success_count: int = 0
    failure_count: int = 0
    failed_files: list[str] = field(default_factory=list)


class RenderTimeoutError(Exception):
    """渲染超时错误。"""
    pass


class AudioRenderer:
    """通过 pedalboard 加载 Vital VST3 插件渲染音频。

    使用固定 MIDI 输入（单音符）将 Vital 预设渲染为 WAV 文件。
    支持单个渲染和批量渲染，批量渲染时对单个失败进行容错处理。
    """

    def __init__(self, vital_vst_path: Path, config: RenderConfig | None = None) -> None:
        """加载 Vital VST3 插件。

        Args:
            vital_vst_path: Vital VST3 插件路径
            config: 渲染配置，为 None 时使用默认配置

        Raises:
            FileNotFoundError: VST3 插件路径不存在
            RuntimeError: 插件加载失败
        """
        self._vst_path = Path(vital_vst_path)
        self._config = config or RenderConfig()

        if not self._vst_path.exists():
            raise FileNotFoundError(
                f"Vital VST3 plugin not found: {self._vst_path}"
            )

        try:
            from pedalboard import load_plugin
            self._plugin = load_plugin(str(self._vst_path))
            logger.info("Loaded Vital VST3 plugin from %s", self._vst_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Vital VST3 plugin from {self._vst_path}: {e}"
            ) from e

        # 记录插件默认 raw_value，用于校准参数映射
        self._default_raw: dict[str, float] = {}
        for name, param in self._plugin.parameters.items():
            self._default_raw[name] = param.raw_value

    def _load_preset_into_plugin(self, preset_path: Path) -> None:
        """将预设参数加载到 VST 插件中。

        先恢复插件到默认状态，然后只设置已知映射的参数。
        对于非线性映射的参数，跳过以保持插件默认值。

        Args:
            preset_path: .vital 预设文件路径

        Raises:
            FileNotFoundError: 预设文件不存在
            ValueError: 预设文件格式无效
        """
        preset_path = Path(preset_path)
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")

        try:
            raw = json.loads(preset_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(
                f"Failed to read preset '{preset_path}': {e}"
            ) from e

        settings = raw.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError(
                f"Invalid preset '{preset_path}': 'settings' must be a dict"
            )

        plugin_params = self._plugin.parameters
        applied = 0
        skipped = 0

        # 先恢复所有参数到默认值（避免上一个预设的残留影响）
        for name, default_raw in self._default_raw.items():
            if name in plugin_params:
                try:
                    plugin_params[name].raw_value = default_raw
                except Exception:
                    pass

        for vital_name, vital_value in settings.items():
            # 跳过非数值参数（modulations, wavetables 等嵌套结构）
            if not isinstance(vital_value, (int, float)):
                continue

            pb_name = _vital_name_to_pedalboard(vital_name)
            if pb_name is None or pb_name not in plugin_params:
                skipped += 1
                continue

            param = plugin_params[pb_name]
            pb_min = param.min_value
            pb_max = param.max_value

            try:
                raw_val = _vital_value_to_raw(
                    vital_name, float(vital_value), pb_min, pb_max
                )
                if raw_val is not None:
                    param.raw_value = raw_val
                    applied += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        logger.debug(
            "Loaded preset %s: %d params applied, %d skipped",
            preset_path.name, applied, skipped,
        )

    def _generate_midi_audio(self) -> np.ndarray:
        """生成 MIDI 事件并通过 VST 插件渲染音频。

        使用 pedalboard VST3Plugin 的 MIDI 合成模式：
        plugin(midi_messages, duration, sample_rate, num_channels)

        Returns:
            渲染后的单声道音频数据，shape 为 (1, num_samples)
        """
        # 构建 MIDI 消息：Note On at t=0, Note Off near end
        note_on = self._create_midi_note_on(
            self._config.midi_note, self._config.velocity
        )
        note_off_time = max(0.0, self._config.duration_sec - 0.1)
        note_off = self._create_midi_note_off(
            self._config.midi_note, note_off_time
        )
        midi_messages = note_on + note_off

        # 使用 VST3Plugin MIDI 合成模式渲染
        rendered = self._plugin(
            midi_messages,
            duration=self._config.duration_sec,
            sample_rate=self._config.sample_rate,
            num_channels=2,
        )

        # rendered shape: (num_channels, num_samples) — 转换为单声道
        if rendered.ndim == 2 and rendered.shape[0] > 1:
            mono = np.mean(rendered, axis=0, keepdims=True)
        elif rendered.ndim == 1:
            mono = rendered.reshape(1, -1)
        else:
            mono = rendered

        return mono

    @staticmethod
    def _create_midi_note_on(note: int, velocity: int) -> list:
        """创建 MIDI Note On 消息。"""
        # pedalboard 使用 (midi_bytes, timestamp_seconds) 格式
        status = 0x90  # Note On, channel 0
        return [(bytes([status, note, velocity]), 0.0)]

    @staticmethod
    def _create_midi_note_off(note: int, time_sec: float = 1.9) -> list:
        """创建 MIDI Note Off 消息。

        Args:
            note: MIDI 音符编号
            time_sec: Note Off 时间点（秒）
        """
        # pedalboard 使用 (midi_bytes, timestamp_seconds) 格式
        status = 0x80  # Note Off, channel 0
        return [(bytes([status, note, 0]), time_sec)]

    def _write_wav(self, audio: np.ndarray, output_path: Path) -> None:
        """将音频数据写入 WAV 文件。

        Args:
            audio: 音频数据，shape 为 (1, num_samples) 或 (num_samples,)
            output_path: 输出 WAV 文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确保 audio 是 2D: (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        from pedalboard.io import AudioFile

        with AudioFile(
            str(output_path),
            "w",
            samplerate=self._config.sample_rate,
            num_channels=1,
        ) as f:
            f.write(audio)

        logger.debug("Wrote WAV: %s", output_path)

    def render_preset(self, preset_path: Path, output_path: Path) -> bool:
        """渲染单个预设为 WAV 文件。

        加载预设到 VST 插件，生成 MIDI 事件，渲染音频，
        输出为 44100Hz 单声道 WAV。

        Args:
            preset_path: .vital 预设文件路径
            output_path: 输出 WAV 文件路径

        Returns:
            True 如果渲染成功，False 如果失败
        """
        preset_path = Path(preset_path)
        output_path = Path(output_path)

        try:
            self._load_preset_into_plugin(preset_path)
            audio = self._generate_midi_audio()
            self._write_wav(audio, output_path)
            logger.info(
                "Successfully rendered: %s -> %s",
                preset_path.name,
                output_path.name,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to render preset '%s': %s", preset_path.name, e
            )
            return False

    def _render_with_timeout(
        self, preset_path: Path, output_path: Path
    ) -> bool:
        """带超时的单个预设渲染。

        使用线程实现超时控制（兼容 macOS）。

        Args:
            preset_path: .vital 预设文件路径
            output_path: 输出 WAV 文件路径

        Returns:
            True 如果渲染成功，False 如果失败或超时
        """
        result = [False]
        error = [None]

        def _render_worker():
            try:
                result[0] = self.render_preset(preset_path, output_path)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_render_worker, daemon=True)
        thread.start()
        thread.join(timeout=self._config.timeout_sec)

        if thread.is_alive():
            logger.error(
                "Render timed out after %.1fs for preset: %s",
                self._config.timeout_sec,
                preset_path.name,
            )
            return False

        if error[0] is not None:
            logger.error(
                "Render error for preset '%s': %s",
                preset_path.name,
                error[0],
            )
            return False

        return result[0]

    def render_batch(
        self, preset_dir: Path, output_dir: Path
    ) -> RenderSummary:
        """批量渲染目录下所有 .vital 文件。

        遍历 preset_dir 中的所有 .vital 文件，逐个渲染为 WAV，
        保存到 output_dir。单个渲染失败或超时时跳过并记录错误，
        继续处理剩余文件。

        输出文件名与源预设同名（.vital → .wav）。

        Args:
            preset_dir: 包含 .vital 预设文件的目录
            output_dir: 输出 WAV 文件的目录

        Returns:
            RenderSummary 包含成功数、失败数和失败文件列表
        """
        preset_dir = Path(preset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有 .vital 文件
        vital_files = sorted(preset_dir.glob("*.vital"))

        if not vital_files:
            logger.warning("No .vital files found in %s", preset_dir)
            return RenderSummary(
                success_count=0, failure_count=0, failed_files=[]
            )

        logger.info(
            "Starting batch render: %d presets from %s",
            len(vital_files),
            preset_dir,
        )

        summary = RenderSummary()

        for preset_path in vital_files:
            # 输出文件名：同名 .wav
            output_filename = preset_path.stem + ".wav"
            output_path = output_dir / output_filename

            success = self._render_with_timeout(preset_path, output_path)

            if success:
                summary.success_count += 1
            else:
                summary.failure_count += 1
                summary.failed_files.append(preset_path.name)

        logger.info(
            "Batch render complete: %d success, %d failed",
            summary.success_count,
            summary.failure_count,
        )

        return summary
