"""
ADSR 自适应时序计算模块。

根据 Vital preset 的 ADSR 包络参数动态计算渲染时长和 MIDI 时序，
确保 attack、decay、sustain、release 四个阶段都能完整展开。

纯计算模块，不依赖 VST 插件（除了读取 preset JSON）。
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveTiming:
    """自适应渲染时序参数（不可变）。"""

    attack_sec: float
    decay_sec: float
    sustain: float
    release_sec: float
    note_off: float
    total_duration: float


@dataclass(frozen=True)
class AdaptiveConfig:
    """自适应渲染配置参数。"""

    sustain_margin: float = 0.2
    tail_margin: float = 0.1
    min_note_off: float = 0.3
    min_duration: float = 1.0
    max_duration: float = 30.0
    target_length_sec: float | None = None

    def __post_init__(self) -> None:
        if self.sustain_margin < 0:
            raise ValueError(
                f"sustain_margin must be >= 0, got {self.sustain_margin}"
            )
        if self.tail_margin < 0:
            raise ValueError(
                f"tail_margin must be >= 0, got {self.tail_margin}"
            )
        if self.min_note_off < 0:
            raise ValueError(
                f"min_note_off must be >= 0, got {self.min_note_off}"
            )
        if self.min_duration <= 0:
            raise ValueError(
                f"min_duration must be > 0, got {self.min_duration}"
            )
        if self.max_duration < self.min_duration:
            raise ValueError(
                f"max_duration ({self.max_duration}) must be >= "
                f"min_duration ({self.min_duration})"
            )
        if self.target_length_sec is not None and self.target_length_sec <= 0:
            raise ValueError(
                f"target_length_sec must be > 0 when set, "
                f"got {self.target_length_sec}"
            )


class AdaptiveTimingCalculator:
    """ADSR 自适应时序计算器。"""

    ENV_MAX_TIME: float = 32.0
    ENV_POWER: float = 4.0

    DEFAULTS: dict[str, float] = {
        "env_1_attack": 0.0,
        "env_1_decay": 0.0,
        "env_1_sustain": 1.0,
        "env_1_release": 0.0,
    }

    def __init__(self, config: AdaptiveConfig | None = None) -> None:
        self._config = config or AdaptiveConfig()

    @staticmethod
    def power_law_to_seconds(raw: float) -> float:
        """Vital JSON 原始值 → 秒数。time = 32 * raw^4"""
        return 32.0 * raw ** 4.0

    @staticmethod
    def seconds_to_power_law(seconds: float) -> float:
        """秒数 → Vital JSON 原始值。raw = (time / 32)^0.25"""
        return (seconds / 32.0) ** 0.25

    # env_1_sustain 不需要 power-law 映射，它本身就是 [0, 1] 的电平值
    _TIME_PARAMS: set[str] = {"env_1_attack", "env_1_decay", "env_1_release"}

    def extract_adsr_from_preset(self, preset_path: Path) -> dict[str, float]:
        """从 .vital JSON 提取 env_1_* ADSR 参数原始值。

        自动检测值的类型：
        - 值在 [0, 1] 范围内：视为 Vital raw 值，直接返回
        - 值 > 1.0（时间参数）：视为物理秒数，反向映射为 raw 值
          （数据生产流水线的 CORE_PARAMS 使用 [0, 4] 秒数范围）
        - 缺失键使用默认值，非数值类型使用默认值并 warning
        - 负值 clamp 到 0

        Args:
            preset_path: .vital 文件路径

        Returns:
            包含 env_1_attack, env_1_decay, env_1_sustain, env_1_release 的字典
            （时间参数保证为 [0, 1] 范围的 raw 值）

        Raises:
            ValueError: JSON 解析失败或缺少 settings 键
        """
        preset_path = Path(preset_path)
        try:
            text = preset_path.read_text(encoding="utf-8")
            raw_json = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read preset '{preset_path}': {e}")

        settings = raw_json.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}

        result: dict[str, float] = {}
        for key, default in self.DEFAULTS.items():
            raw_val = settings.get(key)
            if raw_val is None:
                result[key] = default
                continue

            # Non-numeric → use default with warning
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                warnings.warn(
                    f"Non-numeric value for '{key}': {raw_val!r}, "
                    f"using default {default}",
                    stacklevel=2,
                )
                result[key] = default
                continue

            # 负值 clamp 到 0
            if val < 0.0:
                warnings.warn(
                    f"Value for '{key}' ({val}) below 0, clamping to 0.0",
                    stacklevel=2,
                )
                val = 0.0

            # 时间参数（attack/decay/release）：值 > 1.0 说明是物理秒数，
            # 需要反向映射为 Vital raw 值
            if key in self._TIME_PARAMS and val > 1.0:
                seconds = min(val, self.ENV_MAX_TIME)  # clamp 到 32s 上限
                val = self.seconds_to_power_law(seconds)
                logger.debug(
                    "Detected physical seconds for '%s': %.3fs → raw=%.4f",
                    key, seconds, val,
                )

            # sustain 参数 clamp 到 [0, 1]
            if key == "env_1_sustain" and val > 1.0:
                val = 1.0

            result[key] = val

        return result

    def compute_timing_from_values(
        self,
        attack_raw: float,
        decay_raw: float,
        sustain: float,
        release_raw: float,
    ) -> AdaptiveTiming:
        """从 ADSR 原始值直接计算时序（便于测试）。

        Args:
            attack_raw: attack 原始值 [0, 1]
            decay_raw: decay 原始值 [0, 1]
            sustain: sustain 级别 [0, 1]
            release_raw: release 原始值 [0, 1]

        Returns:
            计算后的 AdaptiveTiming
        """
        attack_sec = self.power_law_to_seconds(attack_raw)
        decay_sec = self.power_law_to_seconds(decay_raw)
        release_sec = self.power_law_to_seconds(release_raw)

        note_off = max(
            self._config.min_note_off,
            attack_sec + decay_sec + self._config.sustain_margin,
        )

        total_duration = note_off + release_sec + self._config.tail_margin
        total_duration = max(
            self._config.min_duration,
            min(self._config.max_duration, total_duration),
        )

        # 超过 max_duration 时按比例缩减 note_off
        if note_off >= total_duration:
            note_off = total_duration - self._config.tail_margin
            note_off = max(self._config.min_note_off, note_off)

        return AdaptiveTiming(
            attack_sec=attack_sec,
            decay_sec=decay_sec,
            sustain=sustain,
            release_sec=release_sec,
            note_off=note_off,
            total_duration=total_duration,
        )

    def compute_timing(self, preset_path: Path) -> AdaptiveTiming:
        """计算自适应渲染时序。主入口方法。

        Args:
            preset_path: .vital 文件路径

        Returns:
            计算后的 AdaptiveTiming
        """
        adsr = self.extract_adsr_from_preset(preset_path)
        return self.compute_timing_from_values(
            attack_raw=adsr["env_1_attack"],
            decay_raw=adsr["env_1_decay"],
            sustain=adsr["env_1_sustain"],
            release_raw=adsr["env_1_release"],
        )
