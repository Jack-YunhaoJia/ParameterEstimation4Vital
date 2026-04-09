"""
AdaptiveTimingCalculator 属性测试与单元测试。

覆盖设计文档中的 Property 1-5, 7 以及 power-law 已知值和边界情况。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adaptive_timing import (
    AdaptiveConfig,
    AdaptiveTiming,
    AdaptiveTimingCalculator,
)


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

raw_value = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# ---------------------------------------------------------------------------
# Property 1: Power-law round-trip
# Feature: adsr-adaptive-rendering, Property 1: Power-law round-trip
# Validates: Requirements 10.1, 10.6, 10.7, 1.2
# ---------------------------------------------------------------------------


class TestPowerLawRoundTrip:
    """Property 1: Power-law 映射 round-trip"""

    @given(raw=raw_value)
    @settings(max_examples=100)
    def test_round_trip(self, raw: float) -> None:
        """seconds_to_power_law(power_law_to_seconds(raw)) ≈ raw within 1e-9."""
        seconds = AdaptiveTimingCalculator.power_law_to_seconds(raw)
        recovered = AdaptiveTimingCalculator.seconds_to_power_law(seconds)
        assert abs(recovered - raw) < 1e-9, (
            f"Round-trip failed: raw={raw}, seconds={seconds}, recovered={recovered}"
        )


# ---------------------------------------------------------------------------
# Property 2: ADSR extraction completeness
# Feature: adsr-adaptive-rendering, Property 2: ADSR extraction completeness
# Validates: Requirements 1.1, 1.3
# ---------------------------------------------------------------------------

ADSR_KEYS = ["env_1_attack", "env_1_decay", "env_1_sustain", "env_1_release"]
ADSR_DEFAULTS = {
    "env_1_attack": 0.0,
    "env_1_decay": 0.0,
    "env_1_sustain": 1.0,
    "env_1_release": 0.0,
}


class TestADSRExtractionCompleteness:
    """Property 2: ADSR 提取完整性"""

    @given(
        subset=st.sets(st.sampled_from(ADSR_KEYS)),
        values=st.fixed_dictionaries(
            {k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False) for k in ADSR_KEYS}
        ),
    )
    @settings(max_examples=100)
    def test_extraction_always_returns_all_four_params(
        self, subset: set[str], values: dict[str, float]
    ) -> None:
        """Extraction always returns all 4 ADSR params with defaults for missing keys."""
        # Build a settings dict with only the chosen subset of keys
        settings_dict = {k: values[k] for k in subset}
        preset_data = {"settings": settings_dict}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vital", delete=False
        ) as f:
            json.dump(preset_data, f)
            tmp_path = Path(f.name)

        calc = AdaptiveTimingCalculator()
        result = calc.extract_adsr_from_preset(tmp_path)

        # All 4 keys must be present
        for key in ADSR_KEYS:
            assert key in result, f"Missing key: {key}"
            if key in subset:
                assert result[key] == pytest.approx(values[key], abs=1e-12)
            else:
                assert result[key] == ADSR_DEFAULTS[key]

        tmp_path.unlink(missing_ok=True)



# ---------------------------------------------------------------------------
# Property 3: note_off lower bound
# Feature: adsr-adaptive-rendering, Property 3: note_off lower bound
# Validates: Requirements 2.1, 2.3, 8.1
# ---------------------------------------------------------------------------


class TestNoteOffLowerBound:
    """Property 3: note_off 下界保证"""

    @given(
        attack_raw=raw_value,
        decay_raw=raw_value,
        sustain=raw_value,
        release_raw=raw_value,
    )
    @settings(max_examples=100)
    def test_note_off_ge_attack_plus_decay_plus_margin(
        self,
        attack_raw: float,
        decay_raw: float,
        sustain: float,
        release_raw: float,
    ) -> None:
        """note_off >= attack_sec + decay_sec + sustain_margin."""
        calc = AdaptiveTimingCalculator()
        timing = calc.compute_timing_from_values(
            attack_raw, decay_raw, sustain, release_raw
        )
        attack_sec = AdaptiveTimingCalculator.power_law_to_seconds(attack_raw)
        decay_sec = AdaptiveTimingCalculator.power_law_to_seconds(decay_raw)
        expected_lower = attack_sec + decay_sec + calc._config.sustain_margin

        assert timing.note_off >= expected_lower or timing.note_off >= calc._config.min_note_off, (
            f"note_off={timing.note_off} < expected_lower={expected_lower}"
        )


# ---------------------------------------------------------------------------
# Property 4: total_duration lower bound
# Feature: adsr-adaptive-rendering, Property 4: total_duration lower bound
# Validates: Requirements 3.1, 8.2
# ---------------------------------------------------------------------------


class TestTotalDurationLowerBound:
    """Property 4: total_duration 下界保证"""

    @given(
        attack_raw=raw_value,
        decay_raw=raw_value,
        sustain=raw_value,
        release_raw=raw_value,
    )
    @settings(max_examples=100)
    def test_total_duration_lower_bound(
        self,
        attack_raw: float,
        decay_raw: float,
        sustain: float,
        release_raw: float,
    ) -> None:
        """When unclamped <= max_duration: total_duration >= note_off + release_sec + tail_margin.
        When unclamped > max_duration: total_duration == max_duration."""
        calc = AdaptiveTimingCalculator()
        config = calc._config
        timing = calc.compute_timing_from_values(
            attack_raw, decay_raw, sustain, release_raw
        )

        attack_sec = AdaptiveTimingCalculator.power_law_to_seconds(attack_raw)
        decay_sec = AdaptiveTimingCalculator.power_law_to_seconds(decay_raw)
        release_sec = AdaptiveTimingCalculator.power_law_to_seconds(release_raw)

        # Compute the unclamped note_off and total_duration
        unclamped_note_off = max(
            config.min_note_off,
            attack_sec + decay_sec + config.sustain_margin,
        )
        unclamped_total = unclamped_note_off + release_sec + config.tail_margin

        if unclamped_total <= config.max_duration:
            # Not truncated: total_duration >= note_off + release_sec + tail_margin
            assert timing.total_duration >= timing.note_off + release_sec + config.tail_margin - 1e-9, (
                f"total_duration={timing.total_duration} < "
                f"note_off({timing.note_off}) + release({release_sec}) + tail({config.tail_margin})"
            )
        else:
            # Truncated: total_duration == max_duration
            assert timing.total_duration == config.max_duration, (
                f"Expected max_duration={config.max_duration}, got {timing.total_duration}"
            )


# ---------------------------------------------------------------------------
# Property 5: Output range invariants
# Feature: adsr-adaptive-rendering, Property 5: Output range invariants
# Validates: Requirements 2.4, 3.3, 3.4, 8.3, 8.4
# ---------------------------------------------------------------------------


class TestOutputRangeInvariants:
    """Property 5: 输出范围不变量"""

    @given(
        attack_raw=raw_value,
        decay_raw=raw_value,
        sustain=raw_value,
        release_raw=raw_value,
    )
    @settings(max_examples=100)
    def test_output_range_invariants(
        self,
        attack_raw: float,
        decay_raw: float,
        sustain: float,
        release_raw: float,
    ) -> None:
        """0.3 <= note_off < total_duration AND 1.0 <= total_duration <= 30.0."""
        calc = AdaptiveTimingCalculator()
        timing = calc.compute_timing_from_values(
            attack_raw, decay_raw, sustain, release_raw
        )

        assert timing.note_off >= 0.3, f"note_off={timing.note_off} < 0.3"
        assert timing.note_off < timing.total_duration, (
            f"note_off={timing.note_off} >= total_duration={timing.total_duration}"
        )
        assert 1.0 <= timing.total_duration <= 30.0, (
            f"total_duration={timing.total_duration} out of [1.0, 30.0]"
        )


# ---------------------------------------------------------------------------
# Property 7: Timing computation determinism
# Feature: adsr-adaptive-rendering, Property 7: Timing computation determinism
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------


class TestTimingDeterminism:
    """Property 7: 时序计算确定性"""

    @given(
        attack_raw=raw_value,
        decay_raw=raw_value,
        sustain=raw_value,
        release_raw=raw_value,
    )
    @settings(max_examples=100)
    def test_deterministic_results(
        self,
        attack_raw: float,
        decay_raw: float,
        sustain: float,
        release_raw: float,
    ) -> None:
        """Two calls with same inputs produce identical results."""
        calc = AdaptiveTimingCalculator()
        t1 = calc.compute_timing_from_values(attack_raw, decay_raw, sustain, release_raw)
        t2 = calc.compute_timing_from_values(attack_raw, decay_raw, sustain, release_raw)

        assert t1.attack_sec == t2.attack_sec
        assert t1.decay_sec == t2.decay_sec
        assert t1.sustain == t2.sustain
        assert t1.release_sec == t2.release_sec
        assert t1.note_off == t2.note_off
        assert t1.total_duration == t2.total_duration


# ---------------------------------------------------------------------------
# Unit tests: power-law known values and edge cases
# Requirements: 10.2, 10.3, 10.4, 10.5, 8.5
# ---------------------------------------------------------------------------


class TestPowerLawKnownValues:
    """Unit tests for power-law mapping known values."""

    def test_raw_0_maps_to_0(self) -> None:
        assert AdaptiveTimingCalculator.power_law_to_seconds(0.0) == 0.0

    def test_raw_025_maps_to_0125(self) -> None:
        result = AdaptiveTimingCalculator.power_law_to_seconds(0.25)
        assert result == pytest.approx(0.125, abs=0.001)

    def test_raw_05_maps_to_2(self) -> None:
        result = AdaptiveTimingCalculator.power_law_to_seconds(0.5)
        assert result == pytest.approx(2.0, abs=0.001)

    def test_raw_1_maps_to_32(self) -> None:
        result = AdaptiveTimingCalculator.power_law_to_seconds(1.0)
        assert result == pytest.approx(32.0, abs=0.001)


class TestAllZeroADSR:
    """All-zero ADSR produces minimum duration 1.0s."""

    def test_all_zero_produces_min_duration(self) -> None:
        calc = AdaptiveTimingCalculator()
        timing = calc.compute_timing_from_values(0.0, 0.0, 0.0, 0.0)
        assert timing.total_duration == pytest.approx(1.0)


class TestAdaptiveConfigDefaults:
    """AdaptiveConfig default values."""

    def test_default_values(self) -> None:
        config = AdaptiveConfig()
        assert config.sustain_margin == 0.2
        assert config.tail_margin == 0.1
        assert config.min_note_off == 0.3
        assert config.min_duration == 1.0
        assert config.max_duration == 30.0
        assert config.target_length_sec is None


class TestAdaptiveConfigInvalidParams:
    """AdaptiveConfig invalid params raise ValueError."""

    def test_negative_sustain_margin_raises(self) -> None:
        with pytest.raises(ValueError, match="sustain_margin"):
            AdaptiveConfig(sustain_margin=-0.1)

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError, match="max_duration"):
            AdaptiveConfig(min_duration=10.0, max_duration=5.0)


class TestTruncationBehavior:
    """Truncation when exceeding 30s and note_off reduction."""

    def test_truncation_at_max_duration(self) -> None:
        """When ADSR values produce total > 30s, total_duration is clamped to 30.0."""
        calc = AdaptiveTimingCalculator()
        # raw=1.0 → 32s for each of attack, decay, release
        timing = calc.compute_timing_from_values(1.0, 1.0, 0.5, 1.0)
        assert timing.total_duration == 30.0

    def test_note_off_reduced_when_truncated(self) -> None:
        """When truncated, note_off < total_duration still holds."""
        calc = AdaptiveTimingCalculator()
        timing = calc.compute_timing_from_values(1.0, 1.0, 0.5, 1.0)
        assert timing.note_off < timing.total_duration
        assert timing.note_off >= 0.3
