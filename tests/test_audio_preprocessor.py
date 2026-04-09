"""
AudioPreprocessor 单元测试。

验证音频预处理流水线的核心逻辑：RMS 计算、削波检测、DC 偏移消除、
峰值归一化、重采样、尾部裁剪和完整处理流水线。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.audio_preprocessor import (
    AudioPreprocessor,
    PreprocessConfig,
    PreprocessResult,
)


class TestPreprocessConfig:
    """PreprocessConfig dataclass 测试。"""

    def test_default_values(self) -> None:
        config = PreprocessConfig()
        assert config.silence_threshold_db == -60.0
        assert config.clipping_threshold == 0.99
        assert config.clipping_ratio_limit == 0.10
        assert config.target_peak_db == -1.0
        assert config.target_sample_rate == 16000
        assert config.tail_silence_threshold_db == -50.0
        assert config.min_duration_sec == 0.5

    def test_custom_values(self) -> None:
        config = PreprocessConfig(
            silence_threshold_db=-50.0,
            target_sample_rate=22050,
        )
        assert config.silence_threshold_db == -50.0
        assert config.target_sample_rate == 22050


class TestPreprocessResult:
    """PreprocessResult dataclass 测试。"""

    def test_creation(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = PreprocessResult(
            audio=audio,
            original_rms_db=-20.0,
            clipping_ratio=0.01,
            is_filtered=False,
            filter_reason=None,
            sample_rate=16000,
        )
        assert result.audio is not None
        assert result.original_rms_db == -20.0
        assert not result.is_filtered

    def test_filtered_result(self) -> None:
        result = PreprocessResult(
            audio=None,
            original_rms_db=-70.0,
            clipping_ratio=0.0,
            is_filtered=True,
            filter_reason="silence",
            sample_rate=16000,
        )
        assert result.audio is None
        assert result.is_filtered
        assert result.filter_reason == "silence"


class TestComputeRmsDb:
    """compute_rms_db 静态方法测试。"""

    def test_all_zero_returns_neg_inf(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        assert AudioPreprocessor.compute_rms_db(audio) == float("-inf")

    def test_full_scale_sine(self) -> None:
        # 满幅正弦波 RMS ≈ -3.01 dBFS
        t = np.linspace(0, 1, 44100, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms_db = AudioPreprocessor.compute_rms_db(audio)
        assert -3.1 < rms_db < -2.9

    def test_dc_signal(self) -> None:
        # 恒定 0.5 信号 → RMS = 0.5 → -6.02 dBFS
        audio = np.full(1000, 0.5, dtype=np.float32)
        rms_db = AudioPreprocessor.compute_rms_db(audio)
        assert abs(rms_db - (-6.02)) < 0.1

    def test_very_quiet_signal(self) -> None:
        audio = np.full(1000, 1e-4, dtype=np.float32)
        rms_db = AudioPreprocessor.compute_rms_db(audio)
        assert rms_db < -60.0


class TestComputeClippingRatio:
    """compute_clipping_ratio 静态方法测试。"""

    def test_no_clipping(self) -> None:
        audio = np.full(1000, 0.5, dtype=np.float32)
        assert AudioPreprocessor.compute_clipping_ratio(audio) == 0.0

    def test_all_clipping(self) -> None:
        audio = np.full(1000, 1.0, dtype=np.float32)
        assert AudioPreprocessor.compute_clipping_ratio(audio) == 1.0

    def test_partial_clipping(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        audio[:10] = 1.0  # 10% at exactly 1.0 (> 0.99)
        ratio = AudioPreprocessor.compute_clipping_ratio(audio)
        assert abs(ratio - 0.1) < 1e-6

    def test_empty_audio(self) -> None:
        audio = np.array([], dtype=np.float32)
        assert AudioPreprocessor.compute_clipping_ratio(audio) == 0.0

    def test_custom_threshold(self) -> None:
        audio = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        ratio = AudioPreprocessor.compute_clipping_ratio(audio, threshold=0.75)
        assert abs(ratio - 0.4) < 1e-6  # 0.8 and 0.9 exceed 0.75


class TestRemoveDcOffset:
    """remove_dc_offset 静态方法测试。"""

    def test_removes_offset(self) -> None:
        audio = np.full(1000, 0.3, dtype=np.float32)
        result = AudioPreprocessor.remove_dc_offset(audio)
        assert abs(np.mean(result)) < 1e-7

    def test_zero_mean_unchanged(self) -> None:
        t = np.linspace(0, 1, 1000, endpoint=False)
        audio = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        result = AudioPreprocessor.remove_dc_offset(audio)
        assert abs(np.mean(result)) < 1e-7

    def test_preserves_shape(self) -> None:
        audio = np.random.randn(500).astype(np.float32) + 0.5
        result = AudioPreprocessor.remove_dc_offset(audio)
        assert result.shape == audio.shape


class TestPeakNormalize:
    """peak_normalize 静态方法测试。"""

    def test_normalizes_to_target(self) -> None:
        audio = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        result = AudioPreprocessor.peak_normalize(audio, target_db=-1.0)
        peak_db = 20.0 * np.log10(np.max(np.abs(result)))
        assert abs(peak_db - (-1.0)) < 0.01

    def test_all_zero_returns_zero(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = AudioPreprocessor.peak_normalize(audio)
        np.testing.assert_array_equal(result, np.zeros(100))

    def test_target_zero_db(self) -> None:
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = AudioPreprocessor.peak_normalize(audio, target_db=0.0)
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-6


class TestResample:
    """resample 静态方法测试。"""

    def test_same_rate_returns_copy(self) -> None:
        audio = np.random.randn(1000).astype(np.float32)
        result = AudioPreprocessor.resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)
        assert result is not audio  # should be a copy

    def test_downsample_44100_to_16000(self) -> None:
        duration = 2.0
        orig_sr = 44100
        target_sr = 16000
        audio = np.random.randn(int(duration * orig_sr)).astype(np.float32)
        result = AudioPreprocessor.resample(audio, orig_sr, target_sr)
        expected_len = round(len(audio) * target_sr / orig_sr)
        assert abs(len(result) - expected_len) <= 1

    def test_output_is_float32(self) -> None:
        audio = np.random.randn(1000).astype(np.float32)
        result = AudioPreprocessor.resample(audio, 44100, 16000)
        assert result.dtype == np.float32


class TestTrimTailSilence:
    """trim_tail_silence 方法测试。"""

    def test_preserves_min_duration(self) -> None:
        # 全静音音频 → 应保留至少 0.5 秒
        sr = 16000
        audio = np.zeros(sr * 2, dtype=np.float32)  # 2 秒静音
        preprocessor = AudioPreprocessor()
        result = preprocessor.trim_tail_silence(audio, sr)
        min_samples = int(0.5 * sr)
        assert len(result) >= min_samples

    def test_trims_tail_silence(self) -> None:
        sr = 16000
        # 1 秒有效音频 + 1 秒静音
        active = np.random.randn(sr).astype(np.float32) * 0.5
        silence = np.zeros(sr, dtype=np.float32)
        audio = np.concatenate([active, silence])
        preprocessor = AudioPreprocessor()
        result = preprocessor.trim_tail_silence(audio, sr)
        # 应该裁剪掉大部分尾部静音
        assert len(result) < len(audio)
        assert len(result) >= sr  # 至少保留有效部分

    def test_short_audio_not_trimmed(self) -> None:
        sr = 16000
        min_samples = int(0.5 * sr)
        audio = np.zeros(min_samples - 1, dtype=np.float32)
        preprocessor = AudioPreprocessor()
        result = preprocessor.trim_tail_silence(audio, sr)
        assert len(result) == len(audio)


class TestProcess:
    """process 方法完整流水线测试。"""

    def test_silence_filtered(self) -> None:
        """全零音频应被标记为静音过滤。"""
        audio = np.zeros(44100, dtype=np.float32)
        preprocessor = AudioPreprocessor()
        result = preprocessor.process(audio, 44100)
        assert result.is_filtered
        assert result.filter_reason == "silence"
        assert result.audio is None
        assert result.original_rms_db == float("-inf")

    def test_clipping_filtered(self) -> None:
        """高削波比例音频应被标记为削波过滤。"""
        # 创建 >10% 削波的音频（非静音）
        audio = np.random.randn(1000).astype(np.float32) * 0.5
        audio[:200] = 1.0  # 20% 削波
        preprocessor = AudioPreprocessor()
        result = preprocessor.process(audio, 44100)
        assert result.is_filtered
        assert result.filter_reason == "clipping"
        assert result.audio is None

    def test_normal_audio_processed(self) -> None:
        """正常音频应通过所有检测并被处理。"""
        t = np.linspace(0, 1, 44100, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        preprocessor = AudioPreprocessor()
        result = preprocessor.process(audio, 44100)
        assert not result.is_filtered
        assert result.filter_reason is None
        assert result.audio is not None
        assert result.sample_rate == 16000

    def test_output_sample_rate(self) -> None:
        """处理后的音频应为目标采样率。"""
        t = np.linspace(0, 2, 44100 * 2, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        config = PreprocessConfig(target_sample_rate=22050)
        preprocessor = AudioPreprocessor(config)
        result = preprocessor.process(audio, 44100)
        assert result.sample_rate == 22050

    def test_result_contains_statistics(self) -> None:
        """结果应包含原始 RMS 和削波比例统计。"""
        t = np.linspace(0, 1, 44100, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        preprocessor = AudioPreprocessor()
        result = preprocessor.process(audio, 44100)
        assert isinstance(result.original_rms_db, float)
        assert isinstance(result.clipping_ratio, float)
        assert result.clipping_ratio == 0.0

    def test_default_config_used_when_none(self) -> None:
        """不传入配置时应使用默认配置。"""
        preprocessor = AudioPreprocessor()
        assert preprocessor.config.silence_threshold_db == -60.0
        assert preprocessor.config.target_sample_rate == 16000
