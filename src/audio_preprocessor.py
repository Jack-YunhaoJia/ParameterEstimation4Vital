"""
音频预处理模块。

对渲染后的原始音频执行标准化预处理流水线：
静音检测 → 削波检测 → DC 偏移消除 → 峰值归一化 → 重采样 → 尾部裁剪
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """音频预处理结果。"""

    audio: np.ndarray | None  # 处理后的音频数据（被过滤时为 None）
    original_rms_db: float  # 原始 RMS（dBFS）
    clipping_ratio: float  # 削波样本比例
    is_filtered: bool  # 是否被过滤
    filter_reason: str | None  # 过滤原因（silence/clipping/None）
    sample_rate: int  # 输出采样率


@dataclass
class PreprocessConfig:
    """音频预处理配置。"""

    silence_threshold_db: float = -60.0  # 静音检测阈值（dBFS）
    clipping_threshold: float = 0.99  # 削波检测绝对值阈值
    clipping_ratio_limit: float = 0.10  # 削波比例上限
    target_peak_db: float = -1.0  # 峰值归一化目标（dBFS）
    target_sample_rate: int = 16000  # 目标采样率
    tail_silence_threshold_db: float = -50.0  # 尾部静音裁剪阈值
    min_duration_sec: float = 0.5  # 最小有效音频时长（秒）


class AudioPreprocessor:
    """音频预处理流水线。

    处理顺序：静音检测 → 削波检测 → DC 偏移消除 → 峰值归一化 → 重采样 → 尾部裁剪
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """初始化预处理器。

        Args:
            config: 预处理配置，None 时使用默认配置。
        """
        self.config = config or PreprocessConfig()

    @staticmethod
    def compute_rms_db(audio: np.ndarray) -> float:
        """计算音频 RMS 能量值（dBFS）。

        Args:
            audio: 输入音频数据（1D float array）

        Returns:
            RMS 能量值（dBFS）。全零音频返回 -inf。
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0.0:
            return float("-inf")
        return float(20.0 * np.log10(rms))

    @staticmethod
    def compute_clipping_ratio(
        audio: np.ndarray, threshold: float = 0.99
    ) -> float:
        """计算削波样本比例。

        Args:
            audio: 输入音频数据（1D float array）
            threshold: 削波检测绝对值阈值

        Returns:
            绝对值超过阈值的样本占总样本数的比例。
        """
        if len(audio) == 0:
            return 0.0
        return float(np.mean(np.abs(audio) > threshold))

    @staticmethod
    def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
        """移除 DC 偏移（减去信号均值）。

        Args:
            audio: 输入音频数据（1D float array）

        Returns:
            去除 DC 偏移后的音频数据。
        """
        return audio - np.mean(audio)

    @staticmethod
    def peak_normalize(
        audio: np.ndarray, target_db: float = -1.0
    ) -> np.ndarray:
        """峰值归一化至目标 dBFS。

        Args:
            audio: 输入音频数据（1D float array）
            target_db: 目标峰值（dBFS）

        Returns:
            归一化后的音频数据。全零音频原样返回。
        """
        peak = np.max(np.abs(audio))
        if peak == 0.0:
            return audio.copy()
        target_linear = 10.0 ** (target_db / 20.0)
        return audio * (target_linear / peak)

    @staticmethod
    def resample(
        audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """使用抗混叠低通滤波器重采样。

        使用 scipy.signal.resample_poly 进行高质量重采样。

        Args:
            audio: 输入音频数据（1D float array）
            orig_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            重采样后的音频数据。
        """
        if orig_sr == target_sr:
            return audio.copy()

        common = gcd(orig_sr, target_sr)
        up = target_sr // common
        down = orig_sr // common

        resampled = resample_poly(audio, up, down)
        return resampled.astype(np.float32)

    def trim_tail_silence(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """裁剪尾部静音段，保留最少 min_duration_sec 秒。

        从音频末尾向前扫描，找到最后一个 RMS 高于阈值的帧，
        裁剪其后的静音段。输出长度不少于 min_duration_sec × sample_rate。

        Args:
            audio: 输入音频数据（1D float array）
            sample_rate: 采样率

        Returns:
            裁剪后的音频数据。
        """
        min_samples = int(self.config.min_duration_sec * sample_rate)
        # 保证至少保留 min_samples
        min_samples = max(min_samples, 1)

        if len(audio) <= min_samples:
            return audio.copy()

        # 使用帧级 RMS 检测尾部静音
        frame_size = int(0.01 * sample_rate)  # 10ms 帧
        if frame_size < 1:
            frame_size = 1

        threshold_db = self.config.tail_silence_threshold_db

        # 从末尾向前扫描，找到最后一个非静音帧
        last_active = min_samples  # 至少保留 min_samples
        n_frames = len(audio) // frame_size

        for i in range(n_frames - 1, -1, -1):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            frame_rms_db = AudioPreprocessor.compute_rms_db(frame)
            if frame_rms_db > threshold_db:
                # 保留到这个帧的末尾
                last_active = end
                break

        # 确保至少保留 min_samples
        trim_point = max(last_active, min_samples)
        # 不超过原始长度
        trim_point = min(trim_point, len(audio))

        return audio[:trim_point].copy()

    def process(
        self, audio: np.ndarray, sample_rate: int
    ) -> PreprocessResult:
        """执行完整预处理流水线。

        处理顺序：静音检测 → 削波检测 → DC 偏移消除 → 峰值归一化 → 重采样 → 尾部裁剪

        Args:
            audio: 原始音频数据（1D float array）
            sample_rate: 输入采样率

        Returns:
            PreprocessResult 包含处理后的音频和统计信息。
        """
        config = self.config

        # 1. 计算原始统计
        original_rms_db = self.compute_rms_db(audio)
        clipping_ratio = self.compute_clipping_ratio(
            audio, config.clipping_threshold
        )

        # 2. 静音检测
        if original_rms_db < config.silence_threshold_db:
            logger.info(
                "Sample filtered: silence (RMS=%.1f dBFS < %.1f dBFS)",
                original_rms_db,
                config.silence_threshold_db,
            )
            return PreprocessResult(
                audio=None,
                original_rms_db=original_rms_db,
                clipping_ratio=clipping_ratio,
                is_filtered=True,
                filter_reason="silence",
                sample_rate=config.target_sample_rate,
            )

        # 3. 削波检测
        if clipping_ratio > config.clipping_ratio_limit:
            logger.info(
                "Sample filtered: clipping (ratio=%.3f > %.3f)",
                clipping_ratio,
                config.clipping_ratio_limit,
            )
            return PreprocessResult(
                audio=None,
                original_rms_db=original_rms_db,
                clipping_ratio=clipping_ratio,
                is_filtered=True,
                filter_reason="clipping",
                sample_rate=config.target_sample_rate,
            )

        # 4. DC 偏移消除
        processed = self.remove_dc_offset(audio)

        # 5. 峰值归一化
        processed = self.peak_normalize(processed, config.target_peak_db)

        # 6. 重采样
        if sample_rate != config.target_sample_rate:
            processed = self.resample(
                processed, sample_rate, config.target_sample_rate
            )

        # 7. 尾部静音裁剪
        processed = self.trim_tail_silence(processed, config.target_sample_rate)

        return PreprocessResult(
            audio=processed,
            original_rms_db=original_rms_db,
            clipping_ratio=clipping_ratio,
            is_filtered=False,
            filter_reason=None,
            sample_rate=config.target_sample_rate,
        )
