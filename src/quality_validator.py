"""
数据质量验证模块。

对生成的音频和 embedding 执行自动化质量检查：
- 单样本质量检测：静音检测、削波检测、频谱熵检测
- 数据集级验证：PCA 坍缩检测、近重复样本检测、有效样本不足警告
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from src.audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SampleQualityResult:
    """单个样本的质量检查结果。"""

    sample_id: str
    is_valid: bool
    rms_db: float
    clipping_ratio: float
    spectral_entropy: float | None
    filter_reason: str | None  # silence/clipping/low_entropy/None


@dataclass
class DatasetQualityReport:
    """数据集质量报告。"""

    total_samples: int
    valid_samples: int
    filtered_samples: int
    filter_reasons: dict[str, int] = field(default_factory=dict)  # 原因 → 数量
    pca_variance_ratio: list[float] | None = None  # 前 50 个主成分方差解释比
    pca_top10_cumulative: float | None = None  # 前 10 主成分累积方差
    pca_collapse_warning: bool = False  # 维度坍缩警告
    near_duplicate_count: int = 0  # 近重复样本对数
    near_duplicate_ratio: float = 0.0  # 近重复占比
    insufficient_samples_warning: bool = False  # 有效样本不足警告
    target_samples: int = 0


class QualityValidator:
    """数据质量验证器。

    提供单样本音频质量检测和数据集级 embedding 质量验证。
    """

    def __init__(
        self,
        silence_threshold_db: float = -60.0,
        clipping_ratio_limit: float = 0.10,
        spectral_entropy_threshold: float = 1.0,
        near_duplicate_threshold: float = 0.999,
        pca_collapse_threshold: float = 0.95,
    ) -> None:
        """初始化质量验证器。

        Args:
            silence_threshold_db: 静音检测阈值（dBFS）
            clipping_ratio_limit: 削波比例上限
            spectral_entropy_threshold: 频谱熵阈值（低于此值视为内容过于简单）
            near_duplicate_threshold: 近重复检测的 cosine similarity 阈值
            pca_collapse_threshold: PCA 坍缩警告阈值（前 10 主成分累积方差）
        """
        self.silence_threshold_db = silence_threshold_db
        self.clipping_ratio_limit = clipping_ratio_limit
        self.spectral_entropy_threshold = spectral_entropy_threshold
        self.near_duplicate_threshold = near_duplicate_threshold
        self.pca_collapse_threshold = pca_collapse_threshold

    def compute_spectral_entropy(
        self, audio: np.ndarray, sample_rate: int
    ) -> float:
        """计算频谱熵。

        使用 FFT 计算功率谱密度，然后计算归一化 Shannon 熵。
        纯正弦波返回低熵值，白噪声返回高熵值。

        步骤：
        1. FFT → 取幅度平方得到功率谱密度
        2. 归一化为概率分布（除以总功率）
        3. 计算 Shannon 熵：H = -sum(p * log(p))
        4. 除以 log(N) 归一化到 [0, 1]

        Args:
            audio: 输入音频数据（1D float array）
            sample_rate: 采样率

        Returns:
            归一化频谱熵，范围 [0, 1]。0 表示纯音，1 表示白噪声。
        """
        # 计算 FFT，只取正频率部分
        fft_result = np.fft.rfft(audio)
        # 功率谱密度（幅度平方）
        psd = np.abs(fft_result) ** 2

        # 移除直流分量（索引 0）
        psd = psd[1:]

        # 避免全零功率谱
        total_power = np.sum(psd)
        if total_power == 0.0:
            return 0.0

        # 归一化为概率分布
        prob = psd / total_power

        # 过滤零概率分量，避免 log(0)
        nonzero_mask = prob > 0
        prob_nonzero = prob[nonzero_mask]

        # 计算 Shannon 熵
        entropy = -np.sum(prob_nonzero * np.log(prob_nonzero))

        # 归一化：除以 log(N)，其中 N 是频率分量数
        n_bins = len(psd)
        if n_bins <= 1:
            return 0.0

        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / max_entropy

        return float(normalized_entropy)

    def validate_sample(
        self, audio: np.ndarray, sample_rate: int, sample_id: str
    ) -> SampleQualityResult:
        """验证单个音频样本。

        按优先级执行检测：静音 → 削波 → 频谱熵。
        第一个触发的过滤条件即为最终过滤原因。

        Args:
            audio: 输入音频数据（1D float array）
            sample_rate: 采样率
            sample_id: 样本唯一标识

        Returns:
            SampleQualityResult 包含检测结果。
        """
        # 计算 RMS（复用 AudioPreprocessor 的逻辑）
        rms_db = AudioPreprocessor.compute_rms_db(audio)

        # 计算削波比例（复用 AudioPreprocessor 的逻辑）
        clipping_ratio = AudioPreprocessor.compute_clipping_ratio(audio)

        # 1. 静音检测
        if rms_db < self.silence_threshold_db:
            logger.info(
                "样本 %s 被过滤：静音 (RMS=%.1f dBFS < %.1f dBFS)",
                sample_id,
                rms_db,
                self.silence_threshold_db,
            )
            return SampleQualityResult(
                sample_id=sample_id,
                is_valid=False,
                rms_db=rms_db,
                clipping_ratio=clipping_ratio,
                spectral_entropy=None,
                filter_reason="silence",
            )

        # 2. 削波检测
        if clipping_ratio > self.clipping_ratio_limit:
            logger.info(
                "样本 %s 被过滤：削波 (ratio=%.3f > %.3f)",
                sample_id,
                clipping_ratio,
                self.clipping_ratio_limit,
            )
            return SampleQualityResult(
                sample_id=sample_id,
                is_valid=False,
                rms_db=rms_db,
                clipping_ratio=clipping_ratio,
                spectral_entropy=None,
                filter_reason="clipping",
            )

        # 3. 频谱熵检测
        spectral_entropy = self.compute_spectral_entropy(audio, sample_rate)
        if spectral_entropy < self.spectral_entropy_threshold:
            logger.info(
                "样本 %s 被过滤：低频谱熵 (entropy=%.3f < %.3f)",
                sample_id,
                spectral_entropy,
                self.spectral_entropy_threshold,
            )
            return SampleQualityResult(
                sample_id=sample_id,
                is_valid=False,
                rms_db=rms_db,
                clipping_ratio=clipping_ratio,
                spectral_entropy=spectral_entropy,
                filter_reason="low_entropy",
            )

        # 所有检测通过
        return SampleQualityResult(
            sample_id=sample_id,
            is_valid=True,
            rms_db=rms_db,
            clipping_ratio=clipping_ratio,
            spectral_entropy=spectral_entropy,
            filter_reason=None,
        )

    def detect_near_duplicates(
        self, embeddings: np.ndarray, threshold: float | None = None
    ) -> int:
        """检测近重复样本对数量。

        计算 pairwise cosine similarity，返回超过阈值的样本对数。

        Args:
            embeddings: (N, D) embedding 矩阵
            threshold: cosine similarity 阈值，None 时使用实例默认值

        Returns:
            超过阈值的样本对数量。
        """
        if threshold is None:
            threshold = self.near_duplicate_threshold

        n = embeddings.shape[0]
        if n < 2:
            return 0

        # 计算 pairwise cosine similarity
        sim_matrix = cosine_similarity(embeddings)

        # 只统计上三角（不含对角线），避免重复计数
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    count += 1

        return count

    def validate_embeddings(
        self,
        embeddings: np.ndarray,
        target_samples: int,
        valid_mask: np.ndarray,
    ) -> DatasetQualityReport:
        """验证 embedding 矩阵质量。

        只处理 valid_mask 为 True 的样本，执行：
        1. PCA 坍缩检测
        2. 近重复检测
        3. 有效样本不足检测

        Args:
            embeddings: (N, D) embedding 矩阵
            target_samples: 目标有效样本数
            valid_mask: (N,) bool 数组，标记有效样本

        Returns:
            DatasetQualityReport 包含验证结果。
        """
        total_samples = len(valid_mask)
        valid_count = int(np.sum(valid_mask))
        filtered_count = total_samples - valid_count

        # 提取有效样本的 embedding
        valid_embeddings = embeddings[valid_mask]

        # PCA 坍缩检测
        pca_variance_ratio = None
        pca_top10_cumulative = None
        pca_collapse_warning = False

        if valid_count >= 2:
            n_components = min(50, valid_count, valid_embeddings.shape[1])
            pca = PCA(n_components=n_components)
            pca.fit(valid_embeddings)

            pca_variance_ratio = pca.explained_variance_ratio_.tolist()

            # 计算前 10 主成分累积方差
            top10_count = min(10, len(pca_variance_ratio))
            pca_top10_cumulative = float(
                sum(pca_variance_ratio[:top10_count])
            )

            # 坍缩警告：前 10 主成分累积方差 > 阈值
            if pca_top10_cumulative > self.pca_collapse_threshold:
                pca_collapse_warning = True
                logger.warning(
                    "PCA 维度坍缩警告：前 %d 主成分累积方差 %.3f > %.3f",
                    top10_count,
                    pca_top10_cumulative,
                    self.pca_collapse_threshold,
                )

        # 近重复检测
        near_duplicate_count = 0
        near_duplicate_ratio = 0.0

        if valid_count >= 2:
            near_duplicate_count = self.detect_near_duplicates(
                valid_embeddings
            )
            # 总样本对数 = C(N, 2) = N*(N-1)/2
            total_pairs = valid_count * (valid_count - 1) // 2
            if total_pairs > 0:
                near_duplicate_ratio = near_duplicate_count / total_pairs

            if near_duplicate_count > 0:
                logger.info(
                    "检测到 %d 对近重复样本 (占比 %.4f)",
                    near_duplicate_count,
                    near_duplicate_ratio,
                )

        # 有效样本不足检测
        insufficient_samples_warning = valid_count < 0.8 * target_samples
        if insufficient_samples_warning:
            logger.warning(
                "有效样本不足警告：%d < %.0f (目标 %d 的 80%%)",
                valid_count,
                0.8 * target_samples,
                target_samples,
            )

        return DatasetQualityReport(
            total_samples=total_samples,
            valid_samples=valid_count,
            filtered_samples=filtered_count,
            filter_reasons={},  # 由调用方填充具体过滤原因统计
            pca_variance_ratio=pca_variance_ratio,
            pca_top10_cumulative=pca_top10_cumulative,
            pca_collapse_warning=pca_collapse_warning,
            near_duplicate_count=near_duplicate_count,
            near_duplicate_ratio=near_duplicate_ratio,
            insufficient_samples_warning=insufficient_samples_warning,
            target_samples=target_samples,
        )
