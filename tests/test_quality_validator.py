"""
QualityValidator 单元测试。

验证数据质量验证器的核心逻辑：频谱熵计算、单样本质量检测、
近重复检测和 embedding 矩阵验证。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quality_validator import (
    DatasetQualityReport,
    QualityValidator,
    SampleQualityResult,
)


class TestComputeSpectralEntropy:
    """频谱熵计算测试。"""

    def test_pure_sine_low_entropy(self) -> None:
        """纯正弦波应返回低熵值（能量集中在单一频率）。"""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440 Hz 纯正弦波
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        validator = QualityValidator()
        entropy = validator.compute_spectral_entropy(audio, sr)

        # 纯正弦波的频谱熵应该很低
        assert entropy < 0.3, f"纯正弦波熵应 < 0.3，实际 {entropy:.4f}"

    def test_white_noise_high_entropy(self) -> None:
        """白噪声应返回高熵值（能量均匀分布在所有频率）。"""
        sr = 16000
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(sr).astype(np.float32)

        validator = QualityValidator()
        entropy = validator.compute_spectral_entropy(audio, sr)

        # 白噪声的频谱熵应该接近 1.0
        assert entropy > 0.9, f"白噪声熵应 > 0.9，实际 {entropy:.4f}"

    def test_zero_audio_returns_zero(self) -> None:
        """全零音频应返回 0 熵。"""
        audio = np.zeros(16000, dtype=np.float32)
        validator = QualityValidator()
        entropy = validator.compute_spectral_entropy(audio, 16000)
        assert entropy == 0.0

    def test_entropy_range(self) -> None:
        """频谱熵应在 [0, 1] 范围内。"""
        sr = 16000
        rng = np.random.default_rng(123)
        audio = rng.standard_normal(sr).astype(np.float32)

        validator = QualityValidator()
        entropy = validator.compute_spectral_entropy(audio, sr)

        assert 0.0 <= entropy <= 1.0


class TestValidateSample:
    """单样本质量检测测试。"""

    def test_silent_audio_filtered(self) -> None:
        """静音音频应被标记为 silence 过滤。"""
        # 非常安静的音频（RMS 远低于 -60 dBFS）
        audio = np.ones(16000, dtype=np.float32) * 1e-5

        validator = QualityValidator()
        result = validator.validate_sample(audio, 16000, "test_silent")

        assert not result.is_valid
        assert result.filter_reason == "silence"
        assert result.spectral_entropy is None

    def test_clipping_audio_filtered(self) -> None:
        """削波音频应被标记为 clipping 过滤。"""
        # 大量削波的音频（超过 10% 的样本绝对值 > 0.99）
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(16000).astype(np.float32) * 0.5
        # 将前 20% 设为削波值
        n_clip = int(0.2 * len(audio))
        audio[:n_clip] = 1.0

        validator = QualityValidator()
        result = validator.validate_sample(audio, 16000, "test_clipping")

        assert not result.is_valid
        assert result.filter_reason == "clipping"
        assert result.clipping_ratio > 0.10

    def test_low_entropy_audio_filtered(self) -> None:
        """低频谱熵音频应被标记为 low_entropy 过滤。"""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # 纯正弦波：高 RMS、无削波、但频谱熵很低
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        # 使用较高的频谱熵阈值确保纯正弦波被过滤
        validator = QualityValidator(spectral_entropy_threshold=1.0)
        result = validator.validate_sample(audio, sr, "test_low_entropy")

        assert not result.is_valid
        assert result.filter_reason == "low_entropy"
        assert result.spectral_entropy is not None
        assert result.spectral_entropy < 1.0

    def test_normal_audio_passes(self) -> None:
        """正常音频（足够响、无削波、高熵）应通过所有检测。"""
        sr = 16000
        rng = np.random.default_rng(42)
        # 白噪声：高 RMS、无削波、高频谱熵
        audio = (rng.standard_normal(sr) * 0.3).astype(np.float32)

        # 使用较低的频谱熵阈值
        validator = QualityValidator(spectral_entropy_threshold=0.1)
        result = validator.validate_sample(audio, sr, "test_normal")

        assert result.is_valid
        assert result.filter_reason is None
        assert result.spectral_entropy is not None
        assert result.spectral_entropy > 0.1

    def test_filter_priority_silence_first(self) -> None:
        """静音检测优先于削波检测。"""
        # 非常安静但有削波特征的音频
        audio = np.zeros(16000, dtype=np.float32)

        validator = QualityValidator()
        result = validator.validate_sample(audio, 16000, "test_priority")

        assert not result.is_valid
        assert result.filter_reason == "silence"

    def test_result_contains_all_fields(self) -> None:
        """结果应包含所有必要字段。"""
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(16000) * 0.3).astype(np.float32)

        validator = QualityValidator(spectral_entropy_threshold=0.1)
        result = validator.validate_sample(audio, 16000, "test_fields")

        assert isinstance(result, SampleQualityResult)
        assert result.sample_id == "test_fields"
        assert isinstance(result.rms_db, float)
        assert isinstance(result.clipping_ratio, float)


class TestDetectNearDuplicates:
    """近重复检测测试。"""

    def test_identical_vectors_detected(self) -> None:
        """完全相同的向量应被检测为近重复。"""
        # 3 个相同的向量 → 3 对近重复 (C(3,2) = 3)
        vec = np.random.default_rng(42).standard_normal(128).astype(np.float32)
        embeddings = np.stack([vec, vec, vec])

        validator = QualityValidator()
        count = validator.detect_near_duplicates(embeddings)

        assert count == 3

    def test_random_vectors_not_detected(self) -> None:
        """随机正交向量不应被检测为近重复。"""
        rng = np.random.default_rng(42)
        # 高维随机向量的 cosine similarity 通常很低
        embeddings = rng.standard_normal((10, 1024)).astype(np.float32)

        validator = QualityValidator()
        count = validator.detect_near_duplicates(embeddings)

        assert count == 0

    def test_single_vector_returns_zero(self) -> None:
        """单个向量应返回 0 对近重复。"""
        embeddings = np.random.default_rng(42).standard_normal((1, 128)).astype(
            np.float32
        )

        validator = QualityValidator()
        count = validator.detect_near_duplicates(embeddings)

        assert count == 0

    def test_custom_threshold(self) -> None:
        """自定义阈值应正确生效。"""
        rng = np.random.default_rng(42)
        # 创建两个非常相似但不完全相同的向量
        vec = rng.standard_normal(128).astype(np.float32)
        vec_similar = vec + rng.standard_normal(128).astype(np.float32) * 0.001
        embeddings = np.stack([vec, vec_similar])

        validator = QualityValidator()
        # 使用较低阈值应检测到
        count_low = validator.detect_near_duplicates(embeddings, threshold=0.99)
        # 使用极高阈值可能检测不到
        count_high = validator.detect_near_duplicates(
            embeddings, threshold=0.99999
        )

        assert count_low >= count_high


class TestValidateEmbeddings:
    """Embedding 矩阵验证测试。"""

    def test_pca_collapse_warning(self) -> None:
        """当前 10 主成分累积方差 > 0.95 时应发出坍缩警告。"""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_dims = 128

        # 创建低秩矩阵（大部分方差集中在少数维度）
        # 使用 5 个主方向生成数据
        base = rng.standard_normal((n_samples, 5))
        projection = rng.standard_normal((5, n_dims)) * 0.1
        embeddings = base @ projection
        # 添加极小噪声
        embeddings += rng.standard_normal((n_samples, n_dims)) * 0.001

        valid_mask = np.ones(n_samples, dtype=bool)

        validator = QualityValidator(pca_collapse_threshold=0.95)
        report = validator.validate_embeddings(embeddings, 100, valid_mask)

        assert report.pca_collapse_warning is True
        assert report.pca_top10_cumulative is not None
        assert report.pca_top10_cumulative > 0.95

    def test_no_pca_collapse_warning(self) -> None:
        """高维均匀分布的 embedding 不应触发坍缩警告。"""
        rng = np.random.default_rng(42)
        n_samples = 200
        n_dims = 128

        # 高维随机数据，方差均匀分布
        embeddings = rng.standard_normal((n_samples, n_dims))
        valid_mask = np.ones(n_samples, dtype=bool)

        validator = QualityValidator(pca_collapse_threshold=0.95)
        report = validator.validate_embeddings(embeddings, 200, valid_mask)

        assert report.pca_collapse_warning is False

    def test_insufficient_samples_warning(self) -> None:
        """有效样本 < 80% 目标时应发出不足警告。"""
        rng = np.random.default_rng(42)
        n_samples = 100
        embeddings = rng.standard_normal((n_samples, 64))

        # 只有 50 个有效样本，目标 100 → 50 < 80
        valid_mask = np.zeros(n_samples, dtype=bool)
        valid_mask[:50] = True

        validator = QualityValidator()
        report = validator.validate_embeddings(embeddings, 100, valid_mask)

        assert report.insufficient_samples_warning is True
        assert report.valid_samples == 50
        assert report.filtered_samples == 50

    def test_no_insufficient_samples_warning(self) -> None:
        """有效样本 >= 80% 目标时不应发出不足警告。"""
        rng = np.random.default_rng(42)
        n_samples = 100
        embeddings = rng.standard_normal((n_samples, 64))

        # 85 个有效样本，目标 100 → 85 >= 80
        valid_mask = np.zeros(n_samples, dtype=bool)
        valid_mask[:85] = True

        validator = QualityValidator()
        report = validator.validate_embeddings(embeddings, 100, valid_mask)

        assert report.insufficient_samples_warning is False

    def test_near_duplicate_detection_in_embeddings(self) -> None:
        """validate_embeddings 应检测到近重复样本。"""
        rng = np.random.default_rng(42)
        n_samples = 20
        n_dims = 64

        embeddings = rng.standard_normal((n_samples, n_dims))
        # 让前两个向量完全相同
        embeddings[1] = embeddings[0].copy()

        valid_mask = np.ones(n_samples, dtype=bool)

        validator = QualityValidator()
        report = validator.validate_embeddings(embeddings, 20, valid_mask)

        assert report.near_duplicate_count >= 1
        assert report.near_duplicate_ratio > 0.0

    def test_valid_mask_filtering(self) -> None:
        """validate_embeddings 应只处理 valid_mask 为 True 的样本。"""
        rng = np.random.default_rng(42)
        n_samples = 50
        n_dims = 64

        embeddings = rng.standard_normal((n_samples, n_dims))

        # 只有前 30 个有效
        valid_mask = np.zeros(n_samples, dtype=bool)
        valid_mask[:30] = True

        validator = QualityValidator()
        report = validator.validate_embeddings(embeddings, 50, valid_mask)

        assert report.total_samples == 50
        assert report.valid_samples == 30
        assert report.filtered_samples == 20
        assert report.target_samples == 50

    def test_report_type(self) -> None:
        """返回值应为 DatasetQualityReport 类型。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((20, 64))
        valid_mask = np.ones(20, dtype=bool)

        validator = QualityValidator()
        report = validator.validate_embeddings(embeddings, 20, valid_mask)

        assert isinstance(report, DatasetQualityReport)
        assert report.pca_variance_ratio is not None
        assert len(report.pca_variance_ratio) > 0

    def test_few_valid_samples(self) -> None:
        """极少有效样本时不应崩溃。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 64))

        # 只有 1 个有效样本
        valid_mask = np.zeros(10, dtype=bool)
        valid_mask[0] = True

        validator = QualityValidator()
        report = validator.validate_embeddings(embeddings, 10, valid_mask)

        assert report.valid_samples == 1
        # PCA 需要至少 2 个样本
        assert report.pca_variance_ratio is None
        assert report.near_duplicate_count == 0
