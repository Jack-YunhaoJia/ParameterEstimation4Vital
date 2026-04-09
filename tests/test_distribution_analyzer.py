"""
DistributionAnalyzer 单元测试。

验证分布分析器的核心逻辑：PCA 降维、cosine similarity 统计、
参数 KS 检验、多样性警告和 JSON 报告序列化。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.distribution_analyzer import DistributionAnalyzer, DistributionReport
from src.training_data import CORE_PARAMS


class TestAnalyzeEmbeddingsPCA:
    """analyze_embeddings PCA 输出测试。"""

    def test_correct_number_of_components(self) -> None:
        """PCA 应输出 min(50, N, D) 个主成分。"""
        rng = np.random.default_rng(42)
        # N=100, D=1024 → 应输出 50 个主成分
        embeddings = rng.standard_normal((100, 1024)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        assert len(result["pca_variance_ratios"]) == 50
        assert len(result["pca_cumulative_ratios"]) == 50

    def test_variance_ratios_in_valid_range(self) -> None:
        """每个方差解释比应在 [0, 1] 范围内。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 1024)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        for ratio in result["pca_variance_ratios"]:
            assert 0.0 <= ratio <= 1.0, f"方差解释比 {ratio} 不在 [0, 1] 范围内"

    def test_cumulative_ratios_monotonic(self) -> None:
        """累积方差解释比应单调递增。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 1024)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        cumulative = result["pca_cumulative_ratios"]
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1], (
                f"累积比在索引 {i} 处不单调: {cumulative[i-1]} > {cumulative[i]}"
            )

    def test_cumulative_ratios_not_exceed_one(self) -> None:
        """累积方差解释比不应超过 1.0。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 1024)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        for ratio in result["pca_cumulative_ratios"]:
            assert ratio <= 1.0 + 1e-7, f"累积比 {ratio} 超过 1.0"

    def test_fewer_samples_than_50(self) -> None:
        """当 N < 50 时，PCA 应输出 N 个主成分。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((20, 1024)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        assert len(result["pca_variance_ratios"]) == 20


class TestAnalyzeEmbeddingsCosineSim:
    """analyze_embeddings cosine similarity 统计测试。"""

    def test_min_le_mean_le_max(self) -> None:
        """min ≤ mean ≤ max 应成立。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 128)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        assert result["cosine_sim_min"] <= result["cosine_sim_mean"]
        assert result["cosine_sim_mean"] <= result["cosine_sim_max"]

    def test_std_non_negative(self) -> None:
        """标准差应 ≥ 0。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 128)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        assert result["cosine_sim_std"] >= 0.0

    def test_quantiles_ordered(self) -> None:
        """分位数应满足 25% ≤ 50% ≤ 75%。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 128)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        q = result["cosine_sim_quantiles"]
        assert q["25%"] <= q["50%"] <= q["75%"]

    def test_values_in_valid_range(self) -> None:
        """cosine similarity 值应在 [-1, 1] 范围内。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 128)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        assert result["cosine_sim_min"] >= -1.0 - 1e-7
        assert result["cosine_sim_max"] <= 1.0 + 1e-7


class TestAnalyzeParameters:
    """analyze_parameters KS 检验测试。"""

    def test_ks_results_cover_all_params(self) -> None:
        """KS 检验应覆盖所有 45 个参数。"""
        rng = np.random.default_rng(42)
        params = rng.uniform(size=(200, 45)).astype(np.float32)
        # 将参数缩放到各自的值域
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_parameters(params)

        assert len(result["param_ks_results"]) == 45
        assert len(result["param_stats"]) == 45

    def test_ks_statistic_in_valid_range(self) -> None:
        """KS statistic 应在 [0, 1] 范围内。"""
        rng = np.random.default_rng(42)
        params = rng.uniform(size=(200, 45)).astype(np.float32)
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_parameters(params)

        for name, ks in result["param_ks_results"].items():
            assert 0.0 <= ks["statistic"] <= 1.0, f"{name} statistic 不在 [0, 1]"
            assert 0.0 <= ks["pvalue"] <= 1.0, f"{name} pvalue 不在 [0, 1]"

    def test_uniform_params_high_pvalue(self) -> None:
        """均匀随机参数的 KS 检验应有较高的 p-value。"""
        rng = np.random.default_rng(42)
        params = rng.uniform(size=(500, 45)).astype(np.float32)
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_parameters(params)

        # 大部分参数的 p-value 应 > 0.01（均匀分布的 KS 检验）
        high_pvalue_count = sum(
            1 for ks in result["param_ks_results"].values() if ks["pvalue"] > 0.01
        )
        # 至少 80% 的参数应通过 KS 检验
        assert high_pvalue_count >= 36, (
            f"仅 {high_pvalue_count}/45 个参数通过 KS 检验"
        )


class TestGenerateReportDiversityWarning:
    """generate_report 多样性警告测试。"""

    def test_diversity_warning_true(self) -> None:
        """当平均 cosine sim > 0.95 时应发出警告。"""
        rng = np.random.default_rng(42)
        # 构造高度相似的 embedding：基向量 + 微小扰动
        base = rng.standard_normal((1, 128)).astype(np.float32)
        noise = rng.standard_normal((50, 128)).astype(np.float32) * 0.01
        embeddings = base + noise  # 所有向量几乎相同

        params = rng.uniform(size=(50, 45)).astype(np.float32)
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        report = analyzer.generate_report(embeddings, params)

        assert report.diversity_warning is True
        assert report.cosine_sim_mean > 0.95

    def test_no_diversity_warning(self) -> None:
        """随机 embedding 不应触发多样性警告。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 128)).astype(np.float32)
        params = rng.uniform(size=(100, 45)).astype(np.float32)
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        report = analyzer.generate_report(embeddings, params)

        assert report.diversity_warning is False


class TestSaveReportRoundTrip:
    """save_report JSON 序列化 round-trip 测试。"""

    def test_save_and_load_preserves_values(self, tmp_path: Path) -> None:
        """JSON 保存后重新加载应保留数值。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((60, 128)).astype(np.float32)
        params = rng.uniform(size=(60, 45)).astype(np.float32)
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        report = analyzer.generate_report(embeddings, params)

        # 保存
        output_path = tmp_path / "report.json"
        analyzer.save_report(report, output_path)

        # 加载
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)

        # 验证 PCA 数据
        assert len(loaded["pca_variance_ratios"]) == len(report.pca_variance_ratios)
        for orig, loaded_val in zip(
            report.pca_variance_ratios, loaded["pca_variance_ratios"]
        ):
            assert abs(orig - loaded_val) < 1e-6

        # 验证 cosine similarity 统计
        assert abs(loaded["cosine_sim_mean"] - report.cosine_sim_mean) < 1e-6
        assert abs(loaded["cosine_sim_std"] - report.cosine_sim_std) < 1e-6
        assert abs(loaded["cosine_sim_min"] - report.cosine_sim_min) < 1e-6
        assert abs(loaded["cosine_sim_max"] - report.cosine_sim_max) < 1e-6

        # 验证 diversity_warning
        assert loaded["diversity_warning"] == report.diversity_warning

        # 验证参数统计
        assert set(loaded["param_stats"].keys()) == set(report.param_stats.keys())
        assert set(loaded["param_ks_results"].keys()) == set(
            report.param_ks_results.keys()
        )

    def test_json_format(self, tmp_path: Path) -> None:
        """JSON 文件应使用 indent=2 和 UTF-8 编码。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((30, 64)).astype(np.float32)
        params = rng.uniform(size=(30, 45)).astype(np.float32)
        for i, (_, p_min, p_max) in enumerate(CORE_PARAMS):
            params[:, i] = params[:, i] * (p_max - p_min) + p_min

        analyzer = DistributionAnalyzer()
        report = analyzer.generate_report(embeddings, params)

        output_path = tmp_path / "report.json"
        analyzer.save_report(report, output_path)

        # 验证文件可读且格式正确
        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("{\n")  # indent=2 格式
        loaded = json.loads(content)
        assert isinstance(loaded, dict)


class TestSmallEmbeddingMatrix:
    """小 embedding 矩阵边界情况测试。"""

    def test_two_samples(self) -> None:
        """2 个样本应正常工作（只有 1 对 cosine similarity）。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((2, 64)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        # 只有 1 对，mean == min == max
        assert result["cosine_sim_mean"] == result["cosine_sim_min"]
        assert result["cosine_sim_mean"] == result["cosine_sim_max"]
        # PCA 应输出 2 个主成分
        assert len(result["pca_variance_ratios"]) == 2

    def test_three_samples(self) -> None:
        """3 个样本应正常工作（3 对 cosine similarity）。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((3, 64)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        assert len(result["pca_variance_ratios"]) == 3
        assert result["cosine_sim_min"] <= result["cosine_sim_mean"]
        assert result["cosine_sim_mean"] <= result["cosine_sim_max"]

    def test_small_dimension(self) -> None:
        """低维 embedding 应正常工作。"""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((100, 10)).astype(np.float32)
        analyzer = DistributionAnalyzer()
        result = analyzer.analyze_embeddings(embeddings)

        # min(50, 100, 10) = 10
        assert len(result["pca_variance_ratios"]) == 10
