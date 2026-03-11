"""
分布分析模块。

分析 embedding 空间分布（PCA 降维、cosine similarity 分布）和
参数覆盖率（分布统计、KS 检验），生成结构化 JSON 报告。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from src.training_data import CORE_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class DistributionReport:
    """分布分析报告。"""

    # PCA 分析
    pca_variance_ratios: list[float]  # 前 50 主成分方差解释比
    pca_cumulative_ratios: list[float]  # 累积方差解释比
    # Cosine similarity 分布
    cosine_sim_mean: float
    cosine_sim_std: float
    cosine_sim_min: float
    cosine_sim_max: float
    cosine_sim_quantiles: dict[str, float]  # "25%", "50%", "75%"
    diversity_warning: bool  # 平均 cosine sim > threshold
    # 参数覆盖率
    param_stats: dict[str, dict[str, float]]  # 参数名 → {mean, std, min, max}
    param_ks_results: dict[str, dict[str, float]]  # 参数名 → {statistic, pvalue}


class DistributionAnalyzer:
    """数据集分布分析器。

    分析 embedding 空间分布和参数覆盖率，生成结构化报告。
    """

    def __init__(self, diversity_threshold: float = 0.95) -> None:
        """初始化分析器。

        Args:
            diversity_threshold: cosine similarity 均值超过此阈值时发出多样性警告
        """
        self.diversity_threshold = diversity_threshold


    def analyze_embeddings(self, embeddings: np.ndarray) -> dict:
        """分析 embedding 空间分布。

        执行 PCA 降维和 pairwise cosine similarity 计算。

        Args:
            embeddings: (N, D) embedding 矩阵

        Returns:
            包含 PCA 和 cosine similarity 分析结果的字典
        """
        n_samples, n_features = embeddings.shape

        # PCA 降维：取 min(50, N, D) 个主成分
        n_components = min(50, n_samples, n_features)
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)

        variance_ratios = pca.explained_variance_ratio_.tolist()
        cumulative_ratios = np.cumsum(variance_ratios).tolist()

        # 计算 pairwise cosine similarity，只取上三角（不含对角线）
        sim_matrix = cosine_similarity(embeddings)
        # 提取上三角元素（不含对角线）
        triu_indices = np.triu_indices(n_samples, k=1)
        sim_values = sim_matrix[triu_indices]

        # 计算分布统计
        sim_mean = float(np.mean(sim_values))
        sim_std = float(np.std(sim_values))
        sim_min = float(np.min(sim_values))
        sim_max = float(np.max(sim_values))
        quantiles = {
            "25%": float(np.percentile(sim_values, 25)),
            "50%": float(np.percentile(sim_values, 50)),
            "75%": float(np.percentile(sim_values, 75)),
        }

        return {
            "pca_variance_ratios": variance_ratios,
            "pca_cumulative_ratios": cumulative_ratios,
            "cosine_sim_mean": sim_mean,
            "cosine_sim_std": sim_std,
            "cosine_sim_min": sim_min,
            "cosine_sim_max": sim_max,
            "cosine_sim_quantiles": quantiles,
        }

    def analyze_parameters(self, params: np.ndarray) -> dict:
        """分析参数覆盖率。

        对每个参数维度计算分布统计（mean, std, min, max），
        并使用 KS 检验与均匀分布进行比较。

        Args:
            params: (N, 45) 参数矩阵

        Returns:
            包含参数统计和 KS 检验结果的字典
        """
        param_stats: dict[str, dict[str, float]] = {}
        param_ks_results: dict[str, dict[str, float]] = {}

        for i, (name, p_min, p_max) in enumerate(CORE_PARAMS):
            col = params[:, i]

            # 分布统计
            param_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }

            # 归一化到 [0, 1] 后与均匀分布进行 KS 检验
            if p_max > p_min:
                normalized = (col - p_min) / (p_max - p_min)
            else:
                # 参数范围为零时，归一化为全 0
                normalized = np.zeros_like(col)

            ks_stat, ks_pvalue = stats.kstest(normalized, "uniform", args=(0, 1))
            param_ks_results[name] = {
                "statistic": float(ks_stat),
                "pvalue": float(ks_pvalue),
            }

        return {
            "param_stats": param_stats,
            "param_ks_results": param_ks_results,
        }

    def generate_report(
        self, embeddings: np.ndarray, params: np.ndarray
    ) -> DistributionReport:
        """生成完整分布分析报告。

        组合 embedding 分析和参数分析，当平均 cosine similarity
        超过 diversity_threshold 时设置 diversity_warning = True。

        Args:
            embeddings: (N, D) embedding 矩阵
            params: (N, 45) 参数矩阵

        Returns:
            DistributionReport 完整分布分析报告
        """
        # 分析 embedding 空间
        emb_result = self.analyze_embeddings(embeddings)
        # 分析参数覆盖率
        param_result = self.analyze_parameters(params)

        # 判断多样性警告
        diversity_warning = emb_result["cosine_sim_mean"] > self.diversity_threshold

        return DistributionReport(
            pca_variance_ratios=emb_result["pca_variance_ratios"],
            pca_cumulative_ratios=emb_result["pca_cumulative_ratios"],
            cosine_sim_mean=emb_result["cosine_sim_mean"],
            cosine_sim_std=emb_result["cosine_sim_std"],
            cosine_sim_min=emb_result["cosine_sim_min"],
            cosine_sim_max=emb_result["cosine_sim_max"],
            cosine_sim_quantiles=emb_result["cosine_sim_quantiles"],
            diversity_warning=diversity_warning,
            param_stats=param_result["param_stats"],
            param_ks_results=param_result["param_ks_results"],
        )

    def save_report(self, report: DistributionReport, output_path: Path) -> None:
        """保存报告为 JSON 文件。

        处理 numpy 类型序列化，使用 indent=2 和 ensure_ascii=False。

        Args:
            report: 分布分析报告
            output_path: JSON 输出路径
        """

        def _numpy_serializer(obj: object) -> object:
            """处理 numpy 类型的 JSON 序列化。"""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"无法序列化类型: {type(obj)}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_dict = asdict(report)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=_numpy_serializer)

        logger.info("分布分析报告已保存至 %s", output_path)
