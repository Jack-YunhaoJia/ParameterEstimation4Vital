"""
区分能力评估模块。

通过 cosine similarity 和线性分类器（Logistic Regression）量化
MuQ embedding 对效果器开关状态的区分能力。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding_extractor import EmbeddingResult
from src.preset_parser import PresetParser

logger = logging.getLogger(__name__)


@dataclass
class EffectDiscriminationResult:
    """单个效果器的区分能力评估结果。

    Attributes:
        effect_name: 效果器开关名称
        cosine_similarity: 开/关 embedding 之间的 cosine similarity 均值
        classification_accuracy: Logistic Regression 二分类准确率
        is_distinguishable: 准确率 >= 0.75
        is_too_similar: cosine_similarity > 0.99（严格大于）
    """

    effect_name: str
    cosine_similarity: float
    classification_accuracy: float
    is_distinguishable: bool
    is_too_similar: bool


@dataclass
class FeasibilityReport:
    """可行性评估报告。

    Attributes:
        results: 各效果器的区分能力评估结果列表
        pass_count: 准确率 >= 75% 的效果器数量
        is_feasible: pass_count >= 6
        recommendation: 总体建议文本
    """

    results: list[EffectDiscriminationResult] = field(default_factory=list)
    pass_count: int = 0
    is_feasible: bool = False
    recommendation: str = ""


class Discriminator:
    """效果器开关区分能力评估器。

    对每个 Effect_Switch 计算 cosine similarity 和 Logistic Regression
    分类准确率，生成可行性报告。
    """

    ACCURACY_THRESHOLD: float = 0.75
    SIMILARITY_WARNING: float = 0.99
    MIN_PASS_COUNT: int = 6

    def evaluate_effect(
        self,
        on_embeddings: np.ndarray,
        off_embeddings: np.ndarray,
        effect_name: str,
    ) -> EffectDiscriminationResult:
        """评估单个效果器的区分能力。

        计算开/关 embedding 组之间的 cosine similarity 均值，
        并使用 Logistic Regression 进行二分类评估。

        Args:
            on_embeddings: 开启状态的 embedding 数组，shape (n_on, dim)
            off_embeddings: 关闭状态的 embedding 数组，shape (n_off, dim)
            effect_name: 效果器开关名称

        Returns:
            EffectDiscriminationResult 包含 cosine similarity、分类准确率等
        """
        # 确保 2D
        if on_embeddings.ndim == 1:
            on_embeddings = on_embeddings.reshape(1, -1)
        if off_embeddings.ndim == 1:
            off_embeddings = off_embeddings.reshape(1, -1)

        # 计算 cosine similarity 均值（所有 on/off 对之间）
        sim_matrix = cosine_similarity(on_embeddings, off_embeddings)
        cos_sim_mean = float(np.mean(sim_matrix))

        # 使用 Logistic Regression 进行二分类
        X = np.vstack([on_embeddings, off_embeddings])
        y = np.concatenate([
            np.ones(len(on_embeddings)),
            np.zeros(len(off_embeddings)),
        ])

        # 训练并评估（在训练集上评估，因为样本量很小）
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X, y)
        accuracy = float(clf.score(X, y))

        is_distinguishable = accuracy >= self.ACCURACY_THRESHOLD
        is_too_similar = cos_sim_mean > self.SIMILARITY_WARNING

        logger.debug(
            "Effect '%s': cosine_sim=%.4f, accuracy=%.4f, "
            "distinguishable=%s, too_similar=%s",
            effect_name,
            cos_sim_mean,
            accuracy,
            is_distinguishable,
            is_too_similar,
        )

        return EffectDiscriminationResult(
            effect_name=effect_name,
            cosine_similarity=cos_sim_mean,
            classification_accuracy=accuracy,
            is_distinguishable=is_distinguishable,
            is_too_similar=is_too_similar,
        )

    def evaluate_all(
        self,
        embeddings: EmbeddingResult,
    ) -> FeasibilityReport:
        """评估所有 9 个效果器，生成可行性报告。

        从 EmbeddingResult 中按文件名解析各效果器的开/关 embedding，
        逐个评估后按分类准确率降序排列，生成可行性报告。

        文件名格式：{effect_name}_{state}.wav
        例如：chorus_on_1.0.wav（开启）、chorus_on_0.0.wav（关闭）
        base_patch.wav 作为额外的"关闭"参考。

        Args:
            embeddings: EmbeddingResult，包含文件名到 embedding 的映射

        Returns:
            FeasibilityReport 包含所有效果器的评估结果和可行性判定
        """
        # 提取 base_patch embedding 作为额外的 off 参考
        base_patch_embedding = None
        for fname, emb in embeddings.embeddings.items():
            if fname.startswith("base_patch"):
                base_patch_embedding = emb
                break

        results: list[EffectDiscriminationResult] = []

        for effect_name in PresetParser.EFFECT_SWITCHES:
            on_list: list[np.ndarray] = []
            off_list: list[np.ndarray] = []

            for fname, emb in embeddings.embeddings.items():
                # 跳过 base_patch（单独处理）
                if fname.startswith("base_patch"):
                    continue

                # 解析文件名：{effect_name}_{state}.wav
                # 例如 chorus_on_1.0.wav → effect=chorus_on, state=1.0
                stem = fname.rsplit(".", 1)[0]  # 去掉 .wav
                parts = stem.rsplit("_", 1)  # 分离 state
                if len(parts) != 2:
                    continue

                file_effect = parts[0]
                state_str = parts[1]

                if file_effect != effect_name:
                    continue

                try:
                    state_val = float(state_str)
                except ValueError:
                    continue

                if state_val >= 0.5:
                    on_list.append(emb)
                else:
                    off_list.append(emb)

            # 将 base_patch 加入 off 组
            if base_patch_embedding is not None:
                off_list.append(base_patch_embedding)

            if not on_list or not off_list:
                logger.warning(
                    "Skipping effect '%s': insufficient embeddings "
                    "(on=%d, off=%d)",
                    effect_name,
                    len(on_list),
                    len(off_list),
                )
                continue

            on_arr = np.stack(on_list)
            off_arr = np.stack(off_list)

            result = self.evaluate_effect(on_arr, off_arr, effect_name)
            results.append(result)

        # 按分类准确率降序排列
        results.sort(key=lambda r: r.classification_accuracy, reverse=True)

        # 计算可行性
        pass_count = sum(
            1 for r in results if r.classification_accuracy >= self.ACCURACY_THRESHOLD
        )
        is_feasible = pass_count >= self.MIN_PASS_COUNT

        # 生成建议
        if is_feasible:
            recommendation = (
                f"Feasible: {pass_count}/{len(results)} effects achieve "
                f"≥{self.ACCURACY_THRESHOLD:.0%} accuracy. "
                f"MuQ embeddings can distinguish effect switch states. "
                f"Proceed to Phase 1."
            )
        else:
            recommendation = (
                f"Not feasible: only {pass_count}/{len(results)} effects "
                f"achieve ≥{self.ACCURACY_THRESHOLD:.0%} accuracy "
                f"(minimum {self.MIN_PASS_COUNT} required). "
                f"Consider alternative audio representations or "
                f"verify effect configurations produce audible differences."
            )

        # 追加 too_similar 警告
        too_similar_effects = [r.effect_name for r in results if r.is_too_similar]
        if too_similar_effects:
            recommendation += (
                f" Warning: {', '.join(too_similar_effects)} have "
                f"cosine similarity > {self.SIMILARITY_WARNING} — "
                f"representation cannot distinguish these effects. "
                f"Check if they produce audible differences in the base patch."
            )

        logger.info(
            "Feasibility report: %d/%d pass, feasible=%s",
            pass_count,
            len(results),
            is_feasible,
        )

        return FeasibilityReport(
            results=results,
            pass_count=pass_count,
            is_feasible=is_feasible,
            recommendation=recommendation,
        )
