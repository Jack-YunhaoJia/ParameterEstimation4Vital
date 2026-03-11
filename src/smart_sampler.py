"""
智能参数采样模块。

使用拉丁超立方采样（LHS）和分层采样策略替代朴素均匀采样，
提升参数空间覆盖率。支持 LHS 连续参数采样 + 分层效果器开关采样。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kstest
from scipy.stats.qmc import LatinHypercube

from src.training_data import (
    CORE_PARAMS,
    EFFECT_SWITCH_INDICES,
    EFFECT_SWITCH_NAMES,
    NUM_PARAMS,
)

logger = logging.getLogger(__name__)


@dataclass
class SamplingReport:
    """采样覆盖率报告。"""

    n_samples: int
    strategy: str  # "lhs" / "stratified"
    per_param_ks_statistic: dict[str, float] = field(default_factory=dict)
    per_param_ks_pvalue: dict[str, float] = field(default_factory=dict)
    effect_switch_distribution: dict[int, int] = field(default_factory=dict)
    seed: int = 42


# Identify discrete parameter indices
_FILTER_MODEL_INDEX: int = next(
    i for i, (name, _, _) in enumerate(CORE_PARAMS) if name == "filter_1_model"
)
_FILTER_STYLE_INDEX: int = next(
    i for i, (name, _, _) in enumerate(CORE_PARAMS) if name == "filter_1_style"
)

# All discrete parameter indices (effect switches + filter_model + filter_style)
_DISCRETE_INDICES: set[int] = set(EFFECT_SWITCH_INDICES) | {
    _FILTER_MODEL_INDEX,
    _FILTER_STYLE_INDEX,
}


class SmartSampler:
    """智能参数采样器。

    支持 LHS 连续参数采样 + 分层效果器开关采样。
    使用 CORE_PARAMS 定义（从 training_data.py 导入）。
    """

    def __init__(self, seed: int = 42) -> None:
        """初始化采样器。

        Args:
            seed: 随机种子，确保采样结果可复现。
        """
        self.seed = seed

    def sample_lhs(self, n: int) -> np.ndarray:
        """使用拉丁超立方采样生成 n 个参数向量。

        连续参数使用 LHS，离散参数（filter_model, filter_style）
        使用均匀离散采样后取整，效果器开关四舍五入为 0/1。

        Args:
            n: 采样数量

        Returns:
            (n, 45) 参数矩阵，float32
        """
        # LHS generates samples in [0, 1]^d
        sampler = LatinHypercube(d=NUM_PARAMS, seed=self.seed)
        unit_samples = sampler.random(n=n)  # (n, 45) in [0, 1]

        # Scale to parameter ranges
        params = np.empty((n, NUM_PARAMS), dtype=np.float32)
        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            params[:, col] = lo + unit_samples[:, col] * (hi - lo)

        # Post-process discrete parameters
        self._discretize_params(params)

        return params

    def sample_stratified_switches(self, n: int) -> np.ndarray:
        """分层采样效果器开关组合。

        按活跃效果器数量（0-9）分层，每层样本数与该层组合数成正比。
        连续参数仍使用 LHS。

        Args:
            n: 总采样数量

        Returns:
            (n, 45) 参数矩阵，float32
        """
        rng = np.random.default_rng(self.seed)
        num_switches = len(EFFECT_SWITCH_INDICES)  # 9

        # Compute layer sizes proportional to C(9, k)
        total_combinations = sum(math.comb(num_switches, k) for k in range(num_switches + 1))
        layer_sizes = []
        allocated = 0
        for k in range(num_switches + 1):
            if k == num_switches:
                # Last layer gets the remainder to ensure exact total
                layer_sizes.append(n - allocated)
            else:
                size = round(n * math.comb(num_switches, k) / total_combinations)
                layer_sizes.append(size)
                allocated += size

        # Generate LHS samples for continuous parameters
        sampler = LatinHypercube(d=NUM_PARAMS, seed=self.seed)
        unit_samples = sampler.random(n=n)  # (n, 45) in [0, 1]

        params = np.empty((n, NUM_PARAMS), dtype=np.float32)
        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            params[:, col] = lo + unit_samples[:, col] * (hi - lo)

        # Post-process discrete parameters (filter_model, filter_style)
        params[:, _FILTER_MODEL_INDEX] = np.floor(params[:, _FILTER_MODEL_INDEX]).clip(0, 5).astype(np.float32)
        params[:, _FILTER_STYLE_INDEX] = np.floor(params[:, _FILTER_STYLE_INDEX]).clip(0, 3).astype(np.float32)

        # Assign effect switches by stratified layers
        row = 0
        for k, layer_n in enumerate(layer_sizes):
            if layer_n <= 0:
                continue
            for _ in range(layer_n):
                # Generate a random switch combination with exactly k active switches
                switch_values = np.zeros(num_switches, dtype=np.float32)
                if k > 0:
                    active_indices = rng.choice(num_switches, size=k, replace=False)
                    switch_values[active_indices] = 1.0
                for si, param_idx in enumerate(EFFECT_SWITCH_INDICES):
                    params[row, param_idx] = switch_values[si]
                row += 1

        return params

    def sample(self, n: int, strategy: str = "lhs_stratified") -> np.ndarray:
        """统一采样入口。

        Args:
            n: 采样数量
            strategy:
                - "lhs": 纯 LHS 采样
                - "lhs_stratified": LHS + 分层效果器开关（默认）

        Returns:
            (n, 45) 参数矩阵
        """
        if strategy == "lhs":
            return self.sample_lhs(n)
        elif strategy == "lhs_stratified":
            return self.sample_stratified_switches(n)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy!r}. Use 'lhs' or 'lhs_stratified'.")

    def generate_report(self, params: np.ndarray) -> SamplingReport:
        """生成采样覆盖率报告。

        对每个参数维度计算 KS 检验统计量和 p-value（与均匀分布比较），
        统计效果器开关分布（按活跃效果器数量分组）。

        Args:
            params: (n, 45) 参数矩阵

        Returns:
            SamplingReport
        """
        ks_stats: dict[str, float] = {}
        ks_pvalues: dict[str, float] = {}

        for col, (name, lo, hi) in enumerate(CORE_PARAMS):
            # Normalize to [0, 1] for KS test against uniform(0, 1)
            if hi > lo:
                normalized = (params[:, col] - lo) / (hi - lo)
            else:
                normalized = params[:, col]
            stat, pvalue = kstest(normalized, "uniform", args=(0, 1))
            ks_stats[name] = float(stat)
            ks_pvalues[name] = float(pvalue)

        # Effect switch distribution: count by number of active switches
        switch_matrix = params[:, EFFECT_SWITCH_INDICES]  # (n, 9)
        active_counts = switch_matrix.sum(axis=1).astype(int)  # (n,)
        distribution: dict[int, int] = {}
        for k in range(len(EFFECT_SWITCH_INDICES) + 1):
            count = int((active_counts == k).sum())
            if count > 0:
                distribution[k] = count

        return SamplingReport(
            n_samples=params.shape[0],
            strategy="lhs",  # Will be overridden by caller if needed
            per_param_ks_statistic=ks_stats,
            per_param_ks_pvalue=ks_pvalues,
            effect_switch_distribution=distribution,
            seed=self.seed,
        )

    @staticmethod
    def _discretize_params(params: np.ndarray) -> None:
        """Post-process discrete parameters in-place.

        - Effect switches: round to 0 or 1
        - filter_1_model: round to integers in {0, 1, 2, 3, 4, 5}
        - filter_1_style: round to integers in {0, 1, 2, 3}
        """
        # Effect switches → binary 0/1
        for idx in EFFECT_SWITCH_INDICES:
            params[:, idx] = np.round(params[:, idx]).clip(0, 1)

        # filter_1_model → integers {0, 1, 2, 3, 4, 5}
        params[:, _FILTER_MODEL_INDEX] = np.floor(
            params[:, _FILTER_MODEL_INDEX]
        ).clip(0, 5).astype(np.float32)

        # filter_1_style → integers {0, 1, 2, 3}
        params[:, _FILTER_STYLE_INDEX] = np.floor(
            params[:, _FILTER_STYLE_INDEX]
        ).clip(0, 3).astype(np.float32)
