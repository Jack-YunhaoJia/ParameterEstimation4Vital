"""
参数回归模块。

MLP 回归模型：从 MuQ embedding (1024-dim) 预测约 45 个核心合成器参数。
包含训练循环、评估逻辑和预设导出功能。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.preset_parser import PresetParser, VitalPreset
from src.training_data import CORE_PARAMS, NUM_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """回归模型评估指标。

    Attributes:
        per_param_mae: 每个参数的平均绝对误差 (param_name -> MAE)
        overall_mae: 所有参数的平均 MAE
        spectral_loss: 多频谱损失（如果不可用则为 0.0）
    """

    per_param_mae: dict[str, float] = field(default_factory=dict)
    overall_mae: float = 0.0
    spectral_loss: float = 0.0


class ParameterRegressor(nn.Module):
    """MLP 回归模型：MuQ embedding → 45 个核心参数。

    Architecture:
        input(1024) → Linear(512) → ReLU → Dropout(0.3)
                     → Linear(256) → ReLU → Dropout(0.3)
                     → Linear(45) → Sigmoid

    Sigmoid ensures output values are constrained to [0, 1]
    (normalized parameter space).
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 45) -> None:
        """初始化 MLP 回归模型。

        Args:
            input_dim: MuQ embedding 维度 (default 1024)
            output_dim: 核心参数数量 (default 45)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """预测参数向量。

        Args:
            embedding: 输入 MuQ embedding, shape (batch_size, input_dim)

        Returns:
            预测的归一化参数向量, shape (batch_size, output_dim),
            所有值在 [0, 1] 范围内
        """
        return self.mlp(embedding)

    def export_preset(
        self,
        predicted_params: torch.Tensor,
        parser: PresetParser,
    ) -> VitalPreset:
        """将预测的归一化参数导出为 Vital 预设。

        将 [0, 1] 归一化参数反映射到各参数的原始值域 [min, max]，
        然后构建有效的 VitalPreset 对象。

        Args:
            predicted_params: 1-D tensor of shape (output_dim,),
                values in [0, 1] (normalized)
            parser: PresetParser 实例（用于格式兼容性）

        Returns:
            包含反归一化参数值的 VitalPreset 对象
        """
        params_np = predicted_params.detach().cpu().numpy()

        # Build settings from denormalized parameters
        settings: dict[str, Any] = {}
        for i, (name, lo, hi) in enumerate(CORE_PARAMS):
            if i < len(params_np):
                # Denormalize: value = lo + normalized * (hi - lo)
                normalized = float(params_np[i])
                denormalized = lo + normalized * (hi - lo)
                settings[name] = denormalized

        return VitalPreset(
            settings=settings,
            modulations=[],
            extra={
                "author": "ParameterRegressor",
                "comments": "Auto-generated from predicted parameters",
                "macro1": "",
                "macro2": "",
                "macro3": "",
                "macro4": "",
                "preset_name": "Predicted Preset",
                "preset_style": "",
            },
        )


def check_phase0_feasibility(report_path: Path) -> bool:
    """检查 Phase 0 可行性门控。

    读取 Phase 0 的 report.json，检查 is_feasible 字段。

    Args:
        report_path: Phase 0 报告文件路径 (report.json)

    Returns:
        True 如果 Phase 0 判定可行

    Raises:
        FeasibilityGateError: 如果报告不存在或判定不可行
    """
    if not report_path.exists():
        raise FeasibilityGateError(
            f"Phase 0 report not found: {report_path}. "
            "Run Phase 0 first before starting Phase 1."
        )

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise FeasibilityGateError(
            f"Cannot read Phase 0 report: {e}"
        )

    is_feasible = report.get("is_feasible", False)
    if not is_feasible:
        recommendation = report.get("recommendation", "No recommendation available.")
        raise FeasibilityGateError(
            "Phase 0 judged NOT feasible. Cannot start Phase 1 training. "
            f"Recommendation: {recommendation}"
        )

    return True


class FeasibilityGateError(Exception):
    """Phase 0 可行性门控错误。"""
    pass


def train_model(
    model: ParameterRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """训练回归模型。

    使用 MSE 损失和 Adam 优化器训练 MLP 模型。

    Args:
        model: ParameterRegressor 模型实例
        train_loader: 训练数据 DataLoader，每个 batch 为 (embeddings, params)
        val_loader: 验证数据 DataLoader（可选）
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备 ("cpu" 或 "cuda")

    Returns:
        训练历史字典，包含 train_loss 和可选的 val_loss 列表
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for embeddings, params in train_loader:
            embeddings = embeddings.to(device)
            params = params.to(device)

            optimizer.zero_grad()
            predictions = model(embeddings)
            loss = criterion(predictions, params)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for embeddings, params in val_loader:
                    embeddings = embeddings.to(device)
                    params = params.to(device)
                    predictions = model(embeddings)
                    loss = criterion(predictions, params)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            history["val_loss"].append(avg_val_loss)

            logger.info(
                "Epoch %d/%d - train_loss: %.6f, val_loss: %.6f",
                epoch + 1, epochs, avg_train_loss, avg_val_loss,
            )
        else:
            logger.info(
                "Epoch %d/%d - train_loss: %.6f",
                epoch + 1, epochs, avg_train_loss,
            )

    return history


def evaluate_model(
    model: ParameterRegressor,
    test_loader: DataLoader,
    param_names: list[str] | None = None,
    device: str = "cpu",
) -> RegressionMetrics:
    """评估回归模型。

    计算每个参数的 MAE 和整体 MAE。

    Args:
        model: 训练好的 ParameterRegressor 模型
        test_loader: 测试数据 DataLoader，每个 batch 为 (embeddings, params)
        param_names: 参数名称列表（用于 per_param_mae 的键）
        device: 计算设备

    Returns:
        RegressionMetrics 包含每个参数的 MAE 和整体 MAE
    """
    if param_names is None:
        param_names = [name for name, _, _ in CORE_PARAMS]

    model = model.to(device)
    model.eval()

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for embeddings, params in test_loader:
            embeddings = embeddings.to(device)
            params = params.to(device)
            predictions = model(embeddings)
            all_predictions.append(predictions.cpu())
            all_targets.append(params.cpu())

    if not all_predictions:
        return RegressionMetrics(
            per_param_mae={},
            overall_mae=0.0,
            spectral_loss=0.0,
        )

    predictions = torch.cat(all_predictions, dim=0)  # (N, 45)
    targets = torch.cat(all_targets, dim=0)  # (N, 45)

    # Per-parameter MAE
    abs_errors = torch.abs(predictions - targets)  # (N, 45)
    per_param_mae_values = abs_errors.mean(dim=0)  # (45,)

    per_param_mae: dict[str, float] = {}
    for i, name in enumerate(param_names):
        if i < per_param_mae_values.shape[0]:
            per_param_mae[name] = float(per_param_mae_values[i])

    overall_mae = float(per_param_mae_values.mean())

    return RegressionMetrics(
        per_param_mae=per_param_mae,
        overall_mae=overall_mae,
        spectral_loss=0.0,  # Spectral loss requires audio rendering
    )
