"""阶段级检查点管理器。

负责在流水线各阶段之间保存和恢复中间状态，支持原子写入、
向后兼容旧版 checkpoint.json、以及可选的检查点清理策略。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 阶段执行顺序
PHASE_ORDER: list[str] = [
    "sampling",
    "rendering",
    "preprocessing",
    "embedding",
    "validation",
    "saving",
    "analysis",
]

# 使用 NPZ 格式的阶段
_NPZ_PHASES: set[str] = {"sampling", "embedding"}


class CheckpointManager:
    """阶段级检查点管理器。"""

    def __init__(
        self,
        output_dir: str | Path,
        keep_checkpoints: bool = True,
    ) -> None:
        """
        Args:
            output_dir: 实验输出目录
            keep_checkpoints: 是否保留历史检查点文件
        """
        self.output_dir = Path(output_dir)
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.master_path = self.checkpoint_dir / "checkpoint_master.json"

        self._master = self._load_or_init_master()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_or_init_master(self) -> dict[str, Any]:
        """Load existing master checkpoint or create a new one.

        Also detects legacy checkpoint.json (version 2) and migrates it
        by marking the 'rendering' phase as completed.
        """
        if self.master_path.exists():
            try:
                with open(self.master_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "checkpoint_master.json 读取失败，将重新初始化: %s", e
                )

        master = self._new_master()

        # 向后兼容：检测旧版 checkpoint.json (version 2)
        legacy_path = self.output_dir / "checkpoint.json"
        if legacy_path.exists():
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    legacy = json.load(f)
                if legacy.get("version") == 2:
                    now = datetime.now(timezone.utc).isoformat()
                    master["phases"]["rendering"] = {
                        "status": "completed",
                        "completed_at": now,
                        "checkpoint_file": str(legacy_path),
                        "input_hash": None,
                        "output_count": len(legacy.get("samples", [])),
                        "file_size_bytes": legacy_path.stat().st_size,
                    }
                    logger.info(
                        "检测到旧版 checkpoint.json (version 2)，"
                        "已自动标记 rendering 阶段为已完成"
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("旧版 checkpoint.json 读取失败: %s", e)

        self._save_master(master)
        return master

    @staticmethod
    def _new_master() -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "version": 3,
            "created_at": now,
            "updated_at": now,
            "phases": {},
        }

    def _save_master(self, master: dict[str, Any] | None = None) -> None:
        if master is None:
            master = self._master
        master["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.master_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.master_path)

    def _phase_file(self, phase_name: str) -> Path:
        ext = ".npz" if phase_name in _NPZ_PHASES else ".json"
        return self.checkpoint_dir / f"{phase_name}{ext}"

    def _output_count(self, data: dict[str, Any]) -> int:
        """Estimate the number of output items from phase data."""
        for v in data.values():
            if isinstance(v, np.ndarray):
                return len(v)
        if isinstance(data, dict):
            return len(data)
        return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete_phase(
        self,
        phase_name: str,
        data: dict[str, Any],
        input_hash: str | None = None,
    ) -> Path:
        """标记阶段完成并持久化输出数据。

        使用原子写入（先写 .tmp 再 os.replace()）。
        写入完成后更新 checkpoint_master.json。
        如果 keep_checkpoints=False，删除前一阶段的检查点文件
        （embedding .npz 除外）。

        Args:
            phase_name: 阶段名称（必须在 PHASE_ORDER 中）
            data: 阶段输出数据
            input_hash: 输入数据的哈希摘要（可选）

        Returns:
            检查点文件路径
        """
        if phase_name not in PHASE_ORDER:
            raise ValueError(
                f"未知阶段名称: {phase_name!r}，"
                f"合法值: {PHASE_ORDER}"
            )

        dest = self._phase_file(phase_name)
        tmp = dest.with_suffix(dest.suffix + ".tmp")

        # 原子写入阶段检查点
        if phase_name in _NPZ_PHASES:
            np.savez(tmp, **data)
            # np.savez may append .npz if suffix isn't already .npz
            actual_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npz")
            if actual_tmp != tmp and actual_tmp.exists():
                os.replace(actual_tmp, dest)
            else:
                os.replace(tmp, dest)
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, dest)

        file_size = dest.stat().st_size
        now = datetime.now(timezone.utc).isoformat()

        self._master["phases"][phase_name] = {
            "status": "completed",
            "completed_at": now,
            "checkpoint_file": str(dest),
            "input_hash": input_hash,
            "output_count": self._output_count(data),
            "file_size_bytes": file_size,
        }
        self._save_master()

        logger.info(
            "阶段 %s 检查点已保存: %s (%d bytes)",
            phase_name,
            dest,
            file_size,
        )

        # 清理前一阶段检查点（如果配置为不保留）
        if not self.keep_checkpoints:
            idx = PHASE_ORDER.index(phase_name)
            if idx > 0:
                prev_phase = PHASE_ORDER[idx - 1]
                prev_file = self._phase_file(prev_phase)
                # embedding .npz 文件始终保留
                if prev_file.exists() and prev_file.suffix != ".npz":
                    prev_file.unlink()
                    logger.info(
                        "已删除前一阶段检查点: %s", prev_file
                    )

        return dest

    def load_phase_data(self, phase_name: str) -> dict[str, Any]:
        """加载指定阶段的检查点数据。

        Args:
            phase_name: 阶段名称

        Returns:
            阶段输出数据

        Raises:
            FileNotFoundError: 检查点文件不存在
            ValueError: 检查点文件损坏
        """
        if phase_name not in PHASE_ORDER:
            raise ValueError(
                f"未知阶段名称: {phase_name!r}，"
                f"合法值: {PHASE_ORDER}"
            )

        dest = self._phase_file(phase_name)
        if not dest.exists():
            raise FileNotFoundError(
                f"阶段 {phase_name!r} 的检查点文件不存在: {dest}"
            )

        try:
            if phase_name in _NPZ_PHASES:
                npz = np.load(dest, allow_pickle=False)
                return dict(npz)
            else:
                with open(dest, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            raise ValueError(
                f"阶段 {phase_name!r} 的检查点文件损坏: {dest}, 错误: {e}"
            ) from e

    def get_completed_phases(self) -> list[str]:
        """返回已完成的阶段列表（按执行顺序）。"""
        completed = []
        for phase in PHASE_ORDER:
            info = self._master.get("phases", {}).get(phase, {})
            if info.get("status") == "completed":
                completed.append(phase)
        return completed

    def get_resume_phase(self) -> str | None:
        """自动检测下一个需要执行的阶段。

        Returns:
            下一个未完成阶段的名称，全部完成时返回 None
        """
        completed = set(self.get_completed_phases())
        for phase in PHASE_ORDER:
            if phase not in completed:
                return phase
        return None

    def validate_phase_checkpoint(self, phase_name: str) -> bool:
        """验证指定阶段的检查点文件是否存在且可读。"""
        if phase_name not in PHASE_ORDER:
            return False
        dest = self._phase_file(phase_name)
        if not dest.exists():
            return False
        try:
            if phase_name in _NPZ_PHASES:
                npz = np.load(dest, allow_pickle=False)
                list(npz.keys())
            else:
                with open(dest, "r", encoding="utf-8") as f:
                    json.load(f)
            return True
        except Exception:
            return False

    def get_checkpoint_size(self, phase_name: str) -> int:
        """返回指定阶段检查点文件的字节大小。

        Raises:
            FileNotFoundError: 检查点文件不存在
        """
        dest = self._phase_file(phase_name)
        if not dest.exists():
            raise FileNotFoundError(
                f"阶段 {phase_name!r} 的检查点文件不存在: {dest}"
            )
        return dest.stat().st_size

    def validate_resume_from(self, phase_name: str) -> None:
        """验证指定阶段的所有前置检查点是否存在且完整。

        检查 PHASE_ORDER 中 phase_name 之前的所有阶段的检查点文件
        是否存在且可读。如果任何前置检查点缺失或损坏，抛出
        FileNotFoundError 并附带缺失文件路径和需要重新执行的阶段名称。

        Args:
            phase_name: 要恢复执行的阶段名称

        Raises:
            ValueError: phase_name 不在 PHASE_ORDER 中
            FileNotFoundError: 前置检查点文件缺失或损坏
        """
        if phase_name not in PHASE_ORDER:
            raise ValueError(
                f"未知阶段名称: {phase_name!r}，"
                f"合法值: {PHASE_ORDER}"
            )

        idx = PHASE_ORDER.index(phase_name)
        for i in range(idx):
            prereq = PHASE_ORDER[i]
            if not self.validate_phase_checkpoint(prereq):
                missing_file = self._phase_file(prereq)
                raise FileNotFoundError(
                    f"前置阶段 {prereq!r} 的检查点文件缺失或损坏: "
                    f"{missing_file}，请从阶段 {prereq!r} 重新执行"
                )

    def get_phases_to_execute(self, resume_from: str | None) -> list[str]:
        """返回从指定阶段开始需要执行的阶段列表。

        Args:
            resume_from: 恢复执行的起始阶段名称，为 None 时返回全部阶段

        Returns:
            需要执行的阶段名称列表（按执行顺序）

        Raises:
            ValueError: resume_from 不在 PHASE_ORDER 中
        """
        if resume_from is None:
            return list(PHASE_ORDER)

        if resume_from not in PHASE_ORDER:
            raise ValueError(
                f"未知阶段名称: {resume_from!r}，"
                f"合法值: {PHASE_ORDER}"
            )

        idx = PHASE_ORDER.index(resume_from)
        return list(PHASE_ORDER[idx:])

    # ------------------------------------------------------------------
    # Embedding persistence (incremental batch save / finalize / load)
    # ------------------------------------------------------------------

    def save_embedding_batch(
        self,
        sample_ids: list[str],
        embeddings: np.ndarray,
    ) -> None:
        """将一个批次的 Embedding 追加保存到 embedding_partial.npz。

        如果 partial 文件已存在，先加载已有数据，拼接后重新保存。

        Args:
            sample_ids: 本批次的样本 ID 列表，长度为 M
            embeddings: 本批次的 Embedding 矩阵，形状 (M, D)，float32
        """
        partial_path = self.checkpoint_dir / "embedding_partial.npz"

        new_ids = np.array(sample_ids, dtype=str)
        new_emb = np.asarray(embeddings, dtype=np.float32)

        if partial_path.exists():
            try:
                existing = np.load(partial_path, allow_pickle=False)
                old_ids = existing["sample_ids"]
                old_emb = existing["embeddings"]
                merged_ids = np.concatenate([old_ids, new_ids])
                merged_emb = np.concatenate([old_emb, new_emb], axis=0)
            except Exception:
                # Partial file corrupted — start fresh with this batch
                logger.warning(
                    "embedding_partial.npz 加载失败，将从当前批次重新开始"
                )
                merged_ids = new_ids
                merged_emb = new_emb
        else:
            merged_ids = new_ids
            merged_emb = new_emb

        # Atomic write: tmp then replace
        tmp_path = partial_path.with_suffix(".npz.tmp")
        np.savez(tmp_path, sample_ids=merged_ids, embeddings=merged_emb)
        # np.savez may append .npz to the path
        actual_tmp = tmp_path if tmp_path.exists() else Path(str(tmp_path) + ".npz")
        if actual_tmp.exists():
            os.replace(actual_tmp, partial_path)
        # Clean up any leftover tmp variant
        if tmp_path.exists() and tmp_path != partial_path:
            tmp_path.unlink(missing_ok=True)

        logger.info(
            "Embedding 批次已保存: %d 条新增, 累计 %d 条",
            len(new_ids),
            len(merged_ids),
        )

    def finalize_embeddings(self) -> None:
        """将 embedding_partial.npz 合并为最终的 embedding.npz。

        将 partial 文件重命名为最终文件路径。
        """
        partial_path = self.checkpoint_dir / "embedding_partial.npz"
        final_path = self.checkpoint_dir / "embedding.npz"

        if not partial_path.exists():
            raise FileNotFoundError(
                f"embedding_partial.npz 不存在: {partial_path}"
            )

        os.replace(partial_path, final_path)
        logger.info("Embedding 已合并为最终文件: %s", final_path)

    def load_partial_embeddings(self) -> tuple[list[str], np.ndarray]:
        """加载已保存的部分 Embedding 数据。

        优先加载 embedding_partial.npz，如果不存在则尝试加载
        最终的 embedding.npz。

        Returns:
            (sample_ids, embeddings) 元组：
            - sample_ids: 样本 ID 列表
            - embeddings: Embedding 矩阵，形状 (N, D)，float32
        """
        partial_path = self.checkpoint_dir / "embedding_partial.npz"
        final_path = self.checkpoint_dir / "embedding.npz"

        target = partial_path if partial_path.exists() else final_path

        if not target.exists():
            return [], np.empty((0, 0), dtype=np.float32)

        try:
            data = np.load(target, allow_pickle=False)
            sample_ids = list(data["sample_ids"].astype(str))
            embeddings = data["embeddings"].astype(np.float32)
            return sample_ids, embeddings
        except Exception:
            logger.warning("Embedding 文件加载失败: %s", target)
            return [], np.empty((0, 0), dtype=np.float32)

    def get_missing_sample_ids(self, expected_ids: list[str]) -> list[str]:
        """计算缺失的 sample_id 集合。

        返回 expected_ids 中尚未保存到 partial/final embedding 文件的 ID。

        Args:
            expected_ids: 期望的完整 sample_id 列表

        Returns:
            缺失的 sample_id 列表（保持 expected_ids 中的顺序）
        """
        saved_ids, _ = self.load_partial_embeddings()
        saved_set = set(saved_ids)
        return [sid for sid in expected_ids if sid not in saved_set]


    # ------------------------------------------------------------------
    # Preprocessing stats persistence
    # ------------------------------------------------------------------

    def save_preprocessing_stats(self, stats: dict[str, dict]) -> Path:
        """以 JSON 格式保存预处理统计信息到 checkpoints/preprocessing.json。

        每个样本一条记录，使用 sample_id 作为键。
        记录字段：original_rms_db、clipping_ratio、is_filtered、
        filter_reason、output_sample_rate、resampled。

        使用原子写入（先写 .tmp 再 os.replace()）。

        Args:
            stats: sample_id → 预处理统计字典的映射

        Returns:
            写入的文件路径
        """
        dest = self.checkpoint_dir / "preprocessing.json"
        tmp = dest.with_suffix(".json.tmp")

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        os.replace(tmp, dest)

        logger.info(
            "预处理统计已保存: %d 条记录, %d bytes",
            len(stats),
            dest.stat().st_size,
        )
        return dest

    def load_preprocessing_stats(self) -> dict[str, dict]:
        """加载已保存的预处理统计信息。

        Returns:
            sample_id → 预处理统计字典的映射

        Raises:
            FileNotFoundError: 检查点文件不存在
            ValueError: 检查点文件损坏
        """
        dest = self.checkpoint_dir / "preprocessing.json"
        if not dest.exists():
            raise FileNotFoundError(
                f"预处理统计文件不存在: {dest}"
            )
        try:
            with open(dest, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(
                f"预处理统计文件损坏: {dest}, 错误: {e}"
            ) from e

