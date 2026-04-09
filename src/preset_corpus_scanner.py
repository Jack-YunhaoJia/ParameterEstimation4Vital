"""
预设语料扫描模块。

递归扫描指定目录下所有 .vital 文件，对每个文件尝试 JSON 解析验证，
返回有效路径列表和失败路径列表。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """扫描结果。

    Attributes:
        preset_paths: 有效 .vital 文件路径列表（JSON 解析成功）
        total_found: 发现的 .vital 文件总数（有效 + 失败）
        failed_paths: JSON 解析失败的文件路径
        failed_errors: 对应的错误信息
    """

    preset_paths: list[Path] = field(default_factory=list)
    total_found: int = 0
    failed_paths: list[Path] = field(default_factory=list)
    failed_errors: list[str] = field(default_factory=list)


class PresetCorpusScanner:
    """扫描指定目录下所有 .vital 文件并验证 JSON 可解析性。"""

    def scan(self, directory: Path, recursive: bool = True) -> ScanResult:
        """递归扫描目录下所有 .vital 文件。

        对每个 .vital 文件尝试 JSON 解析验证，
        失败的文件记录到 failed_paths 并跳过。
        扫描完成后 log 总数。

        Args:
            directory: 要扫描的目录路径
            recursive: 是否递归扫描子目录，默认 True

        Returns:
            ScanResult 包含有效路径、失败路径和总数
        """
        directory = Path(directory)

        if recursive:
            vital_files = sorted(directory.rglob("*.vital"))
        else:
            vital_files = sorted(directory.glob("*.vital"))

        if not vital_files:
            logger.warning("No .vital files found in directory: %s", directory)
            return ScanResult(
                preset_paths=[],
                total_found=0,
                failed_paths=[],
                failed_errors=[],
            )

        preset_paths: list[Path] = []
        failed_paths: list[Path] = []
        failed_errors: list[str] = []

        for filepath in vital_files:
            try:
                text = filepath.read_text(encoding="utf-8")
                json.loads(text)
                preset_paths.append(filepath)
            except (json.JSONDecodeError, OSError, IOError) as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(
                    "Failed to parse .vital file %s: %s", filepath, error_msg
                )
                failed_paths.append(filepath)
                failed_errors.append(error_msg)

        total_found = len(preset_paths) + len(failed_paths)
        logger.info(
            "Scan complete: %d .vital files found (%d valid, %d failed) in %s",
            total_found,
            len(preset_paths),
            len(failed_paths),
            directory,
        )

        return ScanResult(
            preset_paths=preset_paths,
            total_found=total_found,
            failed_paths=failed_paths,
            failed_errors=failed_errors,
        )
