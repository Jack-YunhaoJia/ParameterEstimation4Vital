#!/usr/bin/env python3
"""
为当前 pilot 语料生成固定 manifest。

使用 PresetCorpusScanner.scan() 扫描 presets/ 目录，
将 9 个真实 preset 路径写入 pilot_manifest.json，
作为 Fidelity Spike、pilot producer run、resume drill 的统一输入。

Usage:
    python3 scripts/generate_pilot_manifest.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preset_corpus_scanner import PresetCorpusScanner


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    presets_dir = _PROJECT_ROOT / "presets"
    output_path = _PROJECT_ROOT / "pilot_manifest.json"

    logger.info("Scanning presets directory: %s", presets_dir)

    scanner = PresetCorpusScanner()
    result = scanner.scan(presets_dir, recursive=False)

    manifest = {
        "scan_directory": str(presets_dir),
        "total_found": result.total_found,
        "preset_paths": [str(p) for p in result.preset_paths],
    }

    if result.failed_paths:
        manifest["failed_paths"] = [str(p) for p in result.failed_paths]
        manifest["failed_errors"] = result.failed_errors

    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote pilot_manifest.json: %d valid presets, %d failed",
        len(result.preset_paths),
        len(result.failed_paths),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
