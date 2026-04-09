#!/usr/bin/env python3
"""
Fidelity Spike 脚本 — 对 9 个真实 preset 执行渲染保真审计。

加载 pilot_manifest.json，使用 PresetRenderAudit 对所有 preset 执行
introspection + 渲染 + RMS 计算，输出审计报告和 backend recommendation。

需要环境变量：
  - VITAL_VST_PATH: Vital VST3 插件路径

Usage:
    python scripts/run_fidelity_spike.py
    python scripts/run_fidelity_spike.py --manifest pilot_manifest.json
    python scripts/run_fidelity_spike.py --override-not-ready
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preset_introspector import PresetIntrospector
from src.preset_parser import PresetParser
from src.preset_render_audit import (
    PresetRenderAudit,
    RenderAuditConfig,
)
from src.renderer_backend import RendererBackend

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for the fidelity spike."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fidelity Spike：对真实 preset 执行渲染保真审计。",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="pilot_manifest.json",
        help="pilot manifest JSON 文件路径（默认: pilot_manifest.json）",
    )
    parser.add_argument(
        "--override-not-ready",
        action="store_true",
        help="强制覆盖 not-ready 判定，即使 gap rate 超过阈值也标记为 acceptable",
    )
    parser.add_argument(
        "--mod-gap-threshold",
        type=float,
        default=0.3,
        help="modulation gap rate 阈值（默认: 0.3）",
    )
    parser.add_argument(
        "--wt-gap-threshold",
        type=float,
        default=0.3,
        help="wavetable gap rate 阈值（默认: 0.3）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the fidelity spike audit."""
    setup_logging()
    args = parse_args(argv)

    # --- Require VITAL_VST_PATH ---
    vital_vst_path = os.environ.get("VITAL_VST_PATH")
    if not vital_vst_path:
        logger.error(
            "环境变量 VITAL_VST_PATH 未设置。"
            "请设置为 Vital VST3 插件路径，例如：\n"
            "  export VITAL_VST_PATH=/Library/Audio/Plug-Ins/VST3/Vital.vst3"
        )
        return 1

    vst_path = Path(vital_vst_path)
    if not vst_path.exists():
        logger.error("VITAL_VST_PATH 指向的路径不存在: %s", vst_path)
        return 1

    # --- Load manifest ---
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest 文件不存在: %s", manifest_path)
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    preset_paths = [Path(p) for p in manifest["preset_paths"]]
    logger.info("Loaded manifest: %d presets from %s", len(preset_paths), manifest_path)

    # Verify all preset files exist
    missing = [p for p in preset_paths if not p.exists()]
    if missing:
        for p in missing:
            logger.warning("Preset 文件不存在: %s", p)
        logger.error("%d preset 文件缺失，中止审计", len(missing))
        return 1

    # --- Build audit components ---
    parser = PresetParser()
    introspector = PresetIntrospector(parser)
    backend = RendererBackend(vst_path)

    audit_config = RenderAuditConfig(
        modulation_gap_threshold=args.mod_gap_threshold,
        wavetable_gap_threshold=args.wt_gap_threshold,
        override_not_ready=args.override_not_ready,
    )

    auditor = PresetRenderAudit(introspector, backend, audit_config)

    # --- Run audit ---
    logger.info("=" * 60)
    logger.info("Fidelity Spike: Auditing %d presets", len(preset_paths))
    logger.info("=" * 60)

    report = auditor.audit_batch(preset_paths)

    # --- Print report ---
    print("\n" + "=" * 60)
    print("Fidelity Spike Audit Report")
    print("=" * 60)
    print(f"  Total presets:               {report.total_presets}")
    print(f"  Render success:              {report.render_success_count}")
    print(f"  Render failure:              {report.render_failure_count}")
    print(f"  With active modulations:     {report.presets_with_active_modulations}")
    print(f"  With wavetables:             {report.presets_with_wavetables}")
    print(f"  Modulation gap count:        {report.modulation_gap_count}")
    print(f"  Wavetable gap count:         {report.wavetable_gap_count}")
    print(f"  Modulation gap rate:         {report.modulation_gap_rate:.2%}")
    print(f"  Wavetable gap rate:          {report.wavetable_gap_rate:.2%}")
    print(f"  Mean RMS (dB):               {report.mean_rms_db:.1f}")
    print(f"  Backend recommendation:      {report.backend_recommendation}")
    print("=" * 60)

    # --- Per-preset details ---
    print("\nPer-Preset Details:")
    print("-" * 60)
    for detail in report.details:
        status = "✓" if detail.render_success else "✗"
        rms_str = f"{detail.rms_db:.1f} dB" if detail.rms_db is not None else "N/A"
        mod_gap = "⚠ mod_gap" if detail.has_modulation_gap else ""
        wt_gap = "⚠ wt_gap" if detail.has_wavetable_gap else ""
        gaps = " ".join(filter(None, [mod_gap, wt_gap]))

        print(f"  [{status}] {Path(detail.preset_path).name}")
        print(f"      RMS: {rms_str}  {gaps}")
        if detail.introspection:
            intr = detail.introspection
            print(
                f"      applied={intr.applied_count} "
                f"skipped={intr.skipped_count} "
                f"unsupported={intr.unsupported_count} "
                f"mods={intr.active_modulation_count} "
                f"wt={intr.wavetable_count}"
            )
        if detail.render_error:
            print(f"      error: {detail.render_error}")
    print("-" * 60)

    # --- Gate decision ---
    if report.backend_recommendation == "backend_replacement_recommended":
        print("\n⚠  GATE: Backend replacement recommended.")
        print("   Use --override-not-ready to force continue.")
        return 2
    else:
        print("\n✓  GATE: Current backend acceptable for pilot production.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
