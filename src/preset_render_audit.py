"""
渲染保真审计模块。

批量执行预设渲染保真审计：对每个预设执行 introspection + 渲染 + RMS 计算，
判断 modulation_gap 和 wavetable_gap，汇总统计并给出 backend_recommendation。
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from src.audio_preprocessor import AudioPreprocessor
from src.preset_introspector import IntrospectionReport, PresetIntrospector

logger = logging.getLogger(__name__)


@runtime_checkable
class RendererBackendProtocol(Protocol):
    """渲染后端协议。

    任何具有 render_single 方法的对象均可作为渲染后端。
    """

    def render_single(self, preset_path: Path, output_path: Path) -> "RenderResultLike":
        """渲染单个预设为 WAV 文件。"""
        ...


@runtime_checkable
class RenderResultLike(Protocol):
    """渲染结果协议。"""

    @property
    def success(self) -> bool: ...

    @property
    def audio(self) -> np.ndarray | None: ...

    @property
    def error(self) -> str | None: ...


@dataclass
class PresetAuditDetail:
    """单个预设的审计详情。"""

    preset_path: str
    render_success: bool
    render_error: str | None
    rms_db: float | None
    has_modulation_gap: bool
    has_wavetable_gap: bool
    introspection: IntrospectionReport | None


@dataclass
class RenderAuditConfig:
    """渲染审计配置。"""

    modulation_gap_threshold: float = 0.3
    wavetable_gap_threshold: float = 0.3
    silence_threshold_db: float = -60.0
    override_not_ready: bool = False


@dataclass
class RenderAuditReport:
    """批量渲染审计报告。"""

    total_presets: int
    render_success_count: int
    render_failure_count: int
    presets_with_active_modulations: int
    presets_with_wavetables: int
    modulation_gap_count: int
    wavetable_gap_count: int
    modulation_gap_rate: float
    wavetable_gap_rate: float
    mean_rms_db: float
    details: list[PresetAuditDetail] = field(default_factory=list)
    backend_recommendation: str = "current_backend_acceptable"


class PresetRenderAudit:
    """批量渲染保真审计。

    对每个预设执行 introspection + 渲染 + RMS 计算，
    判断 modulation_gap 和 wavetable_gap，汇总统计并给出 backend_recommendation。
    """

    def __init__(
        self,
        introspector: PresetIntrospector,
        renderer_backend: RendererBackendProtocol,
        config: RenderAuditConfig | None = None,
    ) -> None:
        self._introspector = introspector
        self._renderer = renderer_backend
        self._config = config or RenderAuditConfig()
        self._parser = introspector._parser

    def audit_batch(self, preset_paths: list[Path]) -> RenderAuditReport:
        """批量审计预设渲染保真度。

        对每个预设：
        1. 解析并执行 introspection
        2. 渲染为 WAV（失败时记录错误并继续）
        3. 计算 RMS energy
        4. 判断 modulation_gap 和 wavetable_gap
        5. 汇总统计，给出 backend_recommendation

        Args:
            preset_paths: 预设文件路径列表

        Returns:
            RenderAuditReport 汇总报告
        """
        details: list[PresetAuditDetail] = []
        rms_values: list[float] = []
        render_success_count = 0
        render_failure_count = 0
        presets_with_active_modulations = 0
        presets_with_wavetables = 0
        modulation_gap_count = 0
        wavetable_gap_count = 0

        with tempfile.TemporaryDirectory(prefix="render_audit_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            for preset_path in preset_paths:
                detail = self._audit_single(preset_path, tmp_path)
                details.append(detail)

                if detail.render_success:
                    render_success_count += 1
                    if detail.rms_db is not None:
                        rms_values.append(detail.rms_db)
                else:
                    render_failure_count += 1

                if detail.introspection is not None:
                    if detail.introspection.active_modulation_count > 0:
                        presets_with_active_modulations += 1
                    if detail.introspection.wavetable_count > 0:
                        presets_with_wavetables += 1

                if detail.has_modulation_gap:
                    modulation_gap_count += 1
                if detail.has_wavetable_gap:
                    wavetable_gap_count += 1

        # Calculate gap rates
        modulation_gap_rate = (
            modulation_gap_count / presets_with_active_modulations
            if presets_with_active_modulations > 0
            else 0.0
        )
        wavetable_gap_rate = (
            wavetable_gap_count / presets_with_wavetables
            if presets_with_wavetables > 0
            else 0.0
        )

        # Calculate mean RMS
        mean_rms_db = float(np.mean(rms_values)) if rms_values else float("-inf")

        # Determine backend recommendation
        backend_recommendation = self._compute_recommendation(
            modulation_gap_rate, wavetable_gap_rate
        )

        total_presets = len(preset_paths)

        report = RenderAuditReport(
            total_presets=total_presets,
            render_success_count=render_success_count,
            render_failure_count=render_failure_count,
            presets_with_active_modulations=presets_with_active_modulations,
            presets_with_wavetables=presets_with_wavetables,
            modulation_gap_count=modulation_gap_count,
            wavetable_gap_count=wavetable_gap_count,
            modulation_gap_rate=modulation_gap_rate,
            wavetable_gap_rate=wavetable_gap_rate,
            mean_rms_db=mean_rms_db,
            details=details,
            backend_recommendation=backend_recommendation,
        )

        logger.info(
            "Render audit complete: %d total, %d success, %d failed, "
            "mod_gap_rate=%.2f, wt_gap_rate=%.2f, recommendation=%s",
            total_presets,
            render_success_count,
            render_failure_count,
            modulation_gap_rate,
            wavetable_gap_rate,
            backend_recommendation,
        )

        return report

    def _audit_single(
        self, preset_path: Path, tmp_dir: Path
    ) -> PresetAuditDetail:
        """审计单个预设。"""
        introspection: IntrospectionReport | None = None
        render_success = False
        render_error: str | None = None
        rms_db: float | None = None
        has_modulation_gap = False
        has_wavetable_gap = False

        # Step 1: Parse and introspect
        try:
            preset = self._parser.parse(preset_path)
            introspection = self._introspector.introspect(
                preset, preset_path=str(preset_path)
            )
        except Exception as exc:
            logger.error(
                "Failed to parse/introspect '%s': %s", preset_path, exc
            )
            return PresetAuditDetail(
                preset_path=str(preset_path),
                render_success=False,
                render_error=f"Parse/introspect error: {exc}",
                rms_db=None,
                has_modulation_gap=False,
                has_wavetable_gap=False,
                introspection=None,
            )

        # Step 2: Determine gaps based on introspection
        # modulation_gap: preset has active modulations but renderer always
        # skips modulations in the current backend
        if introspection.active_modulation_count > 0:
            has_modulation_gap = True

        # wavetable_gap: preset has wavetables but renderer always skips
        # wavetables in the current backend
        if introspection.wavetable_count > 0:
            has_wavetable_gap = True

        # Step 3: Render
        try:
            output_wav = tmp_dir / f"{preset_path.stem}_audit.wav"
            result = self._renderer.render_single(preset_path, output_wav)

            if result.success and result.audio is not None:
                render_success = True
                rms_db = AudioPreprocessor.compute_rms_db(result.audio)
            elif result.success:
                # Success but no audio data — try reading from file
                render_success = True
                render_error = None
            else:
                render_success = False
                render_error = result.error
        except Exception as exc:
            logger.error("Render failed for '%s': %s", preset_path, exc)
            render_success = False
            render_error = str(exc)

        return PresetAuditDetail(
            preset_path=str(preset_path),
            render_success=render_success,
            render_error=render_error,
            rms_db=rms_db,
            has_modulation_gap=has_modulation_gap,
            has_wavetable_gap=has_wavetable_gap,
            introspection=introspection,
        )

    def _compute_recommendation(
        self, modulation_gap_rate: float, wavetable_gap_rate: float
    ) -> str:
        """根据 gap rate 和配置计算 backend recommendation。

        如果 override_not_ready=True，始终返回 "current_backend_acceptable"。
        否则，如果任一 gap_rate 超过阈值，返回 "backend_replacement_recommended"。
        """
        config = self._config

        if config.override_not_ready:
            return "current_backend_acceptable"

        if modulation_gap_rate > config.modulation_gap_threshold:
            return "backend_replacement_recommended"

        if wavetable_gap_rate > config.wavetable_gap_threshold:
            return "backend_replacement_recommended"

        return "current_backend_acceptable"
