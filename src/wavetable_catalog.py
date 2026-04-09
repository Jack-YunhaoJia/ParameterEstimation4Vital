"""
波表目录管理模块。

维护波表 ID 到波表元信息的映射表，支持振荡器波表绑定解析、
来源分类（factory / third_party / embedded）和 JSON 持久化。

分类逻辑基于文件系统扫描：
- factory: 名称匹配 Factory/Wavetables/ 下的 .vitaltable 或 .wav 文件，
  或 "Init"（Vital 内置默认波表）
- third_party: 名称匹配其他 pack 目录下的 .vitaltable 或 .wav 文件
- embedded: 名称不匹配任何磁盘文件（波表数据内嵌在 preset JSON 中）

Requirements: 10.1, 10.2, 10.3, 10.4
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.preset_parser import VitalPreset

logger = logging.getLogger(__name__)

# "Init" is Vital's built-in default wavetable (basic sine), always factory
BUILTIN_FACTORY_NAMES: frozenset[str] = frozenset({"Init"})

# Legacy alias for backward compatibility with existing tests
FACTORY_WAVETABLE_NAMES: frozenset[str] = BUILTIN_FACTORY_NAMES

_OSC_KEYS = ("osc_1", "osc_2", "osc_3")
_WAVETABLE_EXTENSIONS = (".vitaltable", ".wav")


@dataclass
class WavetableEntry:
    """波表目录条目。"""

    wavetable_id: int
    name: str
    source_type: str  # "factory" | "third_party" | "embedded"
    pack_name: str = ""  # pack 名称（third_party 时有值）
    file_path: str = ""  # 磁盘文件路径（factory/third_party 时有值）


class WavetableCatalog:
    """波表 ID 管理和来源分类。

    通过扫描 Vital 安装目录下的 Wavetables/ 子目录（.vitaltable 和 .wav），
    构建名称 → 来源类型的查找表。预设中的波表名称与磁盘文件名（stem）匹配。

    分类规则：
    1. "Init" → factory（Vital 内置默认）
    2. 名称匹配 Factory/Wavetables/ 下的文件 → factory
    3. 名称匹配其他 pack/Wavetables/ 下的文件 → third_party
    4. 无匹配 → embedded（波表数据内嵌在 preset JSON 中）
    """

    def __init__(
        self,
        vital_root: Path | None = None,
        catalog_path: Path | None = None,
    ) -> None:
        """初始化波表目录。

        Args:
            vital_root: Vital 安装根目录（如 /Users/jack/Music/Vital），
                        提供后会自动扫描所有 pack 的 Wavetables/ 目录。
            catalog_path: 可选的已有目录 JSON 路径，存在则自动加载
                         （优先于 vital_root 扫描）。
        """
        self._entries: dict[str, WavetableEntry] = {}
        self._next_id: int = 0
        # name -> (source_type, pack_name, file_path) 从文件系统扫描得到
        self._fs_lookup: dict[str, tuple[str, str, str]] = {}

        if catalog_path is not None:
            catalog_path = Path(catalog_path)
            if catalog_path.exists():
                self.load(catalog_path)
                return

        if vital_root is not None:
            self._scan_filesystem(Path(vital_root))

    def _scan_filesystem(self, vital_root: Path) -> None:
        """扫描 Vital 目录下所有 pack 的 Wavetables/ 子目录。"""
        if not vital_root.exists():
            logger.warning("Vital root directory does not exist: %s", vital_root)
            return

        count = 0
        for pack_dir in sorted(vital_root.iterdir()):
            if not pack_dir.is_dir():
                continue
            wt_dir = pack_dir / "Wavetables"
            if not wt_dir.exists():
                continue

            pack_name = pack_dir.name
            is_factory = pack_name == "Factory"

            for f in wt_dir.rglob("*"):
                if not f.is_file() or f.suffix not in _WAVETABLE_EXTENSIONS:
                    continue
                source_type = "factory" if is_factory else "third_party"
                self._fs_lookup[f.stem] = (source_type, pack_name, str(f))
                count += 1

        logger.info(
            "Scanned %d wavetable files from %s (%d packs)",
            count, vital_root, len({v[1] for v in self._fs_lookup.values()}),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_oscillator_wavetables(
        self, preset: VitalPreset
    ) -> dict[str, WavetableEntry | None]:
        """解析预设中每个振荡器的波表绑定。

        Vital 预设在 ``preset.extra["wavetables"]`` 中存储波表列表，
        索引 0/1/2 分别对应 osc_1/osc_2/osc_3。

        Returns:
            {"osc_1": entry_or_None, "osc_2": ..., "osc_3": ...}
        """
        wavetables: list[dict[str, Any]] = preset.extra.get("wavetables", [])
        result: dict[str, WavetableEntry | None] = {}

        for idx, osc_key in enumerate(_OSC_KEYS):
            if idx < len(wavetables):
                wt_data = wavetables[idx]
                if isinstance(wt_data, dict):
                    result[osc_key] = self._resolve_single(wt_data)
                else:
                    logger.warning(
                        "Wavetable data at index %d for %s is not a dict",
                        idx, osc_key,
                    )
                    result[osc_key] = None
            else:
                result[osc_key] = None

        return result

    def classify_source_type(self, wavetable_data: dict) -> str:
        """分类波表来源类型。

        Args:
            wavetable_data: 波表数据字典，需包含 "name" 字段。

        Returns:
            "factory" | "third_party" | "embedded"
        """
        name = wavetable_data.get("name")
        if not isinstance(name, str) or not name.strip():
            return "embedded"

        # 1. Built-in factory names
        if name in BUILTIN_FACTORY_NAMES:
            return "factory"

        # 2. Filesystem lookup
        if name in self._fs_lookup:
            return self._fs_lookup[name][0]

        # 3. No match → embedded
        return "embedded"

    def save(self, path: Path) -> None:
        """将目录持久化为 wavetable_catalog.json。"""
        path = Path(path)
        data = {
            "next_id": self._next_id,
            "entries": [asdict(e) for e in self._entries.values()],
            "fs_lookup": {
                name: {"source_type": st, "pack_name": pn, "file_path": fp}
                for name, (st, pn, fp) in self._fs_lookup.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved wavetable catalog: %d entries, %d fs lookups to %s",
            len(self._entries), len(self._fs_lookup), path,
        )

    def load(self, path: Path) -> None:
        """从 JSON 文件加载目录。"""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self._entries.clear()
        self._next_id = data.get("next_id", 0)

        for entry_dict in data.get("entries", []):
            entry = WavetableEntry(
                wavetable_id=entry_dict["wavetable_id"],
                name=entry_dict["name"],
                source_type=entry_dict["source_type"],
                pack_name=entry_dict.get("pack_name", ""),
                file_path=entry_dict.get("file_path", ""),
            )
            self._entries[entry.name] = entry
            if entry.wavetable_id >= self._next_id:
                self._next_id = entry.wavetable_id + 1

        # Restore fs_lookup
        self._fs_lookup.clear()
        for name, info in data.get("fs_lookup", {}).items():
            self._fs_lookup[name] = (
                info["source_type"], info["pack_name"], info["file_path"],
            )

        logger.info(
            "Loaded wavetable catalog: %d entries, %d fs lookups from %s",
            len(self._entries), len(self._fs_lookup), path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_single(self, wt_data: dict) -> WavetableEntry:
        """Resolve a single wavetable dict to a WavetableEntry."""
        name = wt_data.get("name")
        if not isinstance(name, str) or not name.strip():
            logger.warning("Wavetable has no valid name, marking as embedded")
            return self._get_or_create_entry("__unnamed__", "embedded", "", "")

        source_type = self.classify_source_type(wt_data)

        pack_name = ""
        file_path = ""
        if name in self._fs_lookup:
            _, pack_name, file_path = self._fs_lookup[name]

        return self._get_or_create_entry(name, source_type, pack_name, file_path)

    def _get_or_create_entry(
        self, name: str, source_type: str,
        pack_name: str = "", file_path: str = "",
    ) -> WavetableEntry:
        """Look up or create a WavetableEntry with auto-assigned ID."""
        if name in self._entries:
            return self._entries[name]

        entry = WavetableEntry(
            wavetable_id=self._next_id,
            name=name,
            source_type=source_type,
            pack_name=pack_name,
            file_path=file_path,
        )
        self._entries[name] = entry
        self._next_id += 1
        return entry
