"""
预设数据集生产模块。

顶层协调模块，串联 scan → parse → wavetable → filter → scope → schema →
split → 逐 preset 处理（RouteGraph → Augment → Write → Render → Preprocess →
Embed）→ flush shard → finalize HDF5 的完整流水线。

支持 pilot / canary / full 三种运行模式和 checkpoint/resume。

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.7
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from src.audio_preprocessor import AudioPreprocessor, PreprocessConfig
from src.mutated_preset_writer import MutatedPresetWriter
from src.preset_corpus_scanner import PresetCorpusScanner
from src.preset_parser import PresetParser, VitalPreset
from src.preset_schema_extractor import CorpusSchema, PresetSchemaExtractor
from src.renderer_backend import RendererBackend
from src.route_graph_builder import RouteGraphBuilder
from src.route_mask_augmenter import MaskedVariant, RouteMaskAugmenter
from src.wavetable_catalog import WavetableCatalog

logger = logging.getLogger(__name__)

RunMode = Literal["pilot", "canary", "full"]

EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProducerConfig:
    """生产配置。"""

    corpus_dir: Path
    output_dir: Path
    vital_vst_path: Path
    vital_root: Path | None = None  # Vital 安装根目录，用于波表文件系统扫描
    run_mode: RunMode = "pilot"
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    factory_only_mode: bool = False
    min_variants: int = 16
    max_variants: int = 64
    pilot_max_base_presets: int = 32
    canary_max_base_presets: int = 256
    pilot_max_variants_per_base: int = 8
    canary_max_variants_per_base: int = 24
    resume: bool = True
    checkpoint_every_n_base_presets: int = 8
    flush_every_n_samples: int = 256
    embedding_device: str = "mps"
    seed: int = 42


@dataclass
class ProductionSummary:
    """生产摘要。"""

    total_base_presets: int
    total_variants: int
    total_rendered_samples: int
    total_valid_samples: int
    filter_reasons: dict[str, int]
    split_counts: dict[str, int]
    wavetable_source_breakdown: dict[str, int]
    elapsed_time_sec: float


@dataclass
class ProductionCheckpoint:
    """长时间生产的可恢复状态。"""

    run_mode: RunMode
    selected_base_preset_ids: list[str]
    completed_base_preset_ids: list[str]
    split_assignment: dict[str, str]
    schema_path: str
    shard_manifest_path: str
    summary_so_far: dict
    last_completed_stage: str



# ---------------------------------------------------------------------------
# Helper: dense float32 parameter encoding
# ---------------------------------------------------------------------------

def encode_preset_params(
    preset: VitalPreset,
    schema: CorpusSchema,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a single preset's parameters into a dense float32 vector.

    Returns:
        (param_values, present_mask) each of shape (D,).
        - param_values: float32 vector; categorical fields encoded as category_id.
        - present_mask: uint8 vector; 1 if the param existed in the original preset.
    """
    D = len(schema.param_names)
    param_values = np.zeros(D, dtype=np.float32)
    present_mask = np.zeros(D, dtype=np.uint8)

    settings = preset.settings

    for j, name in enumerate(schema.param_names):
        if name in settings:
            raw = settings[name]
            ptype = schema.param_types[name]
            encoding = schema.param_value_encoding[name]

            if encoding == "category_id":
                # Categorical: map raw value string to category_id
                cats = schema.category_values.get(name, [])
                raw_str = str(raw)
                if raw_str in cats:
                    param_values[j] = float(cats.index(raw_str))
                else:
                    # Unknown category → use default
                    param_values[j] = schema.default_values[name]
            else:
                # identity: binary / continuous
                if isinstance(raw, (int, float)):
                    param_values[j] = float(raw)
                else:
                    param_values[j] = schema.default_values[name]

            present_mask[j] = 1
        else:
            # Missing → fill with default, mask = 0
            param_values[j] = schema.default_values[name]
            present_mask[j] = 0

    return param_values, present_mask


# ---------------------------------------------------------------------------
# Main producer class
# ---------------------------------------------------------------------------

class PresetDatasetProducer:
    """顶层协调模块，串联整条 preset-first 流水线。"""

    _SPLIT_NAMES = ("train", "val", "test")

    def __init__(self, config: ProducerConfig) -> None:
        self.config = config
        self._parser = PresetParser()
        self._scanner = PresetCorpusScanner()
        self._schema_extractor = PresetSchemaExtractor()
        self._graph_builder = RouteGraphBuilder()
        self._augmenter = RouteMaskAugmenter(
            self._graph_builder,
            min_variants=config.min_variants,
            max_variants=config.max_variants,
        )
        self._writer = MutatedPresetWriter(self._parser)
        self._preprocessor = AudioPreprocessor(PreprocessConfig())
        self._wt_catalog = WavetableCatalog(vital_root=config.vital_root)

        # Lazily initialised (require VST / GPU)
        self._backend: RendererBackend | None = None
        self._embedding_extractor: Any = None

        # Per-run state
        self._schema: CorpusSchema | None = None
        self._split_assignment: dict[str, str] = {}
        self._shard_buffers: dict[str, list[dict[str, Any]]] = {
            s: [] for s in self._SPLIT_NAMES
        }
        self._sample_counter: int = 0
        self._checkpoint: ProductionCheckpoint | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def produce(self) -> ProductionSummary:
        """Execute the full production pipeline."""
        t0 = time.time()
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Scan ---
        scan_result = self._scanner.scan(cfg.corpus_dir)
        logger.info("Scanned %d valid presets", len(scan_result.preset_paths))

        # --- 2. Parse ---
        presets: list[tuple[str, VitalPreset]] = []
        for p in scan_result.preset_paths:
            try:
                preset = self._parser.parse(p)
                base_id = p.stem
                presets.append((base_id, preset))
            except Exception as e:
                logger.error("Failed to parse %s: %s", p, e)

        # --- 3. Wavetable catalog — resolve and inject into settings ---
        wt_breakdown: dict[str, int] = {"factory": 0, "third_party": 0, "embedded": 0}
        for base_id, preset in presets:
            wt_map = self._wt_catalog.resolve_oscillator_wavetables(preset)
            for osc_key, entry in wt_map.items():
                if entry is not None:
                    wt_breakdown[entry.source_type] = (
                        wt_breakdown.get(entry.source_type, 0) + 1
                    )
                    # Inject wavetable fields into settings so schema extraction
                    # and encode_preset_params can see them
                    i = osc_key[-1]  # "osc_1" -> "1"
                    preset.settings[f"osc_{i}_wavetable_id"] = entry.name
                    preset.settings[f"osc_{i}_wavetable_source_type"] = entry.source_type

        # --- 4. Factory-only filter ---
        filter_reasons: dict[str, int] = {}
        if cfg.factory_only_mode:
            filtered: list[tuple[str, VitalPreset]] = []
            for base_id, preset in presets:
                wt_map = self._wt_catalog.resolve_oscillator_wavetables(preset)
                has_non_factory = any(
                    e is not None and e.source_type != "factory"
                    for e in wt_map.values()
                )
                if has_non_factory:
                    filter_reasons["non_factory_wavetable"] = (
                        filter_reasons.get("non_factory_wavetable", 0) + 1
                    )
                else:
                    filtered.append((base_id, preset))
            presets = filtered

        # --- 5. Select run scope ---
        presets = self._select_run_scope(presets)
        base_ids = [bid for bid, _ in presets]
        preset_map = {bid: p for bid, p in presets}
        logger.info(
            "Run scope: %d base presets (mode=%s)", len(base_ids), cfg.run_mode
        )

        # --- 6. Schema ---
        self._schema = self._schema_extractor.extract(
            [p for _, p in presets]
        )
        schema_path = cfg.output_dir / "corpus_schema.json"
        self._schema_extractor.save_schema(self._schema, schema_path)

        # --- 7. Split ---
        self._split_assignment = self._split_by_base_preset(base_ids)
        # Flatten to base_id -> split_name lookup
        id_to_split: dict[str, str] = {}
        for split_name, ids in self._split_assignment.items():
            for bid in ids:
                id_to_split[bid] = split_name

        # --- 8. Checkpoint ---
        self._checkpoint = self._load_or_init_checkpoint(
            base_ids, id_to_split, str(schema_path)
        )
        completed = set(self._checkpoint.completed_base_preset_ids)

        # --- 9. Lazy init heavy resources ---
        self._init_backend()
        self._init_embedding_extractor()

        # --- 10. Per-base-preset processing ---
        total_variants = 0
        total_rendered = 0
        total_valid = 0

        for idx, base_id in enumerate(base_ids):
            if base_id in completed:
                logger.info("Skipping completed base preset: %s", base_id)
                continue

            preset = preset_map[base_id]
            split_name = id_to_split[base_id]

            n_variants, n_rendered, n_valid = self._process_base_preset(
                base_id, preset, split_name
            )
            total_variants += n_variants
            total_rendered += n_rendered
            total_valid += n_valid

            # Mark completed
            self._checkpoint.completed_base_preset_ids.append(base_id)

            # Periodic checkpoint
            n_done = len(self._checkpoint.completed_base_preset_ids)
            if n_done % cfg.checkpoint_every_n_base_presets == 0:
                self._save_checkpoint()

        # Final flush
        self._flush_split_shards()
        self._save_checkpoint()

        # --- 11. Finalize HDF5 ---
        self._finalize_hdf5()

        # --- 12. Summary ---
        elapsed = time.time() - t0
        split_counts = {
            s: len(self._split_assignment.get(s, [])) for s in self._SPLIT_NAMES
        }
        summary = ProductionSummary(
            total_base_presets=len(base_ids),
            total_variants=total_variants,
            total_rendered_samples=total_rendered,
            total_valid_samples=total_valid,
            filter_reasons=filter_reasons,
            split_counts=split_counts,
            wavetable_source_breakdown=wt_breakdown,
            elapsed_time_sec=elapsed,
        )

        summary_path = cfg.output_dir / "production_summary.json"
        summary_path.write_text(
            json.dumps(asdict(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Production complete in %.1fs", elapsed)
        return summary

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _split_by_base_preset(
        self, base_preset_ids: list[str]
    ) -> dict[str, list[str]]:
        """Deterministic seed-based split of base preset IDs.

        1. Deduplicate
        2. Shuffle with seed
        3. Split by ratio (0.8 / 0.1 / 0.1)

        Returns {"train": [...], "val": [...], "test": [...]}.
        """
        unique_ids = sorted(set(base_preset_ids))
        rng = random.Random(self.config.seed)
        rng.shuffle(unique_ids)

        n = len(unique_ids)
        r_train, r_val, _r_test = self.config.split_ratio
        n_train = int(n * r_train)
        n_val = int(n * r_val)

        return {
            "train": unique_ids[:n_train],
            "val": unique_ids[n_train : n_train + n_val],
            "test": unique_ids[n_train + n_val :],
        }

    # ------------------------------------------------------------------
    # Run scope
    # ------------------------------------------------------------------

    def _select_run_scope(
        self, presets: list[tuple[str, VitalPreset]]
    ) -> list[tuple[str, VitalPreset]]:
        """Select base presets for the current run mode."""
        cfg = self.config
        if cfg.run_mode == "pilot":
            limit = cfg.pilot_max_base_presets
        elif cfg.run_mode == "canary":
            limit = cfg.canary_max_base_presets
        else:
            return presets  # full: use all

        # Deterministic selection: sort by base_id, take first `limit`
        sorted_presets = sorted(presets, key=lambda x: x[0])
        return sorted_presets[:limit]

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.config.output_dir / "checkpoint.json"

    def _load_or_init_checkpoint(
        self,
        base_ids: list[str],
        id_to_split: dict[str, str],
        schema_path: str,
    ) -> ProductionCheckpoint:
        """Load existing checkpoint or create a new one."""
        cp_path = self._checkpoint_path()

        if self.config.resume and cp_path.exists():
            try:
                data = json.loads(cp_path.read_text(encoding="utf-8"))
                cp = ProductionCheckpoint(
                    run_mode=data["run_mode"],
                    selected_base_preset_ids=data["selected_base_preset_ids"],
                    completed_base_preset_ids=data["completed_base_preset_ids"],
                    split_assignment=data["split_assignment"],
                    schema_path=data["schema_path"],
                    shard_manifest_path=data.get("shard_manifest_path", ""),
                    summary_so_far=data.get("summary_so_far", {}),
                    last_completed_stage=data.get("last_completed_stage", ""),
                )
                logger.info(
                    "Resumed checkpoint: %d/%d completed",
                    len(cp.completed_base_preset_ids),
                    len(cp.selected_base_preset_ids),
                )
                return cp
            except Exception as e:
                logger.warning("Failed to load checkpoint, starting fresh: %s", e)

        return ProductionCheckpoint(
            run_mode=self.config.run_mode,
            selected_base_preset_ids=list(base_ids),
            completed_base_preset_ids=[],
            split_assignment=id_to_split,
            schema_path=schema_path,
            shard_manifest_path=str(self.config.output_dir / "shard_manifest.json"),
            summary_so_far={},
            last_completed_stage="init",
        )

    def _save_checkpoint(self) -> None:
        """Persist current checkpoint to disk."""
        if self._checkpoint is None:
            return
        cp_path = self._checkpoint_path()
        data = {
            "run_mode": self._checkpoint.run_mode,
            "selected_base_preset_ids": self._checkpoint.selected_base_preset_ids,
            "completed_base_preset_ids": self._checkpoint.completed_base_preset_ids,
            "split_assignment": self._checkpoint.split_assignment,
            "schema_path": self._checkpoint.schema_path,
            "shard_manifest_path": self._checkpoint.shard_manifest_path,
            "summary_so_far": self._checkpoint.summary_so_far,
            "last_completed_stage": self._checkpoint.last_completed_stage,
        }
        cp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("Checkpoint saved: %d completed", len(self._checkpoint.completed_base_preset_ids))


    # ------------------------------------------------------------------
    # Per-base-preset processing
    # ------------------------------------------------------------------

    def _process_base_preset(
        self,
        base_id: str,
        preset: VitalPreset,
        split_name: str,
    ) -> tuple[int, int, int]:
        """Process a single base preset through the full inner pipeline.

        Returns (n_variants, n_rendered, n_valid).
        """
        cfg = self.config
        n_variants = 0
        n_rendered = 0
        n_valid = 0

        # Ensure wavetable fields are injected (they may already be from
        # produce() step 3, but re-inject for safety in case of resume)
        wt_map = self._wt_catalog.resolve_oscillator_wavetables(preset)
        for osc_key, entry in wt_map.items():
            if entry is not None:
                i = osc_key[-1]
                preset.settings[f"osc_{i}_wavetable_id"] = entry.name
                preset.settings[f"osc_{i}_wavetable_source_type"] = entry.source_type

        # a. RouteGraph → Augment
        variants = self._augmenter.augment(
            preset,
            base_id,
            run_mode=cfg.run_mode,
            pilot_max_variants_per_base=cfg.pilot_max_variants_per_base,
            canary_max_variants_per_base=cfg.canary_max_variants_per_base,
        )
        n_variants = len(variants)

        # b. Write mutated presets
        variant_dir = cfg.output_dir / "variants" / base_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        written_paths = self._writer.write_batch(variants, variant_dir)

        # c. Render + preprocess + embed for each variant
        for variant, preset_path in zip(variants, written_paths):
            samples = self._render_and_process_variant(
                variant, preset_path, base_id, split_name
            )
            n_rendered += samples["rendered"]
            n_valid += samples["valid"]

            # Periodic shard flush
            total_buffered = sum(len(b) for b in self._shard_buffers.values())
            if total_buffered >= cfg.flush_every_n_samples:
                self._flush_split_shards()

        return n_variants, n_rendered, n_valid

    def _render_and_process_variant(
        self,
        variant: MaskedVariant,
        preset_path: Path,
        base_id: str,
        split_name: str,
    ) -> dict[str, int]:
        """Render a single variant under multiple MIDI conditions, preprocess, embed."""
        stats = {"rendered": 0, "valid": 0}

        if self._backend is None:
            logger.warning("RendererBackend not initialised; skipping render")
            return stats

        audio_dir = self.config.output_dir / "audio" / base_id
        audio_dir.mkdir(parents=True, exist_ok=True)

        variant_label = f"{base_id}_mask_{variant.metadata.variant_id:04d}"
        render_results = self._backend.render_multi_condition(
            preset_path, audio_dir, variant_label
        )

        for rr in render_results:
            stats["rendered"] += 1

            if not rr.success or rr.audio is None:
                continue

            # Preprocess
            pp = self._preprocessor.process(rr.audio, 44100)
            if pp.is_filtered:
                continue

            # Embed
            embedding = self._extract_embedding(pp.audio, pp.sample_rate)
            if embedding is None:
                continue

            stats["valid"] += 1

            # Encode params
            param_values, present_mask = encode_preset_params(
                variant.preset, self._schema  # type: ignore[arg-type]
            )

            sample_id = str(uuid.uuid4())
            route_mask_json = json.dumps({
                "mask_vector": variant.route_mask.mask_vector,
                "masked_edge_names": variant.route_mask.masked_edge_names,
            })

            record: dict[str, Any] = {
                "param_values": param_values,
                "present_mask": present_mask,
                "embeddings": embedding,
                "base_preset_id": base_id,
                "variant_id": variant.metadata.variant_id,
                "route_mask_json": route_mask_json,
                "sample_id": sample_id,
                "midi_note": rr.midi_note,
                "midi_velocity": rr.midi_velocity,
                "midi_duration": 2.0,
                "rms_db": pp.original_rms_db,
            }
            self._shard_buffers[split_name].append(record)

        return stats

    # ------------------------------------------------------------------
    # Shard flush
    # ------------------------------------------------------------------

    def _flush_split_shards(self) -> None:
        """Flush buffered records to per-split shard files (npz)."""
        for split_name in self._SPLIT_NAMES:
            buf = self._shard_buffers[split_name]
            if not buf:
                continue

            shard_dir = self.config.output_dir / "shards" / split_name
            shard_dir.mkdir(parents=True, exist_ok=True)

            shard_idx = len(list(shard_dir.glob("shard_*.npz")))
            shard_path = shard_dir / f"shard_{shard_idx:04d}.npz"

            arrays: dict[str, Any] = {}
            arrays["param_values"] = np.stack([r["param_values"] for r in buf])
            arrays["present_mask"] = np.stack([r["present_mask"] for r in buf])
            arrays["embeddings"] = np.stack([r["embeddings"] for r in buf])
            arrays["base_preset_id"] = np.array(
                [r["base_preset_id"] for r in buf], dtype=object
            )
            arrays["variant_id"] = np.array(
                [r["variant_id"] for r in buf], dtype=np.int32
            )
            arrays["route_mask_json"] = np.array(
                [r["route_mask_json"] for r in buf], dtype=object
            )
            arrays["sample_id"] = np.array(
                [r["sample_id"] for r in buf], dtype=object
            )
            arrays["midi_note"] = np.array(
                [r["midi_note"] for r in buf], dtype=np.int32
            )
            arrays["midi_velocity"] = np.array(
                [r["midi_velocity"] for r in buf], dtype=np.int32
            )
            arrays["midi_duration"] = np.array(
                [r["midi_duration"] for r in buf], dtype=np.float32
            )
            arrays["rms_db"] = np.array(
                [r["rms_db"] for r in buf], dtype=np.float32
            )

            np.savez(shard_path, **arrays)
            logger.debug(
                "Flushed %d records to %s", len(buf), shard_path
            )
            buf.clear()

    # ------------------------------------------------------------------
    # Finalize HDF5
    # ------------------------------------------------------------------

    def _finalize_hdf5(self) -> None:
        """Merge all shard files into the final HDF5 dataset."""
        schema = self._schema
        if schema is None:
            logger.error("No schema available; cannot finalize HDF5")
            return

        hdf5_path = self.config.output_dir / "preset_corpus_dataset.h5"
        D = len(schema.param_names)

        with h5py.File(hdf5_path, "w") as hf:
            # --- Per-split groups ---
            for split_name in self._SPLIT_NAMES:
                shard_dir = self.config.output_dir / "shards" / split_name
                if not shard_dir.exists():
                    # Create empty group
                    grp = hf.create_group(split_name)
                    self._create_empty_split_datasets(grp, D)
                    continue

                shard_files = sorted(shard_dir.glob("shard_*.npz"))
                if not shard_files:
                    grp = hf.create_group(split_name)
                    self._create_empty_split_datasets(grp, D)
                    continue

                # Collect all shard data
                all_records = self._load_shards(shard_files)
                N = len(all_records["sample_id"])

                grp = hf.create_group(split_name)
                grp.create_dataset(
                    "param_values", data=all_records["param_values"], dtype="float32"
                )
                grp.create_dataset(
                    "present_mask", data=all_records["present_mask"], dtype="uint8"
                )
                grp.create_dataset(
                    "embeddings", data=all_records["embeddings"], dtype="float32"
                )

                # String datasets
                dt_str = h5py.string_dtype()
                grp.create_dataset(
                    "base_preset_id",
                    data=all_records["base_preset_id"],
                    dtype=dt_str,
                )
                grp.create_dataset(
                    "variant_id", data=all_records["variant_id"], dtype="int32"
                )
                grp.create_dataset(
                    "route_mask_json",
                    data=all_records["route_mask_json"],
                    dtype=dt_str,
                )
                grp.create_dataset(
                    "sample_id", data=all_records["sample_id"], dtype=dt_str
                )
                grp.create_dataset(
                    "midi_note", data=all_records["midi_note"], dtype="int32"
                )
                grp.create_dataset(
                    "midi_velocity",
                    data=all_records["midi_velocity"],
                    dtype="int32",
                )
                grp.create_dataset(
                    "midi_duration",
                    data=all_records["midi_duration"],
                    dtype="float32",
                )
                grp.create_dataset(
                    "rms_db", data=all_records["rms_db"], dtype="float32"
                )
                grp.create_dataset(
                    "split_local_index",
                    data=np.arange(N, dtype=np.int32),
                    dtype="int32",
                )

            # --- Metadata group ---
            meta = hf.create_group("metadata")
            dt_str = h5py.string_dtype()

            meta.create_dataset(
                "param_names",
                data=np.array(schema.param_names, dtype=object),
                dtype=dt_str,
            )
            meta.create_dataset(
                "param_types",
                data=np.array(
                    [schema.param_types[n] for n in schema.param_names],
                    dtype=object,
                ),
                dtype=dt_str,
            )
            meta.create_dataset(
                "param_value_encoding",
                data=np.array(
                    [schema.param_value_encoding[n] for n in schema.param_names],
                    dtype=object,
                ),
                dtype=dt_str,
            )
            meta.create_dataset(
                "default_values",
                data=np.array(
                    [schema.default_values[n] for n in schema.param_names],
                    dtype=np.float32,
                ),
                dtype="float32",
            )
            meta.create_dataset(
                "corpus_min",
                data=np.array(
                    [schema.corpus_min[n] for n in schema.param_names],
                    dtype=np.float32,
                ),
                dtype="float32",
            )
            meta.create_dataset(
                "corpus_max",
                data=np.array(
                    [schema.corpus_max[n] for n in schema.param_names],
                    dtype=np.float32,
                ),
                dtype="float32",
            )
            meta.create_dataset(
                "presence_ratio",
                data=np.array(
                    [schema.presence_ratio[n] for n in schema.param_names],
                    dtype=np.float32,
                ),
                dtype="float32",
            )

            # Scalar string datasets
            meta.create_dataset(
                "category_values_json",
                data=json.dumps(schema.category_values, ensure_ascii=False),
                dtype=dt_str,
            )
            meta.create_dataset(
                "route_edge_schema_json",
                data=json.dumps(
                    {"note": "route edge schema placeholder"}, ensure_ascii=False
                ),
                dtype=dt_str,
            )
            meta.create_dataset(
                "corpus_schema_json",
                data=json.dumps(asdict(schema), ensure_ascii=False),
                dtype=dt_str,
            )
            meta.create_dataset(
                "run_mode", data=self.config.run_mode, dtype=dt_str
            )
            meta.create_dataset(
                "checkpoint_manifest_json",
                data=json.dumps(
                    {
                        "completed": (
                            self._checkpoint.completed_base_preset_ids
                            if self._checkpoint
                            else []
                        )
                    },
                    ensure_ascii=False,
                ),
                dtype=dt_str,
            )

        logger.info("Finalized HDF5: %s", hdf5_path)

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_empty_split_datasets(grp: h5py.Group, D: int) -> None:
        """Create empty datasets for a split group with correct dtypes."""
        dt_str = h5py.string_dtype()
        grp.create_dataset("param_values", shape=(0, D), dtype="float32")
        grp.create_dataset("present_mask", shape=(0, D), dtype="uint8")
        grp.create_dataset("embeddings", shape=(0, EMBEDDING_DIM), dtype="float32")
        grp.create_dataset("base_preset_id", shape=(0,), dtype=dt_str)
        grp.create_dataset("variant_id", shape=(0,), dtype="int32")
        grp.create_dataset("route_mask_json", shape=(0,), dtype=dt_str)
        grp.create_dataset("sample_id", shape=(0,), dtype=dt_str)
        grp.create_dataset("midi_note", shape=(0,), dtype="int32")
        grp.create_dataset("midi_velocity", shape=(0,), dtype="int32")
        grp.create_dataset("midi_duration", shape=(0,), dtype="float32")
        grp.create_dataset("rms_db", shape=(0,), dtype="float32")
        grp.create_dataset("split_local_index", shape=(0,), dtype="int32")

    @staticmethod
    def _load_shards(shard_files: list[Path]) -> dict[str, np.ndarray]:
        """Load and concatenate shard npz files."""
        collectors: dict[str, list[np.ndarray]] = {}

        for sf_path in shard_files:
            data = np.load(sf_path, allow_pickle=True)
            for key in data.files:
                if key not in collectors:
                    collectors[key] = []
                collectors[key].append(data[key])

        merged: dict[str, np.ndarray] = {}
        for key, arrays in collectors.items():
            merged[key] = np.concatenate(arrays, axis=0)

        return merged

    # ------------------------------------------------------------------
    # Lazy resource init
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        """Initialise RendererBackend (requires VST plugin)."""
        try:
            self._backend = RendererBackend(self.config.vital_vst_path)
            logger.info("RendererBackend initialised")
        except Exception as e:
            logger.warning("Could not init RendererBackend: %s", e)
            self._backend = None

    def _init_embedding_extractor(self) -> None:
        """Initialise EmbeddingExtractor (requires MuQ model)."""
        try:
            from src.embedding_extractor import EmbeddingExtractor

            self._embedding_extractor = EmbeddingExtractor(
                device=self.config.embedding_device
            )
            logger.info("EmbeddingExtractor initialised")
        except Exception as e:
            logger.warning("Could not init EmbeddingExtractor: %s", e)
            self._embedding_extractor = None

    def _extract_embedding(
        self, audio: np.ndarray | None, sample_rate: int
    ) -> np.ndarray | None:
        """Extract embedding from preprocessed audio, returning None on failure."""
        if audio is None:
            return None
        if self._embedding_extractor is None:
            # Return a zero placeholder so the pipeline can still run
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        try:
            return self._embedding_extractor.extract_waveform(audio, sample_rate)
        except Exception as e:
            logger.warning("Embedding extraction failed: %s", e)
            return None
