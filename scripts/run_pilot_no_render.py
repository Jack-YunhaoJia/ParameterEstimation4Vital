#!/usr/bin/env python3
"""
Pilot 生产脚本（无渲染模式）。

执行完整流水线中不依赖 VST 的所有步骤：
scan → parse → wavetable catalog → schema → split → augment → write variants → HDF5 (参数+掩码)

跳过渲染、预处理和 embedding 提取。生成的 .vital 变异文件可以后续用
独立的渲染脚本处理。

Usage:
    python3 scripts/run_pilot_no_render.py
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
import sys

import h5py
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preset_corpus_scanner import PresetCorpusScanner
from src.preset_parser import PresetParser
from src.preset_schema_extractor import PresetSchemaExtractor
from src.route_graph_builder import RouteGraphBuilder
from src.route_mask_augmenter import RouteMaskAugmenter
from src.mutated_preset_writer import MutatedPresetWriter
from src.wavetable_catalog import WavetableCatalog
from src.preset_dataset_producer import encode_preset_params, EMBEDDING_DIM


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)
    t0 = time.time()

    # --- Config ---
    corpus_dir = _PROJECT_ROOT / "presets"
    output_dir = _PROJECT_ROOT / "experiments" / "pilot_preset_corpus"
    vital_root = Path("/Users/jack/Music/Vital")
    seed = 42
    variant_cap = 8  # pilot: conservative
    split_ratio = (0.8, 0.1, 0.1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Scan ---
    scanner = PresetCorpusScanner()
    scan = scanner.scan(corpus_dir)
    logger.info("Scanned %d presets", len(scan.preset_paths))

    # --- 2. Parse ---
    parser = PresetParser()
    presets: list[tuple[str, "VitalPreset"]] = []
    for p in scan.preset_paths:
        preset = parser.parse(p)
        presets.append((p.stem, preset))

    # --- 3. Wavetable catalog + inject ---
    catalog = WavetableCatalog(vital_root=vital_root)
    wt_breakdown = {"factory": 0, "third_party": 0, "embedded": 0}
    for base_id, preset in presets:
        wt_map = catalog.resolve_oscillator_wavetables(preset)
        for osc_key, entry in wt_map.items():
            if entry is not None:
                wt_breakdown[entry.source_type] = wt_breakdown.get(entry.source_type, 0) + 1
                i = osc_key[-1]
                preset.settings[f"osc_{i}_wavetable_id"] = entry.name
                preset.settings[f"osc_{i}_wavetable_source_type"] = entry.source_type

    catalog.save(output_dir / "wavetable_catalog.json")
    logger.info("Wavetable breakdown: %s", wt_breakdown)

    # --- 4. Schema ---
    extractor = PresetSchemaExtractor(
        inventory_path=_PROJECT_ROOT / "vital_param_inventory.json"
    )
    schema = extractor.extract([p for _, p in presets])
    extractor.save_schema(schema, output_dir / "corpus_schema.json")
    D = len(schema.param_names)
    logger.info("Schema: %d params", D)

    # --- 5. Split ---
    import random
    base_ids = [bid for bid, _ in presets]
    unique_ids = sorted(set(base_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    split_assignment = {
        "train": unique_ids[:n_train],
        "val": unique_ids[n_train:n_train + n_val],
        "test": unique_ids[n_train + n_val:],
    }
    id_to_split = {}
    for s, ids in split_assignment.items():
        for bid in ids:
            id_to_split[bid] = s
    logger.info("Split: train=%d, val=%d, test=%d",
                len(split_assignment["train"]),
                len(split_assignment["val"]),
                len(split_assignment["test"]))

    # --- 6. Augment + Write + Encode ---
    graph_builder = RouteGraphBuilder()
    augmenter = RouteMaskAugmenter(graph_builder, min_variants=4, max_variants=variant_cap)
    writer = MutatedPresetWriter(parser)

    # Buffers per split
    buffers: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    total_variants = 0
    preset_map = {bid: p for bid, p in presets}

    for base_id in base_ids:
        preset = preset_map[base_id]
        split_name = id_to_split[base_id]

        variants = augmenter.augment(
            preset, base_id, run_mode="pilot",
            pilot_max_variants_per_base=variant_cap,
        )
        total_variants += len(variants)

        # Write .vital files
        variant_dir = output_dir / "variants" / base_id
        writer.write_batch(variants, variant_dir)

        # Encode params for each variant
        for variant in variants:
            pv, pm = encode_preset_params(variant.preset, schema)
            route_mask_json = json.dumps({
                "mask_vector": variant.route_mask.mask_vector,
                "masked_edge_names": variant.route_mask.masked_edge_names,
            })
            buffers[split_name].append({
                "param_values": pv,
                "present_mask": pm,
                "base_preset_id": base_id,
                "variant_id": variant.metadata.variant_id,
                "route_mask_json": route_mask_json,
                "sample_id": str(uuid.uuid4()),
            })

        logger.info("  %s: %d variants", base_id, len(variants))

    # --- 7. Write HDF5 (params only, no audio/embeddings) ---
    hdf5_path = output_dir / "preset_corpus_dataset.h5"
    dt_str = h5py.string_dtype()

    with h5py.File(hdf5_path, "w") as hf:
        for split_name in ("train", "val", "test"):
            buf = buffers[split_name]
            grp = hf.create_group(split_name)
            N = len(buf)

            if N == 0:
                grp.create_dataset("param_values", shape=(0, D), dtype="float32")
                grp.create_dataset("present_mask", shape=(0, D), dtype="uint8")
                grp.create_dataset("base_preset_id", shape=(0,), dtype=dt_str)
                grp.create_dataset("variant_id", shape=(0,), dtype="int32")
                grp.create_dataset("route_mask_json", shape=(0,), dtype=dt_str)
                grp.create_dataset("sample_id", shape=(0,), dtype=dt_str)
                grp.create_dataset("split_local_index", shape=(0,), dtype="int32")
                continue

            grp.create_dataset("param_values",
                data=np.stack([r["param_values"] for r in buf]), dtype="float32")
            grp.create_dataset("present_mask",
                data=np.stack([r["present_mask"] for r in buf]), dtype="uint8")
            grp.create_dataset("base_preset_id",
                data=np.array([r["base_preset_id"] for r in buf], dtype=object), dtype=dt_str)
            grp.create_dataset("variant_id",
                data=np.array([r["variant_id"] for r in buf], dtype=np.int32), dtype="int32")
            grp.create_dataset("route_mask_json",
                data=np.array([r["route_mask_json"] for r in buf], dtype=object), dtype=dt_str)
            grp.create_dataset("sample_id",
                data=np.array([r["sample_id"] for r in buf], dtype=object), dtype=dt_str)
            grp.create_dataset("split_local_index",
                data=np.arange(N, dtype=np.int32), dtype="int32")

        # Metadata
        meta = hf.create_group("metadata")
        meta.create_dataset("param_names",
            data=np.array(schema.param_names, dtype=object), dtype=dt_str)
        meta.create_dataset("param_types",
            data=np.array([schema.param_types[n] for n in schema.param_names], dtype=object), dtype=dt_str)
        meta.create_dataset("param_value_encoding",
            data=np.array([schema.param_value_encoding[n] for n in schema.param_names], dtype=object), dtype=dt_str)
        meta.create_dataset("default_values",
            data=np.array([schema.default_values[n] for n in schema.param_names], dtype=np.float32), dtype="float32")
        meta.create_dataset("corpus_min",
            data=np.array([schema.corpus_min[n] for n in schema.param_names], dtype=np.float32), dtype="float32")
        meta.create_dataset("corpus_max",
            data=np.array([schema.corpus_max[n] for n in schema.param_names], dtype=np.float32), dtype="float32")
        meta.create_dataset("presence_ratio",
            data=np.array([schema.presence_ratio[n] for n in schema.param_names], dtype=np.float32), dtype="float32")
        meta.create_dataset("category_values_json",
            data=json.dumps(schema.category_values, ensure_ascii=False), dtype=dt_str)
        meta.create_dataset("corpus_schema_json",
            data=json.dumps(asdict(schema), ensure_ascii=False), dtype=dt_str)
        meta.create_dataset("run_mode", data="pilot", dtype=dt_str)
        meta.create_dataset("wavetable_breakdown_json",
            data=json.dumps(wt_breakdown, ensure_ascii=False), dtype=dt_str)

    elapsed = time.time() - t0

    # --- Summary ---
    summary = {
        "total_base_presets": len(base_ids),
        "total_variants": total_variants,
        "split_counts": {s: len(buffers[s]) for s in ("train", "val", "test")},
        "schema_dimension": D,
        "wavetable_breakdown": wt_breakdown,
        "elapsed_sec": round(elapsed, 1),
        "note": "No rendering — .vital variant files written for offline rendering",
    }
    (output_dir / "production_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("Pilot Production Complete (no-render mode)")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nOutput: {output_dir}")
    print(f"HDF5:   {hdf5_path}")
    print(f"Variants: {output_dir / 'variants'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
