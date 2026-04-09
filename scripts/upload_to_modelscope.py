#!/usr/bin/env python3
"""
上传数据集到魔搭 (ModelScope) 平台。

使用方式：
  # 首次使用前，先登录：
  python3 scripts/upload_to_modelscope.py --login

  # 上传整个生产目录（HDF5 + summary + presets）：
  python3 scripts/upload_to_modelscope.py \
      --repo-id YOUR_USERNAME/vital-synth-dataset \
      --output-dir experiments/expansion_200k

  # 只上传 HDF5 文件：
  python3 scripts/upload_to_modelscope.py \
      --repo-id YOUR_USERNAME/vital-synth-dataset \
      --output-dir experiments/expansion_200k \
      --hdf5-only

参考文档：https://modelscope.cn/docs/datasets/upload
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import shutil
from pathlib import Path

# ModelScope 非 LFS 文件总大小限制约 500MB，留余量用 450MB
BATCH_SIZE_LIMIT_MB = 450


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="上传数据集到魔搭 (ModelScope) 平台"
    )
    parser.add_argument(
        "--login", action="store_true",
        help="登录魔搭平台（需要输入 access token）",
    )
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="魔搭数据集仓库 ID，格式：username/dataset-name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/expansion_200k",
        help="生产输出目录路径",
    )
    parser.add_argument(
        "--hdf5-only", action="store_true",
        help="只上传 HDF5 数据集文件（不上传音频和预设）",
    )
    parser.add_argument(
        "--include-audio", action="store_true",
        help="同时上传 audio/ 目录（警告：可能非常大）",
    )
    parser.add_argument(
        "--include-presets", action="store_true",
        help="同时上传 presets/ 目录",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="魔搭 access token（也可通过 --login 预先登录）",
    )
    parser.add_argument(
        "--commit-message", type=str, default=None,
        help="提交信息",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只检查文件，不实际上传",
    )
    return parser.parse_args(argv)


def check_production_status(output_dir: Path) -> dict | None:
    """读取并验证生产状态。"""
    summary_path = output_dir / "production_summary.json"
    if not summary_path.exists():
        print(f"❌ 未找到 production_summary.json: {summary_path}")
        return None

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    valid = summary.get("valid_samples", 0)
    total = summary.get("total_samples", 0)
    filtered = summary.get("filtered_samples", 0)
    failed = summary.get("failed_samples", 0)

    print("=" * 50)
    print("  生产状态")
    print("=" * 50)
    print(f"  总样本数:   {total:,}")
    print(f"  有效样本:   {valid:,}")
    print(f"  过滤样本:   {filtered:,}")
    print(f"  失败样本:   {failed:,}")

    if "filter_reasons" in summary and summary["filter_reasons"]:
        print("  过滤原因:")
        for reason, count in summary["filter_reasons"].items():
            print(f"    {reason}: {count:,}")

    if "dataset_splits" in summary and summary["dataset_splits"]:
        splits = summary["dataset_splits"]
        print(f"  数据集划分: train={splits.get('train', 0):,}, "
              f"val={splits.get('val', 0):,}, test={splits.get('test', 0):,}")

    print("=" * 50)

    if valid == 0:
        print("⚠️  有效样本为 0，不建议上传。")
        return None

    return summary


def collect_upload_files(
    output_dir: Path, hdf5_only: bool, include_audio: bool, include_presets: bool
) -> list[tuple[Path, str]]:
    """收集需要上传的文件列表。返回 (本地路径, 仓库内路径) 对。"""
    files: list[tuple[Path, str]] = []

    # HDF5 数据集文件
    for hdf5 in output_dir.glob("*.h5"):
        files.append((hdf5, hdf5.name))
    for hdf5 in output_dir.glob("*.hdf5"):
        files.append((hdf5, hdf5.name))

    # production_summary.json
    summary_path = output_dir / "production_summary.json"
    if summary_path.exists():
        files.append((summary_path, "production_summary.json"))

    if hdf5_only:
        return files

    # 配置文件（如果存在）
    for config_name in ["config.yaml", "production_config.yaml"]:
        config_path = output_dir / config_name
        if config_path.exists():
            files.append((config_path, config_name))

    # production.log
    log_path = output_dir / "production.log"
    if log_path.exists():
        files.append((log_path, "production.log"))

    return files


def do_login(token: str | None = None) -> None:
    """登录魔搭平台。"""
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("❌ 请先安装 modelscope: pip install modelscope")
        sys.exit(1)

    api = HubApi()
    if token:
        api.login(token)
        print("✅ 登录成功")
    else:
        t = input("请输入魔搭 access token (从 https://modelscope.cn/my/myaccesstoken 获取): ").strip()
        if not t:
            print("❌ token 不能为空")
            sys.exit(1)
        api.login(t)
        print("✅ 登录成功")


def _upload_dir_in_batches(
    api,
    repo_id: str,
    source_dir: Path,
    path_in_repo: str,
    pattern: str,
    commit_msg: str,
) -> None:
    """将大目录分批上传，每批不超过 BATCH_SIZE_LIMIT_MB。"""
    all_files = sorted(source_dir.glob(pattern))
    if not all_files:
        print(f"  ⚠️  {source_dir.name}/ 目录为空，跳过")
        return

    total_size_mb = sum(f.stat().st_size for f in all_files) / 1024 / 1024
    print(f"  📦 {path_in_repo}/ 目录: {len(all_files):,} 个文件, {total_size_mb:.0f} MB")

    # 按大小分批
    batches: list[list[Path]] = []
    current_batch: list[Path] = []
    current_size = 0
    limit = BATCH_SIZE_LIMIT_MB * 1024 * 1024

    for f in all_files:
        fsize = f.stat().st_size
        if current_batch and current_size + fsize > limit:
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(f)
        current_size += fsize

    if current_batch:
        batches.append(current_batch)

    print(f"  分 {len(batches)} 批上传...")

    for i, batch in enumerate(batches, 1):
        batch_size_mb = sum(f.stat().st_size for f in batch) / 1024 / 1024
        print(f"  [{i}/{len(batches)}] {len(batch):,} 个文件 ({batch_size_mb:.0f} MB)...")

        # 创建临时目录，硬链接本批文件（同一文件系统下不占额外空间）
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for f in batch:
                dest = tmp_path / f.name
                try:
                    dest.hardlink_to(f)
                except OSError:
                    # 跨文件系统时回退到拷贝
                    shutil.copy2(f, dest)

            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(tmp_path),
                path_in_repo=path_in_repo,
                commit_message=f"{commit_msg} (batch {i}/{len(batches)})",
                repo_type="dataset",
            )

    print(f"  ✅ {path_in_repo}/ 上传完成")


def upload(args: argparse.Namespace) -> int:
    """执行上传。"""
    output_dir = Path(args.output_dir)

    # 检查生产状态
    summary = check_production_status(output_dir)
    if summary is None:
        return 1

    if args.dry_run:
        files = collect_upload_files(
            output_dir, args.hdf5_only, args.include_audio, args.include_presets
        )
        print(f"\n📦 将上传 {len(files)} 个文件:")
        total_size = 0
        for local_path, repo_path in files:
            size = local_path.stat().st_size
            total_size += size
            print(f"  {repo_path} ({size / 1024 / 1024:.1f} MB)")

        if args.include_audio:
            audio_dir = output_dir / "audio"
            if audio_dir.exists():
                audio_count = sum(1 for _ in audio_dir.glob("*.wav"))
                print(f"  + audio/ 目录 ({audio_count:,} 个 WAV 文件)")

        if args.include_presets:
            presets_dir = output_dir / "presets"
            if presets_dir.exists():
                preset_count = sum(1 for _ in presets_dir.glob("*.vital"))
                print(f"  + presets/ 目录 ({preset_count:,} 个预设文件)")

        print(f"\n  文件总大小: {total_size / 1024 / 1024:.1f} MB (不含目录)")
        print("  (dry-run 模式，未实际上传)")
        return 0

    if not args.repo_id:
        print("❌ 请指定 --repo-id，格式：username/dataset-name")
        return 1

    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("❌ 请先安装 modelscope: pip install modelscope")
        return 1

    api = HubApi()
    if args.token:
        api.login(args.token)

    repo_id = args.repo_id

    # 上传单个文件
    files = collect_upload_files(
        output_dir, args.hdf5_only, args.include_audio, args.include_presets
    )

    commit_msg = args.commit_message or f"Upload dataset ({summary.get('valid_samples', 0):,} valid samples)"

    print(f"\n🚀 开始上传到 {repo_id}...")

    for local_path, repo_path in files:
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"  上传 {repo_path} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
        )

    # 上传 audio 目录（分批）
    if args.include_audio:
        audio_dir = output_dir / "audio"
        if audio_dir.exists():
            _upload_dir_in_batches(
                api, repo_id, audio_dir, "audio", "*.wav", commit_msg,
            )

    # 上传 presets 目录（分批）
    if args.include_presets:
        presets_dir = output_dir / "presets"
        if presets_dir.exists():
            _upload_dir_in_batches(
                api, repo_id, presets_dir, "presets", "*.vital", commit_msg,
            )

    print(f"\n✅ 上传完成！查看: https://modelscope.cn/datasets/{repo_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.login:
        do_login(args.token)
        return 0

    return upload(args)


if __name__ == "__main__":
    sys.exit(main())
