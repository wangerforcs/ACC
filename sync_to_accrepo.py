#!/usr/bin/env python3
"""
Sync adaptive sources, selected top-level feature files, and extra folder contents
into /home/wangzb/accrepo.

This script is designed to be rerun safely:
- it copies tracked source files from this workspace into the target repo
- it preserves relative paths
- it can optionally clean stale synced files from the target

By default it syncs:
- the whole adaptive/ directory except generated artifacts and caches
- a curated list of top-level files that contain the main new features
- all contents under a curated list of extra folders
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from pathlib import Path


SOURCE_ROOT = Path("/home/wangzb/vlash").resolve()
TARGET_ROOT = Path("/home/wangzb/accrepo").resolve()

DEFAULT_TOP_LEVEL_FILES = [
    "evalpi05_adaptive.py",
    "evalpi05_adaptivefine.py",
    "evalpi05_adaptivefine_norm.py",
    "evalpi05_adaptivefine_balanced.py",
    "evalpi05_adaptivefine_heavy.py",
    "evalpi05_adaptivefine_dynamic.py",
    "evalpi05_power_replay.py",
    "evalpi05_random_monitor.py",
    "evalpi05.yaml",
    "visualutils.py",
    "inspect_logs.py",
    "runtrackers.sh",
    "runtrackersscore.sh",
    "run_inference.py",
    "sync_to_accrepo.py",
    "LIBERO/modified_libero_files.txt",
    "measure_cuda_context_power.py",
    "measure_vla_latency.py",
    "plot_fullout_power.py",
    "export_fullout.py",
]

# 新增：需要整体复制的额外文件夹列表
DEFAULT_EXTRA_FOLDERS = [
    "computeflops",
    "trackersscoregood",
    "power_runs_compiled_valuefuncasync",
    "power_runs_compiled_valuefuncsync",
    "power-runs_nocompiled_valuefuncasync",
    "power-runs_nocompiled_valuefuncsync",
    # 例子：
    # "configs",
    # "tools",
    # "scripts",
]

EXCLUDE_GLOBS = [
    "adaptive/__pycache__/**",
    "adaptive/**/*.pyc",
    "adaptive/dino_libero_results/**",
    "adaptive/tracker_video_results/**",
    "adaptive/sam2_video_results/**",
    "adaptive/sam2_rollout_results/**",
    "adaptive/sam2_image_results/**",
]

MODIFIED_LIBERO_LIST = "LIBERO/modified_libero_files.txt"


def is_excluded(rel_path: Path, patterns: list[str]) -> bool:
    path_str = rel_path.as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False


def iter_files_under_folder(source_root: Path, folder: str, exclude_globs: list[str]) -> list[Path]:
    folder_root = source_root / folder
    if not folder_root.exists() or not folder_root.is_dir():
        return []

    files: list[Path] = []
    for path in folder_root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(source_root)
        if is_excluded(rel_path, exclude_globs):
            continue
        files.append(rel_path)
    return sorted(files)


def iter_adaptive_files(source_root: Path, exclude_globs: list[str]) -> list[Path]:
    return iter_files_under_folder(source_root, "adaptive", exclude_globs)


def iter_top_level_files(source_root: Path, file_list: list[str]) -> list[Path]:
    files: list[Path] = []
    for name in file_list:
        path = source_root / name
        if path.is_file():
            files.append(path.relative_to(source_root))
    return sorted(files)


def iter_modified_libero_files(source_root: Path, list_file: str) -> list[Path]:
    manifest_path = source_root / list_file
    if not manifest_path.is_file():
        return []

    files: list[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        file_path = Path(line)
        try:
            rel_path = file_path.resolve().relative_to(source_root.resolve())
        except Exception:
            try:
                rel_path = Path(line).relative_to(source_root)
            except Exception:
                continue

        candidate = source_root / rel_path
        if candidate.is_file():
            files.append(rel_path)

    return sorted(set(files))


def iter_extra_folder_files(source_root: Path, folder_list: list[str], exclude_globs: list[str]) -> list[Path]:
    files: list[Path] = []
    for folder in folder_list:
        files.extend(iter_files_under_folder(source_root, folder, exclude_globs))
    return sorted(set(files))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src_root: Path, dst_root: Path, rel_path: Path, dry_run: bool) -> dict:
    src = src_root / rel_path
    dst = dst_root / rel_path
    ensure_parent(dst)

    changed = True
    if dst.exists():
        try:
            changed = (
                src.stat().st_size != dst.stat().st_size
                or int(src.stat().st_mtime) != int(dst.stat().st_mtime)
            )
        except OSError:
            changed = True

    if not dry_run:
        shutil.copy2(src, dst)

    return {
        "path": rel_path.as_posix(),
        "changed": changed,
    }


def clean_synced_scope(
    dst_root: Path,
    expected_paths: set[str],
    top_level_files: list[str],
    extra_folders: list[str],
    dry_run: bool,
) -> list[str]:
    removed: list[str] = []

    top_level_set = {Path(p).as_posix() for p in top_level_files}
    extra_folder_prefixes = [f"{Path(folder).as_posix().rstrip('/')}/" for folder in extra_folders]

    for path in dst_root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(dst_root).as_posix()
        if rel == ".sync_manifest.json":
            continue

        in_scope = (
            rel.startswith("adaptive/")
            or rel in top_level_set
            or any(rel.startswith(prefix) for prefix in extra_folder_prefixes)
        )

        if in_scope and rel not in expected_paths:
            removed.append(rel)
            if not dry_run:
                path.unlink()

    return sorted(removed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync adaptive sources, selected top-level files, and extra folder contents into /home/wangzb/accrepo."
    )
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="Only print planned actions without copying files.")
    parser.add_argument("--clean", action="store_true", help="Remove stale files inside the synced scope in the target repo.")
    parser.add_argument(
        "--top-level",
        nargs="*",
        default=DEFAULT_TOP_LEVEL_FILES,
        help="Override the list of top-level files to sync.",
    )
    parser.add_argument(
        "--extra-folders",
        nargs="*",
        default=DEFAULT_EXTRA_FOLDERS,
        help="List of extra folders whose full contents will be synced recursively.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    adaptive_files = iter_adaptive_files(source_root, EXCLUDE_GLOBS)
    top_level_files = iter_top_level_files(source_root, args.top_level)
    modified_libero_files = iter_modified_libero_files(source_root, MODIFIED_LIBERO_LIST)
    extra_folder_files = iter_extra_folder_files(source_root, args.extra_folders, EXCLUDE_GLOBS)

    all_files = sorted(set(adaptive_files + top_level_files + modified_libero_files + extra_folder_files))

    copied = [copy_file(source_root, target_root, rel_path, args.dry_run) for rel_path in all_files]
    expected_paths = {item["path"] for item in copied}

    removed: list[str] = []
    if args.clean:
        removed = clean_synced_scope(
            target_root,
            expected_paths,
            args.top_level,
            args.extra_folders,
            args.dry_run,
        )

    manifest = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "top_level_files": args.top_level,
        "extra_folders": args.extra_folders,
        "adaptive_file_count": len(adaptive_files),
        "top_level_file_count": len(top_level_files),
        "modified_libero_file_count": len(modified_libero_files),
        "extra_folder_file_count": len(extra_folder_files),
        "copied": copied,
        "removed": removed,
        "dry_run": args.dry_run,
        "clean": args.clean,
    }

    manifest_path = target_root / ".sync_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Target repo: {target_root}")
    print(f"Adaptive files synced: {len(adaptive_files)}")
    print(f"Top-level files synced: {len(top_level_files)}")
    print(f"Extra-folder files synced: {len(extra_folder_files)}")
    print(f"Total files synced: {len(all_files)}")
    if args.clean:
        print(f"Stale files removed: {len(removed)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
