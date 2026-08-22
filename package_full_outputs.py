#!/usr/bin/env python3
"""Create one combined ZIP containing the complete generated setup tree."""

from __future__ import annotations

import os
from pathlib import Path
import zipfile


def package_full_outputs(output_base: Path) -> Path:
    output_base = output_base.expanduser().resolve()
    setups_root = output_base / "setups"
    if not setups_root.is_dir():
        raise FileNotFoundError(f"Generated setup directory is missing: {setups_root}")

    files = sorted(
        path
        for path in setups_root.rglob("*")
        if path.is_file() and path.suffix.lower() != ".zip"
    )
    if not files:
        raise RuntimeError(f"No generated output files found under {setups_root}")

    archive_path = output_base / "full_outputs.zip"
    temporary_path = output_base / "full_outputs.zip.incomplete"
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in files:
                archive.write(path, path.relative_to(output_base).as_posix())
        os.replace(temporary_path, archive_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Combined full-output archive: {archive_path}")
    print(f"Archived files: {len(files)}")
    return archive_path


def main() -> None:
    raw_output_base = os.environ.get("PIPELINE_OUTPUT") or os.environ.get("OUTPUT_BASE")
    if not raw_output_base:
        raise ValueError("PIPELINE_OUTPUT or OUTPUT_BASE must be set")
    package_full_outputs(Path(raw_output_base))


if __name__ == "__main__":
    main()
