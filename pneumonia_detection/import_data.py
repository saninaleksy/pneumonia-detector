from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


def download_data(dest_dir: Path, kaggle_dataset: str) -> None:
    dataset_path = dest_dir.expanduser().resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)

    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    if train_dir.exists() and test_dir.exists():
        log.info("Dataset already present in: %s", dataset_path)
        return

    log.info(
        "Dataset not found in %s. Downloading from Kaggle: %s",
        dataset_path,
        kaggle_dataset,
    )

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                kaggle_dataset,
                "--unzip",
                "-p",
                str(dataset_path),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle` and configure "
            "~/.kaggle/kaggle.json (chmod 600)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Kaggle download failed: {exc}") from exc

    zip_files = list(dataset_path.glob("*.zip"))
    for zip_file in zip_files:
        try:
            log.info("Extracting archive: %s", zip_file.name)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(dataset_path)
            zip_file.unlink(missing_ok=True)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Corrupted zip file: {zip_file}") from exc

    normalize_chest_xray_dir(dataset_path)

    if not (train_dir.exists() and test_dir.exists()):
        raise RuntimeError("Expected folders: train/ and test/.")

    log.info("Dataset ready in: %s", dataset_path)


def normalize_chest_xray_dir(dataset_path: Path) -> None:
    dataset_path = dataset_path.expanduser().resolve()

    _cleanup_macos_files_recursive(dataset_path)

    nested = dataset_path / "chest_xray"
    if nested.exists() and nested.is_dir():
        for name in ("train", "test", "val"):
            src = nested / name
            dst = dataset_path / name
            if src.exists() and not dst.exists():
                src.rename(dst)

        _cleanup_macos_files_recursive(nested)
        try:
            shutil.rmtree(nested, ignore_errors=True)
        except Exception:
            pass

    _cleanup_macos_files_recursive(dataset_path)


def _cleanup_macos_files_recursive(root: Path) -> None:
    for macos_dir in root.rglob("__MACOSX"):
        if macos_dir.is_dir():
            shutil.rmtree(macos_dir, ignore_errors=True)

    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink(missing_ok=True)

    for path in root.rglob(".DS_Store"):
        if path.is_file():
            path.unlink(missing_ok=True)
