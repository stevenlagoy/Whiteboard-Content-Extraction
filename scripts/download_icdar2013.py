"""Download and verify a pinned ICDAR 2013 dataset mirror."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "ICDAR"
KAGGLE_DATASET = "gopichandu1/icdar2013"
KAGGLE_VERSION = 1
KAGGLE_METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{KAGGLE_DATASET}"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
EXPECTED_SHA256 = "03b61b4af2000e0275fb56dda2a1d218b5ab6af5505dcfe724af8f759d4dab4e"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading archive to {destination}...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "whiteboard-content-extraction"},
    )
    with urllib.request.urlopen(request) as response:
        with destination.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)


def verify_pinned_dataset() -> None:
    request = urllib.request.Request(
        KAGGLE_METADATA_URL,
        headers={"User-Agent": "whiteboard-content-extraction"},
    )
    with urllib.request.urlopen(request) as response:
        metadata = json.loads(response.read().decode("utf-8"))

    current_version = int(metadata["currentVersionNumber"])
    if current_version != KAGGLE_VERSION:
        raise RuntimeError(
            f"Kaggle mirror version changed from pinned version "
            f"{KAGGLE_VERSION} to {current_version}. "
            "Review and update the pin and SHA-256 before downloading."
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, output: Path) -> None:
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a valid ZIP archive: {archive}")

    output.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {output}...")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract pinned Kaggle ICDAR 2013 dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Dataset destination (default: data/raw/ICDAR).",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    archive = output.parent / "ICDAR2013.zip"
    verify_pinned_dataset()
    download(KAGGLE_DOWNLOAD_URL, archive)

    actual_sha256 = sha256(archive)
    if actual_sha256 != EXPECTED_SHA256:
        archive.unlink()
        raise RuntimeError(
            f"Downloaded dataset checksum mismatch: "
            f"expected {EXPECTED_SHA256}, got {actual_sha256}. "
            "The archive was deleted."
        )

    extract_archive(archive, output)
    archive.unlink()

    print(f"ICDAR data prepared at: {output}")


if __name__ == "__main__":
    main()
