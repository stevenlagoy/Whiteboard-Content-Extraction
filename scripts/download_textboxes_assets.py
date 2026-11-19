"""Download optional TextBoxes assets used by the standalone detection scripts."""

import argparse
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "textboxes"

ICDAR_WEIGHTS_URL = (
    "https://www.dropbox.com/s/g8pjzv2de9gty8g/"
    "TextBoxes_icdar13.caffemodel?dl=1"
)
DEPLOY_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/bhargavaurala/"
    "accessmath-textboxes/master/examples/TextBoxes/deploy.prototxt"
)
VGG_WEIGHTS_URL = (
    "https://www.dropbox.com/s/qxc64az0a21vodt/"
    "VGG_ILSVRC_16_layers_fc_reduced.caffemodel?dl=1"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name}...")
    urllib.request.urlretrieve(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TextBoxes inference assets for OpenCV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Asset directory (default: data/models/textboxes).",
    )
    parser.add_argument(
        "--include-vgg",
        action="store_true",
        help="Also download VGG initialization weights used for training.",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    download(DEPLOY_PROTOTXT_URL, output / "deploy.prototxt")
    download(ICDAR_WEIGHTS_URL, output / "TextBoxes_icdar13.caffemodel")

    if args.include_vgg:
        download(
            VGG_WEIGHTS_URL,
            output / "VGG_ILSVRC_16_layers_fc_reduced.caffemodel",
        )

    print(f"TextBoxes assets saved to: {output}")
    print("OpenCV DNN inference does not require Caffe or accessmath-textboxes.")


if __name__ == "__main__":
    main()
