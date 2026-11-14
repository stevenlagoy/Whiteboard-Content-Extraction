# scripts/extract_frames.py

import argparse
import csv
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_video_frames(video_path, output_dir, fps):
    """Extract frames from a video at the requested sampling rate."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not video_fps or video_fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    interval = max(1, round(video_fps / fps))

    rows = []
    frame_number = 0
    saved_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % interval == 0:
            timestamp = frame_number / video_fps

            filename = f"frame_{saved_number:06d}.jpg"
            output_path = output_dir / filename

            cv2.imwrite(str(output_path), frame)

            rows.append({
                "image": filename,
                "timestamp": timestamp,
                "source_type": "video"
            })

            saved_number += 1

        frame_number += 1

    cap.release()

    return rows


def load_image_directory(input_dir, output_dir):
    """Copy supported images into the processed image directory."""

    rows = []

    images = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No supported images found in {input_dir}")

    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        filename = f"image_{index:06d}{image_path.suffix.lower()}"

        output_path = output_dir / filename

        shutil.copy2(image_path, output_path)

        rows.append({
            "image": filename,
            "timestamp": "",
            "source_type": "image"
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames or prepare an image directory."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Video file or directory containing images."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory for processed images and manifest."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Sampling rate for videos. Default: 1 FPS."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        print("Input type: image directory")

        rows = load_image_directory(
            input_path,
            output_dir
        )

    elif input_path.suffix.lower() in VIDEO_EXTENSIONS:
        print("Input type: video")

        rows = extract_video_frames(
            input_path,
            output_dir,
            args.fps
        )

    else:
        raise ValueError(
            "Input must be a supported video file or image directory."
        )

    manifest_path = output_dir / "frames.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "timestamp", "source_type"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} images.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()