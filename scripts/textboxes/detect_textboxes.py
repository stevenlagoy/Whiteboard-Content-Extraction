# scripts/detect_textboxes.py

import argparse
import csv
import json
from pathlib import Path

import cv2


def create_detector(prototxt, weights):
    """Load a compatible OpenCV TextBoxes CNN model."""

    if not hasattr(cv2, "text"):
        raise RuntimeError(
            "OpenCV text module is unavailable. "
            "Install a compatible opencv-contrib-python build."
        )

    return cv2.text.TextDetectorCNN_create(
        str(prototxt),
        str(weights)
    )


def detect_image(detector, image):
    """Return detected bounding boxes and confidence scores."""

    boxes, confidences = detector.detect(image)

    results = []

    for box, confidence in zip(boxes, confidences):
        x, y, width, height = map(int, box)

        results.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "confidence": float(confidence)
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect text regions in extracted images."
    )

    parser.add_argument("--frames", required=True)
    parser.add_argument("--manifest", required=True)
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "data" / "models" / "textboxes"
    parser.add_argument(
        "--prototxt",
        default=str(model_dir / "deploy.prototxt"),
    )
    parser.add_argument(
        "--weights",
        default=str(model_dir / "TextBoxes_icdar13.caffemodel"),
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    frames_dir = Path(args.frames)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector = create_detector(
        args.prototxt,
        args.weights
    )

    detections = []

    with open(manifest_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_path = frames_dir / row["image"]

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            boxes = detect_image(detector, image)

            image_height, image_width = image.shape[:2]

            for index, box in enumerate(boxes):
                x = max(0, box["x"])
                y = max(0, box["y"])

                x2 = min(
                    image_width,
                    box["x"] + box["width"]
                )

                y2 = min(
                    image_height,
                    box["y"] + box["height"]
                )

                if x2 <= x or y2 <= y:
                    continue

                crop = image[y:y2, x:x2]

                crop_name = (
                    f"{image_path.stem}_crop_{index:03d}.jpg"
                )

                crop_path = crops_dir / crop_name

                cv2.imwrite(str(crop_path), crop)

                detections.append({
                    "source_image": row["image"],
                    "timestamp": (
                        float(row["timestamp"])
                        if row["timestamp"] else None
                    ),
                    "source_type": row["source_type"],
                    "crop": crop_name,
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": x2 - x,
                        "height": y2 - y
                    },
                    "confidence": box["confidence"]
                })

            print(
                f"Processed {row['image']}: "
                f"{len(boxes)} detections"
            )

    results_path = output_dir / "detections.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(detections, file, indent=4)

    print(f"Total detections: {len(detections)}")
    print(f"Crops saved to: {crops_dir}")
    print(f"Detection metadata saved to: {results_path}")


if __name__ == "__main__":
    main()