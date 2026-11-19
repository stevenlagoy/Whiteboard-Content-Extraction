
import cv2
import numpy as np
import argparse

from pathlib import Path


def main():
    try:
        import caffe
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Caffe Python bindings are not installed in the active "
            "Python environment."
        ) from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="relative path to the image you want to detect text from")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    TEXTBOXES_ROOT = PROJECT_ROOT / "accessmath-textboxes"

    MODEL = TEXTBOXES_ROOT / "examples/TextBoxes/deploy.prototxt"

    WEIGHTS = (
            TEXTBOXES_ROOT
            / "examples/TextBoxes/TextBoxes_icdar13.caffemodel"
    )

    IMAGE_PATH = args.image_path

    # CPU inference
    caffe.set_mode_cpu()

    net = caffe.Net(
        str(MODEL),
        caffe.TEST,
        weights=str(WEIGHTS)
    )

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    height, width = image.shape[:2]

    # TextBoxes network expects a 700 x 700 input
    resized = cv2.resize(image, (700, 700))

    # BGR image -> Caffe input format
    blob = resized.astype(np.float32)
    blob -= np.array([104, 117, 123], dtype=np.float32)
    blob = blob.transpose((2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    net.blobs["data"].reshape(*blob.shape)
    net.blobs["data"].data[...] = blob

    output = net.forward()
    detections = output["detection_out"][0, 0]

    for detection in detections:
        confidence = float(detection[2])

        if confidence < 0.5:
            continue

        xmin, ymin, xmax, ymax = detection[3:7]

        # Convert normalized coordinates to original image coordinates
        x1 = int(xmin * width)
        y1 = int(ymin * height)
        x2 = int(xmax * width)
        y2 = int(ymax * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            f"{confidence:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(f"{IMAGE_PATH}_output.jpg", image)

    print("Detection complete!")
    print("Saved visualization to textboxes_output.jpg")


if __name__ == "__main__":
    main()