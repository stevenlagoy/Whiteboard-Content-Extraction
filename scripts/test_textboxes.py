
import caffe
import cv2
import numpy as np

MODEL = "examples/TextBoxes/deploy.prototxt"
WEIGHTS = "examples/TextBoxes/TextBoxes_icdar13.caffemodel"
IMAGE_PATH = "/mnt/c/Users/donof/IdeaProjects/Whiteboard-Content-Extraction/data/raw/ICDAR/test/img_9.jpg"

# CPU inference
caffe.set_mode_cpu()

net = caffe.Net(
    MODEL,
    caffe.TEST,
    weights=WEIGHTS
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

cv2.imwrite("textboxes_output_9.jpg", image)

print("Detection complete!")
print("Saved visualization to textboxes_output.jpg")
