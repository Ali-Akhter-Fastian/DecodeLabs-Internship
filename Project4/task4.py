"""
Path 2: Object Detection with MobileNet-SSD
Pipeline:
    raw image -> blob construction (cv2.dnn.blobFromImage) -> MobileNet-SSD
    forward pass -> softmax confidence scores -> 80% confidence gate
    -> bounding box overlay
"""

import cv2
import numpy as np

IMAGEPATH = "task4/sampleimage.jpg"
PROTOTXT_PATH = "task4/MobileNetSSD_deploy.prototxt"
MODELPATH = "task4/MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.80         # gatekeeper minimum 
OUTPUTPATH = "task4/detectionoutput.png"

# MobileNet-SSD was trained on the Pascal VOC 20-class label dataset
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

np.random.seed(42)
BOX_COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3))


def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODELPATH)
    return net


def build_blob(image):
    """
    Step 1: Blob Construction.
    Performs mean subtraction and scales the image to the 300x300
    """
    
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=0.007843,       
        size=(300, 300),
        mean=127.5
    )
    return blob


def detect_objects(net, image):
    """Here, Runs the forward pass and gives us the raw detection."""
    (h, w) = image.shape[:2]
    blob = build_blob(image)
    net.setInput(blob)
    detections = net.forward()
    return detections, h, w


def filter_and_draw(image, detections, h, w, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Aplies the 80% confidence gate (the softmax output for each detection).
    """
    outputimage = image.copy()
    found_objects = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            label = CLASSES[class_id] if class_id < len(CLASSES) else "unknown"

            # normalized coordinate -> actual pixel coordinate
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            color = [int(c) for c in BOX_COLORS[class_id]]
            cv2.rectangle(outputimage, (x1, y1), (x2, y2), color, 2)
            text = f"{label}: {confidence * 100:.1f}%"
            y_label = y1 - 10 if y1 - 10 > 10 else y1 + 15
            cv2.putText(outputimage, text, (x1, y_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            found_objects.append((label, confidence))
        ## else: below the confidence gate will be dropped

    return outputimage, found_objects


def main():
    print("Loading MobileNet SSD (pre-trained, transfer learning)...")
    net = load_model()

    print(f"Loading image: {IMAGEPATH}")
    image = cv2.imread(IMAGEPATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image from path: {IMAGEPATH}")

    print("Detecting objects...")
    detections, h, w = detect_objects(net, image)

    outputimage, found_objects = filter_and_draw(image, detections, h, w)

    print(f"\nAll the Objects passing the {int(CONFIDENCE_THRESHOLD * 100)}% confidence gate:")
    if not found_objects:
        print("  (none met the threshold)")
    for label, confidence in found_objects:
        print(f"  {label:<15} confidence: {confidence * 100:.1f}%")

    cv2.imwrite(OUTPUTPATH, outputimage)
    print(f"\nAnnotated output saved to: {OUTPUTPATH}")


if __name__ == "__main__":
    main()
