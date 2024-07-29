import cv2
import os

import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(f'best.pt')

cap = cv2.VideoCapture(0)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

if not cap.isOpened():
    print("unable to read camera feed")

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    cv2.imshow('webcam', annotated_image)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
