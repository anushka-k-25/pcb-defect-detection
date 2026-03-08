from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/pcb_defect_model/weights/best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("PCB Defect Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()