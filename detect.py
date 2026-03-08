from ultralytics import YOLO

model = YOLO("runs/detect/pcb_defect_model/weights/best.pt")

results = model.predict(
    source="dataset/images/val/12100003_test.jpg",
    conf=0.25,
    show=True,
    save=True
)