from ultralytics import YOLO

def main():

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="data.yaml",   # dataset config
        epochs=25,          # number of epochs
        imgsz=640,          # image size
        batch=8,            # batch size
        device="0",       # change to 0 if you have GPU
        workers=4,
        name="pcb_defect_model"
    )

if __name__ == "__main__":
    main()