# PCB Defect Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-yellowgreen)](https://github.com/ultralytics/ultralytics)

---

# Project Overview

PCB Defect Detection is an **AI-powered system** designed to automatically detect defects in **Printed Circuit Boards (PCBs)** using deep learning.

The system allows users to:

* Upload PCB images for defect detection
* Use a live webcam (PC only) for real-time detection

This project was developed for hackathon purposes, emphasizing:

* A high-tech professional interface
* Real-time AI detection
* Easy deployment and usage

---

# Features

* High-tech UI with PCB-themed design
* Image Upload for detecting defects on any device
* Live Webcam Mode (PC only) for real-time detection
* YOLOv8 deep learning model trained on PCB defects
* Automatic model download from Google Drive
* Cross-platform support (PC and mobile upload)

---

# Folder Structure

```
pcb-defect-detection/
│
├── app.py                  # Main Streamlit application
├── train.py                # YOLOv8 model training script
├── detect.py               # YOLOv8 detection script
├── realtime_defect.py      # Real-time defect detection using webcam
├── requirements.txt        # Python dependencies
│
├── dataset/                # PCB dataset
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── labels/
│       ├── train/
│       └── val/
│
├── runs/                   # Trained model checkpoints and results
└── pcb_env/                # Virtual environment (ignored in repo)
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/anushka-k-25/pcb-defect-detection.git
cd pcb-defect-detection
```

---

## 2. Create and Activate a Virtual Environment

```bash
python -m venv pcb_env
```

### Linux / macOS

```bash
source pcb_env/bin/activate
```

### Windows

```bash
pcb_env\Scripts\activate.bat
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

### Usage

**Upload Image**

* Works on both PC and mobile
* Upload a PCB image to detect defects

**Live Camera**

* Works on PC only
* Detect PCB defects in real time

---

# Streamlit Cloud Deployment

1. Push the code and `requirements.txt` to GitHub
2. Connect the repository to Streamlit Cloud
3. Deploy the application

The app automatically downloads the YOLO model from Google Drive if it is not present locally.

---

# Model

The system uses a **YOLOv8 object detection model** trained on a PCB defect dataset.

If the model is not available locally, it is downloaded automatically from Google Drive.

```
MODEL_URL = "https://drive.google.com/uc?id=1rTcm45pK3nfKaj94ERMueumaQmZtVqKH"
```

---

# Dependencies

Main libraries used in this project:

* Python 3.10+
* Streamlit
* OpenCV (opencv-python-headless)
* Ultralytics YOLOv8
* Pillow
* NumPy
* Plotly
* Pandas
* gdown (for model download)

---

# Future Improvements

* Improve model accuracy with larger datasets
* Add mobile live camera support
* Deploy using Docker
* Add dashboard analytics for defect tracking

---

# License

This project was created for educational and hackathon purposes.

---
