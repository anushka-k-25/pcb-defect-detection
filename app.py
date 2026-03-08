import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title=" AI PCB Inspector", layout="wide", page_icon="🛠️")

# -------------------------------
# Load YOLO Model
# -------------------------------
model = YOLO(r"runs\detect\pcb_defect_model\weights\best.pt")

# -------------------------------
# Session State
# -------------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "home"
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

# -------------------------------
# CSS Styling - PCB / Electronic Theme
# -------------------------------
st.markdown("""
<style>
/* Background resembling PCB board */
body {
    background-color: #0d0d0d;
    background-image: 
        repeating-linear-gradient(0deg, #0d0d0d, #0d0d0d 20px, #111 20px, #111 21px),
        repeating-linear-gradient(90deg, #0d0d0d, #0d0d0d 20px, #111 20px, #111 21px);
    color:white;
    font-family:'Courier New', monospace;
}

/* Animated glowing header like PCB lights */
h1 {
    font-size: 80px;
    color: #00ffff;
    text-align:center;
    text-shadow: 0 0 10px #0ff, 0 0 20px #0ff, 0 0 30px #0ff;
    animation: glow 1.5s infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 5px #0ff, 0 0 10px #0ff, 0 0 20px #0ff; }
    to { text-shadow: 0 0 20px #0ff, 0 0 30px #0ff, 0 0 50px #0ff; }
}

h2 {
    font-size: 28px;
    text-align:center;
    color:#0ff;
    text-shadow: 0 0 5px #0ff;
}

/* Card style - PCB chip like */
.card {
    background: rgba(0,255,255,0.05);
    border-radius:20px;
    padding:20px;
    backdrop-filter: blur(6px);
    box-shadow: 0 0 15px #0ff, 0 0 30px #0ff inset;
    margin-bottom:20px;
}

/* Buttons - glowing neon LEDs */
.big-button button{
    font-size:22px;
    padding:20px 40px;
    margin:10px;
    border-radius:15px;
    background: linear-gradient(90deg,#00fff7,#0ff);
    color:black;
    font-weight:bold;
    box-shadow:0 0 20px #0ff, 0 0 30px #0ff inset;
    transition: 0.3s;
}
.big-button button:hover {
    box-shadow: 0 0 30px #0ff, 0 0 60px #0ff inset;
    transform: scale(1.05);
}

/* Footer styling */
footer {
    color:#0ff;
    text-align:center;
    font-size:14px;
    margin-top:50px;
}

/* Image captions like digital screens */
.stImage>figcaption {
    color:#0ff;
    text-shadow: 0 0 5px #0ff;
}

/* Custom scrollbar for PCB feel */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #111;
}
::-webkit-scrollbar-thumb {
    background: #0ff;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar - Mode Selection
# -------------------------------
with st.sidebar:
    st.title(" Controls")
    mode = st.radio("Choose Mode:", ["Home", "Upload Image", "Live Webcam"])
    st.session_state.mode = mode

# -------------------------------
# Function: Generate Defect Heatmap
# -------------------------------
def generate_heatmap(image, boxes):
    """Creates heatmap from YOLO bounding boxes and overlays on image."""
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            heatmap[y1:y2, x1:x2] += float(box.conf)
        # Normalize heatmap
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.7, heatmap_color, 0.3, 0)
        return cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)
    else:
        return image

# -------------------------------
# HOME PAGE
# -------------------------------
if st.session_state.mode == "Home":
    st.markdown("<h1>AI PCB Inspector</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Deep Learning Automated PCB Defect Detection</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("###  Project Overview")
    st.info(
        """
        Our system automatically detects PCB defects using **YOLO deep learning model**.  
        **Features:**  
        - Real-time detection with webcam  
        - Upload PCB images and get defect summary  
        - Futuristic heatmap highlighting defects
        """
    )

# -------------------------------
# UPLOAD IMAGE MODE
# -------------------------------
elif st.session_state.mode == "Upload Image":
    st.markdown("<h1> Upload PCB Image</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop PCB image", type=["png","jpg","jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original PCB Image")
            st.image(image, use_column_width=True)

        if st.button("🔍 Detect Defects"):
            results = model(np.array(image))
            output = results[0].plot()
            boxes = results[0].boxes

            # Generate heatmap overlay
            output_heatmap = generate_heatmap(np.array(image), boxes)

            with col2:
                st.subheader("Detection Output")
                st.image(output, caption="Bounding Boxes", use_column_width=True)
                st.image(output_heatmap, caption="Defect Heatmap", use_column_width=True)

                # Defect Summary
                if boxes is not None and len(boxes) > 0:
                    defect_data = []
                    for box in boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        defect_data.append({"Class": cls, "Confidence": conf})
                    df = pd.DataFrame(defect_data)
                    st.success(f"Total Defects Detected: {len(boxes)}")
                    st.bar_chart(df.set_index("Class"))
                else:
                    st.info("No defects detected!")

# -------------------------------
# LIVE WEBCAM MODE
# -------------------------------
elif st.session_state.mode == "Live Webcam":
    st.markdown("<h1>📹 Real-Time PCB Detection</h1>", unsafe_allow_html=True)
    FRAME = st.image([])

    start_btn = st.button("Start Webcam")
    stop_btn = st.button("Stop Webcam")

    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            boxes = results[0].boxes

            # Heatmap overlay
            frame_out = generate_heatmap(frame_rgb, boxes)

            # Live defect counter overlay
            num_defects = len(boxes) if boxes is not None else 0
            cv2.putText(frame_out, f"Defects: {num_defects}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
            FRAME.image(frame_out)
            time.sleep(0.03)
        cap.release()

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("""
### 🤖 AI Model
**YOLOv8-based model** trained on PCB defect dataset.  
Detects: Missing holes, Shorts, Open circuits, Spurious copper.  

**High-tech Heatmap Overlay** highlights defective areas for instant visual inspection.
""")