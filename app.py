import streamlit as st
st.set_page_config(page_title="Deteksi Kesegaran Daging", layout="wide")

import torch
import torchvision.transforms as transforms
import numpy as np
import tempfile
import os
import gdown
import pandas as pd
import joblib
from PIL import Image
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn

# === Inisialisasi Session State ===
if "processed" not in st.session_state:
    st.session_state.processed = False
if "image" not in st.session_state:
    st.session_state.image = None
if "detection_results" not in st.session_state: # Add new session state for results
    st.session_state.detection_results = None
if "uploaded_file" not in st.session_state: # To handle uploaded file state
    st.session_state.uploaded_file = None


# === Konfigurasi ===
os.makedirs("models", exist_ok=True)
vit_path = "models/vit_cnn_daging.pt"
vit_gdrive_id = "1zdTPq9sN3DmSkkRBnRqSD_SCfq-sOTvu"
vit_gdrive_url = f"https://drive.google.com/uc?id={vit_gdrive_id}"

# === Download model ViT jika belum ada ===
def download_vit_model():
    if not os.path.exists(vit_path):
        with st.spinner("Mengunduh model ViT dari Google Drive..."):
            gdown.download(vit_gdrive_url, vit_path, quiet=False)
            st.success("Model ViT berhasil diunduh!")

download_vit_model()

# === Fungsi Rekonstruksi ViT ===
def rebuild_vit_model():
    model = models.vit_b_16(weights=None)
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )
    return model

# === Load Models ===
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov11_daging.pt")
    vit_model = rebuild_vit_model()
    vit_model.load_state_dict(torch.load(vit_path, map_location="cpu"))
    vit_model.eval()
    return yolo_model, vit_model

yolo_model, vit_model = load_models()

# === Load RF Model & Scaler ===
@st.cache_resource
def load_rf_model_and_scaler():
    rf_model = joblib.load("sensor_rf_model.joblib")
    scaler = joblib.load("sensor_rf_scaler.joblib")
    return rf_model, scaler

rf_model, rf_scaler = load_rf_model_and_scaler()

# === Label dan Transformasi ===
class_names = ['Busuk', 'Sedang', 'Segar']
label_map = {0: "Layak Konsumsi", 1: "Perlu Diperiksa", 2: "Tidak Layak"}
label_colors = {
    "Layak Konsumsi": "#d4edda",
    "Perlu Diperiksa": "#fff3cd",
    "Tidak Layak": "#f8d7da"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def encode_visual(label):
    return {"Busuk": 0, "Sedang": 1, "Segar": 2}.get(label, -1)

def predict_from_crop(crop_img):
    image = Image.fromarray(crop_img)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = vit_model(tensor)
        pred_class = torch.argmax(outputs, 1).item()
        confidence = torch.softmax(outputs, dim=1)[0, pred_class].item()
    return class_names[pred_class], confidence

def get_latest_sensor_values():
    url = "https://docs.google.com/spreadsheets/d/1Qs058JpuvYJSBhH16tp7zWabTad6zZSOhleSBIkobpM/export?format=csv"
    df = pd.read_csv(url)
    latest_row = df.iloc[-1]
    return float(latest_row['MQ136']), float(latest_row['MQ137'])

# --- UI ---
st.markdown("<h1 style='color:#2c3e50;'>🔍 Pendeteksi Kualitas Daging </h1>", unsafe_allow_html=True)

option = st.radio("Pilih metode input:", ["Kamera", "Upload Gambar"])

img = None
if option == "Kamera":
    img_file = st.camera_input("Ambil Gambar Daging")
    if img_file and not st.session_state.processed:
        st.session_state.image = Image.open(img_file)
        st.session_state.processed = True
        st.session_state.detection_results = None # Clear previous results
    elif st.session_state.processed:
        img = st.session_state.image
elif option == "Upload Gambar":
    img_file = st.file_uploader("Upload gambar daging", type=["jpg", "jpeg", "png"])
    if img_file:
        # Check if a new file is uploaded or the same file is re-uploaded
        if st.session_state.uploaded_file != img_file.name:
            st.session_state.uploaded_file = img_file.name # Store file name
            st.session_state.image = Image.open(img_file)
            st.session_state.processed = True
            st.session_state.detection_results = None # Clear previous results
        img = st.session_state.image # Always display the current image

if st.session_state.image:
    img = st.session_state.image
    st.image(img, caption="📷 Gambar Input", use_column_width=True)

    # Process image only if results haven't been processed yet for the current image
    if st.session_state.detection_results is None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(tfile.name)

        results = yolo_model(tfile.name)
        boxes = results[0].boxes

        if not boxes:
            st.session_state.detection_results = "no_meat_found"
        else:
            # Store results in session state to avoid re-running predictions on rerun
            st.session_state.detection_results = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                np_img = np.array(img)
                crop = np_img[y1:y2, x1:x2]

                pred_visual, visual_conf = predict_from_crop(crop)
                
                mq136, mq137 = get_latest_sensor_values()
                sensor_input = np.array([[mq136, mq137]])
                sensor_scaled = rf_scaler.transform(sensor_input)
                status_pred = rf_model.predict(sensor_scaled)[0]
                status_text = label_map[status_pred]

                st.session_state.detection_results.append({
                    "crop_image": crop,
                    "pred_visual": pred_visual,
                    "visual_conf": visual_conf,
                    "mq136": mq136,
                    "mq137": mq137,
                    "status_text": status_text
                })
        os.unlink(tfile.name) # Clean up temp file

    # Display results from session state
    if st.session_state.detection_results == "no_meat_found":
        st.warning("Tidak ditemukan objek daging oleh YOLO.")
    elif st.session_state.detection_results:
        for result in st.session_state.detection_results:
            st.markdown("### ✅ Hasil Deteksi")
            st.markdown(
                f"""
                <div style='background-color:{label_colors[result['status_text']]};padding:10px;border-radius:8px;'>
                    <b>Visual:</b> {result['pred_visual']} |
                    <b>H₂S (MQ136):</b> {result['mq136']:.4f} |
                    <b>NH₃ (MQ137):</b> {result['mq137']:.4f} |
                    <b>Status Akhir:</b> {result['status_text']}
                </div>
                """, unsafe_allow_html=True
            )
            st.image(result['crop_image'], caption=f"🧠 Prediksi Visual: *{result['pred_visual']}* (Conf: {result['visual_conf']:.2f})", width=300)

# --- Tombol Reset ---
if st.button("🔄 Clear Foto"):
    st.session_state.processed = False
    st.session_state.image = None
    st.session_state.detection_results = None
    st.session_state.uploaded_file = None # Reset uploaded file state
    st.rerun()
