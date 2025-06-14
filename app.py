
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

# === Custom Styling ===
st.markdown("""
<style>
h1 {
    color: #2c3e50;
}
.section-title {
    background-color: #3498db;
    padding: 10px;
    border-radius: 5px;
    color: white;
    font-size: 20px;
    font-weight: bold;
}
.table-style {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# === Konfigurasi ===
os.makedirs("models", exist_ok=True)
vit_path = "models/vit_cnn_daging.pt"
vit_gdrive_id = "1zdTPq9sN3DmSkkRBnRqSD_SCfq-sOTvu"
vit_gdrive_url = f"https://drive.google.com/uc?id={vit_gdrive_id}"

def download_vit_model():
    if not os.path.exists(vit_path):
        with st.spinner("⏳ Mengunduh model ViT dari Google Drive..."):
            gdown.download(vit_gdrive_url, vit_path, quiet=False)
            st.success("✅ Model ViT berhasil diunduh!")

download_vit_model()

def rebuild_vit_model():
    model = models.vit_b_16(weights=None)
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 3)
    )
    return model

@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov11_daging.pt")
    vit_model = rebuild_vit_model()
    vit_model.load_state_dict(torch.load(vit_path, map_location="cpu"))
    vit_model.eval()
    return yolo_model, vit_model

yolo_model, vit_model = load_models()

@st.cache_resource
def load_rf_model_and_scaler():
    rf_model = joblib.load("sensor_rf_model.joblib")
    scaler = joblib.load("sensor_rf_scaler.joblib")
    return rf_model, scaler

rf_model, rf_scaler = load_rf_model_and_scaler()

class_names = ['Busuk', 'Sedang', 'Segar']
label_map = {0: "Layak Konsumsi", 1: "Perlu Diperiksa", 2: "Tidak Layak"}
status_color = {
    "Layak Konsumsi": "green",
    "Perlu Diperiksa": "orange",
    "Tidak Layak": "red"
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
        pred = torch.argmax(outputs, 1).item()
    return class_names[pred]

def get_latest_sensor_values():
    url = "https://docs.google.com/spreadsheets/d/1Qs058JpuvYJSBhH16tp7zWabTad6zZSOhleSBIkobpM/export?format=csv"
    df = pd.read_csv(url)
    latest_row = df.iloc[-1]
    return float(latest_row['MQ136']), float(latest_row['MQ137'])

# === Antarmuka Streamlit ===
st.title("🥩 Deteksi Kualitas Daging: YOLOv11 + ViT CNN + IoT Multimodal")

option = st.radio("Pilih metode input:", ["📸 Kamera", "📁 Upload Gambar"])

img = None
if option == "📸 Kamera":
    img_file = st.camera_input("Ambil Gambar Daging")
    if img_file:
        img = Image.open(img_file)
elif option == "📁 Upload Gambar":
    img_file = st.file_uploader("Upload gambar daging", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file)

if img:
    st.image(img, caption="🖼️ Gambar Input", use_column_width=True)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tfile.name)

    results = yolo_model(tfile.name)
    boxes = results[0].boxes

    st.subheader("📍 Deteksi dan Klasifikasi")
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        np_img = np.array(img)
        crop = np_img[y1:y2, x1:x2]

        pred_visual = predict_from_crop(crop)
        visual_encoded = encode_visual(pred_visual)

        mq136, mq137 = get_latest_sensor_values()
        sensor_input = np.array([[mq136, mq137]])
        sensor_scaled = rf_scaler.transform(sensor_input)

        status_pred = rf_model.predict(sensor_scaled)[0]
        status_text = label_map[status_pred]
        warna = status_color[status_text]

        st.image(crop, caption=f"Prediksi Visual: **{pred_visual}** (Conf: {conf:.2f})", width=300)
        st.markdown(f"### ✅ Keputusan Akhir: <span style='color:{warna}'>{status_text}</span>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Visual": [pred_visual],
            "MQ136": [mq136],
            "MQ137": [mq137],
            "Status Akhir": [status_text]
        }), use_container_width=True)

# === Footer ===
st.markdown("---")
st.markdown("<center>© 2025 Universitas Hang Tuah Pekanbaru | Smart Meat Quality Detection</center>", unsafe_allow_html=True)
