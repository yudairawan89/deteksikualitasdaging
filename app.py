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

# === Konfigurasi ===
os.makedirs("models", exist_ok=True)
vit_path = "models/vit_cnn_daging.pt"
vit_gdrive_id = "1zdTPq9sN3DmSkkRBnRqSD_SCfq-sOTvu"
vit_gdrive_url = f"https://drive.google.com/uc?id={vit_gdrive_id}"

# === Download model ViT jika belum ada ===
def download_vit_model():
    if not os.path.exists(vit_path):
        with st.spinner("⏳ Mengunduh model ViT dari Google Drive..."):
            gdown.download(vit_gdrive_url, vit_path, quiet=False)
            st.success("✅ Model ViT berhasil diunduh!")

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
    yolo_model = YOLO("yolov11_daging.pt")  # file harus sudah ada di folder
    vit_model = rebuild_vit_model()
    vit_model.load_state_dict(torch.load(vit_path, map_location="cpu"))
    vit_model.eval()
    return yolo_model, vit_model

yolo_model, vit_model = load_models()

# === Load RF Multimodal Model ===
@st.cache_resource
def load_rf_multimodal_model():
    return joblib.load("sensor_rf_model.joblib")

rf_model = load_rf_multimodal_model()

# === Label dan Transformasi ===
class_names = ['Busuk', 'Sedang', 'Segar']
label_map = {0: "Layak Konsumsi", 1: "Perlu Diperiksa", 2: "Tidak Layak"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def encode_visual(label):
    return {"Busuk": 0, "Sedang": 1, "Segar": 2}.get(label, -1)

# === Prediksi dari Crop Bounding Box (Visual Only) ===
def predict_from_crop(crop_img):
    image = Image.fromarray(crop_img)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = vit_model(tensor)
        pred = torch.argmax(outputs, 1).item()
    return class_names[pred]

# === Ambil data sensor dari Google Sheets ===
def get_latest_sensor_values():
    url = "https://docs.google.com/spreadsheets/d/1Qs058JpuvYJSBhH16tp7zWabTad6zZSOhleSBIkobpM/export?format=csv"
    df = pd.read_csv(url)
    latest_row = df.iloc[-1]
    return float(latest_row['MQ136']), float(latest_row['MQ137'])

# === UI Streamlit ===
st.title("🥩 Deteksi Kualitas Daging: YOLOv11-ViT CNN + IoT Multimodal Sensor Fusion")

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

    # Simpan sementara
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tfile.name)

    # === YOLO Detection ===
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

        # Ambil sensor dari Google Sheet
        mq136, mq137 = get_latest_sensor_values()

        # Prediksi akhir berdasarkan visual + sensor
        multimodal_input = pd.DataFrame([[visual_encoded, mq136, mq137]], columns=["Visual", "MQ136", "MQ137"])
        status_pred = rf_model.predict(multimodal_input)[0]
        status_text = label_map[status_pred]

        # Tampilkan hasil
        st.image(crop, caption=f"Prediksi Visual: **{pred_visual}** (Conf: {conf:.2f})", width=300)
        st.markdown("### ✅ Keputusan Akhir")
        st.table({
            "Visual": [pred_visual],
            "MQ136": [mq136],
            "MQ137": [mq137],
            "Status Akhir": [status_text]
        })
