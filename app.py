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
        with st.spinner("\u23f3 Mengunduh model ViT dari Google Drive..."):
            gdown.download(vit_gdrive_url, vit_path, quiet=False)
            st.success("\u2705 Model ViT berhasil diunduh!")

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
        conf = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item()
    return class_names[pred], conf

# === Ambil data sensor dari Google Sheets ===
def get_latest_sensor_values():
    url = "https://docs.google.com/spreadsheets/d/1Qs058JpuvYJSBhH16tp7zWabTad6zZSOhleSBIkobpM/export?format=csv"
    df = pd.read_csv(url)
    latest_row = df.iloc[-1]
    return float(latest_row['MQ136']), float(latest_row['MQ137'])

# === UI Streamlit ===
st.title("\ud83e\udd69 Deteksi Kualitas Daging: YOLOv11 + ViT CNN + IoT Multimodal")

option = st.radio("Pilih metode input:", ["\ud83d\udcf8 Kamera", "\ud83d\udcc1 Upload Gambar"])

img = None
if option == "\ud83d\udcf8 Kamera":
    if "camera_img" not in st.session_state:
        st.session_state["camera_img"] = None

    img_file = st.camera_input("Ambil Gambar Daging")
    if img_file and img_file != st.session_state["camera_img"]:
        st.session_state["camera_img"] = img_file
        st.rerun()

    if st.session_state["camera_img"]:
        img = Image.open(st.session_state["camera_img"])

elif option == "\ud83d\udcc1 Upload Gambar":
    img_file = st.file_uploader("Upload gambar daging", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file)

if img:
    st.image(img, caption="\ud83d\udcf8 Gambar Input", use_column_width=True)

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tfile.name)

    results = yolo_model(tfile.name)
    boxes = results[0].boxes

    st.subheader("\ud83d\udccd Deteksi dan Klasifikasi")
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        np_img = np.array(img)
        crop = np_img[y1:y2, x1:x2]

        pred_visual, visual_conf = predict_from_crop(crop)
        visual_encoded = encode_visual(pred_visual)

        mq136, mq137 = get_latest_sensor_values()
        sensor_input = np.array([[mq136, mq137]])
        sensor_scaled = rf_scaler.transform(sensor_input)

        status_pred = rf_model.predict(sensor_scaled)[0]
        status_text = label_map[status_pred]

        st.image(crop, caption=f"Prediksi Visual: *{pred_visual}* (Conf: {visual_conf:.2f})", width=300)
        st.markdown("### \u2705 Keputusan Akhir")
        st.table({
            "Visual": [pred_visual],
            "MQ136": [mq136],
            "MQ137": [mq137],
            "Status Akhir": [status_text]
        })
