# app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms, models

# === Load Models ===
@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov11_daging.pt")
    vit_model = models.vit_b_16(weights=None)
    vit_model.heads = torch.nn.Sequential(
        torch.nn.Linear(vit_model.heads.head.in_features, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 3)  # Busuk, Sedang, Segar
    )
    vit_model.load_state_dict(torch.load("vit_cnn_daging.pt", map_location="cpu"))
    vit_model.eval()
    return yolo_model, vit_model

yolo_model, vit_model = load_models()

# === Kelas ===
class_names = ["Busuk", "Sedang", "Segar"]

# === Transformasi untuk ViT ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Fungsi Klasifikasi Crop ===
def classify_crop(image_pil):
    input_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        output = vit_model(input_tensor)
        pred = torch.argmax(output, 1).item()
    return class_names[pred]

# === Fungsi Deteksi & Crop dengan YOLO ===
def detect_and_crop(image):
    results = yolo_model(image)
    crops = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            crops.append((crop, (x1, y1, x2, y2)))
    return crops

# === UI Streamlit ===
st.title("Deteksi Kesegaran Daging - YOLO + ViT")

uploaded = st.file_uploader("Upload Gambar Daging", type=["jpg", "jpeg", "png"])
use_camera = st.checkbox("Gunakan Kamera")

if uploaded or use_camera:
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    elif use_camera:
        image = st.camera_input("Ambil Foto Daging")
        if image is not None:
            image = Image.open(image)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            st.stop()

    st.image(image, caption="Gambar Masukan", use_column_width=True)

    # Proses YOLO
    st.subheader("Hasil Deteksi dan Klasifikasi")
    crops = detect_and_crop(image)

    for idx, (crop, (x1, y1, x2, y2)) in enumerate(crops):
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pred_label = classify_crop(crop_pil)
        st.image(crop_pil, caption=f"Region {idx+1} - {pred_label}", width=300)
else:
    st.info("Silakan upload gambar atau gunakan kamera.")
