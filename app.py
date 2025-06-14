import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Fire Prediction RHSEM – IoT", layout="wide")

# === AUTO REFRESH ===
st_autorefresh(interval=3000, key="data_refresh")

# === STYLE KUSTOM ===
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    table {width: 100%; border-collapse: collapse;}
    th, td {border: 1px solid #ddd; padding: 8px;}
    th {background-color: #e0e0e0; text-align: center;}
    td {text-align: center;}
    .section-title {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .scrollable-table {
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

# === FUNGSI BANTUAN ===
def convert_day_to_indonesian(day_name):
    return {'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
            'Sunday': 'Minggu'}.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
            'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
            'November': 'November', 'December': 'Desember'}.get(month_name, month_name)

def convert_to_label(pred):
    return {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}.get(pred, "Unknown")

risk_styles = {
    "Low": ("white", "blue"),
    "Moderate": ("white", "green"),
    "High": ("black", "yellow"),
    "Very High": ("white", "red")
}

# === LOAD MODEL DAN SCALER ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === LOAD DATA TANPA CACHE ===
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

st.cache_data.clear()
df = load_data()


# Tambahkan garis horizontal di atas header
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)


# === HEADER ===
# === HEADER ===
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=170)
with col2:
    st.markdown("""
        <div style='margin-left: 20px;'>
            <h2 style='margin-bottom: 0px;'>Smart Fire Prediction RHSEM – IoT Model</h2>
            <p style='font-size: 16px; line-height: 1.5; margin-top: 8px;'>
                Sistem ini menggunakan Rotational Hybrid Stacking Ensemble Method (RHSEM) untuk memprediksi risiko kebakaran hutan secara real-time dengan tingkat akurasi tinggi.
                Model prediksi dikembangkan dari kombinasi berbagai algoritma pembelajaran mesin yang dioptimalkan menggunakan optimasi hyperparameter untuk meningkatkan performa klasifikasi.
                Data pengujian secara real-time berasal dari perangkat IoT yang mengukur parameter lingkungan seperti suhu, kelembapan, curah hujan, kecepatan angin, dan kelembapan tanah.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_btn = st.columns([10, 1])[1]  # Geser tombol ke kanan
    with col_btn:
        st.markdown(
            "<a href='https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?gid=0#gid=0' target='_blank'>"
            "<button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer;'>Data Cloud</button>"
            "</a>",
            unsafe_allow_html=True
        )


# Tambahkan garis horizontal di bawah header
st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# === PREDIKSI REALTIME ===
st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)

if df is not None and not df.empty:
    df = df.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Permukaan Tanah',
        'Waktu': 'Waktu'
    })

    fitur = [
        'Tavg: Temperatur rata-rata (°C)',
        'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)',
        'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
    ]

    # Prediksi semua data
    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = clean_df[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

    scaled_all = scaler.transform(clean_df)
    predictions = [convert_to_label(p) for p in model.predict(scaled_all)]
    df["Prediksi Kebakaran"] = predictions

    last_row = df.iloc[-1]
    waktu = pd.to_datetime(last_row['Waktu'])
    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))
    tanggal = waktu.strftime(f'%d {bulan} %Y')
    risk_label = last_row["Prediksi Kebakaran"]
    font, bg = risk_styles.get(risk_label, ("black", "white"))

    # TABEL SENSOR
    sensor_df = pd.DataFrame({
        "Variabel": fitur,
        "Value": [f"{last_row[col]:.1f}" for col in fitur]
    })
    st.write("Data Sensor Realtime:")
    st.table(sensor_df)

    # KOTAK PREDIKSI
    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
        f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
        f"<span style='text-decoration: underline; font-size: 22px;'>{risk_label}</span></p>",
        unsafe_allow_html=True
    )

# === TABEL TINGKAT RISIKO ===
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
st.markdown("""
<div class="scrollable-table" style="margin-bottom: 25px;">
<table style='width: 100%; border-collapse: collapse;'>
    <thead>
        <tr>
            <th style='background-color:#e0e0e0;'>Warna</th>
            <th style='background-color:#e0e0e0;'>Tingkat Resiko / Intensitas</th>
            <th style='background-color:#e0e0e0;'>Keterangan</th>
        </tr>
    </thead>
    <tbody>
        <tr style='background-color:blue; color:white;'>
            <td>Blue</td><td>Low</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran rendah. Intensitas api pada kategori rendah. Api mudah dikendalikan, cenderung akan padam dengan sendirinya.</td>
        </tr>
        <tr style='background-color:green; color:white;'>
            <td>Green</td><td>Moderate</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sedang. Intensitas api pada kategori sedang. Api relatif masih cukup mudah dikendalikan.</td>
        </tr>
        <tr style='background-color:yellow; color:black;'>
            <td>Yellow</td><td>High</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran tinggi. Intensitas api pada kategori tinggi. Api sulit dikendalikan.</td>
        </tr>
        <tr style='background-color:red; color:white;'>
            <td>Red</td><td>Very High</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sangat tinggi. Intensitas api pada kategori sangat tinggi. Api sangat sulit dikendalikan.</td>
        </tr>
    </tbody>
</table>
</div>
""", unsafe_allow_html=True)




# === TABEL DATA SENSOR ===
st.markdown("<div class='section-title' style='margin-top: 30px;'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
st.dataframe(df[['Waktu'] + fitur + ['Prediksi Kebakaran']], use_container_width=True)




# === EXPORT TO XLSX ===
output = BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='DataSensor')
xlsx_data = output.getvalue()

st.download_button(
    label="📥 Download Hasil Prediksi Kebakaran sebagai XLSX",
    data=xlsx_data,
    file_name='hasil_prediksi_kebakaran.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)





# === PREDIKSI MANUAL ===
st.markdown("<div class='section-title'>Pengujian Menggunakan Data Manual</div>", unsafe_allow_html=True)

# Inisialisasi default
if "manual_input" not in st.session_state:
    st.session_state.manual_input = {
        "suhu": 30.0,
        "kelembapan": 65.0,
        "curah": 10.0,
        "angin": 3.0,
        "tanah": 50.0
    }
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

# Input manual dari state
col1, col2, col3 = st.columns(3)
with col1:
    suhu = st.number_input("Suhu Udara (°C)", value=st.session_state.manual_input["suhu"], key="suhu_input")
    kelembapan = st.number_input("Kelembapan Udara (%)", value=st.session_state.manual_input["kelembapan"], key="kelembapan_input")
with col2:
    curah = st.number_input("Curah Hujan (mm)", value=st.session_state.manual_input["curah"], key="curah_input")
    angin = st.number_input("Kecepatan Angin (m/s)", value=st.session_state.manual_input["angin"], key="angin_input")
with col3:
    tanah = st.number_input("Kelembaban Tanah (%)", value=st.session_state.manual_input["tanah"], key="tanah_input")

# Tombol aksi berdempetan
btn_pred, btn_reset, _ = st.columns([1, 1, 8])
with btn_pred:
    if st.button("🔍 Prediksi Manual"):
        input_df = pd.DataFrame([{
            'Tavg: Temperatur rata-rata (°C)': suhu,
            'RH_avg: Kelembapan rata-rata (%)': kelembapan,
            'RR: Curah hujan (mm)': curah,
            'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
            'Kelembaban Permukaan Tanah': tanah
        }])
        scaled_manual = scaler.transform(input_df)
        st.session_state.manual_result = convert_to_label(model.predict(scaled_manual)[0])
        st.session_state.manual_input.update({
            "suhu": suhu,
            "kelembapan": kelembapan,
            "curah": curah,
            "angin": angin,
            "tanah": tanah
        })

with btn_reset:
    if st.button("🧼 Reset Manual"):
        st.session_state.manual_input = {
            "suhu": 0.0,
            "kelembapan": 0.0,
            "curah": 0.0,
            "angin": 0.0,
            "tanah": 0.0
        }
        st.session_state.manual_result = None
        # Jalankan ulang aplikasi untuk refresh dan pindahkan kursor
        st.experimental_rerun()

# Menampilkan hasil prediksi manual
if st.session_state.manual_result:
    hasil = st.session_state.manual_result
    font, bg = risk_styles.get(hasil, ("black", "white"))
    st.markdown(
        f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px;'>"
        f"Prediksi Risiko Kebakaran: <b>{hasil}</b></p>",
        unsafe_allow_html=True
    )



# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)

st.markdown("""
<div style='
    margin-top: 20px;
    background-color: black;
    padding: 10px 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
'>
    <p style='margin: 0; font-size: 30px; font-weight: bold; line-height: 1.2;'>Smart Fire Prediction RHSEM – IoT Model</p>
    <p style='margin: 0; font-size: 13px; line-height: 1.2;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
</div>
""", unsafe_allow_html=True)




