import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzy_system import compute_fuzzy_risk

# Konfigurasi Halaman
st.set_page_config(page_title="Air Pollution Forecasting System", page_icon="🌬️", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        with open('models/selected_features.json', 'r') as f:
            ga_res = json.load(f)
        with open('models/pso_best_params.json', 'r') as f:
            pso_res = json.load(f)
        with open('models/model_comparison_results.json', 'r') as f:
            comp_res = json.load(f)
        scaler = joblib.load('models/scaler_air_quality.pkl')
        return ga_res, pso_res, comp_res, scaler
    except Exception as e:
        return None, None, None, None

ga_res, pso_res, comp_res, scaler = load_artifacts()

# Navigasi Sidebar
st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Modul System:", [
    "1.📌 Project Overview",
    "2.📊 Dataset Explorer",
    "3.🧬 GA Optimization",
    "4.🐝 PSO Optimization",
    "5.🧠 Neural Network",
    "6.🎛️ Fuzzy Inference",
    "7.🔮 Prediction & Simulation",
    "8.📈 Model Comparison"
])

# ---------------------------------------------------------
# MENU 1: PROJECT OVERVIEW
# ---------------------------------------------------------
if menu == "1.📌 Project Overview":
    st.markdown('<div class="main-title">Intelligent Air Pollution Forecasting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hybrid Intelligent System: GA - PSO - RNN - Fuzzy Logic - Streamlit</div>', unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 Ringkasan Proyek")
        st.write("""
        Sistem kecerdasan buatan hibrida ini dirancang untuk memprediksi tingkat polusi udara (fokus pada konsentrasi **CO**) 
        menggunakan **UCI Air Quality Dataset (ID 360)**. Sistem menggabungkan 4 komponen utama:
        - **Genetic Algorithm (GA):** Untuk seleksi fitur sensor optimal.
        - **Particle Swarm Optimization (PSO):** Untuk optimasi hiperparameter & *sequence length* model time-series.
        - **Recurrent Neural Network (RNN):** Engine peramalan time-series berbasis deep learning.
        - **Fuzzy Inference System (FIS):** Sistem pendukung keputusan pemetaan indeks risiko kesehatan.
        """)
    
    with col2:
        st.subheader("👥 Anggota Kelompok 9")
        st.info("""
        - **Ariyo Arianto** (04231013)
        - **Jhosua George Kamagi** (04231043)
        - **Vandri Afli Navi Almas** (04231089)
        - **Bernadt Jansen Rajagukguk** (04221062)
        """)

# ---------------------------------------------------------
# MENU 2: DATASET EXPLORER
# ---------------------------------------------------------
elif menu == "2.📊 Dataset Explorer":
    st.header("📊 Dataset Explorer (Air Quality UCI ID 360)")
    st.write("Eksplorasi atribut sensor polusi udara dan strategi penanganan *missing values* (`-200`).")
    st.divider()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Observasi", "9,357 Jam")
    m2.metric("Jumlah Atribut Sensor", "13 Fitur")
    m3.metric("Missing Value Flag", "-200")
    m4.metric("Periode Data", "Mar 2004 - Feb 2005")
    
    st.subheader("1. Pembersihan Missing Value (-200)")
    st.write("""
    Pada dataset asli UCI ID 360, nilai yang tidak terekam oleh sensor ditandai dengan angka **`-200`**. 
    Tahapan pembersihan data yang dilakukan:
    1. Mengganti seluruh nilai `-200` menjadi `NaN`.
    2. Menghapus kolom `NMHC(GT)` karena persentase missing value melebihi 90%.
    3. Melakukan **Time-Series Linear Interpolation** dilanjutkan **Forward/Backward Fill** untuk mempertahankan sifat kronologis data.
    """)
    
    st.subheader("2. Statistik Deskriptif Atribut Utama")
    stats_data = {
        'Atribut Sensor': ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'T', 'RH', 'AH'],
        'Deskripsi': ['Karbon Monoksida (True)', 'Respon Sensor S1', 'Benzena (True)', 'Respon Sensor S2', 'Nitrogen Oksida', 'Respon Sensor S3', 'Nitrogen Dioksida', 'Suhu (°C)', 'Kelembapan Relatif (%)', 'Kelembapan Absolut'],
        'Rata-rata': [2.15, 1099.8, 10.08, 939.1, 246.9, 835.5, 113.1, 18.3, 49.2, 1.02],
        'Min': [0.10, 647.0, 0.10, 383.0, 2.0, 322.0, 2.0, -1.9, 9.2, 0.18],
        'Max': [11.90, 2040.0, 63.70, 2214.0, 1479.0, 2683.0, 340.0, 44.6, 88.7, 2.23]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# ---------------------------------------------------------
# MENU 3: GA OPTIMIZATION
# ---------------------------------------------------------
elif menu == "3.🧬 GA Optimization":
    st.header("🧬 Genetic Algorithm (GA) Feature Selection")
    st.write("Seleksi atribut sensor terbaik untuk mengurangi redundansi data masukan RNN.")
    st.divider()
    
    if ga_res:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Fitur Terpilih (Selected)")
            for f in ga_res['selected_features']:
                st.success(f"• {f}")
        with c2:
            st.subheader("❌ Fitur Dieliminasi (Eliminated)")
            for f in ga_res['eliminated_features']:
                st.error(f"• {f}")
        
        st.subheader("Statistik Optimasi GA")
        g1, g2, g3 = st.columns(3)
        g1.metric("Akurasi Fitness Terbaik", f"{ga_res['best_fitness']:.4f}")
        g2.metric("Rata-rata Fitness", f"{ga_res['mean_fitness']:.4f}")
        g3.metric("Standar Deviasi", f"{ga_res['std_fitness']:.4f}")

# ---------------------------------------------------------
# MENU 4: PSO OPTIMIZATION
# ---------------------------------------------------------
elif menu == "4.🐝 PSO Optimization":
    st.header("🐝 Particle Swarm Optimization (PSO)")
    st.write("Optimasi hiperparameter arsitektur Recurrent Neural Network secara otomatis.")
    st.divider()
    
    if pso_res:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Sequence Length (Window)", f"{pso_res['seq_len']} Jam")
        p2.metric("Hidden Units", f"{pso_res['hidden_units']}")
        p3.metric("Learning Rate", f"{pso_res['learning_rate']:.6f}")
        p4.metric("Dropout Rate", f"{pso_res['dropout']:.4f}")
        
        st.subheader("Statistik Konvergensi PSO (10 Independent Runs)")
        ps1, ps2, ps3 = st.columns(3)
        ps1.metric("Best Loss (RMSE)", f"{pso_res['stats']['best_cost']:.4f}")
        ps2.metric("Mean Loss", f"{pso_res['stats']['mean_cost']:.4f}")
        ps3.metric("Std Dev Loss", f"{pso_res['stats']['std_cost']:.4f}")

# ---------------------------------------------------------
# MENU 5: NEURAL NETWORK
# ---------------------------------------------------------
elif menu == "5.🧠 Neural Network":
    st.header("🧠 Arsitektur Recurrent Neural Network (RNN)")
    st.write("Engine peramalan time-series yang dikonfigurasi berdasarkan hasil optimasi PSO.")
    st.divider()
    
    if pso_res:
        st.subheader("Metrik Evaluasi Model Terbaik (Test Set)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error (MAE)", f"{pso_res['metrics']['mae']:.4f}")
        m2.metric("Root Mean Squared Error (RMSE)", f"{pso_res['metrics']['rmse']:.4f}")
        m3.metric("R² Score (Koefisien Determinasi)", f"{pso_res['metrics']['r2']:.4f}")
        
        st.subheader("Spesifikasi Arsitektur")
        st.json({
            "Input Features": len(ga_res['selected_features']) if ga_res else 6,
            "Sequence Length": pso_res['seq_len'],
            "RNN Layers": 2,
            "Hidden Units": pso_res['hidden_units'],
            "Dropout": pso_res['dropout'],
            "Optimizer": "Adam",
            "Loss Function": "MSE Loss"
        })

# ---------------------------------------------------------
# MENU 6: FUZZY INFERENCE
# ---------------------------------------------------------
elif menu == "6.🎛️ Fuzzy Inference":
    st.header("🎛️ Fuzzy Inference System (FIS)")
    st.write("Sistem inferensi logika fuzzy Mamdani untuk memetakan hasil prediksi konsentrasi CO menjadi indeks risiko kesehatan.")
    st.divider()
    
    st.subheader("1. Fungsi Keanggotaan Input: Konsentrasi CO (mg/m³)")
    mf_data = {
        'Kategori Fuzzy': ['Very Low (Sangat Rendah)', 'Low (Rendah)', 'Medium (Sedang)', 'High (Tinggi)', 'Very High (Sangat Tinggi)'],
        'Rentang Nilai CO': ['0.0 - 3.0 mg/m³', '1.5 - 5.5 mg/m³', '4.0 - 9.0 mg/m³', '7.5 - 12.5 mg/m³', '>= 11.0 mg/m³'],
        'Bentuk Kurva': ['Triangular', 'Triangular', 'Triangular', 'Triangular', 'Trapezoidal']
    }
    st.table(pd.DataFrame(mf_data))
    
    st.subheader("2. Klasifikasi Risiko Kesehatan (Output Defuzzifikasi)")
    risk_table = {
        'Skor Risk Index': ['0 - 21', '22 - 47', '48 - 67', '68 - 85', '86 - 100'],
        'Status Kualitas Udara': ['🟢 Good', '🟡 Moderate', '🟠 Unhealthy', '🔴 Very Unhealthy', '🟣 Hazardous'],
        'Rekomendasi Kesehatan': [
            'Aman untuk aktivitas luar ruangan.',
            'Kelompok sensitif sebaiknya mengurangi aktivitas fisik berat di luar.',
            'Gunakan masker saat berada di luar ruangan.',
            'Hindari aktivitas di luar ruangan.',
            'Tingkat BAHAYA! Wajib tetap berada di dalam ruangan.'
        ]
    }
    st.dataframe(pd.DataFrame(risk_table), use_container_width=True)

# ---------------------------------------------------------
# MENU 7: PREDICTION & SIMULATION
# ---------------------------------------------------------
elif menu == "7.🔮 Prediction & Simulation":
    st.header("🔮 Simulasi Prediksi & Inferensi Hybrid")
    st.write("Simulasikan nilai prediksi konsentrasi CO untuk menghasilkan analisis risiko kesehatan otomatis secara real-time.")
    st.divider()
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("Input Parameter Prediksi")
        val_co = st.slider("Prediksi Konsentrasi CO (mg/m³)", 0.0, 15.0, 4.2, 0.1)
        st.write(f"Nilai Terpilih: **{val_co:.1f} mg/m³**")
        btn = st.button("🚀 Jalankan Inferensi Hybrid", use_container_width=True)
    
    with col_output:
        st.subheader("Hasil Inferensi Logika Fuzzy")
        if btn or val_co:
            score, label, icon, desc = compute_fuzzy_risk(val_co)
            
            st.metric("Skor Risk Index (Defuzzifikasi)", f"{score:.1f} / 100")
            st.subheader(f"Status: {icon} {label}")
            
            if label == "Good":
                st.success(desc)
            elif label == "Moderate":
                st.warning(desc)
            elif label == "Unhealthy":
                st.warning(desc)
            else:
                st.error(desc)

# ---------------------------------------------------------
# MENU 8: MODEL COMPARISON
# ---------------------------------------------------------
elif menu == "8.📈 Model Comparison":
    st.header("📈 Perbandingan 5 Model Pembanding Wajib")
    st.write("Evaluasi kinerja baseline RNN hingga Hybrid System (GA-PSO-RNN-Fuzzy) pada data uji.")
    st.divider()
    
    if comp_res:
        df_comp = pd.DataFrame(comp_res).T
        st.dataframe(df_comp.style.highlight_min(subset=['MAE', 'RMSE'], color='#D1FAE5').highlight_max(subset=['R2_Score'], color='#D1FAE5'), use_container_width=True)
        
        st.subheader("Keterangan Model:")
        st.write("""
        - **Model A (Baseline RNN):** Model RNN awal tanpa seleksi fitur dan optimasi hiperparameter.
        - **Model B (GA + RNN):** Arsitektur RNN dengan masukan fitur terpilih dari GA.
        - **Model C (PSO + RNN):** Arsitektur RNN dengan hiperparameter dioptimalkan oleh PSO.
        - **Model D (GA + PSO + RNN):** Penggabungan GA (fitur) dan PSO (hiperparameter) pada RNN.
        - **Model E (Full Hybrid System):** GA-PSO-RNN terintegrasi penuh dengan Fuzzy Inference System.
        """)
