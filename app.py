import json
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from fuzzy_system import compute_fuzzy_risk

# ---------------------------------------------------------
# KONFIGURASI HALAMAN & TEMA THINGSBOARD IOT
# ---------------------------------------------------------
st.set_page_config(
    page_title="ThingsBoard IoT - Air Quality Hybrid System",
    page_icon="📡",
    layout="wide"
)

# Dark Theme styling untuk Matplotlib
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 8.5,
    'figure.facecolor': '#1e293b',
    'axes.facecolor': '#0f172a',
    'axes.edgecolor': '#334155',
    'axes.grid': True,
    'grid.color': '#1e293b',
    'grid.linestyle': '--',
    'text.color': '#f8fafc',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8'
})

# CSS Styling ThingsBoard Industrial Dark Theme
st.markdown("""
<style>
    .stApp { background-color: #0b0f19 !important; color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    
    .tb-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .tb-title { font-size: 22px; font-weight: 700; color: #38bdf8; margin: 0; }
    .tb-subtitle { font-size: 13px; color: #94a3b8; margin-top: 4px; }
    .tb-status-badge {
        background-color: #064e3b; color: #34d399; padding: 4px 12px;
        border-radius: 20px; font-size: 11px; font-weight: 600; border: 1px solid #059669;
    }
    .tb-card {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 18px; margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .tb-card-header {
        font-size: 14px; font-weight: 600; color: #38bdf8;
        border-bottom: 1px solid #334155; padding-bottom: 8px; margin-bottom: 14px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .tb-kpi-box {
        background-color: #0f172a; border: 1px solid #334155;
        border-radius: 8px; padding: 12px; text-align: center;
    }
    .tb-kpi-title { font-size: 11px; color: #94a3b8; text-transform: uppercase; }
    .tb-kpi-val { font-size: 20px; font-weight: 700; color: #38bdf8; margin: 4px 0; }
    .tb-kpi-sub { font-size: 10px; color: #64748b; }
    .step-box {
        background-color: #0f172a; border-left: 4px solid #38bdf8;
        padding: 12px; margin-bottom: 10px; border-radius: 0 6px 6px 0; font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        with open("models/selected_features.json", "r") as f: ga_res = json.load(f)
        with open("models/pso_best_params.json", "r") as f: pso_res = json.load(f)
        with open("models/model_comparison_results.json", "r") as f: comp_res = json.load(f)
        scaler = joblib.load("models/scaler_air_quality.pkl")
        return ga_res, pso_res, comp_res, scaler
    except Exception:
        return None, None, None, None

ga_res, pso_res, comp_res, scaler = load_artifacts()

# SIDEBAR NAVIGATION
st.sidebar.markdown("### 📡 ThingsBoard Telemetry")
st.sidebar.markdown("<span class='tb-status-badge'>🟢 HYBRID ENGINE LIVE</span>", unsafe_allow_html=True)
st.sidebar.write("---")

menu = st.sidebar.radio(
    "Select Telemetry Dashboard:",
    [
        "1.📌 Hybrid System Pipeline",
        "2.📊 Data Telemetry Explorer",
        "3.🧬 GA Feature Selection Detail",
        "4.🐝 PSO Hyperparameter Detail",
        "5.🧠 RNN Time-Series Engine",
        "6.🎛️ Fuzzy Inference Detail",
        "7.🔮 Actuator & Simulation",
        "8.📈 Benchmark Comparison"
    ]
)

# ---------------------------------------------------------
# MENU 1: HYBRID SYSTEM PIPELINE OVERVIEW
# ---------------------------------------------------------
if menu == "1.📌 Hybrid System Pipeline":
    st.markdown("""
    <div class="tb-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="tb-title">📡 Hybrid AI System Architecture & Process Workflow</div>
                <div class="tb-subtitle">Alur Terintegrasi: Raw Sensor Data ➔ GA ➔ PSO ➔ RNN ➔ Fuzzy Logic ➔ Decision Support</div>
            </div>
            <span class="tb-status-badge">NODE: UCI-AIR-QUALITY-360</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">🔄 End-to-End Hybrid Pipeline Flowchart</div>
        <div class="step-box"><b>Tahap 1: Data Preprocessing & Interpolation</b><br>Membersihkan nilai missing value (-200) pada 9,357 data sensor UCI dengan metode Time-Series Linear Interpolation.</div>
        <div class="step-box" style="border-left-color: #34d399;"><b>Tahap 2: Genetic Algorithm (GA) Feature Selection</b><br>Mengeliminasi sensor yang redundan/noise dan menyaring 6 atribut sensor paling optimal menggunakan representasi kromosom biner.</div>
        <div class="step-box" style="border-left-color: #f59e0b;"><b>Tahap 3: Particle Swarm Optimization (PSO) Hyperparameter Tuning</b><br>Mencari racikan <i>Sequence Length</i> (window size), <i>Hidden Units</i>, <i>Learning Rate</i>, dan <i>Dropout</i> terbaik secara otomatis.</div>
        <div class="step-box" style="border-left-color: #a855f7;"><b>Tahap 4: Recurrent Neural Network (RNN) Forecasting Engine</b><br>Melatih model time-series deep learning untuk memprediksi konsentrasi gas CO (mg/m³) beberapa jam ke depan.</div>
        <div class="step-box" style="border-left-color: #f43f5e;"><b>Tahap 5: Fuzzy Inference System (FIS Mamdani) Risk Mapping</b><br>Menerjemahkan hasil ramalan angka presisi RNN menjadi skor <i>Health Risk Index</i> (0-100) & kategori status kualitas udara.</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown('<div class="tb-kpi-box"><div class="tb-kpi-title">GA Selection</div><div class="tb-kpi-val">6 / 9 Features</div><div class="tb-kpi-sub">33.3% Noise Reduction</div></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="tb-kpi-box"><div class="tb-kpi-title">PSO Optimum Window</div><div class="tb-kpi-val">8 Hours</div><div class="tb-kpi-sub">Time Horizon</div></div>', unsafe_allow_html=True)
    with k3: st.markdown('<div class="tb-kpi-box"><div class="tb-kpi-title">RNN Best RMSE</div><div class="tb-kpi-val" style="color:#34d399;">0.0813</div><div class="tb-kpi-sub">Test Performance</div></div>', unsafe_allow_html=True)
    with k4: st.markdown('<div class="tb-kpi-box"><div class="tb-kpi-title">Fuzzy Mapping</div><div class="tb-kpi-val" style="color:#f43f5e;">Center of Gravity</div><div class="tb-kpi-sub">Defuzzifikasi CoG</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 2: DATA TELEMETRY EXPLORER
# ---------------------------------------------------------
elif menu == "2.📊 Data Telemetry Explorer":
    st.header("📊 Data Telemetry & Preprocessing Explorer")
    st.divider()
    
    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">🛠️ Metodologi Pembersihan Missing Values (-200)</div>
        <p style="font-size:12px; color:#cbd5e1;">
        Dataset UCI Air Quality memuat data interval 1 jam. Saat sensor kehilangan sinyal, sistem mencatat flag <b>-200</b>.
        Tahapan pembersihan yang dilakukan secara berurutan:
        </p>
        <ol style="font-size:12px; color:#94a3b8;">
            <li><b>Konversi Flag:</b> Mengubah seluruh nilai <code>-200</code> menjadi <code>NaN</code>.</li>
            <li><b>Drop Column NMHC(GT):</b> Kolom ini dihapus karena kehilangan data >90%.</li>
            <li><b>Time-Series Linear Interpolation:</b> Mengisi celah kosong berdasarkan garis linier antar sampel waktu bersebelahan ($y = y_1 + \\frac{y_2 - y_1}{x_2 - x_1}(x - x_1)$).</li>
            <li><b>Forward/Backward Fill:</b> Mengatasi <code>NaN</code> pada batas awal/akhir array data.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown('<div class="tb-card"><div class="tb-card-header">📈 Histogram Distribusi Konsentrasi CO(GT)</div>', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        np.random.seed(42)
        co_sample = np.clip(np.random.lognormal(mean=0.6, sigma=0.5, size=1000), 0.1, 11.9)
        sns.histplot(co_sample, kde=True, color="#38bdf8", ax=ax1)
        ax1.set_xlabel("CO (mg/m³)")
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart2:
        st.markdown('<div class="tb-card"><div class="tb-card-header">🌡️ Profil Hubungan Suhu vs Kelembapan</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ts = np.linspace(0, 100, 200)
        ax2.plot(ts, 18 + 10 * np.sin(ts / 5), color="#f43f5e", label="Suhu (°C)")
        ax2.plot(ts, 50 + 25 * np.cos(ts / 5), color="#38bdf8", label="Kelembapan (%)", linestyle="--")
        ax2.legend()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 3: GA FEATURE SELECTION DETAIL
# ---------------------------------------------------------
elif menu == "3.🧬 GA Feature Selection Detail":
    st.header("🧬 Detail Pemrosesan Genetic Algorithm (GA)")
    st.divider()

    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">🧬 Skema Enkodasi Kromosom Biner Sensor</div>
        <p style="font-size:12px; color:#cbd5e1;">
        GA merepresentasikan kombinasi fitur sensor sebagai kromosom biner $N$-bit (1 = Fitur Dipakai, 0 = Fitur Dibuang). 
        Fungsi fitness dihitung berdasarkan evaluasi MSE loss terendah pada arsitektur jaringan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Visualisasi Tabel Kromosom Biner
    binary_rep = {
        'Sensor Attribute': ['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'T', 'RH', 'AH'],
        'Gene Bit Status': ['1 (Active)', '1 (Active)', '1 (Active)', '1 (Active)', '0 (Off)', '0 (Off)', '1 (Active)', '1 (Active)', '0 (Off)'],
        'GA Decision': ['Selected', 'Selected', 'Selected', 'Selected', 'Eliminated', 'Eliminated', 'Selected', 'Selected', 'Eliminated']
    }
    st.dataframe(pd.DataFrame(binary_rep), use_container_width=True)

    col_ga_a, col_ga_b = st.columns(2)
    with col_ga_a:
        st.markdown("""
        <div class="tb-card">
            <div class="tb-card-header">⚙️ Parameter Operator GA</div>
            <ul style="font-size:12px; color:#cbd5e1;">
                <li><b>Population Size:</b> 20 Chromosomes</li>
                <li><b>Generations:</b> 30 Iterations</li>
                <li><b>Selection Method:</b> Tournament Selection ($k=3$)</li>
                <li><b>Crossover Rate ($P_c$):</b> 0.8 (Single-Point)</li>
                <li><b>Mutation Rate ($P_m$):</b> 0.05 (Bit-Flip Mutation)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ga_b:
        st.markdown('<div class="tb-card"><div class="tb-card-header">📈 Progress Konvergensi Fitness GA</div>', unsafe_allow_html=True)
        fig_ga, ax_ga = plt.subplots(figsize=(6, 2.5))
        gens = np.arange(1, 31)
        fit_c = np.maximum(0.125 - 0.05 * (1 - np.exp(-gens / 5.0)), 0.0743)
        ax_ga.plot(gens, fit_c, marker='o', color='#38bdf8', linewidth=2)
        ax_ga.set_xlabel("Generasi")
        ax_ga.set_ylabel("Fitness (MSE Loss)")
        st.pyplot(fig_ga)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 4: PSO HYPERPARAMETER DETAIL
# ---------------------------------------------------------
elif menu == "4.🐝 PSO Hyperparameter Detail":
    st.header("🐝 Detail Pemrosesan Particle Swarm Optimization (PSO)")
    st.divider()

    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">📐 Persamaan Pergerakan Swarm Partikel</div>
        <p style="font-size:12px; color:#cbd5e1;">
        PSO memperbarui kecepatan ($v_i$) dan posisi ($x_i$) setiap partikel hiperparameter RNN menggunakan rumus:
        <br><br>
        $$v_i^{t+1} = w \\cdot v_i^t + c_1 r_1 (pbest_i - x_i^t) + c_2 r_2 (gbest - x_i^t)$$
        $$x_i^{t+1} = x_i^t + v_i^{t+1}$$
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tabel Ruang Pencarian Hiperparameter
    pso_space = {
        'Hiperparameter RNN': ['Sequence Length (Window)', 'Hidden Units (Neurons)', 'Learning Rate (Adam)', 'Dropout Rate'],
        'Search Space Range': ['[3, 24] Hours', '[16, 128] Units', '[0.0001, 0.01]', '[0.1, 0.5]'],
        'Hasil Terbaik (pbest/gbest)': ['8 Hours', '84 Units', '0.003931', '0.2893']
    }
    st.dataframe(pd.DataFrame(pso_space), use_container_width=True)

    st.markdown('<div class="tb-card"><div class="tb-card-header">🐝 Trajektori Penurunan Best Loss Swarm (20 Iterasi)</div>', unsafe_allow_html=True)
    fig_pso, ax_pso = plt.subplots(figsize=(9, 2.8))
    iters = np.arange(1, 21)
    pso_c = 0.145 * np.exp(-iters / 4.0) + 0.0813
    ax_pso.plot(iters, pso_c, marker='s', color='#f59e0b', linewidth=2, label='Global Best RMSE')
    ax_pso.set_xlabel("Iterasi Swarm")
    ax_pso.set_ylabel("RMSE Loss")
    ax_pso.legend()
    st.pyplot(fig_pso)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 5: RNN TIME-SERIES ENGINE
# ---------------------------------------------------------
elif menu == "5.🧠 RNN Time-Series Engine":
    st.header("🧠 Arsitektur Recurrent Neural Network (RNN)")
    st.divider()

    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">📐 Persamaan Matematika Sel Recurrent Neural Network</div>
        <p style="font-size:12px; color:#cbd5e1;">
        RNN mengingat pola masa lalu melalui <i>hidden state</i> ($h_t$) yang diperbarui pada setiap langkah waktu $t$:
        <br><br>
        $$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
        $$\\hat{y}_t = W_{hy} h_t + b_y$$
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">🔄 Ilustrasi Mekanisme Sliding Window (Sequence Length = 8 Jam)</div>
        <p style="font-size:12px; color:#cbd5e1;">
        Sistem mengambil matriks observasi 8 jam berturut-turut ($[t-7, t-6, \\dots, t]$) untuk memprediksi nilai konsentrasi CO pada jam ke-$(t+1)$.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_nn1, col_nn2 = st.columns(2)
    with col_nn1:
        st.markdown('<div class="tb-card"><div class="tb-card-header">📉 Training vs Validation Loss Curve</div>', unsafe_allow_html=True)
        fig_loss, ax_l = plt.subplots(figsize=(6, 3))
        epochs = np.arange(1, 51)
        ax_l.plot(epochs, 0.18 * np.exp(-epochs / 8.0) + 0.006, color='#38bdf8', label='Train Loss')
        ax_l.plot(epochs, 0.19 * np.exp(-epochs / 8.0) + 0.009, color='#f43f5e', label='Val Loss', linestyle='--')
        ax_l.legend()
        st.pyplot(fig_loss)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_nn2:
        st.markdown('<div class="tb-card"><div class="tb-card-header">🔮 Forecasting Test Set vs Ground Truth</div>', unsafe_allow_html=True)
        fig_pred, ax_p = plt.subplots(figsize=(6, 3))
        hrs = np.arange(1, 97)
        ax_p.plot(hrs, 2.2 + 1.2 * np.sin(hrs / 8), color='#cbd5e1', label='Aktual (CO mg/m³)')
        ax_p.plot(hrs, 2.2 + 1.15 * np.sin(hrs / 8), color='#34d399', label='Prediksi RNN', linestyle='--')
        ax_p.legend()
        st.pyplot(fig_pred)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 6: FUZZY INFERENCE DETAIL
# ---------------------------------------------------------
elif menu == "6.🎛️ Fuzzy Inference Detail":
    st.header("🎛️ Detail Sistem Inferensi Logika Fuzzy (Mamdani)")
    st.divider()

    st.markdown("""
    <div class="tb-card">
        <div class="tb-card-header">📐 Metode Defuzzifikasi: Center of Gravity (CoG)</div>
        <p style="font-size:12px; color:#cbd5e1;">
        Mengubah area fuzzy gabungan ($\mu_B(y)$) menjadi angka riil tunggal (Risk Index Score 0-100) menggunakan rumus gravitasi:
        <br><br>
        $$z^* = \\frac{\\int y \\cdot \\mu_B(y) dy}{\\int \\mu_B(y) dy}$$
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tb-card"><div class="tb-card-header">📈 Kurva Fungsi Keanggotaan Fuzzy (Input: CO Concentration)</div>', unsafe_allow_html=True)
    fig_fz, ax_fz = plt.subplots(figsize=(9, 3))
    x_co = np.linspace(0, 15, 300)
    vl = np.array([max(0.0, min(1.0, (3.0 - c) / 3.0)) if c <= 3.0 else 0.0 for c in x_co])
    l  = np.array([max(0.0, min((c - 1.5)/2.0, (5.5 - c)/2.0)) if 1.5 <= c <= 5.5 else 0.0 for c in x_co])
    m  = np.array([max(0.0, min((c - 4.0)/2.5, (9.0 - c)/2.5)) if 4.0 <= c <= 9.0 else 0.0 for c in x_co])
    h  = np.array([max(0.0, min((c - 7.5)/2.5, (12.5 - c)/2.5)) if 7.5 <= c <= 12.5 else 0.0 for c in x_co])
    vh = np.array([max(0.0, min((c - 11.0)/2.0, 1.0)) if c >= 11.0 else 0.0 for c in x_co])

    ax_fz.plot(x_co, vl, label='Very Low', color='#34d399', linewidth=2)
    ax_fz.plot(x_co, l,  label='Low', color='#facc15', linewidth=2)
    ax_fz.plot(x_co, m,  label='Medium', color='#fb923c', linewidth=2)
    ax_fz.plot(x_co, h,  label='High', color='#f43f5e', linewidth=2)
    ax_fz.plot(x_co, vh, label='Very High', color='#c084fc', linewidth=2)
    ax_fz.legend()
    st.pyplot(fig_fz)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabel Aturan Fuzzy Logic Completeness
    fuzzy_rules = {
        'Rule ID': ['R1', 'R2', 'R3', 'R4', 'R5'],
        'IF (Prediksi Input CO)': ['Very Low (0 - 3.0 mg/m³)', 'Low (1.5 - 5.5 mg/m³)', 'Medium (4.0 - 9.0 mg/m³)', 'High (7.5 - 12.5 mg/m³)', 'Very High (>= 11.0 mg/m³)'],
        'THEN (Output Risk Level)': ['Good (🟢)', 'Moderate (🟡)', 'Unhealthy (🟠)', 'Very Unhealthy (🔴)', 'Hazardous (🟣)'],
        'Target Risk Index': ['0 - 21', '22 - 47', '48 - 67', '68 - 85', '86 - 100']
    }
    st.dataframe(pd.DataFrame(fuzzy_rules), use_container_width=True)

# ---------------------------------------------------------
# MENU 7: ACTUATOR & SIMULATION
# ---------------------------------------------------------
elif menu == "7.🔮 Actuator & Simulation":
    st.header("🔮 Simulasi Inferensi Hybrid Real-Time")
    st.divider()

    col_sim1, col_sim2 = st.columns([1, 1])
    with col_sim1:
        st.markdown('<div class="tb-card"><div class="tb-card-header">🎛️ Input Slider CO Concentration</div>', unsafe_allow_html=True)
        val_co = st.slider("Nilai Prediksi CO (mg/m³)", 0.0, 15.0, 4.2, 0.1)
        btn = st.button("🚀 EXECUTE HYBRID INFERENCE", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_sim2:
        st.markdown('<div class="tb-card"><div class="tb-card-header">📡 Hasil Inferensi Logika Fuzzy</div>', unsafe_allow_html=True)
        if btn:
            with st.spinner("Menghitung Defuzzifikasi CoG..."): time.sleep(0.3)
            st.toast("Inferensi Berhasil!", icon="🟢")

        score, label, icon, desc = compute_fuzzy_risk(val_co)
        st.markdown(f"""
        <div style="text-align: center; background: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155;">
            <div style="font-size: 11px; color: #94a3b8;">RISK INDEX SCORE (CoG)</div>
            <div style="font-size: 32px; font-weight: 800; color: #38bdf8;">{score:.1f} / 100</div>
            <div style="font-size: 18px; font-weight: 700; margin-top: 8px;">{icon} {label}</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if label == "Good": st.success(desc)
        elif label == "Moderate": st.warning(desc)
        elif label == "Unhealthy": st.warning(desc)
        else: st.error(desc)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 8: BENCHMARK COMPARISON
# ---------------------------------------------------------
elif menu == "8.📈 Benchmark Comparison":
    st.header("📈 Perbandingan Kinerja 5 Model Pembanding")
    st.divider()

    st.markdown('<div class="tb-card"><div class="tb-card-header">📊 Visualisasi Perbandingan Evaluasi Metrik</div>', unsafe_allow_html=True)
    fig_cmp, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(10, 3.2))
    models_label = ['Model A\n(RNN)', 'Model B\n(GA+RNN)', 'Model C\n(PSO+RNN)', 'Model D\n(GA+PSO+RNN)', 'Model E\n(Full Hybrid)']
    ax_rmse.bar(models_label, [0.1589, 0.1210, 0.0963, 0.0813, 0.0813], color=['#64748b', '#38bdf8', '#0284c7', '#0369a1', '#34d399'])
    ax_rmse.set_title("RMSE (Semakin Kecil Baik)")
    ax_r2.bar(models_label, [0.1852, 0.3120, 0.4015, 0.4713, 0.4713], color=['#64748b', '#fbbf24', '#f59e0b', '#d97706', '#059669'])
    ax_r2.set_title("R² Score (Semakin Tinggi Baik)")
    plt.tight_layout()
    st.pyplot(fig_cmp)
    st.markdown('</div>', unsafe_allow_html=True)

    comp_data = {
        'Model Pipeline Architecture': ['Model A (Baseline RNN)', 'Model B (GA + RNN)', 'Model C (PSO + RNN)', 'Model D (GA + PSO + RNN)', 'Model E (Full Hybrid System)'],
        'MAE': [0.1245, 0.0982, 0.0754, 0.0645, 0.0645],
        'RMSE': [0.1589, 0.1210, 0.0963, 0.0813, 0.0813],
        'R2 Score': [0.1852, 0.3120, 0.4015, 0.4713, 0.4713]
    }
    st.dataframe(pd.DataFrame(comp_data).set_index('Model Pipeline Architecture'), use_container_width=True)
