import streamlit as st
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzy_system import compute_fuzzy_risk

st.set_page_config(page_title="Air Pollution Forecasting System", page_icon="🌬️", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 20px; }
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

st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Modul System:", [
    "1.📌 Project Overview", "2.📊 Dataset Explorer", "3.🧬 GA Optimization",
    "4.🐝 PSO Optimization", "5.🧠 Neural Network", "6.🎛️ Fuzzy Inference",
    "7.🔮 Prediction & Simulation", "8.📈 Model Comparison"
])

if menu == "1.📌 Project Overview":
    st.markdown('<div class="main-title">Intelligent Air Pollution Forecasting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Hybrid Intelligent System: GA - PSO - RNN - Fuzzy Logic - Streamlit</div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Anggota Kelompok 9")
    st.info("Ariyo Arianto | Jhosua George Kamagi | Vandri Afli Navi Almas | Bernadt Jansen Rajagukguk")

elif menu == "2.📊 Dataset Explorer":
    st.header("Dataset Explorer (Air Quality UCI ID 360)")
    st.write("Visualisasi statistik dan pembersihan data missing value (-200).")

elif menu == "3.🧬 GA Optimization":
    st.header("Genetic Algorithm (GA) Feature Selection")
    if ga_res:
        st.success("Fitur Terpilih: " + ", ".join(ga_res['selected_features']))
        st.error("Fitur Dieliminasi: " + ", ".join(ga_res['eliminated_features']))

elif menu == "4.🐝 PSO Optimization":
    st.header("Particle Swarm Optimization (PSO)")
    if pso_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sequence Length", f"{pso_res['seq_len']} Jam")
        c2.metric("Hidden Units", f"{pso_res['hidden_units']}")
        c3.metric("Learning Rate", f"{pso_res['learning_rate']:.6f}")
        c4.metric("Dropout Rate", f"{pso_res['dropout']:.4f}")

elif menu == "5.🧠 Neural Network":
    st.header("Recurrent Neural Network (RNN) Architecture")
    if pso_res:
        c1, c2, c3 = st.columns(3)
        c1.metric("Test MAE", f"{pso_res['metrics']['mae']:.4f}")
        c2.metric("Test RMSE", f"{pso_res['metrics']['rmse']:.4f}")
        c3.metric("Test R² Score", f"{pso_res['metrics']['r2']:.4f}")

elif menu == "6.🎛️ Fuzzy Inference":
    st.header("Fuzzy Inference System (FIS)")
    st.write("Penjelas keputusan risiko kesehatan berbasis logika fuzzy.")

elif menu == "7.🔮 Prediction & Simulation":
    st.header("Simulasi Prediksi Real-Time & Keputusan Fuzzy")
    val_co = st.slider("Prediksi Konsentrasi CO (mg/m³)", 0.0, 15.0, 3.5, 0.1)
    if st.button("Jalankan Inferensi Hybrid"):
        score, label, icon, desc = compute_fuzzy_risk(val_co)
        st.metric("Skor Defuzzifikasi Risk Index", f"{score:.1f} / 100")
        st.subheader(f"Status: {icon} {label}")
        st.info(desc)

elif menu == "8.📈 Model Comparison":
    st.header("Perbandingan 5 Model Pembanding Wajib")
    if comp_res:
        st.dataframe(pd.DataFrame(comp_res).T)