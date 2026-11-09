import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from fuzzy_system import compute_fuzzy_risk

# Konfigurasi Halaman
st.set_page_config(page_title="Air Pollution Forecasting System", page_icon="🌬️", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 28px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title {
