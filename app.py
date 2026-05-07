import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
)

# ============================================================
# LOAD MODEL & DATA
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()

FEATURES = [
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude",
]
TARGET = "Y house price of unit area"

# ============================================================
# HELPER
# ============================================================

def format_rupiah(value):
    """Format angka ke Rupiah."""
    return f"Rp {value:,.0f}".replace(",", ".")


# ============================================================
# SIDEBAR — INPUT PREDIKSI
# ============================================================
st.sidebar.header("📋 Input Data Rumah")

house_age = st.sidebar.slider(
    "Usia Rumah (tahun)",
    min_value=0.0,
    max_value=50.0,
    value=17.0,
    step=0.1,
)

mrt_distance = st.sidebar.slider(
    "Jarak ke Stasiun MRT Terdekat (meter)",
    min_value=0.0,
    max_value=7000.0,
    value=1000.0,
    step=10.0,
)

convenience_stores = st.sidebar.slider(
    "Jumlah Minimarket di Sekitar",
    min_value=0,
    max_value=10,
    value=4,
    step=1,
)

latitude = st.sidebar.slider(
    "Latitude",
    min_value=24.93,
    max_value=25.02,
    value=24.97,
    step=0.001,
    format="%.5f",
)

longitude = st.sidebar.slider(
    "Longitude",
    min_value=121.47,
    max_value=121.57,
    value=121.53,
    step=0.001,
    format="%.5f",
)

# ============================================================
# PREDIKSI
# ============================================================
input_data = pd.DataFrame(
    [[house_age, mrt_distance, convenience_stores, latitude, longitude]],
    columns=FEATURES,
)
prediction = model.predict(input_data)[0]

# ============================================================
# MAIN PAGE
# ============================================================
st.title("🏠 Prediksi Harga Rumah")
st.markdown("Aplikasi ini memprediksi **harga rumah per unit area** menggunakan model **Linear Regression**.")

# --- Hasil Prediksi ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="💰 Prediksi Harga", value=format_rupiah(prediction))

with col2:
    avg_price = df[TARGET].mean()
    diff_pct = ((prediction - avg_price) / avg_price) * 100
    st.metric(
        label="📊 Rata-rata Harga Dataset",
        value=format_rupiah(avg_price),
        delta=f"{diff_pct:+.1f}% dari rata-rata",
    )

with col3:
    st.metric(label="📐 Model R² Score", value="0.566")

# --- Detail Input ---
st.markdown("---")
st.subheader("📝 Detail Input Anda")

input_df = pd.DataFrame(
    {
        "Parameter": [
            "Usia Rumah (tahun)",
            "Jarak ke MRT (meter)",
            "Jumlah Minimarket",
            "Latitude",
            "Longitude",
        ],
        "Nilai": [
            f"{house_age:.1f}",
            f"{mrt_distance:.1f}",
            str(convenience_stores),
            f"{latitude:.5f}",
            f"{longitude:.5f}",
        ],
    }
)
st.dataframe(input_df, use_container_width=True, hide_index=True)

# --- Visualisasi ---
st.markdown("---")
st.subheader("📈 Eksplorasi Data")

tab1, tab2, tab3 = st.tabs(["Distribusi Harga", "Korelasi Fitur", "Data Mentah"])

with tab1:
    st.bar_chart(
        df[TARGET].value_counts(bins=30).sort_index(),
        use_container_width=True,
    )

with tab2:
    chart_feature = st.selectbox(
        "Pilih fitur untuk scatter plot:",
        FEATURES,
    )
    scatter_df = df[[chart_feature, TARGET]].dropna()
    st.scatter_chart(scatter_df, x=chart_feature, y=TARGET, use_container_width=True)

with tab3:
    st.dataframe(df, use_container_width=True, height=400)

# --- Koefisien Model ---
st.markdown("---")
st.subheader("🔍 Koefisien Model Linear Regression")

coef_df = pd.DataFrame(
    {
        "Fitur": FEATURES + ["Intercept"],
        "Koefisien": list(model.coef_) + [model.intercept_],
    }
)
st.dataframe(coef_df, use_container_width=True, hide_index=True)

# --- Footer ---
st.markdown("---")
st.caption("Dibuat dengan Streamlit • Model: Linear Regression (scikit-learn)")
