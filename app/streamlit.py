import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import date

# -----------------------------
# Load Artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Load model
    model_path = os.path.join(ROOT_DIR, "artifacts/models/random_forest_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load scaler
    scaler_path = os.path.join(ROOT_DIR, "artifacts/encoders/scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Get feature order from scaler
    FEATURE_ORDER = scaler.feature_names_in_

    return model, scaler, FEATURE_ORDER

model, scaler, FEATURE_ORDER = load_artifacts()

# -----------------------------
# Custom CSS for background, pop-ups, and floating icons
# -----------------------------
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1502082553048-f009c37129b9?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Transparent main container */
    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
    }

    /* Pop-up style buttons */
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin-top: 10px;
        transition: transform 0.2s;
    }

    .stButton>button:hover {
        transform: scale(1.05);
    }

    /* Floating icons animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }

    .floating-icon {
        position: fixed;
        width: 50px;
        animation: float 3s ease-in-out infinite;
        z-index: 9999;
    }
    .icon1 { top: 10%; left: 5%; animation-delay: 0s; }
    .icon2 { top: 30%; left: 20%; animation-delay: 1s; }
    .icon3 { top: 60%; left: 50%; animation-delay: 2s; }
    .icon4 { top: 80%; left: 70%; animation-delay: 1.5s; }
    </style>

    <!-- Floating icons -->
    <img class="floating-icon icon1" src="https://cdn-icons-png.flaticon.com/512/1164/1164954.png" />
    <img class="floating-icon icon2" src="https://cdn-icons-png.flaticon.com/512/869/869869.png" />
    <img class="floating-icon icon3" src="https://cdn-icons-png.flaticon.com/512/1164/1164954.png" />
    <img class="floating-icon icon4" src="https://cdn-icons-png.flaticon.com/512/869/869869.png" />
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Solar & Wind Energy Predictor",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Solar & Wind Energy Production Predictor")

# -----------------------------
# Sidebar Inputs
# -----------------------------
with st.sidebar:
    st.header("📥 Input Parameters")
    selected_date = st.date_input("📅 Select Date", value=date.today())
    start_hour = st.slider("⏰ Start Hour", 0, 23, 8)
    end_hour = st.slider("⏰ End Hour", 1, 24, 17)
    source = st.selectbox("🔌 Energy Source", ["Solar", "Wind"])

# -----------------------------
# Feature Engineering
# -----------------------------
day_of_year = selected_date.timetuple().tm_yday
day_name = selected_date.strftime("%A")
month_name = selected_date.strftime("%B")

def get_season(month):
    if month in ["December", "January", "February"]:
        return "Winter"
    elif month in ["March", "April", "May"]:
        return "Spring"
    elif month in ["June", "July", "August"]:
        return "Summer"
    else:
        return "Fall"

season = get_season(month_name)

# -----------------------------
# Build input dataframe
# -----------------------------
row = {col:0 for col in FEATURE_ORDER}
row["Start_Hour"] = start_hour
row["End_Hour"] = end_hour
row["Day_of_Year"] = day_of_year
row[f"Source_{source}"] = 1
row[f"Day_Name_{day_name}"] = 1
row[f"Month_Name_{month_name}"] = 1
row[f"Season_{season}"] = 1

input_df = pd.DataFrame([row], columns=FEATURE_ORDER)

# -----------------------------
# Scale features
# -----------------------------
scaled_input = scaler.transform(input_df)

# -----------------------------
# Predict
# -----------------------------
if st.button("🚀 Predict Energy Production"):
    prediction = model.predict(scaled_input)[0]

    # Main metric
    st.metric(label="🔋 Estimated Energy Production (Units)", value=f"{prediction:.2f}")

    # Pop-up info columns
    col1, col2, col3 = st.columns(3)
    col1.info(f"📅 Date: {selected_date}")
    col2.info(f"🕒 Time: {start_hour}:00 - {end_hour}:00")
    col3.info(f"🌞 Source: {source}")

    # Visuals for Solar/Wind
    if source == "Solar":
        st.image(
            "https://images.unsplash.com/photo-1581090700227-7e4f12699ebd?auto=format&fit=crop&w=800&q=60",
            caption="Solar Energy",
            use_column_width=True
        )
    else:
        st.image(
            "https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=800&q=60",
            caption="Wind Energy",
            use_column_width=True
        )