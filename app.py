import streamlit as st
import joblib
import pandas as pd
import os

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    model_path = "model.pkl"

    if not os.path.exists(model_path):
        st.error("model.pkl not found!")
        return None

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ===== UI =====
st.title("Tourism Prediction App")
st.write("Enter Customer Details")

# FIXED FEATURES (same as training)
age = st.number_input("Age", value=35)
income = st.number_input("Monthly Income", value=150000)
trips = st.number_input("Number of Trips", value=10)
passport = st.selectbox("Passport (0 = No, 1 = Yes)", [0, 1])
city = st.selectbox("City Tier (1/2/3)", [1, 2, 3])

# ===== PREDICTION =====
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded.")
    else:
        input_df = pd.DataFrame([{
            "Age": age,
            "MonthlyIncome": income,
            "NumberOfTrips": trips,
            "Passport": passport,
            "CityTier": city
        }])

        try:
            prediction = model.predict(input_df)[0]

            if prediction == 1:
                st.success("Customer is likely to purchase")
            else:
                st.warning("Customer is NOT likely to purchase")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
