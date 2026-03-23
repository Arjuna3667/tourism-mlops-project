import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

st.title("Tourism Package Prediction")

# Load model
try:
    model_path = hf_hub_download(
        repo_id="Arjuna3667/tourism-model",
        filename="model.pkl"
    )
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Inputs
age = st.number_input("Age", 18, 70)
income = st.number_input("Monthly Income", 10000, 300000)
trips = st.number_input("Number of Trips", 0, 20)
passport = st.selectbox("Passport", [0, 1])
city = st.selectbox("City Tier", [1, 2, 3])

# Prediction
if st.button("Predict"):
    try:
        columns = model.feature_names

        input_df = pd.DataFrame(
            np.ones((1, len(columns))),
            columns=columns
        )

        input_df["Age"] = age
        input_df["MonthlyIncome"] = income
        input_df["NumberOfTrips"] = trips
        input_df["Passport"] = passport
        input_df["CityTier"] = city

        input_df = input_df.astype(float)

        prob = model.predict_proba(input_df)[0][1]

        st.write(f"Purchase Probability: {prob:.2f}")

        if prob > 0.3:
            st.success("Customer will purchase")
        else:
            st.error("Customer will NOT purchase")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
