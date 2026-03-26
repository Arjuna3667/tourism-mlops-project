import streamlit as st
import os
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = "./train.csv"
MODELS_FOLDER = "./models"

def load_models(models_folder=MODELS_FOLDER):
    models = {}
    if not os.path.exists(models_folder):
        st.error(f"Models folder '{models_folder}' not found!")
        return models
    files = [f for f in os.listdir(models_folder) if f.endswith(".pkl")]
    if not files:
        st.warning(f"No .pkl model files found in '{models_folder}'")
        return models
    for file in files:
        model_name = file.replace(".pkl", "")
        try:
            models[model_name] = joblib.load(os.path.join(models_folder, file))
        except Exception as e:
            st.error(f"Failed to load {file}: {e}")
    return models

def get_feature_names(dataset_path=DATASET_PATH):
    if not os.path.exists(dataset_path):
        return []
    df = pd.read_csv(dataset_path)
    return [col for col in df.columns if col != 'ProdTaken']  # FIXED TARGET COLUMN

st.title("Tourism Prediction App")

models = load_models()

if models:
    model_name = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_name]

    feature_names = get_feature_names()

    input_data = {}
    for feature in feature_names:
        val = st.text_input(feature, "0")
        try:
            input_data[feature] = float(val)
        except:
            input_data[feature] = 0.0

    if st.button("Predict"):
        try:
            df = pd.DataFrame([input_data])
            prediction = model.predict(df)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("No models found.")
