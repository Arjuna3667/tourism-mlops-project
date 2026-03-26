import streamlit as st
import os
import joblib
import pandas as pd
import logging
import warnings
import sys

warnings.filterwarnings("ignore")

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

sys.stderr = DummyFile()
import mlflow
sys.stderr = sys.__stderr__

logging.getLogger("mlflow").setLevel(logging.CRITICAL)

DATASET_PATH = "./train.csv"
MODELS_FOLDER = "./models"

def load_models(models_folder=MODELS_FOLDER):
    models = {}
    if not os.path.exists(models_folder):
        st.error(f"Models folder '{models_folder}' not found!")
        return models
    for file in os.listdir(models_folder):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            models[model_name] = joblib.load(os.path.join(models_folder, file))
    return models

def get_feature_names(dataset_path=DATASET_PATH):
    if not os.path.exists(dataset_path):
        return []
    df = pd.read_csv(dataset_path)
    return [col for col in df.columns if col != 'target']

st.title("Tourism Prediction App")

models = load_models()

if models:
    model_name = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_name]

    feature_names = get_feature_names()

    input_data = {}
    for feature in feature_names:
        input_data[feature] = float(st.text_input(feature, "0"))

    if st.button("Predict"):
        try:
            df = pd.DataFrame([input_data])
            prediction = model.predict(df)
            st.success(f"Predicted Value: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.warning("No models found.")
