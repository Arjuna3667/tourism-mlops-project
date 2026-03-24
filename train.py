import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
pd.read_csv("data/train.csv")

# KEEP ONLY REQUIRED COLUMNS
df = df[[
    "Age",
    "MonthlyIncome",
    "NumberOfTrips",
    "Passport",
    "CityTier",
    "ProdTaken"
]]

# Split
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

# Handle imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_res, y_res)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
