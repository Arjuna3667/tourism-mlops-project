import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

print("STEP 1: Listing files in repo:", os.listdir())

# Load data
df = pd.read_csv("data/train.csv")
print("STEP 2: Data loaded successfully, shape:", df.shape)

# KEEP ONLY REQUIRED COLUMNS
df = df[[
    "Age",
    "MonthlyIncome",
    "NumberOfTrips",
    "Passport",
    "CityTier",
    "ProdTaken"
]]

print("STEP 3: Columns selected")

# Split
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

# Handle imbalance
print("STEP 4: Applying SMOTE...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("STEP 5: SMOTE done, shape:", X_res.shape)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_res, y_res)

print("STEP 6: Model trained")

# Save model
joblib.dump(model, "model.pkl")

print("MODEL SAVED SUCCESSFULLY")
print("FILES AFTER SAVE:", os.listdir())
