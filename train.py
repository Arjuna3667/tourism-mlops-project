import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset from Hugging Face
df = pd.read_csv(
    "https://huggingface.co/datasets/Arjuna3667/tourism-package-data/resolve/main/tourism.csv"
)

X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]

# Balance data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train model
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_res, y_res)

# Save feature names
model.feature_names = X.columns.tolist()

# Save model
joblib.dump(model, "model.pkl")
