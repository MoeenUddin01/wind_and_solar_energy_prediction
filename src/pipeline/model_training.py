from __future__ import annotations

import logging
from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.loader import DataLoader

# -----------------------------
# Logger setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "Production"
TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_DIR = Path("artifacts/models")
MODEL_FILE = MODEL_DIR / "random_forest_model.pkl"

METRICS_DIR = Path("artifacts/model_matrices")
METRICS_FILE = METRICS_DIR / "model_performance.csv"

# -----------------------------
# Load processed data
# -----------------------------
loader = DataLoader()
df = loader.load_processed_data(fill_target_nan=True)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in processed dataset!")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
logger.info("Data loaded | Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
logger.info("Model trained successfully")

# -----------------------------
# Evaluate model on test set
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

logger.info("Evaluation on test set | MAE: %.2f | MSE: %.2f | RMSE: %.2f | R2: %.2f",
            mae, mse, rmse, r2)

# -----------------------------
# Save trained model
# -----------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
logger.info("Model saved to %s", MODEL_FILE)

# -----------------------------
# Save evaluation metrics
# -----------------------------
METRICS_DIR.mkdir(parents=True, exist_ok=True)
metrics_df = pd.DataFrame([{
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Train size": X_train.shape[0],
    "Test size": X_test.shape[0]
}])
metrics_df.to_csv(METRICS_FILE, index=False)
logger.info("Model performance metrics saved to %s", METRICS_FILE)
