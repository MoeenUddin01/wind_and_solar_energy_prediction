from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path
from loader import DataLoader  # Make sure this path is correct

# Initialize loader
loader = DataLoader()

# Load processed dataset
df = loader.load_processed_data(fill_target_nan=False)  # NaNs already filled in preprocessing

# Separate features and target
target_col = "Production"
X = df.drop(columns=[target_col])
y = df[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Combine scaled features with target
df_scaled = pd.concat([X_scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

# Save scaled dataset
scaled_path = Path("data/processed/train_scaled.csv")  # Match the trainer expectation
loader.save_csv(df_scaled, scaled_path)
