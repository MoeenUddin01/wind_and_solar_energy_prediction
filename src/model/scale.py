# src/model/scale.py
from __future__ import annotations
from pathlib import Path
import pickle
import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.loader import DataLoader

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class TrainingScaler:
    """
    Scales ONLY training data.

    - Fits scaler on training features
    - Saves scaler for future inference
    - Saves scaled training dataset
    """

    SCALER_PATH = Path("artifacts/encoders/scaler.pkl")
    TRAIN_SCALED_PATH = Path("data/processed/train_scaled.csv")

    def __init__(self, target_col: str, base_dir: Path | str = "."):
        self.target_col = target_col
        self.base_dir = Path(base_dir)
        self.dl = DataLoader(base_dir=self.base_dir)
        self.scaler = StandardScaler()

    def fit_transform(self) -> pd.DataFrame:
        # Load already-split training data
        train_df = self.dl.load_train_data()
        logger.info("Loaded training data: %s", train_df.shape)

        X_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]

        # Fit scaler ONLY on training features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        logger.info("Scaler fitted on training features")

        train_scaled = pd.concat([X_scaled, y_train.reset_index(drop=True)], axis=1)

        # Save scaler
        self.SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info("Scaler saved at %s", self.SCALER_PATH)

        # Save scaled training data
        self.dl.save_csv(train_scaled, self.TRAIN_SCALED_PATH)
        logger.info("Scaled training data saved")

        return train_scaled


if __name__ == "__main__":
    scaler = TrainingScaler(target_col="Energy_Production")
    scaler.fit_transform()
