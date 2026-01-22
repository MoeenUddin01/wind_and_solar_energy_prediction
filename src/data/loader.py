from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Iterable
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Logger setup
# -----------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -----------------------------
# DataLoader class
# -----------------------------
class DataLoader:
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    PROCESSED_DATA_FILE = "processed_energy_dataset.csv"

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def _resolve(self, path: Path | str) -> Path:
        return path if isinstance(path, Path) and path.is_absolute() else self.base_dir / path

    # -----------------------------
    # Read CSV
    # -----------------------------
    def read_csv(
        self, 
        path: Path | str, 
        *, 
        usecols: Optional[Iterable[str]] = None,
        dtype: Optional[dict] = None,
        nrows: Optional[int] = None
    ) -> pd.DataFrame:
        full_path = self._resolve(path)
        if not full_path.exists():
            raise FileNotFoundError(f"CSV file not found: {full_path}")
        df = pd.read_csv(full_path, usecols=usecols, dtype=dtype, nrows=nrows)
        logger.info("Loaded %s | shape=%s", full_path, df.shape)
        return df

    # -----------------------------
    # Save CSV
    # -----------------------------
    def save_csv(self, df: pd.DataFrame, path: Path | str) -> None:
        full_path = self._resolve(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full_path, index=False)
        logger.info("Saved dataset to %s", full_path)

    # -----------------------------
    # Load processed data
    # -----------------------------
    def load_processed_data(self, fill_target_nan: bool = True) -> pd.DataFrame:
        df = self.read_csv(self.PROCESSED_DATA_DIR / self.PROCESSED_DATA_FILE)
        if fill_target_nan and "Production" in df.columns:
            df["Production"].fillna(df["Production"].mean(), inplace=True)
            logger.info("Filled NaNs in target column 'Production' with mean")
        return df

    # -----------------------------
    # Load train/test split (for regression)
    # -----------------------------
    def load_train_data(
        self,
        target_col: str = "Production",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        df = self.load_processed_data(fill_target_nan=True)

        if target_col not in df.columns:
            logger.error("Target column '%s' not found in dataset!", target_col)
            raise KeyError(f"Target column '{target_col}' not found")

        # Split into train/test (no stratify for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop(columns=[target_col]),
            df[target_col],
            test_size=test_size,
            random_state=random_state
        )

        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test
