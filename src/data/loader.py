from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ------------------------------------------------------
# Logger setup
# ------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataLoader:
    """
    Professional & human-friendly DataLoader for Solar & Wind Energy Production.

    Provides:
    - Safe CSV reading/writing
    - Standardized project paths
    - Sampling & train-test splitting
    - Logging & exception handling
    """

    # --------------------------------------------------
    # Standard paths
    # --------------------------------------------------
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    PROCESSED_DATA_FILE = "processed_energy_dataset.csv"

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    # --------------------------------------------------
    # Path resolution
    # --------------------------------------------------
    def _resolve(self, path: Path | str) -> Path:
        return path if isinstance(path, Path) and path.is_absolute() else self.base_dir / path

    # --------------------------------------------------
    # CSV reading
    # --------------------------------------------------
    def read_csv(
        self,
        path: Path | str,
        *,
        usecols: Optional[Iterable[str]] = None,
        dtype: Optional[dict] = None,
        nrows: Optional[int] = None,
        chunksize: Optional[int] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        full_path = self._resolve(path)

        try:
            if not full_path.exists():
                raise FileNotFoundError(f"CSV file not found: {full_path}")

            if chunksize:
                iterator = pd.read_csv(
                    full_path,
                    usecols=usecols,
                    dtype=dtype,
                    chunksize=chunksize,
                )
                if show_progress and _HAS_TQDM:
                    df = pd.concat(
                        tqdm(iterator, desc=f"Loading {full_path.name}"),
                        ignore_index=True,
                    )
                else:
                    df = pd.concat(iterator, ignore_index=True)
            else:
                df = pd.read_csv(
                    full_path,
                    usecols=usecols,
                    dtype=dtype,
                    nrows=nrows,
                )

            logger.info("Loaded %s | shape=%s", full_path, df.shape)
            return df

        except pd.errors.ParserError as e:
            logger.error("CSV parsing failed for %s", full_path)
            raise RuntimeError("Invalid CSV format") from e

        except Exception as e:
            logger.exception("Unexpected error while reading %s", full_path)
            raise

    # --------------------------------------------------
    # CSV saving
    # --------------------------------------------------
    def save_csv(self, df: pd.DataFrame, path: Path | str, *, index: bool = False) -> None:
        full_path = self._resolve(path)

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(full_path, index=index)
            logger.info("Saved dataset to %s", full_path)

        except PermissionError:
            logger.error("Permission denied while saving %s", full_path)
            raise

        except Exception:
            logger.exception("Failed to save CSV to %s", full_path)
            raise

    # --------------------------------------------------
    # Load processed dataset
    # --------------------------------------------------
    def load_processed_data(self) -> pd.DataFrame:
        try:
            return self.read_csv(self.PROCESSED_DATA_DIR / self.PROCESSED_DATA_FILE)
        except FileNotFoundError:
            logger.error("Processed dataset not found")
            raise

    # --------------------------------------------------
    # Random sample
    # --------------------------------------------------
    def sample(self, df: pd.DataFrame, *, frac: float = 0.1, random_state: int = 42) -> pd.DataFrame:
        if not 0 < frac <= 1:
            logger.error("Invalid sampling fraction: %s", frac)
            raise ValueError("`frac` must be between 0 and 1")
        return df.sample(frac=frac, random_state=random_state)

    # --------------------------------------------------
    # Train/Test split
    # --------------------------------------------------
    def split(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if target_col not in df.columns:
            logger.error("Target column '%s' not found", target_col)
            raise KeyError(f"Target column '{target_col}' not found")

        X = df.drop(columns=target_col)
        y = df[target_col]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
            )
            logger.info("Train-test split | train=%s test=%s", X_train.shape, X_test.shape)
            return X_train, X_test, y_train, y_test

        except ValueError:
            logger.error("Train-test split failed")
            raise

        except Exception:
            logger.exception("Unexpected error during split")
            raise


# Backward compatibility
DaraLoader = DataLoader
