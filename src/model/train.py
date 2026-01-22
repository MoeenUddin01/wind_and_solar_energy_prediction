from pathlib import Path
import logging

from src.loader import DataLoader
from src.model.model import EnergyProductionModel  # corrected import

# Logger
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains the Energy Production ML model.
    DOES NOT save the model.
    """

    TRAIN_SCALED_PATH = Path("data/processed/train_scaled.csv")  # match scaling

    def __init__(self, target_col: str = "Production"):
        self.target_col = target_col
        self.loader = DataLoader()

    def train(self):
        # Load training data
        train_df = self.loader.read_csv(self.TRAIN_SCALED_PATH)
        logger.info("Training data loaded: %s", train_df.shape)

        X_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]

        # Build model
        model = EnergyProductionModel().build()
        logger.info("RandomForest model initialized")

        # Train
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        return model
