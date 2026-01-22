from pathlib import Path
import logging
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Logger setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Evaluates a trained model and saves metrics/figures.
    """

    OUTPUT_DIR = Path("artifacts/model_matrices")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def __init__(self, model, X_test, y_test, class_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names

    def evaluate(self):
        """
        Evaluate the model and return metrics.
        """
        y_pred = self.model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        logger.info("Model Accuracy: %.4f", accuracy)
        return accuracy, report, cm

    def save_classification_report(self, report, filename="classification_report.txt"):
        """
        Save classification report as text file.
        """
        file_path = self.OUTPUT_DIR / filename
        with open(file_path, "w") as f:
            for label, metrics in report.items():
                f.write(f"{label}: {metrics}\n")
        logger.info("Classification report saved at %s", file_path)

    def plot_confusion_matrix(self, cm, filename="confusion_matrix.png"):
        """
        Plot and save the confusion matrix as an image.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        file_path = self.OUTPUT_DIR / filename
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        logger.info("Confusion matrix saved at %s", file_path)

    def plot_metrics_bar(self, report, filename="metrics_bar_chart.png"):
        """
        Optional: plot precision, recall, f1-score for each class.
        """
        import pandas as pd

        metrics_df = pd.DataFrame(report).transpose()
        # Only keep class labels, drop 'accuracy', 'macro avg', 'weighted avg'
        metrics_df = metrics_df[~metrics_df.index.isin(["accuracy", "macro avg", "weighted avg"])]
        metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(8, 6))
        plt.title("Precision, Recall, F1-Score per Class")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        file_path = self.OUTPUT_DIR / filename
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        logger.info("Metrics bar chart saved at %s", file_path)
