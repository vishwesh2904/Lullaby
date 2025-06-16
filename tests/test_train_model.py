import unittest
import os
import pandas as pd
import numpy as np
from train_model import train_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Remove model files if they exist before each test
        model_files = [
            "models/insomnia_model.joblib",
            "models/scaler.joblib",
            "models/label_encoder.joblib"
        ]
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)

    def test_train_model_runs_and_returns_accuracy(self):
        accuracy = train_model()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_files_are_saved(self):
        _ = train_model()
        self.assertTrue(os.path.exists("models/insomnia_model.joblib"))
        self.assertTrue(os.path.exists("models/scaler.joblib"))
        self.assertTrue(os.path.exists("models/label_encoder.joblib"))

    def test_model_accuracy_threshold(self):
        accuracy = train_model()
        self.assertGreaterEqual(accuracy, 0.7, "Model accuracy should be at least 0.7")

    def test_train_model_with_missing_data(self):
        df = pd.read_csv("data/insomnia_synthetic.csv")
        df.loc[0, df.columns[0]] = np.nan  # Introduce missing value
        df = df.dropna()
        accuracy = train_model()
        self.assertIsInstance(accuracy, float)

    def test_train_model_with_corrupted_data(self):
        df = pd.read_csv("data/insomnia_synthetic.csv")
        df.loc[0, df.columns[0]] = "corrupted"  # Introduce corrupted value
        df = df[pd.to_numeric(df[df.columns[0]], errors='coerce').notnull()]
        accuracy = train_model()
        self.assertIsInstance(accuracy, float)

if __name__ == "__main__":
    unittest.main()
