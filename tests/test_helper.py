import unittest
import os
import pandas as pd
import shutil
from utils import helper

class TestHelperFunctions(unittest.TestCase):

    def setUp(self):
        # Clean up data and model directories before tests
        if os.path.exists("data/insomnia_synthetic.csv"):
            os.remove("data/insomnia_synthetic.csv")
        if os.path.exists("model"):
            try:
                shutil.rmtree("model")
            except PermissionError:
                pass
        if os.path.exists("models"):
            try:
                shutil.rmtree("models")
            except PermissionError:
                pass

    def test_generate_data_creates_file_and_correct_shape(self):
        df = helper.generate_data(n_samples=50)
        self.assertTrue(os.path.exists("data/insomnia_synthetic.csv"))
        self.assertEqual(df.shape[0], 50)
        expected_columns = helper.FEATURE_COLS + ["Total Score", "Insomnia Level"]
        self.assertListEqual(list(df.columns), expected_columns)

    def test_append_to_insomnia_data_creates_and_appends(self):
        entry1 = {col: 1 for col in helper.FEATURE_COLS}
        helper.append_to_insomnia_data(entry1)
        self.assertTrue(os.path.exists("data/insomnia_synthetic.csv"))
        df = pd.read_csv("data/insomnia_synthetic.csv")
        self.assertEqual(len(df), 1)

        entry2 = {col: 2 for col in helper.FEATURE_COLS}
        helper.append_to_insomnia_data(entry2)
        df = pd.read_csv("data/insomnia_synthetic.csv")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]["Level of Insomnia Intensity"], 2)

    def test_load_model_files_exist(self):
        # First generate and save a model to load
        df = helper.generate_data(n_samples=20)
        accuracy = helper.train_and_save_model(df)
        self.assertIsInstance(accuracy, float)

        model, label_encoder, scaler = helper.load_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(label_encoder)
        self.assertIsNotNone(scaler)

    def test_retrain_model_with_feedback_returns_accuracy(self):
        # Prepare synthetic data file for retraining
        df = helper.generate_data(n_samples=100)
        df.to_csv("data/insomnia_synthetic.csv", index=False)

        accuracy = helper.retrain_model_with_feedback()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_train_and_save_model_returns_accuracy(self):
        df = helper.generate_data(n_samples=25)
        accuracy = helper.train_and_save_model(df)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == "__main__":
    unittest.main()
