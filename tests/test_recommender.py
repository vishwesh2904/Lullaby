import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from utils.recommender import predict_insomnia_level

class TestPredictInsomniaLevel(unittest.TestCase):

    @patch('utils.recommender.load_model')
    def test_all_zero_inversely_related_features(self, mock_load_model):
        # Mock model, label_encoder, scaler
        mock_model = MagicMock()
        mock_label_encoder = MagicMock()
        mock_scaler = MagicMock()

        # Setup label_encoder classes to include "Mild" and "Negligible"
        mock_label_encoder.classes_ = np.array(["Negligible", "Mild", "Moderate", "High", "Severe"])

        # Setup model.predict to return index of "Mild"
        mock_model.predict.return_value = [1]

        # Setup scaler.transform to return input as is
        mock_scaler.transform.side_effect = lambda x: x

        mock_load_model.return_value = (mock_model, mock_label_encoder, mock_scaler)

        # Create input data with all four inversely related features zero
        input_data = pd.DataFrame({
            "Sleep Efficiency": [0],
            "Sleep Wellness Practices": [0],
            "Coping Skills": [0],
            "Emotion Regulation": [0],
            # Add other features with non-zero values to test isolation
            "Other Feature 1": [1],
            "Other Feature 2": [2]
        })

        # Add missing features from FEATURE_COLS with zeros if needed
        from utils.helper import FEATURE_COLS
        for col in FEATURE_COLS:
            if col not in input_data.columns:
                input_data[col] = 0

        result = predict_insomnia_level(input_data)
        self.assertNotEqual(result, "Negligible", "Should not return 'Negligible' when all inversely related features are zero")
        self.assertEqual(result, "Mild", "Should return 'Mild' for all zero inversely related features")

    @patch('utils.recommender.load_model')
    def test_typical_input(self, mock_load_model):
        mock_model = MagicMock()
        mock_label_encoder = MagicMock()
        mock_scaler = MagicMock()

        mock_label_encoder.classes_ = np.array(["Negligible", "Mild", "Moderate", "High", "Severe"])
        mock_label_encoder.inverse_transform.return_value = ["Moderate"]
        mock_model.predict.return_value = [2]  # "Moderate"
        mock_scaler.transform.side_effect = lambda x: x

        mock_load_model.return_value = (mock_model, mock_label_encoder, mock_scaler)

        input_data = pd.DataFrame({
            "Sleep Efficiency": [5],
            "Sleep Wellness Practices": [3],
            "Coping Skills": [4],
            "Emotion Regulation": [2],
            "Other Feature 1": [1],
            "Other Feature 2": [2]
        })

        from utils.helper import FEATURE_COLS
        for col in FEATURE_COLS:
            if col not in input_data.columns:
                input_data[col] = 0

        result = predict_insomnia_level(input_data)
        self.assertEqual(result, "Moderate")

if __name__ == "__main__":
    unittest.main()
