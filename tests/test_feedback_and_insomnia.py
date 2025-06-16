import unittest
import os
import pandas as pd
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feedback import save_feedback, update_insomnia_synthetic_with_questionnaire
from utils import helper

class TestFeedbackAndInsomnia(unittest.TestCase):
    feedback_csv = "data/test_feedback.csv"
    insomnia_csv = "data/test_insomnia.csv"

    def setUp(self):
        # Remove test files if they exist
        if os.path.exists(self.feedback_csv):
            os.remove(self.feedback_csv)
        if os.path.exists(self.insomnia_csv):
            os.remove(self.insomnia_csv)

    def tearDown(self):
        # Remove test files after tests
        if os.path.exists(self.feedback_csv):
            os.remove(self.feedback_csv)
        if os.path.exists(self.insomnia_csv):
            os.remove(self.insomnia_csv)

    def test_save_feedback_creates_file_and_appends(self):
        data1 = {
            "Level of Insomnia Intensity": 3,
            "Sleep Efficiency": 2,
            "Degree of Depression": 0,
            "Sleep Wellness Practices": 0,
            "Sleep-Related Negative Thinking": 0,
            "Anxious Thinking Regarding Sleep": 0,
            "Stress Level": 0,
            "Coping Skills": 0,
            "Emotion Regulation": 0,
            "Recommended Song": "Lullaby by XYZ",
            "Rating": 4
        }
        save_feedback(data1, self.feedback_csv)
        self.assertTrue(os.path.exists(self.feedback_csv))
        df = pd.read_csv(self.feedback_csv)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Recommended Song"], "Lullaby by XYZ")

        data2 = data1.copy()
        data2["Rating"] = 5
        save_feedback(data2, self.feedback_csv)
        df = pd.read_csv(self.feedback_csv)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]["Rating"], 5)

    @patch.object(helper, "retrain_model_with_feedback")
    def test_update_insomnia_synthetic_appends_and_retrains(self, mock_retrain):
        data = {
            "Level of Insomnia Intensity": 3,
            "Sleep Efficiency": 2,
            "Degree of Depression": 0,
            "Sleep Wellness Practices": 0,
            "Sleep-Related Negative Thinking": 0,
            "Anxious Thinking Regarding Sleep": 0,
            "Stress Level": 0,
            "Coping Skills": 0,
            "Emotion Regulation": 0,
            "Total Score": 20,
            "Insomnia Level": "Moderate"
        }
        # Create empty insomnia file to start fresh
        if os.path.exists(self.insomnia_csv):
            os.remove(self.insomnia_csv)
        # Save initial empty dataframe with columns
        pd.DataFrame(columns=data.keys()).to_csv(self.insomnia_csv, index=False)

        # Mock retrain to return dummy accuracy values
        mock_retrain.return_value = (0.95, 0.90)

        result = update_insomnia_synthetic_with_questionnaire(data, self.insomnia_csv)
        self.assertTrue(result)
        df = pd.read_csv(self.insomnia_csv)
        self.assertIn("Level of Insomnia Intensity", df.columns)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Insomnia Level"], "Moderate")

if __name__ == "__main__":
    unittest.main()
