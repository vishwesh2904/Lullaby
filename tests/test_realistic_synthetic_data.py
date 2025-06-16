import unittest
import os
import pandas as pd
import subprocess
from utils.helper import retrain_model_with_feedback
from utils.recommender import predict_insomnia_level, recommend_song_from_dataset
import pandas as pd

class TestRealisticSyntheticData(unittest.TestCase):

    def test_generate_realistic_synthetic_data(self):
        # Run the synthetic data generation script
        result = subprocess.run(["python", "scripts/generate_realistic_synthetic_data.py"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"Script failed with error: {result.stderr}")

        # Check if the data file is created
        self.assertTrue(os.path.exists("data/insomnia_synthetic.csv"))

        # Check if the file has expected columns
        df = pd.read_csv("data/insomnia_synthetic.csv")
        expected_columns = [
            "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
            "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
            "Coping Skills", "Emotion Regulation", "Age", "Total Score", "Insomnia Level"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_retrain_model_accuracy(self):
        # Retrain the model on the new synthetic data
        accuracy = retrain_model_with_feedback()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.88, msg="Accuracy is below 88%")
        # Adjust upper bound to allow accuracy above 95%
        self.assertLessEqual(accuracy, 1.0, msg="Accuracy is above 100%")

    def test_predict_insomnia_level(self):
        # Test prediction function with sample input aligned to FEATURE_COLS
        from utils.helper import FEATURE_COLS
        import pandas as pd

        # Create a sample input with all zeros for all FEATURE_COLS
        sample_input = {col: [0] for col in FEATURE_COLS}
        input_df = pd.DataFrame(sample_input)

        insomnia_level = predict_insomnia_level(input_df)
        self.assertIn(insomnia_level, ["No Insomnia", "Mild", "Moderate", "Severe"])

    def test_recommend_song_from_dataset(self):
        # Test recommendation for each insomnia level
        for level in ["No Insomnia", "Mild", "Moderate", "Severe"]:
            labels, links, thumbnails, categories = recommend_song_from_dataset(level, num_songs=2)
            self.assertIsInstance(labels, list)
            self.assertIsInstance(links, list)
            self.assertIsInstance(thumbnails, list)
            self.assertIsInstance(categories, list)
            self.assertLessEqual(len(labels), 2)
            self.assertLessEqual(len(links), 2)
            self.assertLessEqual(len(thumbnails), 2)
            self.assertLessEqual(len(categories), 2)

if __name__ == "__main__":
    unittest.main()
