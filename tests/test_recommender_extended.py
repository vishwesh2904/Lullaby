import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from utils.recommender import recommend_song_from_dataset, predict_insomnia_level

class TestRecommendSongFromDataset(unittest.TestCase):

    def setUp(self):
        # Sample lullaby dataset mock
        self.sample_df = pd.DataFrame({
            'song_name': ['Song1', 'Song2', 'Song3'],
            'artist_name': ['Artist1', 'Artist2', 'Artist3'],
            'bpm': [60, 70, 80],
            'category': ['Calm', 'Calm', 'Energetic'],
            'energy': [0.1, 0.2, 0.9],
            'happiness': [0.5, 0.6, 0.1],
            'danceability': [0.3, 0.4, 0.2],
            'accousticness': [0.7, 0.8, 0.1],
            'score': [3, 4, 2]
        })

    @patch('utils.recommender.load_lullaby_dataset')
    @patch('utils.recommender.load_feedback_data')
    def test_recommend_no_insomnia(self, mock_feedback, mock_lullaby):
        mock_lullaby.return_value = self.sample_df
        mock_feedback.return_value = pd.DataFrame()
        labels, links, thumbs, cats = recommend_song_from_dataset("No Insomnia", num_songs=1)
        self.assertIn("negligible insomnia", labels[0].lower())

    @patch('utils.recommender.load_lullaby_dataset')
    @patch('utils.recommender.load_feedback_data')
    def test_recommend_mild_insomnia(self, mock_feedback, mock_lullaby):
        mock_lullaby.return_value = self.sample_df
        mock_feedback.return_value = pd.DataFrame()
        labels, links, thumbs, cats = recommend_song_from_dataset("Mild", num_songs=2)
        self.assertEqual(len(labels), 2)
        self.assertTrue(all(isinstance(label, str) for label in labels))

    @patch('utils.recommender.load_lullaby_dataset')
    @patch('utils.recommender.load_feedback_data')
    def test_recommend_with_category_filter(self, mock_feedback, mock_lullaby):
        mock_lullaby.return_value = self.sample_df
        mock_feedback.return_value = pd.DataFrame()
        labels, links, thumbs, cats = recommend_song_from_dataset("Moderate", num_songs=2, selected_categories=["Calm"])
        self.assertTrue(all(cat.lower() == "calm" for cat in cats))

    @patch('utils.recommender.load_lullaby_dataset')
    @patch('utils.recommender.load_feedback_data')
    def test_recommend_no_songs_found(self, mock_feedback, mock_lullaby):
        mock_lullaby.return_value = pd.DataFrame(columns=self.sample_df.columns)
        mock_feedback.return_value = pd.DataFrame()
        labels, links, thumbs, cats = recommend_song_from_dataset("Severe", num_songs=1)
        self.assertIn("no suitable lullaby", labels[0].lower())

class TestIntegrationPredictionRecommendation(unittest.TestCase):

    @patch('utils.recommender.load_lullaby_dataset')
    @patch('utils.recommender.load_feedback_data')
    @patch('utils.recommender.load_model')
    def test_full_flow(self, mock_load_model, mock_feedback, mock_lullaby):
        # Setup mocks
        mock_lullaby.return_value = pd.DataFrame({
            'song_name': ['Song1'],
            'artist_name': ['Artist1'],
            'bpm': [60],
            'category': ['Calm'],
            'energy': [0.1],
            'happiness': [0.5],
            'danceability': [0.3],
            'accousticness': [0.7],
            'score': [3]
        })
        mock_feedback.return_value = pd.DataFrame()

        mock_model = MagicMock()
        mock_label_encoder = MagicMock()
        mock_scaler = MagicMock()

        mock_label_encoder.classes_ = ["Negligible", "Mild", "Moderate", "High", "Severe"]
        mock_label_encoder.inverse_transform.return_value = ["Mild"]
        mock_model.predict.return_value = [1]
        mock_scaler.transform.side_effect = lambda x: x

        mock_load_model.return_value = (mock_model, mock_label_encoder, mock_scaler)

        input_data = pd.DataFrame({
            "Sleep Efficiency": [5],
            "Sleep Wellness Practices": [3],
            "Coping Skills": [4],
            "Emotion Regulation": [2],
            "Level of Insomnia Intensity": [1],
            "Degree of Depression": [1],
            "Sleep-Related Negative Thinking": [1],
            "Anxious Thinking Regarding Sleep": [1],
            "Stress Level": [1],
            "Age": [30]
        })

        insomnia_level = predict_insomnia_level(input_data)
        labels, links, thumbs, cats = recommend_song_from_dataset(insomnia_level, num_songs=1)

        self.assertEqual(insomnia_level, "Mild")
        self.assertTrue(len(labels) > 0)
        self.assertTrue(all(isinstance(label, str) for label in labels))

if __name__ == "__main__":
    unittest.main()
