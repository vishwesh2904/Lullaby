import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app
import pandas as pd

class TestInputUsage(unittest.TestCase):

    @patch("app.st")
    @patch("utils.helper.load_model")
    def test_all_inputs_used_in_prediction(self, mock_load_model, mock_st):
        # Setup mocks
        mock_st.session_state = MagicMock()
        mock_st.session_state.logged_in = True
        mock_st.session_state.username = "testuser"
        mock_st.session_state.get = MagicMock(return_value="Home")

        # Mock model, label_encoder, scaler
        mock_model = MagicMock()
        mock_label_encoder = MagicMock()
        mock_scaler = MagicMock()

        # Setup scaler.transform to check input DataFrame
        def scaler_transform_side_effect(input_df):
            # Assert input_df has all expected columns
            expected_columns = [
                "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
                "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
                "Coping Skills", "Emotion Regulation", "Age"
            ]
            assert list(input_df.columns) == expected_columns, "Input columns mismatch"
            # Assert input_df has correct number of features
            assert input_df.shape[1] == 10, "Input feature count mismatch"
            # Assert input_df values are floats or ints
            for col in expected_columns:
                val = input_df.iloc[0][col]
                assert isinstance(val, (float, int)), f"Value for {col} is not numeric"
            # Return dummy scaled array for prediction
            return [[0.0]*10]

        mock_scaler.transform.side_effect = scaler_transform_side_effect
        mock_model.predict.return_value = [0]
        mock_label_encoder.inverse_transform.return_value = ["No Insomnia"]

        mock_load_model.return_value = (mock_model, mock_label_encoder, mock_scaler)

        # Setup st.number_input to provide test inputs
        test_inputs = [1.0, 2.0, 3.0, 1.5, 0.5, 2.5, 3.5, 1.0, 0.0, 30]
        mock_st.number_input.side_effect = test_inputs + [1]  # last for num_songs
        mock_st.button.return_value = True
        mock_st.success = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.image = MagicMock()
        mock_st.error = MagicMock()
        mock_st.expander = MagicMock(return_value=MagicMock())
        mock_st.form_submit_button = MagicMock(return_value=False)

        # Run main function which triggers prediction
        app.main()

if __name__ == "__main__":
    unittest.main()
