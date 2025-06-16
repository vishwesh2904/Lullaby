import unittest
from unittest.mock import patch, MagicMock
import app
import pandas as pd

class TestAppIntegration(unittest.TestCase):

    @patch("app.st")
    @patch("app.load_model")
    @patch("app.recommend_song_from_dataset")
    @patch("app.save_insomnia_entry")
    @patch("app.collect_feedback")
    def test_predict_and_recommend_flow(self, mock_collect_feedback, mock_save_entry, mock_recommend, mock_load_model, mock_st):
        # Setup mocks
        mock_st.session_state = MagicMock()
        mock_st.session_state.logged_in = True
        mock_st.session_state.username = "testuser"
        mock_st.session_state.get = MagicMock(return_value="Home")
        mock_load_model.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_recommend.return_value = (["Song1", "Song2"], ["link1", "link2"], ["thumb1", "thumb2"])
        mock_st.number_input.side_effect = [2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 1.0, 25, 3]  # inputs + num_songs
        mock_st.button.return_value = True
        mock_st.success = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.image = MagicMock()
        mock_st.error = MagicMock()
        mock_st.expander = MagicMock(return_value=MagicMock())
        mock_st.form_submit_button = MagicMock(return_value=False)

        # Run main function
        app.main()

        # Assertions
        mock_load_model.assert_called_once()
        mock_save_entry.assert_called_once()
        mock_recommend.assert_called_once()
        mock_collect_feedback.assert_called_once()
        mock_st.success.assert_any_call("🧠 Predicted Insomnia Level: **No Insomnia**")
        mock_st.success.assert_any_call("📁 Your input has been saved to the dataset.")
