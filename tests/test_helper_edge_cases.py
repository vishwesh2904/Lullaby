import unittest
import os
import pandas as pd
from utils import helper

class TestHelperEdgeCases(unittest.TestCase):

    def setUp(self):
        # Clean up insomnia_synthetic.csv before each test
        if os.path.exists("data/insomnia_synthetic.csv"):
            os.remove("data/insomnia_synthetic.csv")

    def test_append_to_insomnia_data_with_empty_entry(self):
        # Append empty dict should create file with no rows
        helper.append_to_insomnia_data({})
        self.assertTrue(os.path.exists("data/insomnia_synthetic.csv"))
        try:
            df = pd.read_csv("data/insomnia_synthetic.csv")
            self.assertEqual(len(df), 0)
        except pd.errors.EmptyDataError:
            # If file is empty, treat as zero rows
            self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
