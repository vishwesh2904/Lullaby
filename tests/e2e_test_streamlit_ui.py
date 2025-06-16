import unittest
from playwright.sync_api import sync_playwright
import subprocess
import time
import os
import signal

class TestStreamlitE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the Streamlit app as a subprocess
        cls.process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        # Wait for the server to start
        time.sleep(10)

    @classmethod
    def tearDownClass(cls):
        # Terminate the Streamlit app subprocess
        if cls.process:
            if os.name == 'nt':
                cls.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                cls.process.terminate()
            cls.process.wait()

    def test_home_page_prediction_and_recommendation(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8501")

            # Wait for login/signup sidebar
            page.wait_for_selector("text=Authentication")

            # Signup flow
            page.click("text=Signup")
            page.fill("input[placeholder='Choose a Username']", "testuser")
            page.fill("input[placeholder='Email']", "testuser@example.com")
            page.fill("input[placeholder='Password']", "password123")
            page.click("text=Signup")

            # Wait for login success or error
            page.wait_for_timeout(3000)

            # If logged in, proceed to Home page inputs
            if page.is_visible("text=🧠 Insomnia Detection & Smart Lullaby Recommender"):
                # Fill inputs for symptoms and age
                for i in range(9):
                    selector = f"input[key='input_{i}']"
                    page.fill(selector, "2.0")
                page.fill("input[key='input_9']", "25")  # Age
                page.fill("input[key='num_songs']", "3")

                # Click Predict & Recommend button
                page.click("text=Predict & Recommend")

                # Wait for prediction result
                page.wait_for_selector("text=Predicted Insomnia Level")

                # Check for recommended songs
                self.assertTrue(page.is_visible("text=Recommended Song 1"))

            browser.close()

if __name__ == "__main__":
    unittest.main()
