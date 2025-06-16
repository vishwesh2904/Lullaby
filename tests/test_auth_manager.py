import unittest
import os
import pandas as pd
import tempfile
from auth import auth_manager

class TestAuthManager(unittest.TestCase):

    def setUp(self):
        # Use a temporary file for DB_PATH
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.test_dir.name, "users.xlsx")
        auth_manager.DB_PATH = self.test_db_path

    def tearDown(self):
        self.test_dir.cleanup()

    def test_create_user_table_creates_file(self):
        auth_manager.create_user_table()
        self.assertTrue(os.path.exists(self.test_db_path))

    def test_signup_and_login_user(self):
        auth_manager.create_user_table()
        result = auth_manager.signup_user("user1", "user1@example.com", "password")
        self.assertTrue(result)

        # Duplicate username or email should fail
        result2 = auth_manager.signup_user("user1", "user2@example.com", "password")
        self.assertFalse(result2)
        result3 = auth_manager.signup_user("user2", "user1@example.com", "password")
        self.assertFalse(result3)

        # Test login with correct and incorrect credentials
        login_success = auth_manager.login_user("user1@example.com", "password")
        self.assertTrue(login_success)
        login_fail = auth_manager.login_user("user1@example.com", "wrongpassword")
        self.assertFalse(login_fail)

    def test_get_username_by_email(self):
        auth_manager.create_user_table()
        auth_manager.signup_user("user1", "user1@example.com", "password")
        username = auth_manager.get_username_by_email("user1@example.com")
        self.assertEqual(username, "user1")
        username_none = auth_manager.get_username_by_email("nonexistent@example.com")
        self.assertIsNone(username_none)

if __name__ == "__main__":
    unittest.main()
