import unittest
import os
import tempfile
from auth import auth_manager

class TestAuthManagerEdgeCases(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_db_path = os.path.join(self.test_dir.name, "users.xlsx")
        auth_manager.DB_PATH = self.test_db_path

    def tearDown(self):
        self.test_dir.cleanup()

    def test_signup_user_with_empty_username_email_password(self):
        auth_manager.create_user_table()
        result = auth_manager.signup_user("", "", "")
        self.assertTrue(result)  # Currently no validation, so it allows empty strings

    def test_login_user_with_nonexistent_email(self):
        auth_manager.create_user_table()
        login_result = auth_manager.login_user("nonexistent@example.com", "password")
        self.assertFalse(login_result)

    def test_get_username_by_email_with_nonexistent_email(self):
        auth_manager.create_user_table()
        username = auth_manager.get_username_by_email("nonexistent@example.com")
        self.assertIsNone(username)

if __name__ == "__main__":
    unittest.main()
