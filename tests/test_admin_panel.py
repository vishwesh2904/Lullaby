import sys
import os

# Add the parent directory to sys.path to allow imports from admin package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import admin.admin_panel as admin_panel

def test_admin_panel_import():
    assert admin_panel is not None
