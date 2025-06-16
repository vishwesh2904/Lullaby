import pandas as pd
import hashlib
import os
from datetime import datetime

DB_PATH = "auth/users.xlsx"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_table():
    os.makedirs("auth", exist_ok=True)
    if not os.path.exists(DB_PATH):
        df = pd.DataFrame(columns=["username", "email", "password"])
        df.to_excel(DB_PATH, index=False)

def load_users():
    if os.path.exists(DB_PATH):
        return pd.read_excel(DB_PATH)
    else:
        return pd.DataFrame(columns=["username", "email", "password"])

def save_users(df):
    df.to_excel(DB_PATH, index=False)

def signup_user(username, email, password):
    df = load_users()
    if (df["username"] == username).any() or (df["email"] == email).any():
        return False
    hashed = hash_password(password)
    new_user = pd.DataFrame([{"username": username, "email": email, "password": hashed}])
    df = pd.concat([df, new_user], ignore_index=True)
    save_users(df)
    return True

def login_user(email, password):
    df = load_users()
    hashed = hash_password(password)
    user = df[(df["email"] == email) & (df["password"] == hashed)]
    return not user.empty

def get_username_by_email(email):
    df = load_users()
    user = df[df["email"] == email]
    if not user.empty:
        return user.iloc[0]["username"]
    return None

def save_prediction(username, insomnia_level):
    # This function can be updated similarly if needed
    pass

def save_song_feedback(username, song_name, feedback):
    # This function can be updated similarly if needed
    pass
