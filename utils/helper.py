import os
import pandas as pd
import joblib
import base64
import requests
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
import logging
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --- Constants ---
FEATURE_COLS = [
    "Level of Insomnia Intensity",
    "Sleep Efficiency",
    "Degree of Depression",
    "Sleep Wellness Practices",
    "Sleep-Related Negative Thinking",
    "Anxious Thinking Regarding Sleep",
    "Stress Level",
    "Coping Skills",
    "Emotion Regulation",
    "Age"
]

from scripts.generate_realistic_synthetic_data import generate_realistic_insomnia_synthetic_data

# --- Synthetic Data Regeneration ---
def regenerate_dataset(n_samples=10000):
    """
    Regenerate the insomnia synthetic dataset using the improved generation function.
    """
    generate_realistic_insomnia_synthetic_data(n_samples)

# --- Initial Model Training ---
def train_and_save_model(df: pd.DataFrame) -> float:
    """
    Train a RandomForest model on the provided DataFrame and save artifacts.
    """
    X = df[FEATURE_COLS].copy()
    y = df["Insomnia Level"]

    # Invert inversely related features before scaling
    inversely_related = [
        "Sleep Efficiency",
        "Sleep Wellness Practices",
        "Coping Skills",
        "Emotion Regulation"
    ]
    for feat in inversely_related:
        if feat in X.columns:
            X[feat] = -X[feat]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler.feature_names_in_ = np.array(X.columns)

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=2,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_scaled, y_encoded)

    # Persist artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/insomnia_model.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Evaluation
    accuracy = model.score(X_scaled, y_encoded)
    y_pred = model.predict(X_scaled)
    f1 = f1_score(y_encoded, y_pred, average='weighted')
    cm = confusion_matrix(y_encoded, y_pred)

    print("="*40)
    print("Model Evaluation Metrics:")
    print("="*40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("="*40)

    return accuracy

# --- Load Model Helper ---
@st.cache_resource
def load_model():
    """
    Load model, encoder, scaler; validate dataset schema.
    """
    # Paths
    model_path = "models/insomnia_model.joblib"
    encoder_path = "models/label_encoder.joblib"
    scaler_path = "models/scaler.joblib"

    # Ensure artifacts exist
    if not (os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Missing model artifacts in 'models/' directory.")

    # Load
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    # Validate data
    COLUMNS = FEATURE_COLS + ["Total Score", "Insomnia Level"]
    df = pd.read_csv("data/insomnia_synthetic.csv", on_bad_lines="skip", usecols=COLUMNS)   
    expected_cols = FEATURE_COLS + ["Total Score", "Insomnia Level"]
    if set(expected_cols).issubset(df.columns):
        df = df[expected_cols]
    else:
        raise ValueError(
            f"Dataset missing columns. Expected at least: {expected_cols}, Found: {list(df.columns)}"
        )

    dataset_labels = set(df["Insomnia Level"].unique())
    encoder_labels = set(label_encoder.classes_)
    if not dataset_labels.issubset(encoder_labels):
        # Refit encoder on full labels
        new_le = LabelEncoder()
        new_le.fit(df["Insomnia Level"])
        label_encoder = new_le

    return model, label_encoder, scaler

# --- Retrain with Feedback ---
def retrain_model_with_feedback():
    """
    Retrain and tune models using feedback data, with feature selection and SMOTE.
    """
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv("data/insomnia_synthetic.csv", on_bad_lines="skip")
    expected_cols = FEATURE_COLS + ["Total Score", "Insomnia Level"]
    if set(expected_cols).issubset(df.columns):
        df = df[expected_cols]
    else:
        raise ValueError(
            f"Dataset missing columns. Expected at least: {expected_cols}, Found: {list(df.columns)}"
        )

    logging.info(f"Data shape before fill NA: {df.shape}")
    df = df.fillna(df.median(numeric_only=True))
    logging.info(f"Data shape after fill NA: {df.shape}")
    if df.isnull().any().any():
        raise ValueError("Data still contains missing values after imputation.")

    # Encode labels
    le = LabelEncoder()
    df["Insomnia Level Encoded"] = le.fit_transform(df["Insomnia Level"])

    # Features/target
    X = df[FEATURE_COLS].copy()
    y = df["Insomnia Level Encoded"]

    # Invert selected features
    inv = ["Sleep Efficiency", "Sleep Wellness Practices", "Coping Skills", "Emotion Regulation"]
    for feat in inv:
        X[feat] = -X[feat]

    # Feature selection
    selector = SelectKBest(mutual_info_classif, k='all')
    X_sel = selector.fit_transform(X, y)
    sel_feats = [FEATURE_COLS[i] for i, ok in enumerate(selector.get_support()) if ok]
    logging.info(f"Selected features: {sel_feats}")

    # Scale + SMOTE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    if not len(X_res) or not len(y_res):
        raise ValueError("No data after SMOTE resampling.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Parameter grids
    rf_params = {'n_estimators': [200,300], 'max_depth':[20,30,40,None], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,2,4], 'bootstrap':[True,False]}
    xgb_params = {'n_estimators':[100,200,300], 'max_depth':[3,6,9], 'learning_rate':[0.01,0.1,0.2], 'subsample':[0.7,1.0], 'colsample_bytree':[0.7,1.0]}

    rf = RandomForestClassifier(random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_rf = GridSearchCV(rf, rf_params, cv=cv, n_jobs=-1, verbose=1)
    grid_xgb = GridSearchCV(xgb, xgb_params, cv=cv, n_jobs=-1, verbose=1)

    logging.info("Grid search RF...")
    grid_rf.fit(X_train, y_train)
    logging.info("Grid search XGB...")
    grid_xgb.fit(X_train, y_train)

    # Choose best
    acc_rf = grid_rf.best_score_
    acc_xgb = grid_xgb.best_score_
    if acc_xgb > acc_rf:
        best = grid_xgb.best_estimator_
        logging.info("Selected XGBoost")
    else:
        best = grid_rf.best_estimator_
        logging.info("Selected RandomForest")

    # Plot importances
    if hasattr(best, 'feature_importances_'):
        imp = best.feature_importances_
    else:
        imp = best.coef_[0]
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(imp)), imp[idx], align='center')
    plt.xticks(range(len(imp)), [sel_feats[i] for i in idx], rotation=45)
    plt.tight_layout()
    plt.savefig("models/feature_importances.png")
    plt.close()

    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/insomnia_model.joblib")
    joblib.dump(le, "models/label_encoder.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    return best.score(X_test, y_test)

# --- Spotify API Helper ---
def get_spotify_track_link(song_name: str, artist_name: str) -> str | None:
    """
    Fetch Spotify track link via Client Credentials flow.
    """
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Missing Spotify credentials.")
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    token_resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "client_credentials"}
    )
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]
    search = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": f"{song_name} {artist_name}", "type":"track", "limit":1}
    )
    search.raise_for_status()
    items = search.json().get("tracks", {}).get("items", [])
    return items[0]["external_urls"]["spotify"] if items else None

# --- Append Feedback Entry ---
def append_to_insomnia_data(new_entry: list, file_path: str = "data/insomnia_synthetic.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cols = FEATURE_COLS + ["Total Score", "Insomnia Level"]
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, on_bad_lines="skip")
        if len(new_entry) != len(df.columns):
            raise ValueError("New entry column count mismatch.")
        df = pd.concat([df, pd.DataFrame([new_entry], columns=df.columns)], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry], columns=cols)
    df.to_csv(file_path, index=False)

# --- Expose Feature Questions ---
def get_questions() -> list[str]:
    return FEATURE_COLS
