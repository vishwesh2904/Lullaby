import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from joblib import dump
import os
from sklearn.impute import SimpleImputer

FEATURE_COLS = [
    "Level of Insomnia Intensity", "Sleep Efficiency", "Degree of Depression", "Sleep Wellness Practices",
    "Sleep-Related Negative Thinking", "Anxious Thinking Regarding Sleep", "Stress Level",
    "Coping Skills", "Emotion Regulation", "Age"
]

def train_model():
    print("DEBUG: Reading insomnia_synthetic.csv in train_model.py")
    cols_to_load = FEATURE_COLS + ["Total Score", "Insomnia Level"]

    # Safely read only the expected columns, skipping any malformed rows
    try:
        df = pd.read_csv(
            "data/insomnia_synthetic.csv",
            on_bad_lines="skip",      # drop rows with wrong field counts
            usecols=cols_to_load      # load only the 12 columns we need
        )
    except Exception as e:
        print("Error reading CSV:", e)
        raise

    # Ensure we actually got all the columns
    missing = set(cols_to_load) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Clean and filter labels
    df['Insomnia Level'] = df['Insomnia Level'].astype(str).str.strip()
    df = df[~df['Insomnia Level'].isin(['', 'nan', 'NaN', 'None', 'none'])]
    df = df.dropna(subset=['Insomnia Level'])

    # Features and target
    X = df[FEATURE_COLS].values
    y = df['Insomnia Level'].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Remove any unseen labels in the test set
    train_labels = set(y_train)
    mask = [lbl in train_labels for lbl in y_test]
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_scaled = imputer.fit_transform(X_train_scaled)
    X_test_scaled = imputer.transform(X_test_scaled)

    # Define models and hyperparameters
    models = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = ""
    best_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, spec in models.items():
        grid = GridSearchCV(
            spec["model"],
            spec["params"],
            cv=cv,
            scoring='accuracy',
            n_jobs=1,
            verbose=1
        )
        grid.fit(X_train_scaled, y_train_enc)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name
            best_params = grid.best_params_

    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test_enc, y_pred)
    print(f"Best Model: {best_name} with params {best_params}")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    dump(best_model, "models/insomnia_model.joblib")
    dump(scaler, "models/scaler.joblib")
    dump(label_encoder, "models/label_encoder.joblib")
    print("✅ Saved model, scaler, and label encoder to 'models/'")

    return acc

if __name__ == "__main__":
    train_model()
