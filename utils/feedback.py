import pandas as pd
import os

def save_feedback(data, feedback_csv_path="data/feedback.csv"):
    if not os.path.exists(feedback_csv_path):
        df = pd.DataFrame(columns=data.keys())
        df.to_csv(feedback_csv_path, index=False)

    try:
        df = pd.read_csv(feedback_csv_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=data.keys())

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(feedback_csv_path, index=False)

def update_insomnia_synthetic_with_questionnaire(data, insomnia_csv_path="data/insomnia_synthetic.csv"):
    if not os.path.exists(insomnia_csv_path):
        df = pd.DataFrame(columns=data.keys())
        df.to_csv(insomnia_csv_path, index=False)

    try:
        df = pd.read_csv(insomnia_csv_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=data.keys())

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(insomnia_csv_path, index=False)

    from utils.helper import retrain_model_with_feedback
    accuracy = retrain_model_with_feedback()
    return accuracy > 0.0
import pandas as pd
import os
from utils.helper import FEATURE_COLS, retrain_model_with_feedback

# --- Feedback Saving ---
def save_feedback(data: dict, feedback_csv_path: str = "data/feedback.csv") -> None:
    """
    Append a feedback record to feedback.csv, skipping malformed rows.
    """
    os.makedirs(os.path.dirname(feedback_csv_path), exist_ok=True)
    cols = list(data.keys())

    # Initialize file with correct headers if missing
    if not os.path.exists(feedback_csv_path):
        pd.DataFrame(columns=cols).to_csv(feedback_csv_path, index=False)

    # Read existing feedback, skip bad lines, only load known columns
    try:
        df = pd.read_csv(
            feedback_csv_path,
            on_bad_lines="skip",
            usecols=cols
        )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=cols)

    # Append new feedback
    df = pd.concat([df, pd.DataFrame([data], columns=cols)], ignore_index=True)
    df.to_csv(feedback_csv_path, index=False)

# --- Insomnia Synthetic Update ---
def update_insomnia_synthetic_with_questionnaire(
    data: dict,
    insomnia_csv_path: str = "data/insomnia_synthetic.csv"
) -> bool:
    """
    Append questionnaire results to insomnia_synthetic.csv and retrain model.
    """
    os.makedirs(os.path.dirname(insomnia_csv_path), exist_ok=True)

    # Expected columns in the synthetic dataset
    expected_cols = FEATURE_COLS + ["Total Score", "Insomnia Level"]

    # Initialize file if missing
    if not os.path.exists(insomnia_csv_path):
        pd.DataFrame(columns=expected_cols).to_csv(insomnia_csv_path, index=False)

    # Read existing data, skip malformed rows, only load expected columns
    try:
        df = pd.read_csv(
            insomnia_csv_path,
            on_bad_lines="skip",
            usecols=expected_cols
        )
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=expected_cols)

    # Append new row using the correct column order
    df = pd.concat(
        [df, pd.DataFrame([data], columns=expected_cols)],
        ignore_index=True
    )
    df.to_csv(insomnia_csv_path, index=False)

    # Retrain model and return success status
    accuracy = retrain_model_with_feedback()
    return accuracy > 0.0
