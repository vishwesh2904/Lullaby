import pandas as pd
import numpy as np
import os

# --- Constants ---
COLUMNS = [
    "Level of Insomnia Intensity",
    "Sleep Efficiency",
    "Degree of Depression",
    "Sleep Wellness Practices",
    "Sleep-Related Negative Thinking",
    "Anxious Thinking Regarding Sleep",
    "Stress Level",
    "Coping Skills",
    "Emotion Regulation",
    "Age",
    "Total Score",
    "Insomnia Level"
]

INSOMNIA_LEVELS = ["Negligible", "Mild", "Moderate", "High", "Severe"]


def generate_features(level: str) -> list[float]:
    """
    Generate a single feature vector (10 features + age) for a given insomnia level.
    """
    params = {
        "Negligible": {
            "neg": [0.2, 0.2, 0.2, 0.2, 0.2],
            "pos": [3.8, 3.8, 3.8, 3.8],
            "age": (30, 40)
        },
        "Mild": {
            "neg": [1.0, 1.0, 1.0, 1.0, 1.0],
            "pos": [3.0, 3.0, 3.0, 3.0],
            "age": (30, 45)
        },
        "Moderate": {
            "neg": [2.0, 2.0, 2.0, 2.0, 2.0],
            "pos": [2.0, 2.0, 2.0, 2.0],
            "age": (35, 50)
        },
        "High": {
            "neg": [3.0, 3.0, 3.0, 3.0, 3.0],
            "pos": [1.0, 1.0, 1.0, 1.0],
            "age": (40, 60)
        },
        "Severe": {
            "neg": [3.8, 3.8, 3.8, 3.8, 3.8],
            "pos": [0.2, 0.2, 0.2, 0.2],
            "age": (45, 65)
        }
    }
    std_dev = 0.15
    p = params[level]

    # Generate "negative" and "positive" feature values with noise
    neg_vals = [np.clip(np.random.normal(m, std_dev), 0, 4) for m in p["neg"]]
    pos_vals = [np.clip(np.random.normal(m, std_dev), 0, 4) for m in p["pos"]]

    # Random age within given range
    age = int(np.random.uniform(*p["age"]))

    # Assemble features in the same order as COLUMNS[:10]
    features = [
        neg_vals[0],   # Level of Insomnia Intensity
        pos_vals[0],   # Sleep Efficiency
        neg_vals[1],   # Degree of Depression
        pos_vals[1],   # Sleep Wellness Practices
        neg_vals[2],   # Sleep-Related Negative Thinking
        neg_vals[3],   # Anxious Thinking Regarding Sleep
        neg_vals[4],   # Stress Level
        pos_vals[2],   # Coping Skills
        pos_vals[3],   # Emotion Regulation
        age            # Age
    ]

    return features


def generate_realistic_insomnia_synthetic_data(n_samples: int = 10000) -> None:
    """
    Create a synthetic dataset of insomnia features and save to CSV.
    """
    data_rows: list[list] = []
    samples_per_class = n_samples // len(INSOMNIA_LEVELS)

    # Generate balanced samples per class
    for level in INSOMNIA_LEVELS:
        for _ in range(samples_per_class):
            feats = generate_features(level)
            total_score = sum(feats[:-1])  # exclude age
            row = feats + [total_score, level]
            data_rows.append(row)

    # Add fixed edge cases (zeros, fours, borderline)
    for _ in range(50):
        zeros = [0] * 9
        total_zeros = sum(zeros)
        data_rows.append(zeros + [35, total_zeros, "Mild"])

        fours = [4] * 9
        total_fours = sum(fours)
        data_rows.append(fours + [65, total_fours, "Severe"])

        border = [(m + 2.0) / 2 for m in [1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 2.0, 3.0, 3.0]]
        total_border = sum(border)
        data_rows.append(border + [45, total_border, "Moderate"])

    # Filter out any malformed rows
    valid_rows = []
    for idx, row in enumerate(data_rows):
        if len(row) == len(COLUMNS):
            valid_rows.append(row)
        else:
            print(f"Dropping malformed row {idx}: expected {len(COLUMNS)} cols, got {len(row)}")

    # Construct DataFrame and save
    df = pd.DataFrame(valid_rows, columns=COLUMNS)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/insomnia_synthetic.csv", index=False)
    print(f"Generated {len(df)} samples and saved to data/insomnia_synthetic.csv")


if __name__ == "__main__":
    generate_realistic_insomnia_synthetic_data()
