import pandas as pd

def clean_insomnia_data(input_path="data/insomnia_synthetic.csv", output_path="data/insomnia_synthetic_cleaned.csv"):
    # Read the dataset
    df = pd.read_csv(input_path)

    # Drop rows with any missing values
    df_cleaned = df.dropna()

    # Drop unused columns if they exist
    unused_columns = ["Username", "Timestamp"]
    for col in unused_columns:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])

    # Save the cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    clean_insomnia_data()
