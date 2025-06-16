import pandas as pd

def preview_csv(file_path, num_rows=20):
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        print(df.head(num_rows))
    except Exception as e:
        print(f"Error reading CSV file: {e}")

if __name__ == "__main__":
    preview_csv("data/lullaby_songs.csv")
