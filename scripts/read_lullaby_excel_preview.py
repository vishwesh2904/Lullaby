import pandas as pd

def preview_excel(file_path, num_rows=20):
    try:
        df = pd.read_excel(file_path)
        print(df.head(num_rows))
    except Exception as e:
        print(f"Error reading Excel file: {e}")

if __name__ == "__main__":
    preview_excel("data/lullaby_songs.csv")
