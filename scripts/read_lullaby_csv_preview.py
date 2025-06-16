import csv

def preview_csv(file_path, num_lines=20):
    with open(file_path, 'r', encoding='latin1') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(row)
            if i + 1 >= num_lines:
                break

if __name__ == "__main__":
    preview_csv("data/lullaby_songs.csv")
