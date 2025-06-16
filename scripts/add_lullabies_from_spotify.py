import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

def add_lullabies_to_csv(csv_path="data/lullaby_songs.csv", search_queries=None, max_songs_per_query=5):
    if search_queries is None:
        search_queries = ["lullaby", "sleep music", "baby lullaby", "calm lullaby"]

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise Exception("Spotify API credentials are missing. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET")

    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Load existing lullaby songs CSV or create new DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["song_name", "artist_name", "bpm", "spotify_link"])

    new_songs = []

    for query in search_queries:
        results = sp.search(q=query, type="track", limit=max_songs_per_query)
        for item in results['tracks']['items']:
            song_name = item['name']
            artist_name = item['artists'][0]['name']
            spotify_link = item['external_urls']['spotify']

            # Get audio features for BPM
            audio_features = sp.audio_features(item['id'])
            bpm = audio_features[0]['tempo'] if audio_features and audio_features[0] else None

            # Check if song already exists in df
            exists = ((df['song_name'] == song_name) & (df['artist_name'] == artist_name)).any()
            if not exists:
                new_songs.append({
                    "song_name": song_name,
                    "artist_name": artist_name,
                    "bpm": bpm,
                    "spotify_link": spotify_link
                })

    if new_songs:
        new_df = pd.DataFrame(new_songs)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"Added {len(new_songs)} new lullaby songs to {csv_path}")
    else:
        print("No new lullaby songs found to add.")

if __name__ == "__main__":
    add_lullabies_to_csv()
