import pandas as pd
import numpy as np
from utils.helper import get_spotify_track_link
from utils.spotify_utils import get_spotify_thumbnail

import os

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

if not client_id or not client_secret:
    raise Exception("Spotify API credentials are missing. Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET")

import streamlit as st

# --- Load Datasets ---
def load_lullaby_dataset():
    """
    Load lullaby songs CSV, skipping malformed lines.
    Ensures 'spotify_link' column exists for downstream processing.
    """
    df = pd.read_csv(
        "data/lullaby_songs.csv",
        on_bad_lines="skip"
    )
    # Ensure column names are consistent
    # Normalize original column names
    df.columns = df.columns.str.strip()
    # If 'spotify_link' missing, add it
    if 'spotify_link' not in df.columns:
        df['spotify_link'] = None
    return df


def load_feedback_data():
    """
    Load user feedback CSV safely, skipping bad lines.
    """
    file_path = "data/feedback.csv"
    if not os.path.exists(file_path):
        return pd.DataFrame()
    return pd.read_csv(
        file_path,
        on_bad_lines="skip"
    )


def compute_song_scores(df, feedback_df):
    if feedback_df.empty or "Recommended Song" not in feedback_df.columns:
        df['score'] = 1  # Default score
        return df

    rating_df = (
        feedback_df.groupby("Recommended Song")["Rating"]
        .mean()
        .reset_index()
    )
    rating_df.columns = ["song_label", "avg_rating"]

    df['song_label'] = df.apply(
        lambda row: f"{row['song_name']} by {row['artist_name']} (BPM: {row['bpm']})",
        axis=1
    )

    df = df.merge(rating_df, how="left", on="song_label")
    df['avg_rating'] = df['avg_rating'].fillna(3)
    df['score'] = df['avg_rating']

    return df

from utils.helper import FEATURE_COLS
from joblib import load


def load_model():
    try:
        label_encoder = load("models/label_encoder.joblib")
        scaler = load("models/scaler.joblib")
        model = load("models/insomnia_model.joblib")
        return model, label_encoder, scaler
    except FileNotFoundError:
        raise FileNotFoundError(
            "Model files are missing. Please retrain the model using train_model.py."
        )


def predict_insomnia_level(input_data):
    model, label_encoder, scaler = load_model()

    input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    inversely_related = [
        "Sleep Efficiency", "Sleep Wellness Practices",
        "Coping Skills", "Emotion Regulation"
    ]
    for feat in inversely_related:
        if feat in input_data.columns:
            input_data[feat] = -input_data[feat]

    scaled_input = scaler.transform(input_data.values)
    pred = model.predict(scaled_input)[0]
    level = label_encoder.inverse_transform([pred])[0]
    return level


def recommend_song_from_dataset(insomnia_level, num_songs=1, selected_categories=None):
    df = load_lullaby_dataset()

    # Rename and normalize columns for consistency
    rename_map = {
        'Song Name': 'song_name',
        'Artist Name': 'artist_name',
        'Danceability': 'danceability',
        'Acousticness': 'accousticness',
        'Speechiness': 'speechiness',
        'Camelot': 'camelot',
        'BPM': 'bpm',
        'Duration': 'duration',
        'Popularity': 'popularity',
        'Energy': 'energy',
        'Happiness': 'happiness',
        'Instrumentalness': 'instrument',
        'Liveness': 'liveness',
        'Loudness': 'loudness',
        'Category': 'category',
        'spotify_link': 'spotify_link'
    }
    df.rename(columns=rename_map, inplace=True)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
    )

    # Filter by selected categories
    if selected_categories:
        cats = [c.lower() for c in selected_categories]
        df = df[df['category'].str.lower().isin(cats)]

    feedback_df = load_feedback_data()
    df = compute_song_scores(df, feedback_df)

    # Lullaby selection logic ...
    if insomnia_level in ["No Insomnia", "Negligible"]:
        message = "You have negligible insomnia! Enjoy some relaxing tunes on Spotify."
        return [message], ["https://open.spotify.com/"], [], []

    feature_weights = {
        "Mild": {"energy": -0.5, "happiness": 0.7, "danceability": 0, "accousticness": 0},
        "Moderate": {"energy": -0.7, "happiness": 0.3, "danceability": -0.3, "accousticness": 0.5},
        "High": {"energy": -0.8, "happiness": 0.2, "danceability": -0.4, "accousticness": 0.6},
        "Severe": {"energy": -0.9, "happiness": 0.1, "danceability": -0.5, "accousticness": 0.7},
    }

    weights = feature_weights.get(insomnia_level, {})
    df['weighted_score'] = (
        weights.get('energy', 0) * df['energy'] +
        weights.get('happiness', 0) * df['happiness'] +
        weights.get('danceability', 0) * df['danceability'] +
        weights.get('accousticness', 0) * df['accousticness'] +
        df['score']
    )
    filtered = df[df['weighted_score'] > df['weighted_score'].quantile(0.25)]
    if filtered.empty:
        return ["No suitable lullaby found."], [None], [None], [None]

    filtered = filtered.sort_values('weighted_score', ascending=False)

    labels, links, thumbs, cats = [], [], [], []
    for _, song in filtered.head(num_songs).iterrows():
        label = f"{song['song_name']} by {song['artist_name']} (BPM: {song['bpm']})"
        spotify_link = song.get('spotify_link') or get_spotify_track_link(song['song_name'], song['artist_name'])
        thumbnail = get_spotify_thumbnail(spotify_link)
        labels.append(label)
        links.append(spotify_link)
        thumbs.append(thumbnail)
        cats.append(song['category'])

    return labels, links,thumbs, cats
