import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_thumbnail(spotify_link):
    try:
        if "track" in spotify_link:
            track_id = spotify_link.split("/")[-1].split("?")[0]
            track = sp.track(track_id)
            return track["album"]["images"][0]["url"] if track["album"]["images"] else None
        return None
    except Exception as e:
        print(f"Error fetching thumbnail: {e}")
        return None
