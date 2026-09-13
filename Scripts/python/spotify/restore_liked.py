"""Re-like every song in liked_songs_backup.json (newest-liked last, so the
order in Liked Songs comes out roughly as it was). Dates cannot be restored -
Spotify stamps 'added_at' at the time of the call."""
import json, os, sys
import spotipy
from spotipy.oauth2 import SpotifyOAuth

D = os.path.dirname(os.path.abspath(__file__))
tracks = json.load(open(os.path.join(D, "liked_songs_backup.json")))
tracks.reverse()  # oldest first
sp = spotipy.Spotify(requests_timeout=30, retries=5, auth_manager=SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
    scope="user-library-read user-library-modify",
    cache_path=os.path.join(D, ".cache")))
ids = [t["id"] for t in tracks]
for i in range(0, len(ids), 50):
    sp.current_user_saved_tracks_add(ids[i:i + 50])
    print(f"restored {min(i+50, len(ids))}/{len(ids)}", flush=True)
print("done")
