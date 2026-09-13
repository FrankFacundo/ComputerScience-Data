"""List the playlists you created (optionally with their tracks).

Usage:
    python playlists.py            # just the playlists
    python playlists.py --tracks   # playlists + their tracks
"""
import os
import sys

import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ["SPOTIFY_CLIENT_ID"],
    client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
    redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
    scope="playlist-read-private playlist-read-collaborative"))


def my_playlists(only_mine=True):
    """Yield every playlist on /v1/me/playlists, paging 50 at a time."""
    me = sp.current_user()["id"]
    offset = 0
    while True:
        page = sp.current_user_playlists(limit=50, offset=offset)
        for playlist in page["items"]:
            if not only_mine or playlist["owner"]["id"] == me:
                yield playlist
        if page["next"] is None:
            break
        offset += 50


def playlist_tracks(playlist_id):
    """Yield every track of a playlist, paging 100 at a time."""
    offset = 0
    while True:
        page = sp.playlist_items(playlist_id, limit=100, offset=offset)
        for item in page["items"]:
            track = item["track"]
            if track is not None:  # local/removed tracks come back as None
                yield track
        if page["next"] is None:
            break
        offset += 100


with_tracks = "--tracks" in sys.argv

for playlist in my_playlists():
    print(f"{playlist['name']}  ({playlist['tracks']['total']} tracks)  "
          f"{playlist['id']}  public={playlist['public']}")
    if with_tracks:
        for track in playlist_tracks(playlist["id"]):
            artists = ", ".join(a["name"] for a in track["artists"])
            print(f"    {artists} - {track['name']}")
