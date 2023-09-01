import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pprint import pprint

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                                                           client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
                                                    redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
                                               scope="playlist-read-private playlist-read-collaborative user-follow-read user-read-playback-position user-top-read user-read-recently-played user-library-read user-read-private"))

# Get current user info
user = sp.current_user()
# pprint(user)
user_name = user["display_name"]
user_country = user["country"]
print("Username:", user_name)
print("Country:", user_country, "\n")

# Get top 1 tracks
top = sp.current_user_top_tracks(limit=20, offset=0)
top_song = top['items'][1]
# pprint(top_song)
top_song_uri = top_song['uri']
print("Top song URI:", top_song_uri, '\n')

# Get track details
track_info = sp.track(top_song_uri)  # Replace with the Spotify URI of the track you want information about
# pprint(track_info)
track_album_uri = track_info["album"]["uri"]
track_name = track_info['name']
track_artist = track_info['artists'][0]['name']
track_artist_uri = track_info['artists'][0]['uri']
print("Track Name:", track_name)
print("Artist:", track_artist, '\n')
# print("Uri:", track_album_uri, '\n')

# Get album details
album_info = sp.album(track_album_uri)
track_album_genre = album_info["genres"]
print("Track album genre:", track_album_genre, '\n')

# Get artist details
artist_info = sp.artist(track_artist_uri)
artist_genres = artist_info["genres"]
print("Artist genres", artist_genres)
