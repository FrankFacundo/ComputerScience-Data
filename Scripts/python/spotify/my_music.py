import os

import spotipy
import pickle

from dataclasses import dataclass
from typing import List
from collections import OrderedDict

from spotipy.oauth2 import SpotifyOAuth
from pprint import pprint


@dataclass
class Song:
    uri: str
    name: str
    album: str = ""
    artists: List[str] = ""
    year: str = ""
    top_genre: str = ""
    genres: List[str] = ""


class MyMusic:

    def __init__(self):

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
            scope=
            "playlist-read-private playlist-read-collaborative user-follow-read user-read-playback-position user-top-read user-read-recently-played user-library-read user-read-private"
        ))

        # Get current user info
        user = self.sp.current_user()
        # pprint(user)
        user_name = user["display_name"]
        user_country = user["country"]
        print("Username:", user_name)
        print("Country:", user_country, "\n")

    def get_top_tracks(self):
        # Get top 1 tracks
        top = self.sp.current_user_top_tracks(limit=20, offset=0)
        top_song = top['items'][1]
        # pprint(top_song)
        top_song_uri = top_song['uri']
        print("Top song URI:", top_song_uri, '\n')

    def get_track_details(self, track_uri):
        # Get track details
        track_info = self.sp.track(
            track_uri
        )  # Replace with the Spotify URI of the track you want information about
        # pprint(track_info)
        track_name = track_info['name']
        artists = []
        for artist in track_info['artists']:
            artists.append(artist["name"])
        # print("Track Name:", track_name)
        # print("Artist:", artists, '\n')
        # print("Uri:", track_album_uri, '\n')

    def get_album_details(self, album_uri):
        # Get album details
        album_info = self.sp.album(album_uri)
        track_album_genre = album_info["genres"]
        # print("Track album genre:", track_album_genre, '\n')
        return track_album_genre

    def get_artist_details(self, artist_uri):
        # Get artist details
        artist_info = self.sp.artist(artist_uri)
        artist_genres = artist_info["genres"]
        # print("Artist genres", artist_genres)
        return artist_genres

    def get_tracks_liked(self):
        song_list = {}

        total = float('inf')
        offset = 0
        # step = 3
        step = 50
        while total > offset:
            print(f"Offset: {offset}, step: {step}, total: {total}")
            page_results = self.sp.current_user_saved_tracks(limit=step,
                                                             offset=offset)
            offset = offset + step

            total = page_results["total"]
            # total = 1

            for song in page_results['items']:
                song = song["track"]
                artists = OrderedDict()
                genres = []
                for artist in song['artists']:
                    artist_genres = self.get_artist_details(artist["uri"])

                    genres = genres + artist_genres
                    artists[artist["name"]] = artist_genres
                # print(genres)
                top_genre = genres[0] if genres else "Unclassified"

                song_list[song['uri']] = Song(
                    uri=song['uri'],
                    name=song['name'],
                    artists=artists,
                    album=song["album"]["name"],
                    year=song["album"]["release_date"],
                    top_genre=top_genre,
                    genres=genres)
        return song_list


# my_music = MyMusic()
# song_list = my_music.get_tracks_liked()
# print(len(song_list))
# print(song_list)
# with open("song_list.pkl", "wb") as file:
#     pickle.dump(song_list, file)