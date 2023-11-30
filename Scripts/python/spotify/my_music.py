import os

import spotipy
# import pickle

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
    # Chosen by Spotify
    top_genre: str = ""
    # Assigned after some processing
    genre: str = ""
    genres: List[str] = ""


class MyMusic:

    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.environ["SPOTIFY_REDIRECT_URI"],
            scope=
            "playlist-read-private playlist-read-collaborative user-follow-read user-read-playback-position user-top-read user-read-recently-played user-library-read user-read-private playlist-modify-private playlist-modify-public"
        ))
        # Get current user info
        self.user = self.sp.current_user()
        # pprint(user)
        self.user_name = self.user["display_name"]
        self.user_country = self.user["country"]
        print("Username:", self.user_name)
        print("Country:", self.user_country, "\n")
        return

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

    def search_song(self, song_name):
        search_results = {}
        songs = self.sp.search(q=song_name, type="track")["tracks"]["items"]
        for song in songs:
            artists = OrderedDict()
            genres = []
            for artist in song['artists']:
                artist_genres = self.get_artist_details(artist["uri"])

                genres = genres + artist_genres
                artists[artist["name"]] = artist_genres
            # print(genres)
            top_genre = genres[0] if genres else "Unclassified"

            search_results[song['uri']] = Song(
                uri=song['uri'],
                name=song['name'],
                artists=artists,
                album=song["album"]["name"],
                year=song["album"]["release_date"],
                top_genre=top_genre,
                genres=genres)
        return search_results

    def create_playlist(self, playlist_name):
        playlist = self.sp.user_playlist_create(user=self.user["id"],
                                                name=playlist_name,
                                                public=True)
        playlist_id = playlist["id"]
        self.sp.playlist_add_items(playlist_id=playlist_id,
                                   items=['1vW12BfxjOQKYElBm9ttW9'])

    def list_playlists(self):

        owner_playlists = []

        total = float('inf')
        offset = 0
        # step = 3
        step = 50
        while total > offset:
            print(f"Offset: {offset}, step: {step}, total: {total}")
            page_playlists = self.sp.current_user_playlists(limit=step,
                                                            offset=offset)
            offset = offset + step

            total = page_playlists["total"]

            for playlist in page_playlists['items']:
                if playlist['owner']['id'] == 'frankfacundo1002177':
                    playlist_item = {}
                    playlist_item['external_urls'] = playlist['external_urls']
                    playlist_item['name'] = playlist['name']
                    # playlist_item['tracks'] = playlist['tracks']
                    owner_playlists.append(playlist_item)
        return owner_playlists


my_music = MyMusic()
# my_music.create_playlist("ff_test")
playlists = my_music.list_playlists()
pprint(playlists)
print(len(playlists))

# my_music = MyMusic()
# search_results = my_music.search_song("honda costumbres")
# print(len(search_results))
# pprint(search_results)

# song_list = my_music.get_tracks_liked()
# print(len(song_list))
# print(song_list)
# with open("song_list.pkl", "wb") as file:
#     pickle.dump(song_list, file)
