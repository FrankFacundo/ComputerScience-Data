import pickle

from dataclasses import asdict
from pprint import pprint
from typing import Dict
from my_music import Song


class Analysis:

    def __init__(self):
        self.songs: Dict[str, Song] = self.get_songs()
        return

    def get_songs(self) -> Dict[str, Song]:
        with open("song_list.pkl", "rb") as file:
            loaded_song_list: Dict[str, Song] = pickle.load(file)
        return loaded_song_list

    def head(self, n=5):
        for i, song in enumerate(self.songs.values()):
            if i == n: break
            print(f"Song {i}:")
            pprint(asdict(song))
            print('\n')

    def get_top_genre_counts(self):
        # Count the number of songs for each top genre
        genre_counts = {}
        for song in self.songs.values():
            genre = song.top_genre
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1

        # Sort the genre counts in decreasing order
        sorted_genre_counts = sorted(genre_counts.items(),
                                     key=lambda x: x[1],
                                     reverse=True)
        return sorted_genre_counts

    def get_genres_counts(self, songs=None):
        if songs is None:
            songs = self.songs

        genre_possibility_counts = {}

        for song in songs.values():
            genres = song.genres
            for genre in genres:
                if genre in genre_possibility_counts:
                    genre_possibility_counts[genre] += 1
                else:
                    genre_possibility_counts[genre] = 1

        # Sort the genre counts in decreasing order
        genre_possibility_counts = sorted(genre_possibility_counts.items(),
                                          key=lambda x: x[1],
                                          reverse=True)
        return genre_possibility_counts

    def get_songs_by_top_genre(self, top_genre, search_words=False):
        songs = []
        if search_words:
            for song in self.songs.values():
                if top_genre.lower() in song.top_genre.lower():
                    songs.append(song)
        else:
            for song in self.songs.values():
                if top_genre.lower() == song.top_genre.lower():
                    songs.append(song)
        return songs

    def get_songs_by_name(self, name):
        songs = []
        for song in self.songs.values():
            if name.lower() in song.name.lower():
                songs.append(song)
        return songs

    def get_songs_by_artist(self, artist):
        songs = []
        for song in self.songs.values():
            for artist_list in song.artists.keys():
                if artist.lower() in artist_list.lower():
                    songs.append(song)
        return songs

    def count_counts(self):
        count_counts = {}
        for genre_counts in self.get_top_genre_counts():
            if genre_counts[1] in count_counts:
                count_counts[genre_counts[1]] += 1
            else:
                count_counts[genre_counts[1]] = 1
        return count_counts
