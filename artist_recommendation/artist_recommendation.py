import json
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from enum import Enum

'''
Make sure you've run through user_creation/psuedo_user_creation.ipynb first (unless the json files are already in the repo,
which they may be.)

If they're not:

|--------------------------------------------------------------------------------------------------------|
cd to the project root directory, then run:                                                              |
                                                                                                         |
jupyter notebook                                                                                         |
                                                                                                         |
and navigate to user_creation/psuedo_user_creation.ipynb in your browser to run it.                      |
Downloading the pseudo_user files will take a bit!                                                       |
|--------------------------------------------------------------------------------------------------------|

When you're back follow the commands below to run this code, ensure the pseudo-users-avg/elbow-data files
are present in the user_creation directory, they potentially may be just because we pushed them up, if they are
no need to reproduce them.

|--------------------------------------|
pipenv shell                           |
pip3 install numpy scikit-learn scipy  |
clear                                  |
python3 artist_recommendation.py       |
|--------------------------------------|
'''

# Set to true to enable cli prompts for artist names, [optional] ---|
ENABLE_INTERACTIVE_MODE = True # <---------------------------------|
FILE_PATH = '../user_creation/pseudo-users-elbow-data.json'
NAME = 'name'
ARTIST_URI = "artist_uri"
URI = 'uri'

class Match(Enum):
    EXIT = 'exit'
    EXACT = 'exact'
    YES = 'yes'
    PARTIAL = 'partial'

def load_data(file_path):
    with open(file_path, 'r') as f:
        print("\nLoading data from {}..".format(file_path))
        data = json.load(f)
        print('\nData loaded successfully\n')
        return data

def get_artist_data(synthetic_users):
    all_playlists, artist_to_id, id_to_artist, unique_artist_uris = (
        list(), dict(), dict(), set()
    )

    print('Building artist vocabulary and playlist list..\n')
    for user in synthetic_users:
        for playlist in user.get("playlists", []):
            all_playlists.append(playlist)
            for track in playlist.get("tracks", []):
                artist_uri = track.get(ARTIST_URI)
                artist_name = track.get("artist_name")
                if artist_uri:
                    unique_artist_uris.add(artist_uri)
                    if artist_uri not in artist_to_id:
                        artist_id = len(artist_to_id)
                        artist_to_id[artist_uri] = artist_id
                        id_to_artist[artist_id] = {NAME: artist_name, URI: artist_uri}

    print("Found {} playlists and {} unique artists\n".format(len(all_playlists), len(artist_to_id)))
    return all_playlists, artist_to_id, id_to_artist, unique_artist_uris

def make_cooccurrence_matrix(playlists, artist_to_id):
    cooccurrence_counts = defaultdict(int)
    num_artists = len(artist_to_id)
    print("Building co occurrence matrix..")
    for playlist in playlists:
        artists_in_playlist = {
            track[ARTIST_URI] for track
            in playlist["tracks"] if track[ARTIST_URI]
            in artist_to_id
        }
        artists_list = list(artists_in_playlist)
        for i in range(len(artists_list)):
            for j in range(i, len(artists_list)):
                artist1_uri = artists_list[i]
                artist2_uri = artists_list[j]
                artist1_id = artist_to_id[artist1_uri]
                artist2_id = artist_to_id[artist2_uri]
                pair = tuple(sorted((artist1_id, artist2_id)))
                cooccurrence_counts[pair] += 1

    rows, cols, data = (
        list(), list(), list()
    )
    for (artist1_id, artist2_id), count in cooccurrence_counts.items():
        rows.append(artist1_id)
        cols.append(artist2_id)
        data.append(count)
        if artist1_id != artist2_id:
            rows.append(artist2_id)
            cols.append(artist1_id)
            data.append(count)

    cooccurrence_matrix = csr_matrix((data, (rows, cols)), shape=(num_artists, num_artists))
    print("Co occurrence matrix built successfully\n")
    return cooccurrence_matrix

class ArtistRecommender:
    def __init__(self, cooccurrence_matrix, id_to_artist_map, artist_to_id_map):
        self.cooccurrence_matrix = cooccurrence_matrix
        self.id_to_artist = id_to_artist_map
        self.artist_to_id = artist_to_id_map
        self.knn = NearestNeighbors(algorithm='brute', metric='cosine')
        self.knn.fit(self.cooccurrence_matrix)
        print("Artist Recommender initialized\n")

    def get_recs(self, artist_uri, k=10):
        if artist_uri not in self.artist_to_id:
            artist_name = "Unknown Artist"
            for _, info in self.id_to_artist.items():
                if info[URI] == artist_uri:
                    artist_name = info[NAME]
                    break
            print("Artist '{}' with URI {} not found in the dataset.".format(artist_name, artist_uri))
            return list()

        artist_id = self.artist_to_id[artist_uri]
        input_artist_name = self.id_to_artist[artist_id][NAME].lower()
        artist_vector = self.cooccurrence_matrix[artist_id]
        _, indices = self.knn.kneighbors(artist_vector, n_neighbors=k + 5)

        recommendations = list()
        for idx in range(1, len(indices.flatten())):
            if len(recommendations) >= k:
                break
            
            neighbor_id = indices.flatten()[idx]
            recommended_artist = self.id_to_artist[neighbor_id]
            
            if recommended_artist[NAME].lower() != input_artist_name:
                recommendations.append(recommended_artist)
                
        return recommendations

def find_artist_by_name(query, id_to_artist):
    partial_match = None
    for artist_info in id_to_artist.values():
        if query.lower() == artist_info[NAME].lower():
            return Match.EXACT.value, artist_info

    for artist_info in id_to_artist.values():
        if query.lower() in artist_info[NAME].lower():
            partial_match = artist_info
            break

    if partial_match:
        return Match.PARTIAL.value, partial_match
    return None, None

def run_interactive_mode(recommender, id_to_artist):
    while True:
        query = input("Enter an artist name to get recommendations (or type 'exit' to quit): ")
        if query.lower() == Match.EXIT.value:
            break
        match_type, artist_info = find_artist_by_name(query, id_to_artist)

        if match_type.lower() == Match.EXACT.value:
            present(recommender, [artist_info[URI]], id_to_artist, k=10)
        elif match_type.lower() == Match.PARTIAL.value:
            suggestion = input("\nDid you mean '{}'? (yes/no): ".format(artist_info[NAME]))
            if suggestion.lower() == Match.YES.value:
                present(recommender, [artist_info[URI]], id_to_artist, k=10)
            else:
                print("\nOk - please try your search again.\n")
        else:
            print("No artist matching `{}` found in the dataset\n".format(query))

def present(recommender, artists_to_test, id_to_artist, k=10):
    print("\n\t\tARTIST RECOMMENDATION RESULTS\n")
    for artist_uri in artists_to_test:
        if artist_uri in recommender.artist_to_id:
             artist_id = recommender.artist_to_id[artist_uri]
             artist_name = id_to_artist[artist_id][NAME]

        print("Recommendations for {}:".format(artist_name))
        recommendations = recommender.get_recs(artist_uri, k=k)
        for idx, artist in enumerate(recommendations):
            print("\t{}.) {}".format(idx + 1, artist[NAME]))
        print('\n')

synthetic_users_data = load_data(FILE_PATH)
playlists, artist_to_id, id_to_artist, unique_artist_uris = get_artist_data(synthetic_users_data)
cooccurrence_matrix = make_cooccurrence_matrix(playlists, artist_to_id)
recommender = ArtistRecommender(cooccurrence_matrix, id_to_artist, artist_to_id)

if ENABLE_INTERACTIVE_MODE:
    run_interactive_mode(recommender, id_to_artist)
else:
    n_artists_to_test = 1
    artists_to_test = list(unique_artist_uris)[:n_artists_to_test]
    present(recommender, artists_to_test, id_to_artist, k=10) 

'''
Note:

I was a little bit confused why despite accumulating the artist data the 
same every time resulted in random results every time but then I realized
that in python the sets are unordered, unlike C++ where you can use a std::set
so set() is basically like std::unordered_set.
''' 

