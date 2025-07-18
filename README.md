# Spotify Playlist & Artist Recommendation

A music recommendation system using machine learning techniques to suggest similar artists based on playlist cooccurrence patterns.

- Tyler Netek
- Nick Hellmer

## Overview

This project explores how machine learning can be used to build a music recommendation system similar to those used by streaming platforms like Spotify. We focus on artist recommendations based on the hypothesis that artists who frequently appear together in user created playlists have a higher chance to be similar in appeal.

## Features

- **Artist Recommendation**: 
Suggests similar artists based on co-occurrence patterns in playlists
- **User Similarity**: 
Compares synthetic user profiles to find musically similar listeners
- **Interactive Mode**: 
Query the system by artist name to get personalized recommendations

## Data

We use the 'Spotify Million Playlist Dataset' from Kaggle which contains one million real user created playlists. For development and testing we work with a subset of this data, only 1,000 playlists, to prototype our approach


### Artist Recommendation

1. **Cooccurrence Matrix**: We build an artist to artist matrix where each cell represents how often two artists appear in the same playlist
2. **K Nearest Neighbors**: Using cosine similarity, we find the most similar artists to a given input artist

### User Similarity

- **Synthetic Users**: 
We create pseudo-users by clustering similar playlists together
- **Similarity Metrics**: We explore different similarity measures:
   - Standard cosine similarity
   - Scaled cosine similarity accounting for magnitude differences
   - Soft cosine similarity with tunable parameters

## Project Structure

- `artist_recommendation` -
 Contains the artist recommendation implementation
- `user_creation` -
 Scripts for creating synthetic user profiles from playlist data
- `user_similarity` -
 Implementation of user to user similarity metrics

## User Guide

1. Make sure you have the required data files in the `user_creation` directory
2. If needed, run the `user_creation -> psuedo_user_creation.ipynb` notebook to generate synthetic user data
3. Install dependencies:
   ```python
   pipenv shell
   pip3 install numpy scikit-learn scipy
   ```
4. Run the artist recommendation system:
   ```python
   cd artist_recommendation
   python artist_recommendation.py
   ```

## Evaluation Criteria

We evaluate our recommendations qualitatively by examining if the suggested artists share similar genres styles or musical characteristics with the input artist.

## Future Ideas

- Scale up to use the full million playlist dataset
- Incorporate additional features like audio characteristics or genre information
- Implement a user feedback system to improve recommendations over time
- Explore more advanced models beyond co-occurrence based approaches