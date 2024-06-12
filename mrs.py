import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge dataframes
movies = pd.merge(movies, credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

import ast
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert columns
columns_to_convert = ['genres', 'keywords', 'cast', 'crew']
for column in columns_to_convert:
    movies[column] = movies[column].apply(lambda L: [i.replace(" ", "") for i in L])

movies['overview'] = movies['overview'].apply(lambda x:x.split())

# Combine columns
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)


# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index
    if index.empty:
        return ["Movie not found."]
    else:
        index = index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = [movies.iloc[i[0]].title for i in distances[1:6]]
        return recommended_movie_names


# Streamlit UI
st.title("Movie Recommendations")
selected_movie = st.selectbox('Select the movie', movies['ti'
                                                         ''
                                                         'tle'].values)
if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for recommended_movie in recommended_movie_names:
        st.title(recommended_movie)