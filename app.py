import streamlit as st
import pickle

# Load data
movies = pickle.load(open("movies_list.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# Movie recommendation logic
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = [movies.iloc[i[0]].title for i in movie_indices]
    return recommended_movies

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.write("### Recommended Movies:")
    for name in recommendations:
        st.write(f"- {name}")
