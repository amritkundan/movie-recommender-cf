import numpy as np
import pandas as pd
import streamlit as st

from src.movie_recommender_cf.data import REQUIRED_COLUMNS, load_ratings
from src.movie_recommender_cf.models import (
    fit_svd_for_new_user,
    recommend_with_fitted_model,
)
from src.movie_recommender_cf.recommender import recommend_from_ratings
from src.movie_recommender_cf.utils import merge_with_movies


RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"
SVD_PARAMETERS = {
    "n_factors": 100,
    "n_epochs": 20,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": 42,
}


@st.cache_data
def load_app_data(
    ratings_path: str = RATINGS_PATH,
    movies_path: str = MOVIES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load immutable source data once per file version."""
    ratings = load_ratings(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies


@st.cache_data(show_spinner=False)
def quick_recommendations(
    ratings: pd.DataFrame,
    submitted_ratings: tuple[tuple[int, float], ...],
) -> pd.DataFrame:
    """Cache sparse-neighborhood recommendations across page reruns."""
    return recommend_from_ratings(ratings, dict(submitted_ratings), top_k=10)


@st.cache_resource(show_spinner=False)
def cached_svd_model(
    ratings: pd.DataFrame,
    submitted_ratings: tuple[tuple[int, float], ...],
):
    """Train SVD once for each dataset and visitor-rating combination."""
    return fit_svd_for_new_user(
        ratings,
        dict(submitted_ratings),
        model_params=SVD_PARAMETERS,
    )


st.set_page_config(
    page_title="Movie Recommender (Collaborative Filtering)",
    page_icon="🎥",
)

st.title("🎥 Collaborative Filtering Movie Recommender")
st.markdown(
    """
An interactive demo of **Collaborative Filtering** using user–movie ratings.  
You can explore two different recommendation modes below.
"""
)

st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Choose Recommendation Mode:",
    ["Quick (Cosine Similarity)", "Trained Model (SVD)"],
    help="Cosine Similarity = instant demo. SVD = trained matrix factorization model.",
)

cold_start_mode = st.sidebar.radio(
    "🎬 Choose how to pick initial movies:",
    ["Random", "Popular"],
    help="Show either random movies or the most popular ones first.",
)

try:
    df, movies = load_app_data()
except Exception as error:
    st.error(f"Error loading data: {error}")
    st.stop()

movie_popularity = df.groupby("movieId")["rating"].count().sort_values(ascending=False)
popular_movie_ids = movie_popularity.head(300).index.to_numpy()
all_movie_ids = df["movieId"].unique()

st.markdown("### 🎞️ Rate a few movies to get started")

if "sample_movies" not in st.session_state or st.button("🔄 Refresh movie list"):
    sample_pool = popular_movie_ids if cold_start_mode == "Popular" else all_movie_ids
    sample_size = min(10, len(sample_pool))
    st.session_state.sample_movies = np.random.choice(
        sample_pool,
        sample_size,
        replace=False,
    )

user_ratings = {}
columns = st.columns(2)
for index, movie_id in enumerate(st.session_state.sample_movies):
    matching_titles = movies.loc[movies["movieId"] == movie_id, "title"].values
    title = matching_titles[0] if len(matching_titles) else f"Movie ID {movie_id}"

    with columns[index % 2]:
        rating_column, refresh_column = st.columns([4, 1])
        with rating_column:
            rating = st.slider(
                title,
                0.0,
                5.0,
                0.0,
                0.5,
                key=f"rating_{movie_id}",
            )
        with refresh_column:
            if st.button(
                "🔄",
                key=f"refresh_{movie_id}",
                help="Swap this movie for another one",
            ):
                refresh_pool = (
                    popular_movie_ids if cold_start_mode == "Popular" else all_movie_ids
                )
                available = np.setdiff1d(
                    refresh_pool,
                    st.session_state.sample_movies,
                    assume_unique=False,
                )
                if len(available):
                    replacement = np.random.choice(available)
                    sample_index = list(st.session_state.sample_movies).index(movie_id)
                    st.session_state.sample_movies[sample_index] = replacement
                    st.rerun()
        if rating > 0:
            user_ratings[int(movie_id)] = float(rating)

if st.button("🎬 Get Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie to get personalized recommendations.")
        st.stop()

    submitted_ratings = tuple(sorted(user_ratings.items()))
    if mode == "Quick (Cosine Similarity)":
        st.info("🔹 Using Quick Mode: sparse user-based collaborative filtering.")
        recommendations = quick_recommendations(
            df[REQUIRED_COLUMNS],
            submitted_ratings,
        )
    else:
        st.info("🧠 Using Trained Model Mode (SVD). This may take a moment the first time.")
        with st.spinner("Training the model with your ratings..."):
            algorithm, new_user_id = cached_svd_model(
                df[REQUIRED_COLUMNS],
                submitted_ratings,
            )
        recommendations = recommend_with_fitted_model(
            algorithm,
            new_user_id,
            df["movieId"].unique(),
            set(user_ratings),
            top_k=10,
        )

    results = merge_with_movies(recommendations, movies)
    st.success("Here are your recommended movies:")
    st.table(results[["title", "predicted_rating"]])
