try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.movie_recommender_cf.data import load_ratings, get_user_movie_matrix


# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="Movie Recommender (Collaborative Filtering)", page_icon="🎥", layout="centered")

st.title("🎥 Collaborative Filtering Movie Recommender")
st.markdown("""
An interactive demo of **Collaborative Filtering** using user–movie ratings.  
You can explore two different recommendation modes below.
""")


# ------------------------------
# Sidebar Configuration
# ------------------------------
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Choose Recommendation Mode:",
    ["Quick (Cosine Similarity)", "Trained Model (SVD)"],
    help="Cosine Similarity = instant demo. SVD = trained matrix factorization model."
)

st.sidebar.markdown("""
**Quick Mode (Cosine)**  
- No training required  
- Computes user–user similarity directly  
- Great for small or demo datasets  

**SVD Mode (Surprise)**  
- Learns latent factors from ratings  
- Provides more realistic predictions  
- Slower, but powerful for real use cases
""")


# ------------------------------
# Load Data
# ------------------------------
ratings_path = "data/ratings.csv"
movies_path = "data/movies.csv"

try:
    df = load_ratings(ratings_path)
    movies = pd.read_csv(movies_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Ensure consistent columns
if "movieId" not in df.columns or "movieId" not in movies.columns:
    st.error("Missing 'movieId' column in ratings or movies data.")
    st.stop()


# ------------------------------
# Movie Rating Interface
# ------------------------------
st.markdown("### 🎞️ Rate a few movies to get started")

# Keep the same sample list across reruns using session_state
if "sample_movies" not in st.session_state or not isinstance(st.session_state.sample_movies, pd.DataFrame):
    # only include movies that exist in both datasets
    common_ids = df["movieId"].unique()
    available_movies = movies[movies["movieId"].isin(common_ids)]
    if len(available_movies) == 0:
        st.error("No overlapping movie IDs between ratings.csv and movies.csv.")
        st.stop()
    st.session_state.sample_movies = available_movies.sample(10, random_state=42)

# Ensure DataFrame format
sample_df = pd.DataFrame(st.session_state.sample_movies)

user_ratings = {}
for _, row in sample_df.iterrows():
    title = str(row["title"])
    mid = int(row["movieId"])
    rating = st.slider(f"{title}", 0.0, 5.0, 0.0, 0.5, key=f"slider_{mid}")
    if rating > 0:
        user_ratings[mid] = rating


# ------------------------------
# Recommendation Logic
# ------------------------------
if st.button("🎬 Get Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie to get personalized recommendations.")
        st.stop()

    # QUICK MODE (COSINE)
    if mode == "Quick (Cosine Similarity)":
        st.info("🔹 Using Quick Mode: User-based collaborative filtering with cosine similarity.")

        matrix = get_user_movie_matrix(df)
        # Add new user (-1) row
        matrix.loc[-1] = 0
        for mid, r in user_ratings.items():
            if mid in matrix.columns:
                matrix.loc[-1, mid] = r

        sim = cosine_similarity(matrix)
        user_idx = list(matrix.index).index(-1)
        sim_scores = sim[user_idx]
        sim_scores[user_idx] = 0

        weighted_sum = sim_scores @ matrix.values
        norm = np.array([abs(sim_scores).sum()] * matrix.shape[1])
        preds = weighted_sum / (norm + 1e-9)

        recs = pd.Series(preds, index=matrix.columns)
        for mid in user_ratings.keys():
            recs.loc[mid] = -1
        top = recs.sort_values(ascending=False).head(10).reset_index()
        top.columns = ["movieId", "predicted_rating"]

        ## Merge movie titles and drop missing ones
        top = top.merge(movies, on="movieId", how="left")[["title", "predicted_rating"]]
        top = top.dropna(subset=["title"]).head(10)

        if top.empty:
            st.warning("⚠️ No matching movie titles found for recommendations. "
               "You may need to use a larger movies.csv or one that matches your ratings.csv IDs.")
        else:
            st.success("🎯 Recommended Movies:")
            st.table(top)


    # TRAINED MODEL (SVD)
    else:
        if not SURPRISE_AVAILABLE:
            st.warning("⚠️ SVD mode requires the Surprise library, which isn't supported on Streamlit Cloud.\n"
                       "Run this app locally to use the full trained model version.")
            st.stop()

        st.info("🧠 Using Trained Model Mode (SVD via Surprise). This may take a moment...")

        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)

        user_id = int(df["userId"].max()) + 1
        preds = []
        for mid in movies["movieId"].unique():
            if mid not in user_ratings:
                pred = algo.predict(user_id, mid, r_ui=None)
                preds.append((mid, pred.est))
        top_svd = sorted(preds, key=lambda x: x[1], reverse=True)[:10]
        top_df = pd.DataFrame(top_svd, columns=["movieId", "predicted_rating"]).merge(
            movies, on="movieId", how="left"
        )[["title", "predicted_rating"]]

        st.success("🎯 Recommended Movies:")
        st.table(top_df)
