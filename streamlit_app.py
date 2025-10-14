import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Optional import for SVD mode
try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

from src.movie_recommender_cf.data import load_ratings, get_user_movie_matrix


st.set_page_config(page_title="Movie Recommender (Collaborative Filtering)", page_icon="🎥")

st.title("🎥 Collaborative Filtering Movie Recommender")
st.markdown("""
An interactive demo of **Collaborative Filtering** using user–movie ratings.  
You can explore two different recommendation modes below.
""")

# --- Sidebar: Settings ---
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Choose Recommendation Mode:",
    ["Quick (Cosine Similarity)", "Trained Model (SVD)"],
    help="Cosine Similarity = instant demo. SVD = trained matrix factorization model."
)

cold_start_mode = st.sidebar.radio(
    "🎬 Choose how to pick initial movies:",
    ["Random", "Popular"],
    help="Show either random movies or the most popular ones first."
)

# --- Load dataset ---
ratings_path = "data/ratings.csv"
movies_path = "data/movies.csv"

try:
    df = load_ratings(ratings_path)
    movies = pd.read_csv(movies_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Merge ratings with titles
df = df.merge(movies[["movieId", "title"]], on="movieId", how="left")

# Compute popularity (for Popular mode)
movie_popularity = df.groupby("movieId")["rating"].count().sort_values(ascending=False)
popular_movie_ids = movie_popularity.head(300).index.tolist()  # top 300 most rated

st.markdown("### 🎞️ Rate a few movies to get started")

# --- Initialize / refresh movie sample ---
if "sample_movies" not in st.session_state or st.button("🔄 Refresh movie list"):
    if cold_start_mode == "Popular":
        st.session_state.sample_movies = np.random.choice(popular_movie_ids, 10, replace=False)
    else:
        st.session_state.sample_movies = np.random.choice(df["movieId"].unique(), 10, replace=False)

user_ratings = {}

# --- Rating sliders with per-movie refresh buttons ---
cols = st.columns(2)
for i, mid in enumerate(st.session_state.sample_movies):
    title = movies.loc[movies["movieId"] == mid, "title"].values
    title = title[0] if len(title) else f"Movie ID {mid}"

    with cols[i % 2]:
        c1, c2 = st.columns([4, 1])
        with c1:
            rating = st.slider(title, 0.0, 5.0, 0.0, 0.5, key=f"rating_{mid}")
        with c2:
            if st.button("🔄", key=f"refresh_{mid}", help="Swap this movie for another random one"):
                if cold_start_mode == "Popular":
                    available = [m for m in popular_movie_ids if m not in st.session_state.sample_movies]
                else:
                    available = [m for m in df["movieId"].unique() if m not in st.session_state.sample_movies]
                if available:
                    new_movie = np.random.choice(available)
                    idx = list(st.session_state.sample_movies).index(mid)
                    st.session_state.sample_movies[idx] = new_movie
                    st.rerun()
        if rating > 0:
            user_ratings[mid] = rating

# --- Get Recommendations ---
if st.button("🎬 Get Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie to get personalized recommendations.")
        st.stop()

    # QUICK MODE
    if mode == "Quick (Cosine Similarity)":
        st.info("🔹 Using Quick Mode: User-based collaborative filtering with cosine similarity.")
        matrix = get_user_movie_matrix(df)
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
        top = recs.sort_values(ascending=False).head(10)

        results = pd.DataFrame({
            "movieId": top.index,
            "predicted_rating": top.values
        }).merge(movies[["movieId", "title"]], on="movieId", how="left")
        st.success("Here are your recommended movies:")
        st.table(results[["title", "predicted_rating"]])

    # TRAINED MODEL MODE
    else:
        if not SURPRISE_AVAILABLE:
            st.warning("⚠️ SVD mode requires the Surprise library, which isn't supported on Streamlit Cloud.")
            st.stop()

        st.info("🧠 Using Trained Model Mode (SVD via Surprise). This may take a moment...")
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)

        user_id = int(df["userId"].max()) + 1
        preds = []
        for mid in df["movieId"].unique():
            if mid not in user_ratings:
                pred = algo.predict(user_id, mid, r_ui=None)
                preds.append((mid, pred.est))
        top_svd = sorted(preds, key=lambda x: x[1], reverse=True)[:10]

        results = pd.DataFrame(top_svd, columns=["movieId", "predicted_rating"]) \
            .merge(movies[["movieId", "title"]], on="movieId", how="left")
        st.success("Here are your top recommendations:")
        st.table(results[["title", "predicted_rating"]])
