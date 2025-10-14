import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Try importing Surprise (works locally, not on Streamlit Cloud)
try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

from src.movie_recommender_cf.data import load_ratings, get_user_movie_matrix

# --- Page config ---
st.set_page_config(page_title="Movie Recommender (Collaborative Filtering)", page_icon="🎥")

st.title("🎥 Collaborative Filtering Movie Recommender")
st.markdown("""
Explore two recommendation modes:
- **Quick (Cosine Similarity)** — instant demo, no training.
- **Trained Model (SVD)** — smarter recommendations, works locally.
""")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Choose Recommendation Mode:",
    ["Quick (Cosine Similarity)", "Trained Model (SVD)"],
    help="Cosine = instant; SVD = trained model (not available on Streamlit Cloud)"
)

# --- Load data ---
ratings_path = "data/ratings.csv"
movies_path = "data/movies.csv"  # new!

try:
    df = load_ratings(ratings_path)
    movies = pd.read_csv(movies_path)
except Exception as e:
    st.error(str(e))
    st.stop()

# --- Session state for rated movies ---
if "sample_movies" not in st.session_state:
    st.session_state.sample_movies = np.random.choice(df["movieId"].unique(), 10, replace=False)
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

st.markdown("### 🎞️ Rate a few movies to get started")

# Merge movie titles
movie_choices = (
    pd.DataFrame({"movieId": st.session_state.sample_movies})
    .merge(movies, on="movieId", how="left")
)

# --- Rating sliders ---
for _, row in movie_choices.iterrows():
    title = row["title"] if pd.notna(row["title"]) else f"Movie ID {row['movieId']}"
    key = f"rating_{row['movieId']}"
    prev_value = st.session_state.user_ratings.get(row["movieId"], 0.0)
    rating = st.slider(title, 0.0, 5.0, prev_value, 0.5, key=key)
    if rating > 0:
        st.session_state.user_ratings[row["movieId"]] = rating
    elif row["movieId"] in st.session_state.user_ratings:
        del st.session_state.user_ratings[row["movieId"]]

if st.button("🎬 Get Recommendations"):
    user_ratings = st.session_state.user_ratings
    if not user_ratings:
        st.warning("Please rate at least one movie to get personalized recommendations.")
        st.stop()

    # Handle missing Surprise
    if mode == "Trained Model (SVD)" and not SURPRISE_AVAILABLE:
        st.warning(
            "⚠️ SVD mode requires the Surprise library, which isn't supported on Streamlit Cloud. "
            "Run this app locally to use it."
        )
        st.stop()

    # --- Quick mode (Cosine similarity) ---
    if mode == "Quick (Cosine Similarity)":
        st.info("🔹 Using Quick Mode (cosine similarity)")
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

        top_df = pd.DataFrame({"movieId": top.index, "predicted_rating": top.values})
        top_df = top_df.merge(movies, on="movieId", how="left")[["title", "predicted_rating"]]
        st.success("🎯 Recommended Movies:")
        st.table(top_df)

    # --- Trained model (SVD) ---
    elif mode == "Trained Model (SVD)" and SURPRISE_AVAILABLE:
        st.info("🧠 Using Trained Model Mode (SVD via Surprise)")
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
        top_df = pd.DataFrame(top_svd, columns=["movieId", "predicted_rating"])
        top_df = top_df.merge(movies, on="movieId", how="left")[["title", "predicted_rating"]]
        st.success("🎯 Recommended Movies:")
        st.table(top_df)
