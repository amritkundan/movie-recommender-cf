try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

from src.movie_recommender_cf.data import load_ratings, get_user_movie_matrix



st.set_page_config(page_title="Movie Recommender (Collaborative Filtering)", page_icon="🎥")

st.title("🎥 Collaborative Filtering Movie Recommender")
st.markdown("""
An interactive demo of **Collaborative Filtering** using user–movie ratings.  
You can explore two different recommendation modes below.
""")

# --- Sidebar: Mode Selection ---
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

# --- Load Dataset ---
ratings_path = "data/ratings.csv"
try:
    df = load_ratings(ratings_path)
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown("### 🎞️ Rate a few movies to get started")

movie_ids = df["movieId"].unique()
sample_movies = np.random.choice(movie_ids, 10, replace=False)
user_ratings = {}

for mid in sample_movies:
    rating = st.slider(f"Movie ID {mid}", 0.0, 5.0, 0.0, 0.5)
    if rating > 0:
        user_ratings[mid] = rating

if st.button("🎬 Get Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie to get personalized recommendations.")
        st.stop()

    # --- Handle Surprise availability ---
    if mode == "Trained Model (SVD)" and not SURPRISE_AVAILABLE:
        st.warning(
            "⚠️ SVD mode requires the Surprise library, which isn't supported on Streamlit Cloud. "
            "Run this app locally to use the full trained model version."
        )
        st.stop()

    # --- QUICK MODE (COSINE) ---
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
        st.success("Here are your recommended movies (by ID):")
        st.table(top.reset_index().rename(columns={"index": "movieId", 0: "predicted_rating"}))

    # --- TRAINED MODEL (SVD) ---
    elif mode == "Trained Model (SVD)" and SURPRISE_AVAILABLE:
        st.info("🧠 Using Trained Model Mode (SVD via Surprise). This may take a moment...")
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)

        user_id = int(df["userId"].max()) + 1  # assign new user ID
        preds = []
        for mid in df["movieId"].unique():
            if mid not in user_ratings:
                pred = algo.predict(user_id, mid, r_ui=None)
                preds.append((mid, pred.est))
        top_svd = sorted(preds, key=lambda x: x[1], reverse=True)[:10]
        st.success("Here are your top recommendations (by ID):")
        st.table(pd.DataFrame(top_svd, columns=["movieId", "predicted_rating"]))

