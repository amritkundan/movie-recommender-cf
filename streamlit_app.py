from pathlib import Path
import streamlit as st
from src.movie_recommender.data import load_movies_csv
from src.movie_recommender.recommender import ContentRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Content-based Movie Recommender")

csv_path = st.text_input("Path to movies.csv", value="data/movies.csv")
seed_title = st.text_input("Movie title (seed)", value="Avatar")
topk = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)

if st.button("Recommend"):
    try:
        df = load_movies_csv(Path(csv_path))
        model = ContentRecommender(df).fit()
        recs = model.similar(seed_title, k=topk, include_scores=True)
        for title, score in recs:
            st.write(f"- **{title}** — {score:.3f}")
    except Exception as e:
        st.error(str(e))
