import pandas as pd
from src.movie_recommender.recommender import ContentRecommender

def test_fit_and_recommend():
    df = pd.DataFrame([
        {"title": "Avatar", "keywords": "alien future", "cast": "Sam Worthington", "genres": "action sci-fi", "director": "James Cameron"},
        {"title": "Titanic", "keywords": "ship romance", "cast": "Leonardo DiCaprio", "genres": "drama romance", "director": "James Cameron"},
        {"title": "Alien", "keywords": "alien horror", "cast": "Sigourney Weaver", "genres": "horror sci-fi", "director": "Ridley Scott"},
    ])
    model = ContentRecommender(df).fit()
    recs = model.similar("Avatar", k=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert "Titanic" in recs or "Alien" in recs
