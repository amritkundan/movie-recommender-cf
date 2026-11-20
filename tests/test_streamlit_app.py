from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_app_starts_with_both_recommendation_modes(tmp_path, monkeypatch):
    data_directory = tmp_path / "data"
    data_directory.mkdir()

    movie_ids = list(range(10, 22))
    ratings = pd.DataFrame(
        [
            {
                "userId": user_id,
                "movieId": movie_id,
                "rating": float(((user_id + movie_id) % 5) + 1),
            }
            for user_id in range(1, 5)
            for movie_id in movie_ids
        ]
    )
    movies = pd.DataFrame(
        {
            "movieId": movie_ids,
            "title": [f"Movie {movie_id}" for movie_id in movie_ids],
        }
    )
    ratings.to_csv(data_directory / "ratings.csv", index=False)
    movies.to_csv(data_directory / "movies.csv", index=False)

    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "🎥 Collaborative Filtering Movie Recommender"
    assert app.radio[0].options == [
        "Quick (Cosine Similarity)",
        "Trained Model (SVD)",
    ]
