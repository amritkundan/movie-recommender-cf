import numpy as np
import pandas as pd
import pytest

from movie_recommender_cf.data import get_user_movie_matrix
from movie_recommender_cf.recommender import (
    CollaborativeFiltering,
    recommend_from_ratings,
)


def sample_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 10, "rating": 5.0},
            {"userId": 1, "movieId": 20, "rating": 1.0},
            {"userId": 1, "movieId": 30, "rating": 5.0},
            {"userId": 2, "movieId": 10, "rating": 1.0},
            {"userId": 2, "movieId": 20, "rating": 5.0},
            {"userId": 3, "movieId": 10, "rating": 4.5},
            {"userId": 3, "movieId": 20, "rating": 1.0},
            {"userId": 3, "movieId": 30, "rating": 4.0},
            {"userId": 3, "movieId": 40, "rating": 2.0},
        ]
    )


def test_matrix_preserves_missing_ratings():
    matrix = get_user_movie_matrix(sample_ratings())

    assert np.isnan(matrix.loc[2, 30])
    assert matrix.loc[1, 10] == 5.0


def test_user_predictions_normalize_only_over_users_who_rated_item():
    matrix = get_user_movie_matrix(sample_ratings())
    matrix.loc["new"] = np.nan
    matrix.loc["new", [10, 20]] = [5.0, 1.0]

    recommendations = CollaborativeFiltering(matrix).fit().recommend_for_user("new")
    predicted_movie_30 = recommendations.set_index("movieId").loc[30, "predicted_rating"]

    # Users 1 and 3 rated movie 30. User 2 did not, so their missing value
    # must not dilute the denominator as if it were a zero-star rating.
    assert 4.0 <= predicted_movie_30 <= 5.0


def test_recommend_from_ratings_excludes_movies_the_visitor_rated():
    recommendations = recommend_from_ratings(
        sample_ratings(),
        {10: 5.0, 20: 1.0},
        top_k=2,
    )

    assert list(recommendations.columns) == ["movieId", "predicted_rating"]
    assert set(recommendations["movieId"]).isdisjoint({10, 20})
    assert len(recommendations) == 2


def test_recommend_from_ratings_rejects_unknown_movies():
    with pytest.raises(ValueError, match="None of the rated movies"):
        recommend_from_ratings(sample_ratings(), {999: 5.0})
