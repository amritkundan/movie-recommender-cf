import numpy as np
import pandas as pd

from .data import get_user_movie_matrix


def _cosine_on_observed(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity using only coordinates both vectors observed."""
    overlap = np.isfinite(left) & np.isfinite(right)
    if not overlap.any():
        return 0.0

    left_observed = left[overlap]
    right_observed = right[overlap]
    denominator = np.linalg.norm(left_observed) * np.linalg.norm(right_observed)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_observed, right_observed) / denominator)


def _pairwise_observed_cosine(values: np.ndarray) -> np.ndarray:
    """Compute a dense similarity matrix without treating missing data as zero."""
    size = values.shape[0]
    similarities = np.eye(size, dtype=float)
    for left_index in range(size):
        for right_index in range(left_index + 1, size):
            score = _cosine_on_observed(values[left_index], values[right_index])
            similarities[left_index, right_index] = score
            similarities[right_index, left_index] = score
    return similarities


class CollaborativeFiltering:
    def __init__(self, matrix: pd.DataFrame, mode: str = "user"):
        """
        mode: 'user' for user-based, 'item' for item-based CF
        """
        self.mode = mode
        self.matrix = matrix
        self.similarity = None

    def fit(self):
        """Compute similarity over co-rated values only."""
        if self.mode == "user":
            values = self.matrix.to_numpy(dtype=float)
        elif self.mode == "item":
            values = self.matrix.T.to_numpy(dtype=float)
        else:
            raise ValueError("mode must be 'user' or 'item'")
        self.similarity = _pairwise_observed_cosine(values)
        return self

    def recommend_for_user(self, user_id: int, top_k: int = 5) -> pd.DataFrame:
        """Recommend movies for a specific user."""
        if self.similarity is None:
            raise RuntimeError("Model not fitted. Call fit().")

        if user_id not in self.matrix.index:
            raise KeyError(f"User ID {user_id} not found.")

        user_idx = list(self.matrix.index).index(user_id)

        if self.mode == "user":
            sim_scores = np.array(self.similarity[user_idx], copy=True)
            sim_scores[user_idx] = 0.0

            observed = self.matrix.notna().to_numpy(dtype=float)
            values = self.matrix.fillna(0.0).to_numpy(dtype=float)
            weighted_sum = sim_scores @ values
            normalizer = np.abs(sim_scores) @ observed
            preds = np.divide(
                weighted_sum,
                normalizer,
                out=np.full(self.matrix.shape[1], np.nan),
                where=normalizer > 0,
            )
            predicted_ratings = pd.Series(preds, index=self.matrix.columns)
        else:  # item-based
            ratings = self.matrix.loc[user_id]
            observed = ratings.notna().to_numpy()
            weights = self.similarity[:, observed]
            weighted_sum = weights @ ratings.loc[observed].to_numpy(dtype=float)
            normalizer = np.abs(weights).sum(axis=1)
            preds = np.divide(
                weighted_sum,
                normalizer,
                out=np.full(self.matrix.shape[1], np.nan),
                where=normalizer > 0,
            )
            predicted_ratings = pd.Series(preds, index=self.matrix.columns)

        # remove movies already rated
        unseen_movies = self.matrix.loc[user_id].isna()
        recs = (
            predicted_ratings.loc[unseen_movies]
            .dropna()
            .sort_values(ascending=False)
            .head(top_k)
        )
        return recs.rename("predicted_rating").rename_axis("movieId").reset_index()


def recommend_from_ratings(
    historical_ratings: pd.DataFrame,
    user_ratings: dict[int, float],
    top_k: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies for a new visitor from explicit ratings."""
    if not user_ratings:
        raise ValueError("At least one user rating is required.")

    matrix = get_user_movie_matrix(historical_ratings)
    new_user_id = "__new_user__"
    matrix.loc[new_user_id] = np.nan
    for movie_id, rating in user_ratings.items():
        if movie_id in matrix.columns:
            matrix.loc[new_user_id, movie_id] = float(rating)

    if matrix.loc[new_user_id].notna().sum() == 0:
        raise ValueError("None of the rated movies exist in the ratings dataset.")

    return CollaborativeFiltering(matrix, mode="user").fit().recommend_for_user(
        new_user_id,
        top_k=top_k,
    )
