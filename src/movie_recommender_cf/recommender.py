import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, matrix: pd.DataFrame, mode: str = "user"):
        """
        mode: 'user' for user-based, 'item' for item-based CF
        """
        self.mode = mode
        self.matrix = matrix
        self.similarity = None

    def fit(self):
        """Compute similarity matrix"""
        if self.mode == "user":
            self.similarity = cosine_similarity(self.matrix)
        elif self.mode == "item":
            self.similarity = cosine_similarity(self.matrix.T)
        else:
            raise ValueError("mode must be 'user' or 'item'")
        return self

    def recommend_for_user(self, user_id: int, top_k: int = 5) -> pd.DataFrame:
        """Recommend movies for a specific user."""
        if self.similarity is None:
            raise RuntimeError("Model not fitted. Call fit().")

        if user_id not in self.matrix.index:
            raise KeyError(f"User ID {user_id} not found.")

        user_idx = list(self.matrix.index).index(user_id)

        if self.mode == "user":
            sim_scores = self.similarity[user_idx]
            # weighted ratings
            sim_scores = np.array(sim_scores)
            weighted_sum = sim_scores @ self.matrix.values
            norm = np.array([abs(sim_scores).sum()] * self.matrix.shape[1])
            preds = weighted_sum / (norm + 1e-9)
            user_ratings = pd.Series(preds, index=self.matrix.columns)
        else:  # item-based
            user_ratings = self.matrix.loc[user_id]
            preds = self.similarity @ user_ratings.values
            user_ratings = pd.Series(preds, index=self.matrix.columns)

        # remove movies already rated
        rated_movies = self.matrix.loc[user_id]
        unrated = rated_movies[rated_movies == 0]
        recs = user_ratings.loc[unrated.index].sort_values(ascending=False).head(top_k)
        return recs.reset_index().rename(columns={"index": "movieId", 0: "predicted_rating"})
