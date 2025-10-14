from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .features import build_corpus

@dataclass
class ContentRecommender:
    df: pd.DataFrame
    vectorizer: CountVectorizer | None = None
    count_matrix: np.ndarray | None = None
    sim_matrix: np.ndarray | None = None
    title_lower_to_idx: dict | None = None

    def fit(self, ngram_range: tuple[int,int]=(1,2), min_df: int = 1) -> "ContentRecommender":
        corpus = build_corpus(self.df)
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
        self.count_matrix = self.vectorizer.fit_transform(corpus)
        self.sim_matrix = cosine_similarity(self.count_matrix)
        # map lowercase titles for robust lookup
        self.title_lower_to_idx = {t.lower(): i for i, t in enumerate(self.df["title"].tolist())}
        return self

    def _idx_from_title(self, title: str) -> int:
        if self.title_lower_to_idx is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        idx = self.title_lower_to_idx.get(title.lower())
        if idx is None:
            # Try partial match
            candidates = [i for i, t in enumerate(self.df["title"].tolist()) if title.lower() in t.lower()]
            if not candidates:
                raise KeyError(f"Title not found (or close match): {title}")
            idx = candidates[0]
        return idx

    def similar(self, title: str, k: int = 10, include_scores: bool = False) -> List[str] | List[Tuple[str, float]]:
        if self.sim_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        idx = self._idx_from_title(title)
        sims = list(enumerate(self.sim_matrix[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        # exclude the item itself at position 0
        top = sims[1:k+1]
        if include_scores:
            return [(self.df.iloc[i]["title"], float(score)) for i, score in top]
        return [self.df.iloc[i]["title"] for i, _ in top]
