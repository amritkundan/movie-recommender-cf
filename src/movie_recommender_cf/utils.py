import pandas as pd

def merge_with_movies(recs: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    if "movieId" not in movies_df.columns or "title" not in movies_df.columns:
        return recs
    return recs.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
