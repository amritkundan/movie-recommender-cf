import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = ["userId", "movieId", "rating"]

def load_ratings(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def get_user_movie_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot user-movie matrix"""
    matrix = df.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    return matrix
