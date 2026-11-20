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
    """Build a user-item matrix while preserving missing ratings as ``NaN``.

    A zero is a possible rating-scale value in some datasets and, more
    importantly, is not evidence that a user disliked a movie.  Keeping
    missing observations explicit lets recommenders normalize each prediction
    over only the neighbors that actually rated that movie.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return (
        df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
