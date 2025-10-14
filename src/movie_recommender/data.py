from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["title", "keywords", "cast", "genres", "director"]

def load_movies_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Fill NaNs
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].fillna("")
    # Keep an indexable id
    if "index" not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    else:
        df = df.rename(columns={"index": "row_id"})
    # Ensure title is string
    df["title"] = df["title"].astype(str)
    return df
