import pandas as pd

def combine_features(row: pd.Series) -> str:
    return " ".join([
        str(row.get("keywords", "")),
        str(row.get("cast", "")),
        str(row.get("genres", "")),
        str(row.get("director", "")),
    ])

def build_corpus(df: pd.DataFrame) -> pd.Series:
    return df.apply(combine_features, axis=1)
