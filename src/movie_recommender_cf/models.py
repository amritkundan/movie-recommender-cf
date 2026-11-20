from collections import defaultdict
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Dataset, KNNBasic, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV

from .data import REQUIRED_COLUMNS


DEFAULT_PARAMETER_GRIDS = {
    "SVD": {
        "n_factors": [50, 100],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.02, 0.1],
    },
    "KNN": {
        "k": [20, 40],
        "sim_options": {
            "name": ["cosine", "pearson"],
            "user_based": [True, False],
        },
    },
}

MODEL_CLASSES = {"SVD": SVD, "KNN": KNNBasic}


def load_surprise_data(df: pd.DataFrame) -> Dataset:
    """Convert a ratings dataframe to a Surprise dataset."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Ratings data cannot be empty.")

    minimum = float(df["rating"].min())
    maximum = float(df["rating"].max())
    reader = Reader(rating_scale=(minimum, maximum))
    return Dataset.load_from_df(df[REQUIRED_COLUMNS], reader)


def _model_class(model_type: str):
    normalized_type = model_type.upper()
    try:
        return normalized_type, MODEL_CLASSES[normalized_type]
    except KeyError as error:
        raise ValueError("model_type must be 'SVD' or 'KNN'") from error


def fit_surprise_model(
    df: pd.DataFrame,
    model_type: str = "SVD",
    params: dict[str, Any] | None = None,
):
    """Fit one Surprise model on all supplied observations."""
    _, algorithm_class = _model_class(model_type)
    algorithm = algorithm_class(**(params or {}))
    algorithm.fit(load_surprise_data(df).build_full_trainset())
    return algorithm


def train_surprise_model(
    df: pd.DataFrame,
    model_type: str = "SVD",
    param_grid: dict[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Tune on training folds, then evaluate once on untouched test ratings.

    The held-out test partition is created before cross-validation.  Grid search
    receives only the training partition, so its scores cannot influence the
    final test observations.
    """
    normalized_type, algorithm_class = _model_class(model_type)
    train_df, test_df = train_test_split(
        df[REQUIRED_COLUMNS],
        test_size=test_size,
        random_state=random_state,
    )

    tuning_data = load_surprise_data(train_df)
    grid_search = GridSearchCV(
        algorithm_class,
        param_grid or DEFAULT_PARAMETER_GRIDS[normalized_type],
        measures=["rmse"],
        cv=3,
        # A single worker is more predictable on constrained app hosts and
        # avoids process-spawn overhead for this compact demonstration model.
        n_jobs=1,
    )
    grid_search.fit(tuning_data)
    best_params = grid_search.best_params["rmse"]

    # Refit a clean estimator on the complete training partition.  The grid
    # search estimators have each seen only folds of that partition.
    best_algorithm = algorithm_class(**best_params)
    best_algorithm.fit(tuning_data.build_full_trainset())
    testset = list(test_df[REQUIRED_COLUMNS].itertuples(index=False, name=None))
    predictions = best_algorithm.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return best_algorithm, predictions, rmse, best_params


def recommend_for_new_user_svd(
    historical_ratings: pd.DataFrame,
    user_ratings: dict[int, float],
    top_k: int = 10,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fit the visitor into an SVD model and rank their unseen movies."""
    if not user_ratings:
        raise ValueError("At least one user rating is required.")

    new_user_id = int(historical_ratings["userId"].max()) + 1
    visitor_rows = pd.DataFrame(
        [
            {"userId": new_user_id, "movieId": movie_id, "rating": rating}
            for movie_id, rating in user_ratings.items()
        ]
    )
    combined_ratings = pd.concat(
        [historical_ratings[REQUIRED_COLUMNS], visitor_rows],
        ignore_index=True,
    )
    algorithm = fit_surprise_model(
        combined_ratings,
        model_type="SVD",
        params=model_params,
    )

    rated_movie_ids = set(user_ratings)
    estimates = [
        (movie_id, algorithm.predict(new_user_id, movie_id).est)
        for movie_id in historical_ratings["movieId"].unique()
        if movie_id not in rated_movie_ids
    ]
    estimates.sort(key=lambda item: item[1], reverse=True)
    return pd.DataFrame(
        estimates[:top_k],
        columns=["movieId", "predicted_rating"],
    )


def get_top_n(predictions, n: int = 10):
    """Return the top-N recommendation for each user from predictions."""
    top_n = defaultdict(list)
    for user_id, movie_id, _, estimate, _ in predictions:
        top_n[user_id].append((movie_id, estimate))
    for user_id, user_ratings in top_n.items():
        user_ratings.sort(key=lambda item: item[1], reverse=True)
        top_n[user_id] = user_ratings[:n]
    return top_n


def evaluate_model(predictions, k: int = 10, threshold: float = 3.5):
    """Compute precision@K and recall@K."""
    user_est_true = defaultdict(list)
    for user_id, _, true_rating, estimate, _ in predictions:
        user_est_true[user_id].append((estimate, true_rating))

    precisions, recalls = {}, {}
    for user_id, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda item: item[0], reverse=True)
        relevant = sum(true_rating >= threshold for _, true_rating in user_ratings)
        recommended = sum(estimate >= threshold for estimate, _ in user_ratings[:k])
        relevant_and_recommended = sum(
            true_rating >= threshold and estimate >= threshold
            for estimate, true_rating in user_ratings[:k]
        )
        precisions[user_id] = (
            relevant_and_recommended / recommended if recommended else 0.0
        )
        recalls[user_id] = relevant_and_recommended / relevant if relevant else 0.0

    if not precisions:
        return 0.0, 0.0
    precision = sum(precisions.values()) / len(precisions)
    recall = sum(recalls.values()) / len(recalls)
    return precision, recall
