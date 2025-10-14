from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
import pandas as pd
import numpy as np
from collections import defaultdict

def load_surprise_data(df: pd.DataFrame):
    """Convert a pandas dataframe to a Surprise dataset"""
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    return data

def train_surprise_model(df: pd.DataFrame, model_type: str = "SVD", param_grid: dict | None = None):
    """
    Train a collaborative filtering model using the Surprise library.
    
    model_type: 'SVD' (matrix factorization) or 'KNN' (user/item-based CF)
    param_grid: dict for GridSearchCV
    """
    data = load_surprise_data(df)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    if model_type.upper() == "SVD":
        algo = SVD()
        default_grid = {"n_factors": [50, 100], "lr_all": [0.002, 0.005], "reg_all": [0.02, 0.1]}
    elif model_type.upper() == "KNN":
        algo = KNNBasic()
        default_grid = {"k": [20, 40], "sim_options": {"name": ["cosine", "pearson"], "user_based": [True, False]}}
    else:
        raise ValueError("model_type must be 'SVD' or 'KNN'")

    if param_grid is not None:
        default_grid = param_grid

    # Hyperparameter tuning
    gs = GridSearchCV(algo, default_grid, measures=["rmse"], cv=3, n_jobs=-1)
    gs.fit(data)
    best_params = gs.best_params["rmse"]
    best_algo = gs.best_estimator["rmse"]

    best_algo.fit(trainset)
    predictions = best_algo.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"Best {model_type} RMSE: {rmse:.4f}")
    return best_algo, predictions, rmse, best_params


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


def evaluate_model(predictions, k=10, threshold=3.5):
    """Compute precision@K and recall@K"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    print(f"Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")
    return precision, recall
