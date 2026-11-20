from types import SimpleNamespace

import pandas as pd

import movie_recommender_cf.models as models


def sample_ratings() -> pd.DataFrame:
    rows = []
    for user_id in range(1, 7):
        for movie_id in range(10, 16):
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": float(((user_id + movie_id) % 5) + 1),
                }
            )
    return pd.DataFrame(rows)


def test_grid_search_receives_class_and_never_sees_test_partition(monkeypatch):
    ratings = sample_ratings()
    training_rows = ratings.iloc[:30].copy()
    test_rows = ratings.iloc[30:].copy()
    captured = {}

    def fake_split(*args, **kwargs):
        return training_rows, test_rows

    class FakeAlgorithm:
        def __init__(self, **params):
            captured["params"] = params

        def fit(self, trainset):
            captured["refit_raw_ratings"] = list(trainset.all_ratings())
            return self

        def test(self, testset):
            captured["testset"] = testset
            return [SimpleNamespace(r_ui=row[2], est=row[2]) for row in testset]

    class FakeGridSearch:
        def __init__(self, algorithm_class, *args, **kwargs):
            captured["algorithm_class"] = algorithm_class
            self.best_params = {"rmse": {"n_factors": 25}}

        def fit(self, dataset):
            captured["tuning_raw_ratings"] = list(dataset.raw_ratings)

    monkeypatch.setattr(models, "train_test_split", fake_split)
    monkeypatch.setitem(models.MODEL_CLASSES, "SVD", FakeAlgorithm)
    monkeypatch.setattr(models, "GridSearchCV", FakeGridSearch)
    monkeypatch.setattr(models.accuracy, "rmse", lambda predictions, verbose: 0.0)

    _, _, _, best_params = models.train_surprise_model(ratings)

    assert captured["algorithm_class"] is FakeAlgorithm
    assert len(captured["tuning_raw_ratings"]) == len(training_rows)
    assert captured["testset"] == list(
        test_rows[models.REQUIRED_COLUMNS].itertuples(index=False, name=None)
    )
    assert best_params == {"n_factors": 25}


def test_new_user_ratings_are_included_in_svd_training(monkeypatch):
    ratings = sample_ratings()
    captured = {}

    class FakeAlgorithm:
        def predict(self, user_id, movie_id):
            return SimpleNamespace(est=float(movie_id))

    def fake_fit(combined_ratings, model_type, params):
        captured["combined"] = combined_ratings
        captured["model_type"] = model_type
        return FakeAlgorithm()

    monkeypatch.setattr(models, "fit_surprise_model", fake_fit)

    recommendations = models.recommend_for_new_user_svd(
        ratings,
        {10: 5.0, 11: 1.0},
        top_k=2,
    )

    new_user_id = ratings["userId"].max() + 1
    visitor_rows = captured["combined"].query("userId == @new_user_id")
    assert visitor_rows.set_index("movieId")["rating"].to_dict() == {10: 5.0, 11: 1.0}
    assert set(recommendations["movieId"]).isdisjoint({10, 11})
    assert captured["model_type"] == "SVD"
