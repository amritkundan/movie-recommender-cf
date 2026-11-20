# 🎬 Movie Recommender System

An interactive Streamlit app that recommends movies from explicit ratings with
collaborative filtering.

Users rate a small set of familiar titles and can choose between two strategies:

- **Quick mode** compares the visitor with existing users over co-rated movies.
  Missing ratings remain missing and do not dilute similarity or prediction scores.
- **SVD mode** adds the visitor's submitted ratings to the training data before
  fitting a matrix-factorization model. The fitted resource is cached across
  Streamlit reruns for the same dataset and ratings.

## Data

The app expects these files:

- `data/ratings.csv` with `userId`, `movieId`, and `rating` columns
- `data/movies.csv` with `movieId` and `title` columns

Both recommendation modes use only explicit user–movie ratings. Titles are joined
after ranking for display.

## Run locally

Python 3.12 is the deployment target.

```bash
git clone https://github.com/amritkundan/movie-recommender-cf.git
cd movie-recommender-cf
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

`scikit-surprise>=1.1.5` is required and installed by `requirements.txt`, so SVD
mode is available in a standard installation.

## Test

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The tests cover sparse-rating handling, filtering of already-rated movies,
training-only hyperparameter selection, new-user SVD training, and Streamlit
startup.

## Deploy on Streamlit Community Cloud

Deploy `streamlit_app.py`, select Python 3.12 in Advanced settings, and keep
`requirements.txt` as the dependency file. The repository's `runtime.txt` also
documents the intended Python version.

🔗 [Live demo](https://movie-recommender-cf-js9vo4rx9vv6cmhmf5pacw.streamlit.app/)
