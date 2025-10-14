# 🎬 Movie Recommender (Content-Based)

A simple, fast content-based movie recommender using keywords, cast, genres, and director. Built from a Jupyter notebook and refactored into a clean, testable Python package with a CLI and optional Streamlit app.

## Features
- Content-based similarity with `CountVectorizer` + cosine similarity
- CLI to get recommendations from the terminal
- Optional Streamlit UI for quick demos
- Tests + packaging for easy maintenance

## Data
Place your dataset at `data/movies.csv` with columns:
`title, keywords, cast, genres, director`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
