import argparse
from pathlib import Path
from .data import load_movies_csv
from .recommender import ContentRecommender

def main():
    parser = argparse.ArgumentParser(
        description="Content-based Movie Recommender (keywords + cast + genres + director)."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/movies.csv"),
                        help="Path to movies.csv (must contain columns: title, keywords, cast, genres, director).")
    parser.add_argument("--title", type=str, required=True, help="Seed movie title to find similar movies for.")
    parser.add_argument("--topk", type=int, default=10, help="Number of similar movies to return.")
    parser.add_argument("--scores", action="store_true", help="Include similarity scores.")
    args = parser.parse_args()

    df = load_movies_csv(args.csv)
    model = ContentRecommender(df).fit()
    results = model.similar(args.title, k=args.topk, include_scores=args.scores)

    if args.scores:
        for t, s in results:
            print(f"{t}\t{round(s, 4)}")
    else:
        for t in results:
            print(t)

if __name__ == "__main__":
    main()
