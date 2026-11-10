"""
Test pentru extraction/columns.py pe dataset-ul final.

Run: python extraction/test_columns.py   (după python extraction/integrate.py)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.columns import CATEGORIES, EXCLUDED, categories, feature_columns
from extraction.utils import PROJECT_ROOT

DATASET = PROJECT_ROOT / "data" / "processed" / "movies_dataset.parquet"


def test_no_excluded_in_features(df: pd.DataFrame):
    features = set(feature_columns(df))
    for name in ("META", "LEAKAGE", "TARGETS"):
        bad = [c for c in features if name in categories(c)]
        assert not bad, f"coloane {name} printre caracteristici: {bad}"
    for col in ("profitable", "roi", "worldwide_gross", "vote_count", "gross_n_sources", "budget_source"):
        assert col not in features, f"{col} nu are voie să fie caracteristică"
    print(f"Test 1 passed: {len(features)} caracteristici, nicio coloană META / LEAKAGE / TARGETS.")


def test_every_column_once(df: pd.DataFrame):
    problems = {c: categories(c) for c in df.columns if len(categories(c)) != 1}
    assert not problems, ("fiecare coloană trebuie să fie în exact o categorie "
                          f"(0 = necategorisită, >1 = ambiguă): {problems}")
    counts = {k: sum(categories(c) == [k] for c in df.columns) for k in CATEGORIES}
    print(f"Test 2 passed: toate cele {len(df.columns)} coloane au exact o categorie {counts}.")


def test_new_column_fails(df: pd.DataFrame):
    df = df.head(1).assign(coloana_noua=1)
    assert categories("coloana_noua") == [], "o coloană necunoscută nu trebuie să aibă categorie"
    assert "coloana_noua" in feature_columns(df)   # ar ajunge în model fără test
    print("Test 3 passed: o coloană nouă, necategorisită, e prinsă de testul 2.")


if __name__ == "__main__":
    df = pd.read_parquet(DATASET)
    test_no_excluded_in_features(df)
    test_every_column_once(df)
    test_new_column_fails(df)
    print(f"\nToate testele au trecut. Excluse din model: {', '.join(EXCLUDED)}.")
