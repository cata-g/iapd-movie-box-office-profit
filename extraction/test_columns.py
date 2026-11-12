"""
Test pentru extraction/columns.py pe dataset-ul final.

Run: python extraction/test_columns.py   (după python extraction/integrate.py)
     python -m pytest extraction            (fixture-ul `df` citește dataset-ul o singură dată)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.columns import CATEGORIES, EXCLUDED, categories, feature_columns
from extraction.utils import PROJECT_ROOT

DATASET = PROJECT_ROOT / "data" / "processed" / "movies_dataset.parquet"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Dataset-ul final, citit o dată pentru toate testele din fișier (sub pytest)."""
    return pd.read_parquet(DATASET)


def test_no_excluded_in_features(df: pd.DataFrame):
    features = set(feature_columns(df))
    for name in ("META", "LEAKAGE", "TARGETS"):
        bad = [c for c in features if name in categories(c)]
        assert not bad, f"coloane {name} printre caracteristici: {bad}"
    for col in ("profitable", "roi", "worldwide_gross", "vote_count", "gross_n_sources", "budget_source"):
        assert col not in features, f"{col} nu are voie să fie caracteristică"
    # keywords TMDB: semnal de popularitate de după lansare (#54), doar pentru experimentul leakage
    kw = [c for c in df.columns if c.startswith("kw_")]
    assert kw, "dataset-ul trebuie să păstreze coloanele kw_* pentru experimentul leakage"
    assert not [c for c in kw if c in features], "nicio coloană kw_* nu are voie să fie caracteristică"
    assert all(categories(c) == ["LEAKAGE"] for c in kw), "coloanele kw_* trebuie să fie LEAKAGE"
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
    data = pd.read_parquet(DATASET)
    test_no_excluded_in_features(data)
    test_every_column_once(data)
    test_new_column_fails(data)
    print(f"\nToate testele au trecut. Excluse din model: {', '.join(EXCLUDED)}.")
