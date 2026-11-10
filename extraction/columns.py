"""
Categoriile coloanelor din data/processed/movies_dataset.parquet (#45).

    from extraction.columns import feature_columns
    X = df[feature_columns(df)]

Doar FEATURES intră în model. Restul se exclud explicit:
  TARGETS  ținta și ce se calculează din ea (încasările nu se știu înainte de lansare)
  IDS      identificare și text brut (genurile sunt deja one-hot în genre_*)
  META     de unde vin bugetul și încasările, sumele brute per sursă, monede și cursuri,
           calitatea datelor. Prezic ținta (un film cu încasări în toate cele 4 surse e mai
           cunoscut, deci mai des profitabil), dar nu sunt informații despre film: leakage.
  LEAKAGE  vote_average, vote_count: doar pentru experimentul leakage vs. no-leakage

Fiecare coloană trebuie să fie în exact o categorie (extraction/test_columns.py): o coloană
nouă, necategorisită, face testul să pice. Numele cu `*` sunt tipare (fnmatch).
"""

from fnmatch import fnmatchcase

import pandas as pd

TARGETS = ["worldwide_gross", "roi", "profitable"]

IDS = ["imdb_id", "tmdb_id", "title", "release_date", "genres"]

META = [
    # sursa valorii finale și cât de sigură e
    "*_source", "*_n_sources", "*_n_agree", "*_conflict",
    # valorile brute per sursă, în USD
    "budget_tmdb", "budget_dbp", "budget_wd", "budget_wp",
    "gross_wp", "worldwide_gross_dbp", "gross_wd", "revenue_tmdb",
    # suma în moneda ei, moneda ROI-ului și cursul folosit
    "*_amount", "*_currency", "fx_year", "*_fx_rate",
    # calitatea datelor
    "budget_is_range", "gross_is_domestic", "gross_is_rental", "gross_is_single_market",
    "gross_is_local_currency",
]

LEAKAGE = ["vote_average", "vote_count"]

FEATURES = [
    "budget",
    "director_award_noms_before", "director_award_wins_before",
    "cast_award_noms_before", "cast_award_wins_before",
    "major_studio", "major_distributor", "major_any",
    "release_year", "release_month", "release_weekday",
    "genre_*", "runtime_min", "certification", "kw_*",
    "is_english", "original_language", "country_*",
    "is_franchise", "sequel_number", "is_adaptation", "adaptation_type",
    "director_prev_*", "cast_prev_*",
]

CATEGORIES = {"TARGETS": TARGETS, "IDS": IDS, "META": META, "LEAKAGE": LEAKAGE, "FEATURES": FEATURES}
EXCLUDED = ("TARGETS", "IDS", "META", "LEAKAGE")


def _matches(col: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(col, p) for p in patterns)


def categories(col: str) -> list[str]:
    """Categoriile în care intră o coloană (corect: exact una)."""
    return [name for name, patterns in CATEGORIES.items() if _matches(col, patterns)]


def excluded_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(_matches(c, CATEGORIES[k]) for k in EXCLUDED)]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Toate coloanele care nu sunt TARGETS, IDS, META sau LEAKAGE."""
    excluded = set(excluded_columns(df))
    return [c for c in df.columns if c not in excluded]
