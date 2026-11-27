"""
Preprocesare pentru model (#40): dataset-ul final -> matrici gata de antrenare, fără leakage.

    from modeling.preprocessing import load_dataset, temporal_split, build_preprocessor, X_y
    df = load_dataset()                      # threshold=2.5; 2 sau 3 pentru testul de prag
    train, val, test = temporal_split(df)
    X_train, y_train = X_y(train)            # target="profitable" sau "log_roi"
    pre = build_preprocessor("linear")       # "tree": fără scalare
    Xt = pre.fit_transform(X_train)          # fit DOAR pe train
    Xv = pre.transform(X_y(val)[0])

Două etape:
  1. add_features: transformări rând cu rând (log, sin/cos, indicatori), fără parametri
     învățați din date, deci se pot aplica înainte de split.
  2. build_preprocessor: tot ce învață ceva din date (medianele, media/deviația pentru scalare,
     categoriile one-hot) e un ColumnTransformer, ajustat doar pe train.

Strategia pentru fiecare coloană și motivele: docs/preprocessing.md.

Run:
  python modeling/preprocessing.py     # dimensiunile split-urilor și echilibrul claselor
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (activează IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.columns import feature_columns
from extraction.utils import PROJECT_ROOT

DATASET = PROJECT_ROOT / "data" / "processed" / "movies_dataset.parquet"
CPI_PATH = PROJECT_ROOT / "data" / "interim" / "cpi.parquet"

THRESHOLD = 2.5
SEED = 42
# Split temporal pe anul lansării: train <= 2014, validare 2015–2018, test 2019–2024
TRAIN_END, VAL_END = 2014, 2018
MIN_CATEGORY = 50          # categoriile cu mai puțin de 50 de filme în train intră la „infrequent”

COUNTS = ["director_prev_films", "cast_prev_films",
          "director_award_noms_before", "director_award_wins_before",
          "cast_award_noms_before", "cast_award_wins_before"]

NUMERIC = [
    "log_budget", "log_budget_real", "release_year", "runtime_min", "sequel_number", "certification",
    "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    *[f"log1p_{c}" for c in COUNTS],
    "director_prev_mean_log_roi", "director_prev_profitable_rate",
    "cast_prev_mean_log_roi", "cast_prev_profitable_rate",
]
BINARY_PATTERNS = ["genre_", "country_"]
BINARY_FIXED = ["major_studio", "major_distributor", "major_any", "streaming_distributor",
                "is_english", "is_franchise", "is_adaptation",
                "has_director_history", "has_cast_history", "certification_missing"]
CATEGORICAL = ["original_language", "adaptation_type"]

# Caracteristicile din extraction/columns.py care nu intră direct, ci prin altă coloană
REPLACED = {
    "budget": ["log_budget", "log_budget_real"],
    "release_month": ["month_sin", "month_cos"],
    "release_weekday": ["weekday_sin", "weekday_cos"],
    **{c: [f"log1p_{c}"] for c in COUNTS},
    # aceeași informație ca *_prev_mean_log_roi, dar cu ROI-uri extreme (skewness > 50)
    "director_prev_mean_roi": ["director_prev_mean_log_roi"],
    "cast_prev_mean_roi": ["cast_prev_mean_log_roi"],
}

# Doar pentru experimentul leakage vs. no-leakage (#41)
LEAKAGE_NUMERIC = ["vote_average", "log1p_vote_count"]
LEAKAGE_BINARY_PATTERNS = ["kw_"]


def _cyclic(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Poziția pe un cerc: decembrie și ianuarie ajung vecine, nu la 11 unități distanță."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_features(df: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    """Transformări rând cu rând, fără parametri învățați din date (sigure înainte de split)."""
    df = df.copy()
    factor = df["release_year"].map(cpi.set_index("year")["factor_2024"])
    if factor.isna().any():
        raise ValueError(f"lipsește CPI pentru anii {sorted(df.loc[factor.isna(), 'release_year'].unique())}")
    budget = df["budget"].astype(float)
    df["log_budget"] = np.log(budget)
    df["log_budget_real"] = np.log(budget * factor)        # dolari 2024
    df["month_sin"], df["month_cos"] = _cyclic(df["release_month"] - 1, 12)
    df["weekday_sin"], df["weekday_cos"] = _cyclic(df["release_weekday"], 7)
    for c in COUNTS:
        df[f"log1p_{c}"] = np.log1p(df[c])
    # istoricul lipsește exact când nu există filme anterioare (lipsă structurală)
    df["has_director_history"] = (df["director_prev_films"] > 0).astype(int)
    df["has_cast_history"] = (df["cast_prev_films"] > 0).astype(int)
    df["certification_missing"] = df["certification"].isna().astype(int)
    df["log1p_vote_count"] = np.log1p(df["vote_count"])
    # Int64 cu <NA> -> float cu NaN, ca imputerele să le vadă
    for c in ("runtime_min", "sequel_number", "certification"):
        df[c] = df[c].astype("Float64").astype(float)
    return df


def load_dataset(path: Path = DATASET, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Dataset-ul final cu caracteristicile derivate și țintele `profitable` (la prag) și `log_roi`."""
    df = pd.read_parquet(path)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = add_features(df, pd.read_parquet(CPI_PATH))
    df["profitable"] = (df["roi"] >= threshold).astype(int)
    df["log_roi"] = np.log(df["roi"])
    return df


def binary_columns(df: pd.DataFrame, leakage: bool = False) -> list[str]:
    patterns = BINARY_PATTERNS + (LEAKAGE_BINARY_PATTERNS if leakage else [])
    return [c for c in df.columns if any(c.startswith(p) for p in patterns)] + BINARY_FIXED


def model_columns(df: pd.DataFrame, leakage: bool = False) -> dict[str, list[str]]:
    """Coloanele de intrare pe grupe; cu leakage=True se adaugă voturile și keywords."""
    numeric = NUMERIC + (LEAKAGE_NUMERIC if leakage else [])
    return {"numeric": numeric, "binary": binary_columns(df, leakage), "categorical": CATEGORICAL}


def X_y(df: pd.DataFrame, target: str = "profitable", leakage: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    cols = [c for group in model_columns(df, leakage).values() for c in group]
    return df[cols], df[target]


def temporal_split(df: pd.DataFrame, train_end: int = TRAIN_END,
                   val_end: int = VAL_END) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split principal: modelul învață din filmele vechi și e testat pe cele noi."""
    year = df["release_year"]
    return df[year <= train_end], df[(year > train_end) & (year <= val_end)], df[year > val_end]


def stratified_split(df: pd.DataFrame, target: str = "profitable", seed: int = SEED,
                     sizes: tuple[float, float] = (0.15, 0.15)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split aleator 70/15/15 cu aceeași proporție de profitabile; doar pentru comparație."""
    val_size, test_size = sizes
    rest, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                                  stratify=rest[target], random_state=seed)
    return train, val, test


def _imputer(kind: str):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    if kind == "iterative":
        return IterativeImputer(random_state=SEED, max_iter=10)
    raise ValueError(f"imputer necunoscut: {kind}")


def build_preprocessor(model_kind: str = "linear", leakage: bool = False,
                       imputer: str = "median") -> ColumnTransformer:
    """
    model_kind="linear": imputare + StandardScaler (regresie logistică, kNN, SVM).
    model_kind="tree":   imputare, fără scalare (arborii nu depind de scară).
    Coloanele binare (genre_*, country_*, kw_*) se aleg la fit, din X.
    """
    if model_kind not in ("linear", "tree"):
        raise ValueError(f"model_kind necunoscut: {model_kind}")
    # kNN / iterativ au nevoie de date pe aceeași scară: scalăm înainte, apoi imputăm
    steps = [("impute", _imputer(imputer))]
    if imputer in ("knn", "iterative"):
        steps = [("scale", StandardScaler())] + steps
    elif model_kind == "linear":
        steps.append(("scale", StandardScaler()))
    numeric = Pipeline(steps)
    onehot = OneHotEncoder(min_frequency=MIN_CATEGORY, handle_unknown="infrequent_if_exist",
                           sparse_output=False)
    pre = ColumnTransformer([
        ("numeric", numeric, NUMERIC + (LEAKAGE_NUMERIC if leakage else [])),
        ("binary", "passthrough", partial(binary_columns, leakage=leakage)),
        ("categorical", onehot, CATEGORICAL),
    ], verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, splits in (("temporal", temporal_split(df)), ("stratificat", stratified_split(df))):
        for part, s in zip(("train", "validare", "test"), splits):
            rows.append({"split": name, "set": part, "filme": len(s),
                         "% din total": round(100 * len(s) / len(df), 1),
                         "ani": f"{s['release_year'].min()}–{s['release_year'].max()}",
                         "% profitabile": round(100 * s["profitable"].mean(), 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load_dataset()
    print(summary(data).to_string(index=False))
    train, _, _ = temporal_split(data)
    X, _ = X_y(train)
    out = build_preprocessor("linear").fit_transform(X)
    print(f"\nIntrare: {X.shape[1]} coloane -> după preprocesare: {out.shape[1]} coloane")
