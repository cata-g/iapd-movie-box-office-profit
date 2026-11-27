"""
Teste pentru modeling/preprocessing.py (#40), pe dataset-ul final.

Run: python -m pytest modeling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.columns import feature_columns
from modeling.preprocessing import (DATASET, REPLACED, X_y, build_preprocessor, load_dataset,
                                    stratified_split, temporal_split)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_dataset()


@pytest.fixture(scope="module")
def splits(df):
    return temporal_split(df)


def test_temporal_order(df, splits):
    train, val, test = splits
    assert len(train) + len(val) + len(test) == len(df)
    assert train["release_date"].max() < val["release_date"].min()
    assert val["release_date"].max() < test["release_date"].min()


def test_stratified_balance(df):
    rate = df["profitable"].mean()
    for part in stratified_split(df):
        assert abs(part["profitable"].mean() - rate) < 0.005


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_no_missing_after_transform(splits, kind):
    train, val, test = splits
    pre = build_preprocessor(kind).fit(X_y(train)[0])
    for part in splits:
        out = pre.transform(X_y(part)[0])
        assert not out.isna().any().any()
        assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_fit_only_on_train(splits):
    """Statisticile învățate sunt cele din train; datele de validare / test nu le schimbă."""
    train, val, test = splits
    X_train = X_y(train)[0]
    pre = build_preprocessor("linear").fit(X_train)
    numeric = pre.named_transformers_["numeric"]
    cols = pre.transformers_[0][2]
    assert np.allclose(numeric.named_steps["impute"].statistics_, X_train[cols].median())
    filled = X_train[cols].fillna(X_train[cols].median())
    assert np.allclose(numeric.named_steps["scale"].mean_, filled.mean())
    # categoriile one-hot vin tot din train: o limbă nouă în test nu adaugă coloane
    X_test = X_y(test)[0].assign(original_language="xx")
    assert list(pre.transform(X_test).columns) == list(pre.get_feature_names_out())


@pytest.mark.parametrize("target", ["profitable", "log_roi"])
def test_no_leakage_columns(df, target):
    X, _ = X_y(df, target=target)
    for col in X.columns:
        assert not col.startswith("kw_"), col
        assert col not in ("vote_average", "vote_count", "log1p_vote_count", "roi", "log_roi",
                           "profitable", "worldwide_gross")
    X_leak, _ = X_y(df, leakage=True)
    assert {"vote_average", "log1p_vote_count"} <= set(X_leak.columns)
    assert any(c.startswith("kw_") for c in X_leak.columns)


def test_every_feature_used(df):
    """Fiecare caracteristică din extraction/columns.py intră direct sau prin înlocuitorul ei."""
    X, _ = X_y(df)
    raw = pd.read_parquet(DATASET).head(1)
    for col in feature_columns(raw):
        assert col in X.columns or all(r in X.columns for r in REPLACED.get(col, [None])), col
    for col in X.columns:
        assert col in raw.columns or any(col in r for r in REPLACED.values()) or \
            col.startswith("has_") or col == "certification_missing", col


def test_threshold(df):
    assert df["profitable"].mean() == pytest.approx(0.3186, abs=1e-3)
    assert load_dataset(threshold=2)["profitable"].mean() > df["profitable"].mean()


def test_inflation_adjustment(df):
    row = df[df["release_year"] == 2024].iloc[0]
    assert row["log_budget_real"] == pytest.approx(row["log_budget"])
    old = df[df["release_year"] == 1965].iloc[0]
    assert np.exp(old["log_budget_real"] - old["log_budget"]) == pytest.approx(9.96, abs=0.05)
