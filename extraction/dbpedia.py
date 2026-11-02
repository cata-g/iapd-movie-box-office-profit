"""
DBpedia extraction: production budget and worldwide gross (USD only) parsed from
English Wikipedia film infoboxes, linked to imdb_id / tmdb_id via Wikidata.

Run:
  python extraction/dbpedia.py          # full run -> data/interim/dbpedia.parquet (+ .csv)
  python extraction/dbpedia.py --test   # a few known films, printed only

Output: one row per imdb_id, release year 1950-2024 (same window as TMDB, #2).
Rows without imdb_id / year, or duplicates, go to dbpedia_unmatched.csv with a `reason`.

Pipeline:
  1. DBpedia SPARQL: every dbo:Film with a budget and/or gross (dbo: or raw dbp: property) typed as usDollar.
  2. Wikipedia API: article title -> Wikidata QID (follows redirects).
     DBpedia's owl:sameAs links to Wikidata are unreliable, so they are not used.
  3. Wikidata SPARQL: QID -> imdb_id (P345), tmdb_id (P4947), earliest release date (P577).
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import (
    BUDGET_RANGE, GROSS_RANGE, PROJECT_ROOT, YEAR_MAX, YEAR_MIN, _get_logger, fetch, sparql,
)

DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

USD = "http://dbpedia.org/datatype/usDollar"
PAGE_SIZE = 10_000          # Virtuoso's max rows per result
WIKIPEDIA_BATCH = 50        # max titles per API call
WIKIDATA_BATCH = 200

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "dbpedia.parquet"
UNMATCHED_PATH = PROJECT_ROOT / "data" / "interim" / "dbpedia_unmatched.csv"

# gross/budget outside this range is usually a currency/unit error, but real outliers
# exist (Paranormal Activity), so these rows are only flagged, not dropped.
RATIO_RANGE = (0.01, 100)

TEST_FILMS = [
    "Inception",
    "Paranormal_Activity",
    "Avatar_(2009_film)",
    "Titanic_(1997_film)",
    "The_Blair_Witch_Project",
]

log = _get_logger("dbpedia")


# ---------------------------------------------------------------------------
# Step 1: DBpedia
# ---------------------------------------------------------------------------
def _sparql(endpoint: str, source: str, query: str) -> list[dict]:
    bindings = sparql(endpoint, source, query)
    if bindings is None:
        raise RuntimeError(f"SPARQL query failed on {endpoint}, see logs/{source}.log")
    return bindings


def fetch_usd_values(prop: str, only: list[str] | None = None) -> pd.DataFrame:
    """All (film, wiki_url, value) rows for a DBpedia property, USD only, paginated."""
    values = ""
    if only:
        values = "VALUES ?f { %s }" % " ".join(f"<http://dbpedia.org/resource/{t}>" for t in only)
    rows = []
    offset = 0
    while True:
        query = f"""
            SELECT DISTINCT ?f ?w ?v WHERE {{
              {values}
              ?f a dbo:Film ; {prop} ?v ; foaf:isPrimaryTopicOf ?w .
              FILTER(datatype(?v) = <{USD}>)
            }} ORDER BY ?f ?v LIMIT {PAGE_SIZE} OFFSET {offset}"""
        page = _sparql(DBPEDIA_SPARQL, "dbpedia", query)
        rows += [(b["f"]["value"], b["w"]["value"], float(b["v"]["value"])) for b in page]
        log.info("%s: offset %d -> %d rows", prop, offset, len(page))
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.DataFrame(rows, columns=["dbpedia_uri", "wiki_url", "value"])


def _collapse(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row per film. Several values (ranges, conflicting sources) -> keep the max, count them."""
    df = df[df["value"] > 0]
    return df.groupby(["dbpedia_uri", "wiki_url"])["value"].agg(**{name: "max", f"{name}_n_values": "nunique"})


# ---------------------------------------------------------------------------
# Step 2: Wikipedia title -> Wikidata QID
# ---------------------------------------------------------------------------
def titles_to_qids(titles: list[str]) -> dict[str, str]:
    result = {}
    titles = sorted(set(titles))  # stable batches -> cache hits on rerun
    for i in range(0, len(titles), WIKIPEDIA_BATCH):
        batch = titles[i:i + WIKIPEDIA_BATCH]
        data = fetch(WIKIPEDIA_API, source="wikipedia", params={
            "action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
            "redirects": 1, "titles": "|".join(batch), "format": "json", "formatversion": 2,
        })
        if data is None:
            log.warning("Wikipedia batch %d failed", i)
            continue
        q = data.get("query", {})
        renamed = {r["from"]: r["to"] for r in q.get("normalized", []) + q.get("redirects", [])}
        qid_by_page = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", [])}
        for t in batch:
            page = t
            while page in renamed:  # normalized, then redirected
                page = renamed[page]
            if qid_by_page.get(page):
                result[t] = qid_by_page[page]
    return result


# ---------------------------------------------------------------------------
# Step 3: Wikidata QID -> imdb_id, tmdb_id, release date
# ---------------------------------------------------------------------------
def wikidata_ids(qids: list[str]) -> pd.DataFrame:
    rows = []
    qids = sorted(set(qids))
    for i in range(0, len(qids), WIKIDATA_BATCH):
        batch = qids[i:i + WIKIDATA_BATCH]
        query = """
            SELECT ?item ?imdb ?tmdb ?date WHERE {
              VALUES ?item { %s }
              OPTIONAL { ?item wdt:P345 ?imdb . FILTER(STRSTARTS(?imdb, "tt")) }
              OPTIONAL { ?item wdt:P4947 ?tmdb }
              OPTIONAL { ?item wdt:P577 ?date }
            }""" % " ".join(f"wd:{q}" for q in batch)
        for b in _sparql(WIKIDATA_SPARQL, "wikidata", query):
            rows.append({
                "wikidata_id": b["item"]["value"].rsplit("/", 1)[-1],
                "imdb_id": b.get("imdb", {}).get("value"),
                "tmdb_id": b.get("tmdb", {}).get("value"),
                "date": b.get("date", {}).get("value"),
            })
    df = pd.DataFrame(rows, columns=["wikidata_id", "imdb_id", "tmdb_id", "date"])
    df["date"] = pd.to_datetime(df["date"].str[:10], errors="coerce")
    return df.groupby("wikidata_id").agg(
        imdb_id=("imdb_id", "min"),
        imdb_n_ids=("imdb_id", "nunique"),
        tmdb_id=("tmdb_id", "min"),
        release_date_wd=("date", "min"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(only: list[str] | None = None) -> pd.DataFrame:
    budget = _collapse(fetch_usd_values("dbo:budget|dbp:budget", only), "budget_dbp")
    gross = _collapse(fetch_usd_values("dbo:gross|dbp:gross", only), "worldwide_gross_dbp")
    df = budget.join(gross, how="outer").reset_index()
    for col in ["budget_dbp_n_values", "worldwide_gross_dbp_n_values"]:
        df[col] = df[col].fillna(0).astype(int)
    df["dbp_title"] = df["wiki_url"].map(lambda u: unquote(u.rsplit("/wiki/", 1)[-1]).replace("_", " "))
    log.info("DBpedia: %d films", len(df))

    qids = titles_to_qids(df["dbp_title"].tolist())
    df["wikidata_id"] = df["dbp_title"].map(qids)
    log.info("Wikipedia: %d/%d titles -> QID", df["wikidata_id"].notna().sum(), len(df))

    wd = wikidata_ids(df["wikidata_id"].dropna().tolist())
    df = df.join(wd, on="wikidata_id")
    df["imdb_n_ids"] = df["imdb_n_ids"].fillna(0).astype(int)
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")

    df["year"] = df["release_date_wd"].dt.year.astype("Int64")

    for col, (lo, hi) in [("budget_dbp", BUDGET_RANGE), ("worldwide_gross_dbp", GROSS_RANGE)]:
        bad = df[col].notna() & ~df[col].between(lo, hi)
        log.info("%s: %d values outside [%d, %d] -> NaN", col, bad.sum(), lo, hi)
        df[f"{col}_implausible"] = bad
        df.loc[bad, col] = float("nan")

    ratio = df["worldwide_gross_dbp"] / df["budget_dbp"]
    df["suspect_ratio"] = ratio.notna() & ~ratio.between(*RATIO_RANGE)

    return df[[
        "imdb_id", "tmdb_id", "wikidata_id", "dbp_title", "year", "release_date_wd",
        "budget_dbp", "worldwide_gross_dbp",
        "budget_dbp_n_values", "worldwide_gross_dbp_n_values", "imdb_n_ids",
        "budget_dbp_implausible", "worldwide_gross_dbp_implausible", "suspect_ratio",
        "dbpedia_uri", "wiki_url",
    ]].sort_values("dbp_title", ignore_index=True)


def split(df: pd.DataFrame, year_min: int, year_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    matched: one row per imdb_id, year in [year_min, year_max] -> ready to merge into the manifest.
    unmatched: everything else, with a `reason` column.
    """
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason[df["budget_dbp"].isna() & df["worldwide_gross_dbp"].isna()] = "no_plausible_value"
    reason[df["year"].isna()] = "no_year"
    reason[df["year"].notna() & ~df["year"].between(year_min, year_max)] = "year_out_of_range"
    reason[df["imdb_id"].isna()] = "no_imdb_id"

    # Several Wikipedia articles can point to the same imdb_id: keep the most complete row
    ok = df[reason.isna()].copy()
    ok["_filled"] = ok[["budget_dbp", "worldwide_gross_dbp"]].notna().sum(axis=1)
    ok = ok.sort_values(["imdb_id", "_filled"], ascending=[True, False])
    dup = ok.duplicated("imdb_id", keep="first")
    reason[dup[dup].index] = "duplicate_imdb_id"

    matched = df[reason.isna()].sort_values("imdb_id", ignore_index=True)
    unmatched = df[reason.notna()].assign(reason=reason[reason.notna()]).reset_index(drop=True)
    return matched, unmatched


def report(df: pd.DataFrame, unmatched: pd.DataFrame) -> str:
    n = len(df) + len(unmatched)
    has_imdb = n - (unmatched["reason"] == "no_imdb_id").sum()
    lines = [
        f"Total filme DBpedia (orice an): {n}",
        f"Cu imdb_id: {has_imdb} ({has_imdb / n:.1%})",
        f"Excluse: " + ", ".join(f"{k}={v}" for k, v in unmatched["reason"].value_counts().items()),
        f"Păstrate în dbpedia.parquet: {len(df)}",
        f"Cu budget_dbp: {df['budget_dbp'].notna().sum()}",
        f"Cu worldwide_gross_dbp > 0: {(df['worldwide_gross_dbp'] > 0).sum()}",
        f"Cu buget și încasări: {(df['budget_dbp'].notna() & (df['worldwide_gross_dbp'] > 0)).sum()}",
        f"Ambiguități: {(df['budget_dbp_n_values'] > 1).sum()} bugete cu mai multe valori, "
        f"{(df['worldwide_gross_dbp_n_values'] > 1).sum()} încasări cu mai multe valori, "
        f"{(df['imdb_n_ids'] > 1).sum()} filme cu mai multe imdb_id",
        f"Valori neplauzibile șterse (din cei păstrați): {df['budget_dbp_implausible'].sum()} bugete, "
        f"{df['worldwide_gross_dbp_implausible'].sum()} încasări",
        f"Marcate suspect_ratio (încasări/buget < {RATIO_RANGE[0]} sau > {RATIO_RANGE[1]}): {df['suspect_ratio'].sum()}",
        "Pe decenii: " + ", ".join(f"{d}s={c}" for d, c in (df["year"] // 10 * 10).value_counts().sort_index().items()),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only a few known films, no file written")
    parser.add_argument("--year-min", type=int, default=YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=YEAR_MAX)
    args = parser.parse_args()

    if args.test:
        df = build(only=TEST_FILMS)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df.drop(columns=["dbpedia_uri", "wiki_url"]))
    else:
        df, unmatched = split(build(), args.year_min, args.year_max)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
        df.to_csv(OUT_PATH.with_suffix(".csv"), index=False, encoding="utf-8")
        unmatched.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8")
        print(f"Scris: {OUT_PATH} (+ .csv), {UNMATCHED_PATH}\n")
        print(report(df, unmatched))
