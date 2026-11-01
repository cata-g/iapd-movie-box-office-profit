"""
Wikidata extraction: every film with an imdb_id, its production cost (P2130) and
worldwide box office (P2142) in USD, whether it is an adaptation (P144) and its
English Wikipedia article (used by extraction/wikipedia.py).

Run:
  python extraction/wikidata.py          # full run -> data/interim/wikidata.parquet (+ .csv)
  python extraction/wikidata.py --test   # a few known films, printed only

Pipeline:
  1. Film list, one SPARQL query per half-year of P577 (a whole year times out):
     ?film wdt:P31/wdt:P279* wd:Q11424 ; wdt:P345 ?imdb_id, plus the enwiki sitelink.
     A period that still times out is split again (quarters, then months).
  2. Details in batches of QIDs: every P577 (earliest = release date), P2130 and P2142
     with their currency, P2142's "valid in place" qualifier, P144, P4947 (tmdb_id).
  3. Amounts are kept only in USD (wd:Q4917); other currencies are counted, not converted.
     Box office counts as worldwide when it has no place qualifier or place = worldwide.
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import (
    BUDGET_RANGE, GROSS_RANGE, PROJECT_ROOT, YEAR_MAX, YEAR_MIN, _get_logger, sparql,
)

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USD = "Q4917"
WORLDWIDE = "Q13780930"
DETAILS_BATCH = 300

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "wikidata.parquet"

TEST_FILMS = ["Q25188", "Q381028", "Q24871", "Q44578", "Q644554"]  # Inception, Paranormal Activity, Avatar, Titanic, Blair Witch

log = _get_logger("wikidata")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Step 1: film list
# ---------------------------------------------------------------------------
LIST_QUERY = """
SELECT ?film ?imdb ?enwiki WHERE {
  %s
  ?film wdt:P345 ?imdb .
  ?film wdt:P31/wdt:P279* wd:Q11424 .
  OPTIONAL { ?enwiki schema:about ?film ; schema:isPartOf <https://en.wikipedia.org/> }
}"""


def _period_filter(start: date, end: date) -> str:
    return (f'?film wdt:P577 ?d . FILTER(?d >= "{start.isoformat()}T00:00:00Z"^^xsd:dateTime '
            f'&& ?d < "{end.isoformat()}T00:00:00Z"^^xsd:dateTime)')


def _split(start: date, end: date) -> list[tuple[date, date]]:
    months = (end.year - start.year) * 12 + end.month - start.month
    if months <= 1:
        return []
    step = 3 if months > 3 else 1
    out, cur = [], start
    while cur < end:
        m = cur.month - 1 + step
        nxt = date(cur.year + m // 12, m % 12 + 1, 1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def list_films_in_period(start: date, end: date) -> list[dict]:
    rows = sparql(WIKIDATA_SPARQL, "wikidata", LIST_QUERY % _period_filter(start, end))
    if rows is not None:
        log.info("list %s..%s: %d rows", start, end, len(rows))
        return rows
    parts = _split(start, end)
    if not parts:
        raise RuntimeError(f"Wikidata list query failed for {start}..{end}, see logs/wikidata.log")
    log.warning("list %s..%s timed out, splitting into %d parts", start, end, len(parts))
    return [r for s, e in parts for r in list_films_in_period(s, e)]


def list_films(only: list[str] | None = None) -> pd.DataFrame:
    if only:
        rows = sparql(WIKIDATA_SPARQL, "wikidata",
                      LIST_QUERY % ("VALUES ?film { %s }" % " ".join(f"wd:{q}" for q in only)))
    else:
        rows = []
        for year in range(YEAR_MIN, YEAR_MAX + 1):
            rows += list_films_in_period(date(year, 1, 1), date(year, 7, 1))
            rows += list_films_in_period(date(year, 7, 1), date(year + 1, 1, 1))
            print(f"\rLista filme: {year}", end="", flush=True)
        print()
    df = pd.DataFrame([{
        "wikidata_id": _qid(r["film"]["value"]),
        "imdb_id": r["imdb"]["value"],
        "enwiki_title": unquote(r["enwiki"]["value"].rsplit("/wiki/", 1)[-1]).replace("_", " ") if "enwiki" in r else None,
    } for r in rows])
    df = df[df["imdb_id"].str.startswith("tt")]
    return df.groupby("wikidata_id").agg(
        imdb_id=("imdb_id", "min"),
        imdb_n_ids=("imdb_id", "nunique"),
        enwiki_title=("enwiki_title", "first"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Step 2: details per batch of QIDs
# ---------------------------------------------------------------------------
DETAILS_QUERY = """
SELECT ?film ?kind ?amount ?unit ?place ?value WHERE {
  VALUES ?film { %s }
  {
    ?film p:P2130 ?st . ?st wikibase:rank ?rank .
    ?st psv:P2130 [ wikibase:quantityAmount ?amount ; wikibase:quantityUnit ?unit ] .
    BIND("cost" AS ?kind)
  } UNION {
    ?film p:P2142 ?st . ?st wikibase:rank ?rank .
    ?st psv:P2142 [ wikibase:quantityAmount ?amount ; wikibase:quantityUnit ?unit ] .
    OPTIONAL { ?st pq:P3005 ?place }
    BIND("box_office" AS ?kind)
  } UNION {
    ?film wdt:P144 ?value . BIND("based_on" AS ?kind)
  } UNION {
    ?film wdt:P577 ?value . BIND("date" AS ?kind)
  } UNION {
    ?film wdt:P4947 ?value . BIND("tmdb" AS ?kind)
  }
  FILTER(!BOUND(?rank) || ?rank != wikibase:DeprecatedRank)
}"""


def fetch_details(qids: list[str]) -> pd.DataFrame:
    rows = []
    qids = sorted(set(qids))  # stable batches -> cache hits on rerun
    for i in range(0, len(qids), DETAILS_BATCH):
        batch = qids[i:i + DETAILS_BATCH]
        bindings = sparql(WIKIDATA_SPARQL, "wikidata", DETAILS_QUERY % " ".join(f"wd:{q}" for q in batch))
        if bindings is None:
            raise RuntimeError(f"Wikidata details query failed (batch {i}), see logs/wikidata.log")
        for b in bindings:
            rows.append({
                "wikidata_id": _qid(b["film"]["value"]),
                "kind": b["kind"]["value"],
                "amount": float(b["amount"]["value"]) if "amount" in b else None,
                "unit": _qid(b["unit"]["value"]) if "unit" in b else None,
                "place": _qid(b["place"]["value"]) if "place" in b else None,
                "value": b.get("value", {}).get("value"),
            })
        print(f"\rDetalii: {min(i + DETAILS_BATCH, len(qids))}/{len(qids)}", end="", flush=True)
    print()
    return pd.DataFrame(rows, columns=["wikidata_id", "kind", "amount", "unit", "place", "value"])


def _money(details: pd.DataFrame, kind: str, name: str, worldwide_only: bool = False) -> pd.DataFrame:
    d = details[details["kind"] == kind]
    if worldwide_only:
        d = d[d["place"].isna() | (d["place"] == WORLDWIDE)]
    usd = d[(d["unit"] == USD) & (d["amount"] > 0)]
    out = usd.groupby("wikidata_id")["amount"].agg(**{name: "max", f"{name}_n_values": "nunique"})
    # Films that have the amount only in another currency (reported, not converted)
    other = d[~d["wikidata_id"].isin(usd["wikidata_id"]) & (d["unit"] != USD)]
    other_unit = other.groupby("wikidata_id")["unit"].first().rename(f"{name}_other_currency")
    return out.join(other_unit, how="outer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(only: list[str] | None = None) -> pd.DataFrame:
    films = list_films(only)
    log.info("Film list: %d items with imdb_id", len(films))

    details = fetch_details(films["wikidata_id"].tolist())

    dates = details[details["kind"] == "date"].assign(
        release_date_wd=lambda d: pd.to_datetime(d["value"].str[:10], errors="coerce"))
    first_date = dates.groupby("wikidata_id")["release_date_wd"].min()
    tmdb = details[details["kind"] == "tmdb"].groupby("wikidata_id")["value"].min().rename("tmdb_id")
    adaptations = set(details.loc[details["kind"] == "based_on", "wikidata_id"])

    df = (films.set_index("wikidata_id")
          .join(first_date).join(tmdb)
          .join(_money(details, "cost", "budget_wd"))
          .join(_money(details, "box_office", "gross_wd", worldwide_only=True))
          .reset_index())
    df["is_adaptation"] = df["wikidata_id"].isin(adaptations)
    df["year"] = df["release_date_wd"].dt.year.astype("Int64")
    df["tmdb_id"] = pd.to_numeric(df["tmdb_id"], errors="coerce").astype("Int64")
    for col in ["budget_wd_n_values", "gross_wd_n_values"]:
        df[col] = df[col].fillna(0).astype(int)

    for col, (lo, hi) in [("budget_wd", BUDGET_RANGE), ("gross_wd", GROSS_RANGE)]:
        bad = df[col].notna() & ~df[col].between(lo, hi)
        log.info("%s: %d values outside [%d, %d] -> NaN", col, bad.sum(), lo, hi)
        df[f"{col}_implausible"] = bad
        df.loc[bad, col] = float("nan")

    return df[[
        "imdb_id", "wikidata_id", "tmdb_id", "enwiki_title", "year", "release_date_wd",
        "budget_wd", "gross_wd", "is_adaptation",
        "budget_wd_n_values", "gross_wd_n_values", "budget_wd_other_currency", "gross_wd_other_currency",
        "budget_wd_implausible", "gross_wd_implausible", "imdb_n_ids",
    ]]


def finalize(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    """Keep films whose first release is inside the window; one row per imdb_id (most complete)."""
    df = df[df["year"].between(year_min, year_max)].copy()
    df["_filled"] = df[["budget_wd", "gross_wd", "enwiki_title"]].notna().sum(axis=1)
    df = df.sort_values(["imdb_id", "_filled"], ascending=[True, False])
    return df.drop_duplicates("imdb_id").drop(columns="_filled").reset_index(drop=True)


CURRENCY_CODES = {"Q4916": "EUR", "Q25224": "GBP", "Q41044": "RUB", "Q1104069": "CAD", "Q259502": "AUD",
                  "Q8146": "JPY", "Q80524": "INR", "Q39099": "CNY", "Q202040": "KRW", "Q25344": "CHF"}


def report(df: pd.DataFrame) -> str:
    top_other = df["budget_wd_other_currency"].map(lambda q: CURRENCY_CODES.get(q, q)).value_counts().head(5)
    return "\n".join([
        f"Filme cu imdb_id în Wikidata ({YEAR_MIN}–{YEAR_MAX}, unic pe imdb_id): {len(df)}",
        f"Cu articol en.wikipedia: {df['enwiki_title'].notna().sum()}",
        f"Cost în USD (budget_wd): {df['budget_wd'].notna().sum()}",
        f"Cost doar în altă monedă: {df['budget_wd_other_currency'].notna().sum()} "
        f"(top: {', '.join(f'{k}={v}' for k, v in top_other.items())})",
        f"Încasări mondiale în USD (gross_wd): {df['gross_wd'].notna().sum()}",
        f"Încasări doar în altă monedă: {df['gross_wd_other_currency'].notna().sum()}",
        f"Cu buget și încasări în USD: {(df['budget_wd'].notna() & df['gross_wd'].notna()).sum()}",
        f"Ecranizări (P144): {df['is_adaptation'].sum()}",
        f"Valori neplauzibile șterse: {df['budget_wd_implausible'].sum()} bugete, {df['gross_wd_implausible'].sum()} încasări",
        f"Ambiguități: {(df['budget_wd_n_values'] > 1).sum()} costuri cu mai multe valori, "
        f"{(df['gross_wd_n_values'] > 1).sum()} încasări cu mai multe valori, "
        f"{(df['imdb_n_ids'] > 1).sum()} filme cu mai multe imdb_id",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only a few known films, no file written")
    parser.add_argument("--year-min", type=int, default=YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=YEAR_MAX)
    args = parser.parse_args()

    if args.test:
        df = build(only=TEST_FILMS)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df)
    else:
        df = finalize(build(), args.year_min, args.year_max)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
        df.to_csv(OUT_PATH.with_suffix(".csv"), index=False, encoding="utf-8")
        print(f"Scris: {OUT_PATH} (+ .csv)\n")
        print(report(df))
