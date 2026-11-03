"""
Awards extraction (#35): Oscar, Golden Globe and BAFTA nominations and wins of each
film's director(s) and top-3 cast, counted only BEFORE the film's release (no leakage).

Run:
  python extraction/awards.py          # full run -> data/interim/awards.parquet
  python extraction/awards.py --test   # a few known films, printed only

Needs tmdb_movies.parquet / tmdb_extra.parquet (#2, credits) and movies_manifest.json (#5).

Pipeline:
  1. Award categories: every item that is an instance (P31) or part (P361) of
     Academy Awards (Q19020), Golden Globes (Q1011547), British Academy Film Awards (Q732997).
  2. TMDB person id -> Wikidata person via P4985 (TMDb person ID), in batches.
  3. Each person's wins (P166) and nominations (P1411) in those categories, dated with the
     statement's P585, else the ceremony's date (P805 -> P585). Undated statements are dropped.
  4. Per film: count, per role, the events dated strictly before the release date.
     A win also counts as a nomination. Every manifest film gets a row; 0 = no award before.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import PROJECT_ROOT, _get_logger, sparql

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
BATCH = 300

INTERIM = PROJECT_ROOT / "data" / "interim"
OUT_PATH = INTERIM / "awards.parquet"

CEREMONIES = {"oscar": "Q19020", "globe": "Q1011547", "bafta": "Q732997"}
ROLES = {"director": "director_ids", "cast": "cast_top3_ids"}

TEST_FILMS = ["tt1375666", "tt1179904", "tt0120338", "tt0499549", "tt0468569"]  # Inception, Paranormal Activity, Titanic, Avatar, The Dark Knight

log = _get_logger("wikidata")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _batched(template: str, values: list[str], label: str) -> list[dict]:
    rows = []
    values = sorted(set(values))  # stable batches -> cache hits on rerun
    for i in range(0, len(values), BATCH):
        bindings = sparql(WIKIDATA_SPARQL, "wikidata", template % " ".join(values[i:i + BATCH]))
        if bindings is None:
            raise RuntimeError(f"Wikidata {label} query failed (batch {i}), see logs/wikidata.log")
        rows += bindings
        print(f"\r{label}: {min(i + BATCH, len(values))}/{len(values)}", end="", flush=True)
    print()
    return rows


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_films(only: list[str] | None = None) -> pd.DataFrame:
    """Manifest films with release date and TMDB credits (one row per imdb_id)."""
    manifest = json.loads((INTERIM / "movies_manifest.json").read_text(encoding="utf-8"))
    films = pd.DataFrame([{"imdb_id": k, "year": v.get("year")} for k, v in manifest.items() if k])
    if only:
        films = films[films["imdb_id"].isin(only)]

    tmdb = pd.concat([pd.read_parquet(INTERIM / f) for f in ["tmdb_movies.parquet", "tmdb_extra.parquet"]
                      if (INTERIM / f).exists()])
    tmdb = tmdb[["imdb_id", "release_date", "director_ids", "cast_top3_ids"]].drop_duplicates("imdb_id")
    films = films.merge(tmdb, on="imdb_id", how="left")

    # Release date: TMDB, else Jan 1 of the manifest year (conservative: nothing from that year counts)
    films["release"] = pd.to_datetime(films["release_date"], errors="coerce")
    no_date = films["release"].isna() & films["year"].notna()
    films.loc[no_date, "release"] = pd.to_datetime(films.loc[no_date, "year"].astype(int).astype(str) + "-01-01")
    log.info("Films: %d, without TMDB credits: %d, without any date: %d",
             len(films), films["director_ids"].isna().sum(), films["release"].isna().sum())
    return films


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------
def _latest_possible_date(value: str | None, precision: int | None) -> str | None:
    """A date known only to the year (precision 9) or month (10) is stored as the 1st of
    January / of the month; move it to the last day so an award is never counted as
    happening before a release it may actually follow (anti-leakage)."""
    if value is None:
        return None
    d = pd.Timestamp(value[:10])
    if precision is not None and precision <= 9:
        d = pd.Timestamp(year=d.year, month=12, day=31)
    elif precision == 10:
        d = d + pd.offsets.MonthEnd(0)
    return d.strftime("%Y-%m-%d")


def award_categories() -> dict[str, str]:
    """Award category QID -> ceremony key."""
    out = {}
    for key, qid in CEREMONIES.items():
        rows = sparql(WIKIDATA_SPARQL, "wikidata",
                      f"SELECT DISTINCT ?a WHERE {{ ?a wdt:P31|wdt:P361 wd:{qid} }}")
        if rows is None:
            raise RuntimeError("Wikidata award category query failed")
        out |= {_qid(r["a"]["value"]): key for r in rows}
    log.info("Award categories: %d", len(out))
    return out


def people_to_wikidata(tmdb_person_ids: list[int]) -> pd.DataFrame:
    rows = _batched("""
        SELECT ?p ?tmdb WHERE { VALUES ?tmdb { %s } ?p wdt:P4985 ?tmdb . }""",
                    [f'"{i}"' for i in tmdb_person_ids], "Persoane")
    df = pd.DataFrame([{"person": int(r["tmdb"]["value"]), "wikidata_id": _qid(r["p"]["value"])} for r in rows],
                      columns=["person", "wikidata_id"])
    return df.drop_duplicates("person")


def person_awards(qids: list[str], categories: dict[str, str]) -> pd.DataFrame:
    # The category list itself is too long for the URL (HTTP 431): filter by ceremony instead
    ceremonies = " ".join(f"wd:{q}" for q in CEREMONIES.values())
    rows = _batched("""
        SELECT ?p ?kind ?award ?date ?prec ?cerDate ?cerPrec WHERE {
          VALUES ?p { %s }
          VALUES ?ceremony { """ + ceremonies + """ }
          ?award wdt:P31|wdt:P361 ?ceremony .
          { ?p p:P166 ?st . ?st ps:P166 ?award . BIND("win" AS ?kind) }
          UNION
          { ?p p:P1411 ?st . ?st ps:P1411 ?award . BIND("nom" AS ?kind) }
          OPTIONAL { ?st pqv:P585 [ wikibase:timeValue ?date ; wikibase:timePrecision ?prec ] }
          OPTIONAL { ?st pq:P805 ?cer . ?cer p:P585/psv:P585 [ wikibase:timeValue ?cerDate ; wikibase:timePrecision ?cerPrec ] }
        }""", [f"wd:{q}" for q in qids], "Premii")
    df = pd.DataFrame([{
        "wikidata_id": _qid(r["p"]["value"]),
        "kind": r["kind"]["value"],
        "ceremony": categories[_qid(r["award"]["value"])],
        "award": _qid(r["award"]["value"]),
        "date": _latest_possible_date(*(
            (r["date"]["value"], int(r["prec"]["value"])) if "date" in r else
            (r["cerDate"]["value"], int(r["cerPrec"]["value"])) if "cerDate" in r else (None, None))),
    } for r in rows], columns=["wikidata_id", "kind", "ceremony", "award", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log.info("Award statements: %d, undated (dropped): %d", len(df), df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    # One event per person / category / year (a statement can be duplicated with several qualifiers)
    df["year"] = df["date"].dt.year
    df = df.sort_values("date").drop_duplicates(["wikidata_id", "kind", "award", "year"])
    # A win is also a nomination; avoid double counting when both statements exist
    noms = pd.concat([df[df["kind"] == "nom"], df[df["kind"] == "win"].assign(kind="nom")])
    noms = noms.drop_duplicates(["wikidata_id", "award", "year"])
    return pd.concat([noms, df[df["kind"] == "win"]], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build(only: list[str] | None = None) -> pd.DataFrame:
    films = load_films(only)

    # film x person, per role
    credits = []
    for role, col in ROLES.items():
        c = films[["imdb_id", "release", col]].explode(col).dropna(subset=[col])
        credits.append(c.rename(columns={col: "person"}).assign(role=role))
    credits = pd.concat(credits, ignore_index=True)
    credits["person"] = credits["person"].astype(int)

    people = people_to_wikidata(credits["person"].unique().tolist())
    credits = credits.merge(people, on="person", how="left")
    log.info("People: %d unique, found in Wikidata: %d", credits["person"].nunique(), len(people))

    events = person_awards(people["wikidata_id"].tolist(), award_categories())
    hits = credits.dropna(subset=["wikidata_id"]).merge(events, on="wikidata_id")
    hits = hits[hits["date"] < hits["release"]]

    out = films[["imdb_id"]].set_index("imdb_id")
    wide = hits.groupby(["imdb_id", "role", "ceremony", "kind"]).size().unstack(["role", "ceremony", "kind"])
    for role in ROLES:
        for ceremony in CEREMONIES:
            for kind, suffix in [("nom", "noms"), ("win", "wins")]:
                s = wide[(role, ceremony, kind)] if (role, ceremony, kind) in wide.columns else pd.Series(dtype=float)
                out[f"{role}_{ceremony}_{suffix}_before"] = s.reindex(out.index).fillna(0).astype(int)
        found = credits[credits["role"] == role].groupby("imdb_id")["wikidata_id"].count()
        total = credits[credits["role"] == role].groupby("imdb_id")["person"].count()
        out[f"{role}_found_wd"] = found.reindex(out.index).fillna(0).astype(int)
        out[f"{role}_n"] = total.reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


def report(df: pd.DataFrame) -> str:
    films = load_films()
    year = films.set_index("imdb_id")["release"].dt.year.reindex(df["imdb_id"]).values
    any_oscar = (df["director_oscar_noms_before"] + df["cast_oscar_noms_before"]) > 0
    decades = pd.Series(any_oscar.values, index=(pd.Series(year) // 10 * 10).values).groupby(level=0).mean()
    people = df[["director_n", "cast_n", "director_found_wd", "cast_found_wd"]].sum()
    return "\n".join([
        f"Filme: {len(df)} (cu regizor în TMDB: {(df['director_n'] > 0).sum()}, cu actori: {(df['cast_n'] > 0).sum()})",
        f"Persoane găsite în Wikidata (pe credite): regizori {people['director_found_wd'] / people['director_n']:.1%}, "
        f"actori {people['cast_found_wd'] / people['cast_n']:.1%}",
        f"Filme cu regizor nominalizat la Oscar înainte: {(df['director_oscar_noms_before'] > 0).sum()}, "
        f"premiat: {(df['director_oscar_wins_before'] > 0).sum()}",
        f"Filme cu actori (top 3) nominalizați la Oscar înainte: {(df['cast_oscar_noms_before'] > 0).sum()}, "
        f"premiați: {(df['cast_oscar_wins_before'] > 0).sum()}",
        f"Filme cu cel puțin o nominalizare Oscar înainte (regizor sau actori): {any_oscar.sum()} ({any_oscar.mean():.1%})",
        f"Globuri de Aur (nominalizări înainte, regizor/actori): {(df['director_globe_noms_before'] > 0).sum()}/"
        f"{(df['cast_globe_noms_before'] > 0).sum()}; BAFTA: {(df['director_bafta_noms_before'] > 0).sum()}/"
        f"{(df['cast_bafta_noms_before'] > 0).sum()}",
        "Procent filme cu nominalizare Oscar înainte, pe decenii: "
        + ", ".join(f"{int(d)}s={v:.0%}" for d, v in decades.items() if pd.notna(d)),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only a few known films, no file written")
    args = parser.parse_args()

    if args.test:
        df = build(only=TEST_FILMS)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df.set_index("imdb_id").T)
    else:
        df = build()
        df.to_parquet(OUT_PATH, index=False)
        print(f"Scris: {OUT_PATH}\n")
        print(report(df))
