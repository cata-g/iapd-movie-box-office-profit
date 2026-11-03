"""
Issue #27 — Integrare și curățare → dataset final

Pornește de la manifest, unește toate tabelele din data/interim/ pe imdb_id,
alege bugetul și încasările finale, aplică regulile de curățare, calculează
ținta și caracteristicile și scrie:
    data/processed/movies_dataset.csv
    docs/integration_report.md

Usage:
    python extraction/integrate.py                  # prag 2.5
    python extraction/integrate.py --threshold 3
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.utils import (BUDGET_RANGE, GROSS_RANGE, PROJECT_ROOT,
                              YEAR_MAX, YEAR_MIN)

INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_CSV = PROCESSED / "movies_dataset.csv"
REPORT = PROJECT_ROOT / "docs" / "integration_report.md"

# Ordinea de prioritate (CLAUDE.md)
BUDGET_SOURCES = {"tmdb": "budget_tmdb", "dbpedia": "budget_dbp",
                  "wikidata": "budget_wd", "wikipedia": "budget_wp"}
GROSS_SOURCES = {"wikipedia": "gross_wp", "dbpedia": "worldwide_gross_dbp",
                 "wikidata": "gross_wd", "tmdb": "revenue_tmdb"}

CONFLICT_RATIO = 1.25          # max/min > 1.25 → conflict (> 25%)
ROI_EXTREME = (0.001, 100)     # în afara intervalului → păstrat doar dacă e confirmat
MIN_RUNTIME = 40               # sub 40 min = scurtmetraj (fără lansare în cinema)

# ID-uri TMDB ale marilor studiouri și ale diviziilor lor de film
MAJOR_STUDIOS = {
    "Disney": [2, 3166, 3, 6125, 171656, 420, 1, 9195],
    "Warner Bros.": [174, 12, 2785, 4051],
    "Universal": [33],
    "Paramount": [4],
    "Sony/Columbia": [5, 34, 559, 3287, 2251, 63520],
    "20th Century": [25, 127928, 43],
    "Lionsgate": [1632, 35],
}
MAJOR_IDS = {i for ids in MAJOR_STUDIOS.values() for i in ids}

CERTIFICATION = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4}

EUROPE = set("AD AL AT BA BE BG BY CH CS CY CZ DD DE DK EE ES FI FR GB GR HR HU IE IS IT "
             "LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SK SM SU UA VA XC XG YU".split())
ASIA = set("AE AF BD BN BT CN HK ID IL IN IQ IR JO JP KG KH KP KR KW KZ LA LB LK MM MN MO "
           "MV MY NP OM PH PK PS QA SA SG SY TH TJ TL TM TR TW UZ VN YE".split())

TOP_KEYWORDS = 30
TOP_LANGUAGES = 5


# ---------------------------------------------------------------------------
# Încărcare
# ---------------------------------------------------------------------------

def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["imdb_id"].notna() & (df["imdb_id"] != "")]
    return df.drop_duplicates("imdb_id").set_index("imdb_id")


def load_sources(manifest: dict) -> pd.DataFrame:
    tmdb = pd.concat([pd.read_parquet(INTERIM / "tmdb_movies.parquet"),
                      pd.read_parquet(INTERIM / "tmdb_extra.parquet")], ignore_index=True)
    # Același imdb_id poate apărea la mai multe intrări TMDB: păstrăm tmdb_id-ul din manifest
    want = {k: v.get("tmdb_id") for k, v in manifest.items()}
    tmdb["_match"] = [want.get(i) == t for i, t in zip(tmdb["imdb_id"], tmdb["tmdb_id"])]
    tmdb = tmdb.sort_values(["imdb_id", "_match", "tmdb_id"], ascending=[True, False, True])
    tmdb = _dedupe(tmdb.drop(columns="_match"))

    dbp = _dedupe(pd.read_parquet(INTERIM / "dbpedia.parquet"))
    wd = _dedupe(pd.read_parquet(INTERIM / "wikidata.parquet"))
    wp = _dedupe(pd.read_parquet(INTERIM / "wikipedia.parquet"))
    aw = _dedupe(pd.read_parquet(INTERIM / "awards.parquet"))

    ids = sorted(k for k in manifest if k)
    df = pd.DataFrame(index=pd.Index(ids, name="imdb_id"))
    df["manifest_year"] = [manifest[i].get("year") for i in ids]
    df = df.join(tmdb[["tmdb_id", "title", "release_date", "budget_tmdb", "revenue_tmdb",
                       "runtime", "genres", "original_language", "production_countries",
                       "company_ids", "collection_id", "certification_us", "director_ids",
                       "cast_top3_ids", "keywords", "vote_average", "vote_count",
                       "sequel_number"]])
    df = df.join(dbp[["budget_dbp", "worldwide_gross_dbp"]])
    df = df.join(wd[["budget_wd", "gross_wd", "is_adaptation", "adaptation_type"]])
    df = df.join(wp[["budget_wp", "gross_wp", "based_on_wp"]])
    df = df.join(aw)
    return df


# ---------------------------------------------------------------------------
# Buget și încasări finale
# ---------------------------------------------------------------------------

def _plausible(s: pd.Series, lo: float, hi: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(lo, hi))


def choose_value(df: pd.DataFrame, sources: dict, prefix: str) -> pd.DataFrame:
    vals = pd.DataFrame({name: df[col] for name, col in sources.items()})
    first = vals.notna().idxmax(axis=1).where(vals.notna().any(axis=1))
    df[prefix] = vals.bfill(axis=1).iloc[:, 0]
    df[f"{prefix}_source"] = first
    df[f"{prefix}_n_sources"] = vals.notna().sum(axis=1)
    df[f"{prefix}_conflict"] = (vals.max(axis=1) / vals.min(axis=1)) > CONFLICT_RATIO
    # câte surse (inclusiv cea aleasă) sunt la cel mult 25% de valoarea aleasă
    ratio = vals.div(df[prefix], axis=0)
    df[f"{prefix}_n_agree"] = ratio.apply(
        lambda r: r.between(1 / CONFLICT_RATIO, CONFLICT_RATIO)).sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Curățare
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, funnel: list) -> pd.DataFrame:
    def step(name: str, mask: pd.Series, rule: str) -> pd.DataFrame:
        nonlocal df
        removed = int((~mask).sum())
        df = df[mask]
        funnel.append({"pas": name, "regula": rule, "eliminate": removed, "raman": len(df)})
        return df

    funnel.append({"pas": "manifest", "regula": "filme cu buget > 0 în cel puțin o sursă, cu imdb_id",
                   "eliminate": 0, "raman": len(df)})

    # Valori absurde per sursă → lipsă (buget < 10.000 USD = eroare)
    n_budget_err = {}
    for col in BUDGET_SOURCES.values():
        raw = pd.to_numeric(df[col], errors="coerce")
        df[col] = _plausible(raw, *BUDGET_RANGE)
        n_budget_err[col] = int((raw.notna() & (raw > 0) & df[col].isna()).sum())
    for col in GROSS_SOURCES.values():
        df[col] = _plausible(df[col], *GROSS_RANGE)

    df = choose_value(df, BUDGET_SOURCES, "budget")
    df = choose_value(df, GROSS_SOURCES, "worldwide_gross")
    df = df.rename(columns={"worldwide_gross_source": "gross_source",
                            "worldwide_gross_n_sources": "gross_n_sources",
                            "worldwide_gross_conflict": "gross_conflict",
                            "worldwide_gross_n_agree": "gross_n_agree"})

    step("buget", df["budget"].notna(),
         f"fără buget plauzibil ({BUDGET_RANGE[0]:,}–{BUDGET_RANGE[1]:,} USD) în nicio sursă; "
         f"valori per sursă invalidate: " + ", ".join(f"`{k}` {v:,}" for k, v in n_budget_err.items()))
    step("încasări", df["worldwide_gross"].notna(),
         f"fără încasări mondiale plauzibile ({GROSS_RANGE[0]:,}–{GROSS_RANGE[1]:,} USD) în nicio sursă")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    step("dată lansare", df["release_date"].notna(),
         "fără dată completă de lansare în TMDB (necesară pentru lună, zi și istoricul anterior)")
    year = df["release_date"].dt.year
    step("interval ani", year.between(YEAR_MIN, YEAR_MAX), f"lansat în afara {YEAR_MIN}–{YEAR_MAX}")

    is_tv = df["genres"].apply(lambda g: g is not None and "TV Movie" in list(g))
    step("film TV", ~is_tv, "gen TMDB „TV Movie” (fără lansare în cinema)")
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    step("scurtmetraj", ~runtime.between(1, MIN_RUNTIME - 1), f"durată între 1 și {MIN_RUNTIME - 1} min")

    df["roi"] = df["worldwide_gross"] / df["budget"]
    extreme = ~df["roi"].between(*ROI_EXTREME)
    confirmed = (df["budget_n_agree"] >= 2) & (df["gross_n_agree"] >= 2)
    step("ROI extrem", ~extreme | confirmed,
         f"ROI < {ROI_EXTREME[0]} sau > {ROI_EXTREME[1]} fără confirmare "
         "(≥ 2 surse care dau, la ±25%, aceeași valoare pentru buget și pentru încasări)")
    return df


# ---------------------------------------------------------------------------
# Caracteristici
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _as_list(x) -> list:
    return [] if x is None or (isinstance(x, float) and np.isnan(x)) else list(x)


def _country_group(code: str) -> str:
    if code == "US":
        return "us"
    if code in EUROPE:
        return "europe"
    if code in ASIA:
        return "asia"
    return "other"


def prev_history(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Pentru fiecare film și fiecare persoană din `col`: filmele din dataset ale
    persoanei lansate strict înainte. Întoarce (nr. maxim de filme anterioare,
    media ROI per persoană, mediată peste persoanele cu istoric)."""
    films_by_person = defaultdict(list)
    for imdb_id, people, date, roi in zip(df.index, df[col], df["release_date"], df["roi"]):
        for p in _as_list(people):
            films_by_person[p].append((date, roi))
    for p in films_by_person:
        films_by_person[p].sort(key=lambda x: x[0])

    n_prev, mean_roi = [], []
    for people, date in zip(df[col], df["release_date"]):
        counts, means = [], []
        for p in _as_list(people):
            prior = [r for d, r in films_by_person[p] if d < date]
            counts.append(len(prior))
            if prior:
                means.append(float(np.mean(prior)))
        n_prev.append(max(counts) if counts else 0)
        mean_roi.append(float(np.mean(means)) if means else np.nan)
    return pd.Series(n_prev, index=df.index), pd.Series(mean_roi, index=df.index)


def features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["tmdb_id"] = df["tmdb_id"].astype("Int64")
    out["title"] = df["title"]

    # Ținta
    out["budget"] = df["budget"].round().astype("int64")
    out["worldwide_gross"] = df["worldwide_gross"].round().astype("int64")
    out["roi"] = df["roi"].round(4)
    out["profitable"] = (df["roi"] >= threshold).astype(int)

    # 2. Premii înainte de lansare (#35)
    for who in ("director", "cast"):
        for kind in ("noms", "wins"):
            cols = [f"{who}_{a}_{kind}_before" for a in ("oscar", "globe", "bafta")]
            out[f"{who}_award_{kind}_before"] = df[cols].fillna(0).sum(axis=1).astype(int)

    # 3. Mare studio
    out["major_studio"] = df["company_ids"].apply(
        lambda ids: int(any(int(i) in MAJOR_IDS for i in _as_list(ids))))

    # 4–6. Data lansării
    out["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")
    out["release_year"] = df["release_date"].dt.year
    out["release_month"] = df["release_date"].dt.month
    out["release_weekday"] = df["release_date"].dt.day_of_week   # 0 = luni

    # 7. Genuri (one-hot)
    genres = df["genres"].apply(_as_list)
    out["genres"] = genres.apply("|".join)
    for g in sorted({g for gs in genres for g in gs}):
        out[f"genre_{_slug(g)}"] = genres.apply(lambda gs, g=g: int(g in gs))

    # 8. Durată (0 = lipsă)
    rt = pd.to_numeric(df["runtime"], errors="coerce")
    out["runtime_min"] = rt.where(rt > 0).astype("Int64")

    # 9. Clasificare MPAA
    cert = df["certification_us"].astype("string").str.strip().str.upper()
    out["certification"] = cert.map(CERTIFICATION).astype("Int64")

    # 10. Cele mai frecvente keywords (one-hot)
    kws = df["keywords"].apply(lambda k: [w.strip().lower() for w in _as_list(k)])
    top = pd.Series([w for ws in kws for w in set(ws)]).value_counts()
    top = sorted(top.head(TOP_KEYWORDS).items(), key=lambda kv: (-kv[1], kv[0]))
    for w, _ in top:
        out[f"kw_{_slug(w)}"] = kws.apply(lambda ws, w=w: int(w in ws))

    # 11. Limba originală
    lang = df["original_language"].fillna("xx")
    top_lang = [l for l in lang.value_counts().index if l != "en"][:TOP_LANGUAGES]
    out["is_english"] = (lang == "en").astype(int)
    out["original_language"] = lang.where(lang.isin(["en", *top_lang]), "other")

    # 12. Țări de producție (grupate)
    groups = df["production_countries"].apply(lambda cs: {_country_group(c) for c in _as_list(cs)})
    for g in ("us", "europe", "asia", "other"):
        out[f"country_{g}"] = groups.apply(lambda s, g=g: int(g in s))

    # 13–14. Franciză și numărul în serie
    coll = pd.to_numeric(df["collection_id"], errors="coerce").fillna(0)
    out["is_franchise"] = (coll > 0).astype(int)
    seq = pd.to_numeric(df["sequel_number"], errors="coerce")
    out["sequel_number"] = seq.where(seq.notna() | (coll > 0), 0).astype("Int64")

    # 15. Ecranizare
    kw_based = kws.apply(lambda ws: any(w.startswith("based on") for w in ws))
    wp_based = df["based_on_wp"].fillna("").astype(str).str.strip() != ""
    wd_based = df["is_adaptation"].fillna(False).astype(bool)
    out["is_adaptation"] = (wd_based | kw_based | wp_based).astype(int)
    atype = df["adaptation_type"].where(wd_based)
    out["adaptation_type"] = atype.where(atype.notna(),
                                         np.where(out["is_adaptation"] == 1, "other", "none"))

    # 16–18. Istoricul regizorului și al actorilor (doar filme lansate înainte)
    out["director_prev_films"], out["director_prev_mean_roi"] = prev_history(df, "director_ids")
    out["cast_prev_films"], out["cast_prev_mean_roi"] = prev_history(df, "cast_top3_ids")
    out["director_prev_mean_roi"] = out["director_prev_mean_roi"].round(4)
    out["cast_prev_mean_roi"] = out["cast_prev_mean_roi"].round(4)

    # Sursele și valorile brute (nu intră în model)
    for col in ["budget_source", "budget_n_sources", "budget_n_agree", "budget_conflict",
                "gross_source", "gross_n_sources", "gross_n_agree", "gross_conflict",
                *BUDGET_SOURCES.values(), *GROSS_SOURCES.values()]:
        out[col] = df[col]
    for col in [*BUDGET_SOURCES.values(), *GROSS_SOURCES.values()]:
        out[col] = out[col].round().astype("Int64")
    out["budget_conflict"] = out["budget_conflict"].astype(int)
    out["gross_conflict"] = out["gross_conflict"].astype(int)

    # Doar pentru experimentul leakage vs. no-leakage
    out["vote_average"] = df["vote_average"]
    out["vote_count"] = df["vote_count"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------

def _md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                                       for v in r.values()) + " |")
    return "\n".join(lines)


def write_report(out: pd.DataFrame, funnel: list, threshold: float) -> None:
    n = len(out)
    pos = int(out["profitable"].sum())

    def dist(col):
        vc = out[col].value_counts().sort_index()
        return _md_table([{"valoare": k, "filme": int(v), "%": f"{100 * v / n:.1f}"} for k, v in vc.items()])

    feature_cols = ["budget", "director_award_noms_before", "major_studio", "release_year",
                    "release_month", "release_weekday", "genres", "runtime_min", "certification",
                    "kw_*", "original_language", "country_*", "is_franchise", "sequel_number",
                    "is_adaptation", "director_prev_films", "director_prev_mean_roi",
                    "cast_prev_mean_roi"]
    missing = []
    for c in ["runtime_min", "certification", "sequel_number",
              "director_prev_mean_roi", "cast_prev_mean_roi"]:
        missing.append({"coloană": f"`{c}`", "lipsă": int(out[c].isna().sum()),
                        "%": f"{100 * out[c].isna().mean():.1f}"})

    text = f"""# Raport de integrare (#27)

Generat de `python extraction/integrate.py --threshold {threshold}`.

**Rezultat:** {n:,} filme în `data/processed/movies_dataset.csv`, {len(out.columns)} coloane.

## Pâlnia de curățare

{_md_table(funnel)}

Ordinea de prioritate:
- buget: {" → ".join(BUDGET_SOURCES)};
- încasări mondiale: {" → ".join(GROSS_SOURCES)}.

Box Office Mojo și The Numbers sunt excluse (vezi README, „Surse excluse”).

## Ținta

`roi = worldwide_gross / budget`, `profitable = roi >= {threshold}`.

| Clasă | Filme | % |
|---|---|---|
| profitable = 1 | {pos:,} | {100 * pos / n:.1f} |
| profitable = 0 | {n - pos:,} | {100 * (n - pos) / n:.1f} |

ROI median: {out["roi"].median():.2f}.

## Conflicte între surse

Conflict = valoarea maximă depășește valoarea minimă cu peste {round((CONFLICT_RATIO - 1) * 100)}%.

| | Filme cu conflict | % |
|---|---|---|
| buget | {int(out["budget_conflict"].sum()):,} | {100 * out["budget_conflict"].mean():.1f} |
| încasări | {int(out["gross_conflict"].sum()):,} | {100 * out["gross_conflict"].mean():.1f} |

### Câte surse confirmă bugetul

{dist("budget_n_sources")}

### Câte surse confirmă încasările

{dist("gross_n_sources")}

DBpedia și Wikipedia citesc același infobox, deci nu sunt surse independente. Pentru robustețe,
modelul se poate evalua separat doar pe filmele cu `gross_n_sources >= 2`
({int((out["gross_n_sources"] >= 2).sum()):,} filme).

### Sursa valorii finale

{_md_table([{"sursă": k, "buget": int((out["budget_source"] == k).sum()),
             "încasări": int((out["gross_source"] == k).sum())} for k in BUDGET_SOURCES])}

## Valori lipsă în caracteristici

{_md_table(missing)}

`director_prev_mean_roi` și `cast_prev_mean_roi` lipsesc când persoana nu are niciun film
anterior în dataset (debut sau filme fără buget/încasări cunoscute).

## Anti-leakage

- Premiile și istoricul regizorului/actorilor folosesc doar filme lansate **strict înainte**.
- `vote_average` și `vote_count` sunt păstrate doar pentru experimentul leakage vs. no-leakage.
- Istoricul ROI e calculat pe tot datasetul: la o împărțire train/test aleatoare, un film din
  train poate folosi ROI-ul unui film anterior din test. Împărțirea temporală (după an) evită asta.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=2.5, help="prag ROI pentru `profitable`")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # consola Windows (cp1252)

    manifest = json.loads((INTERIM / "movies_manifest.json").read_text(encoding="utf-8"))
    df = load_sources(manifest)
    funnel: list = []
    df = clean(df, funnel)
    out = features(df, args.threshold)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")
    write_report(out, funnel, args.threshold)

    print(f"Scris {OUT_CSV.relative_to(PROJECT_ROOT)}: {len(out):,} filme, {len(out.columns)} coloane")
    print(f"Scris {REPORT.relative_to(PROJECT_ROOT)}")
    for f in funnel:
        print(f"  {f['pas']:15s} -{f['eliminate']:>6,}  -> {f['raman']:>6,}")
    print(f"  profitable: {out['profitable'].mean():.1%}")


if __name__ == "__main__":
    main()
