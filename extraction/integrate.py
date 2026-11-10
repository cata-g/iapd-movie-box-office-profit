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
    python extraction/integrate.py --roi-rule vechi --budget-choice prioritate   # regulile vechi
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

# Ordinea de prioritate (CLAUDE.md); coloanele țin valoarea în USD a fiecărei surse
BUDGET_SOURCES = {"tmdb": "budget_tmdb", "dbpedia": "budget_dbp",
                  "wikidata": "budget_wd", "wikipedia": "budget_wp"}
GROSS_SOURCES = {"wikipedia": "gross_wp", "dbpedia": "worldwide_gross_dbp",
                 "wikidata": "gross_wd", "tmdb": "revenue_tmdb"}
# Suma în moneda ei, per sursă: budget_<src>_amount / budget_<src>_currency (la fel pentru gross)
SRC_SHORT = {"tmdb": "tmdb", "dbpedia": "dbp", "wikidata": "wd", "wikipedia": "wp"}
DBP_SAME_AMOUNT = 0.005        # DBpedia = prima sumă din infobox (±0,5%) -> aceeași monedă ca infobox-ul

# Grupuri de surse independente: DBpedia copiază infobox-ul Wikipedia, deci cele două
# contează o singură dată când numărăm sursele care confirmă o valoare
SOURCE_GROUP = {"tmdb": "tmdb", "wikidata": "wikidata", "dbpedia": "wiki", "wikipedia": "wiki"}

CONFLICT_RATIO = 1.25          # max/min > 1.25 → conflict (> 25%)
ROI_EXTREME = (0.001, 100)     # în afara intervalului → păstrat doar dacă e confirmat

ROI_RULES = ("vechi", "surse-independente", "roi-recalculat")
BUDGET_CHOICES = ("majoritate", "prioritate")
ROI_RULE_TEXT = {
    "vechi": "ROI sub 0,001 sau peste 100 pe care nu-l confirmă 2 surse (probabil o greșeală)",
    "surse-independente": "ROI sub 0,001 sau peste 100 pe care nu-l confirmă 2 surse independente "
                          "(DBpedia și Wikipedia contează o dată; probabil o greșeală)",
    "roi-recalculat": "ROI sub 0,001 sau peste 100 care nu rămâne extrem când bugetul, apoi încasările "
                      "sunt înlocuite cu valoarea unei surse independente (probabil o greșeală)",
}
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

# Etichete de ton TMDB: descriu reacția publicului după vizionare → leakage, nu intră în top keywords
TONE_KEYWORDS = set("""
absurd admiring adoring aggressive ambiguous ambivalent amused angry antagonistic anxious
apathetic appreciative approving arrogant assertive audacious awestruck baffled bewildered
bitter bold callous candid cautionary celebratory cheerful comforting compassionate complex
condescending critical cruel curious defiant depressing desperate disdainful disgusted dramatic
earnest embarrassed empathetic enchanted energetic enthusiastic excited exhilarated exuberant
familiar farcical frightened frustrated grim harsh hilarious hopeful horrified humorous
informative inquisitive inspirational intense introspective joyful joyous lighthearted loving
melancholy melodramatic mischievous mocking nostalgic optimistic outrageous passionate
pessimistic playful powerful provocative reflective regretful relaxed respectful romantic sad
sarcastic satirical sentimental serious shocking sincere skeptical somber suspenseful
sympathetic tense thoughtful tragic vibrant whimsical witty zealous
""".split()) | {"mean spirited"}
TOP_LANGUAGES = 5


# ---------------------------------------------------------------------------
# Încărcare
# ---------------------------------------------------------------------------

def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["imdb_id"].notna() & (df["imdb_id"] != "")]
    return df.drop_duplicates("imdb_id").set_index("imdb_id")


def _currency_ids(wd: pd.DataFrame, wp: pd.DataFrame, kind: str) -> set:
    """Filmele cu o sumă (`kind` = budget / gross) într-o monedă cunoscută în Wikidata sau Wikipedia."""
    return {i for src, p in ((wd, "wd"), (wp, "wp"))
            for i in src.index[src[f"{kind}_{p}_currency"].notna() & (src[f"{kind}_{p}_amount"] > 0)]}


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

    # Manifestul (#5) are doar filmele cu buget în USD; se adaugă cele cu buget doar în altă monedă
    ids = sorted({k for k in manifest if k} | _currency_ids(wd, wp, "budget"))
    df = pd.DataFrame(index=pd.Index(ids, name="imdb_id"))
    df["in_manifest"] = df.index.isin(list(manifest))
    df = df.join(tmdb[["tmdb_id", "title", "release_date", "budget_tmdb", "revenue_tmdb",
                       "runtime", "genres", "original_language", "production_countries",
                       "company_ids", "collection_id", "certification_us", "director_ids",
                       "cast_top3_ids", "keywords", "vote_average", "vote_count",
                       "sequel_number"]])
    df = df.join(dbp[["budget_dbp", "worldwide_gross_dbp"]])
    df = df.join(wd[["budget_wd", "gross_wd", "budget_wd_amount", "budget_wd_currency", "gross_wd_amount",
                     "gross_wd_currency", "is_adaptation", "adaptation_type"]])
    df = df.join(wp[["budget_wp", "gross_wp", "budget_wp_amount", "budget_wp_currency", "budget_wp_note",
                     "budget_wp_range", "gross_wp_amount", "gross_wp_currency", "gross_wp_note",
                     "based_on_wp", "distributor_wp"]])
    df = df.join(aw)
    return native_amounts(df)


def native_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Coloanele budget_<src>_amount / _currency și gross_<src>_amount / _currency.
    TMDB e mereu în USD. DBpedia e marcat USD, dar uneori a citit „A$2,417,000” sau
    „MXN$142.3 million” ca dolari americani: când suma DBpedia e chiar prima sumă din
    infobox-ul Wikipedia, iar aceea e în altă monedă (sau într-un simbol ambiguu), DBpedia
    primește moneda din infobox."""
    df["dbp_currency_fixed"] = False
    for kind, dbp_col, tmdb_col in (("budget", "budget_dbp", "budget_tmdb"),
                                    ("gross", "worldwide_gross_dbp", "revenue_tmdb")):
        tmdb_amount = pd.to_numeric(df[tmdb_col], errors="coerce")
        df[f"{kind}_tmdb_amount"] = tmdb_amount.where(tmdb_amount > 0)
        df[f"{kind}_tmdb_currency"] = np.where(df[f"{kind}_tmdb_amount"].notna(), "USD", None)

        dbp = pd.to_numeric(df[dbp_col], errors="coerce")
        wp_amount, wp_cur = df[f"{kind}_wp_amount"], df[f"{kind}_wp_currency"]
        misread = (dbp.notna() & ((dbp - wp_amount).abs() <= DBP_SAME_AMOUNT * wp_amount)
                   & (wp_cur != "USD") & (wp_cur.notna() | (df[f"{kind}_wp_note"] == "ambiguous")))
        df[f"{kind}_dbp_amount"] = dbp
        df[f"{kind}_dbp_currency"] = np.where(misread, wp_cur, np.where(dbp.notna(), "USD", None))
        df.loc[misread & wp_cur.isna(), f"{kind}_dbp_amount"] = np.nan     # simbol ambiguu: ignorat
        df[f"dbp_{kind}_currency_fixed"] = misread
    return df


def usd_values(df: pd.DataFrame) -> pd.DataFrame:
    """Valoarea în USD a fiecărei surse (coloanele din BUDGET_SOURCES / GROSS_SOURCES): suma
    însăși dacă e în USD; pentru Wikipedia, și suma în USD scrisă în paranteză. Celelalte
    monede rămân fără valoare în USD."""
    for kind, sources in (("budget", BUDGET_SOURCES), ("gross", GROSS_SOURCES)):
        for name, col in sources.items():
            p = SRC_SHORT[name]
            amount, cur = df[f"{kind}_{p}_amount"], df[f"{kind}_{p}_currency"]
            usd = amount.where(cur == "USD")
            if name == "wikipedia":
                usd = usd.fillna(pd.to_numeric(df[col], errors="coerce"))
            df[col] = usd
    return df


def native_pair(df: pd.DataFrame, budget_sources: dict, gross_sources: dict) -> pd.DataFrame:
    """Pentru ROI fără curs de schimb: primul buget (în ordinea de prioritate) într-o monedă
    pe care o are și o sursă de încasări, și primele încasări în aceeași monedă."""
    rows = []
    for i in df.index:
        budgets = [(df.at[i, f"budget_{SRC_SHORT[n]}_amount"], df.at[i, f"budget_{SRC_SHORT[n]}_currency"])
                   for n in budget_sources]
        grosses = [(df.at[i, f"gross_{SRC_SHORT[n]}_amount"], df.at[i, f"gross_{SRC_SHORT[n]}_currency"])
                   for n in gross_sources]
        budgets = [(a, c) for a, c in budgets if isinstance(c, str) and a > 0]
        grosses = [(a, c) for a, c in grosses if isinstance(c, str) and a > 0]
        pair = next(((b, g, c) for b, c in budgets for g, gc in grosses if gc == c), (np.nan, np.nan, None))
        rows.append(pair)
    return pd.DataFrame(rows, index=df.index, columns=["budget_native", "gross_native", "roi_currency"])


# ---------------------------------------------------------------------------
# Buget și încasări finale
# ---------------------------------------------------------------------------

def _plausible(s: pd.Series, lo: float, hi: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(lo, hi))


def _close(a, b) -> bool:
    """a și b sunt la cel mult ±25% una de alta (CONFLICT_RATIO)."""
    return 1 / CONFLICT_RATIO <= a / b <= CONFLICT_RATIO


def majority_choice(vals: pd.DataFrame) -> pd.Series:
    """Sursa aleasă prin majoritate, pentru fiecare film. `vals` are coloanele în ordinea
    de prioritate. Din fiecare grup SOURCE_GROUP intră un reprezentant: sursa grupului cu
    cea mai mare prioritate. Sprijinul unui reprezentant = câți reprezentanți (inclusiv el)
    sunt la ±25% de el. Rămâne sursa prioritară dacă are sprijinul maxim; altfel, cea mai
    prioritară sursă cu sprijin maxim."""
    names = list(vals.columns)
    chosen = []
    for row in vals.to_numpy(dtype=float):
        reps, seen = [], set()
        for name, v in zip(names, row):
            if not np.isnan(v) and SOURCE_GROUP[name] not in seen:
                seen.add(SOURCE_GROUP[name])
                reps.append((name, v))
        if not reps:
            chosen.append(None)
            continue
        support = [sum(_close(v, w) for _, w in reps) for _, v in reps]
        chosen.append(reps[support.index(max(support))][0])   # primul cu maxim = cel mai prioritar
    return pd.Series(chosen, index=vals.index, dtype=object)


def choose_value(df: pd.DataFrame, sources: dict, prefix: str, majority: bool = False) -> pd.DataFrame:
    """Valoarea finală `prefix` și sursa ei. `sources` = {sursă: coloană} în ordinea de prioritate;
    cu majority=True, sursa vine din majority_choice (doar pentru buget)."""
    vals = pd.DataFrame({name: df[col] for name, col in sources.items()})
    if majority:
        first = majority_choice(vals)
    else:
        first = vals.notna().idxmax(axis=1).where(vals.notna().any(axis=1))
    df[prefix] = [vals.at[i, s] if isinstance(s, str) else np.nan for i, s in zip(vals.index, first)]
    df[f"{prefix}_source"] = first
    df[f"{prefix}_n_sources"] = vals.notna().sum(axis=1)
    df[f"{prefix}_conflict"] = (vals.max(axis=1) / vals.min(axis=1)) > CONFLICT_RATIO
    # câte surse sunt la cel mult 25% de valoarea aleasă (inclusiv cea aleasă)
    ratio = vals.div(df[prefix], axis=0)
    agree = ratio.apply(lambda r: r.between(1 / CONFLICT_RATIO, CONFLICT_RATIO))
    df[f"_{prefix}_n_agree_raw"] = agree.sum(axis=1)                 # fiecare sursă separat (regula „vechi”)
    groups = pd.DataFrame({g: agree[[n for n in sources if SOURCE_GROUP[n] == g]].any(axis=1)
                           for g in dict.fromkeys(SOURCE_GROUP[n] for n in sources)})
    df[f"{prefix}_n_agree"] = groups.sum(axis=1)                      # DBpedia + Wikipedia = o sursă
    return df


def roi_confirmed(df: pd.DataFrame, rule: str, budget_sources: dict = None,
                  gross_sources: dict = None) -> pd.Series:
    """True dacă un ROI extrem (în afara ROI_EXTREME) e confirmat și filmul rămâne.
      vechi               2 surse (fiecare separat) confirmă și bugetul, și încasările
      surse-independente  la fel, dar DBpedia și Wikipedia contează o singură dată
      roi-recalculat      ROI-ul rămâne extrem (pe aceeași parte) și cu un buget dintr-un alt
                          grup de surse decât budget_source, și cu încasări dintr-un alt grup
                          decât gross_source; fiecare valoare se confirmă separat, cu toleranța
                          de ±25% aplicată fiecăreia (prag 100 / 1,25² sus, 0,001 · 1,25² jos)"""
    if rule == "vechi":
        return (df["_budget_n_agree_raw"] >= 2) & (df["_gross_n_agree_raw"] >= 2)
    if rule == "surse-independente":
        return (df["budget_n_agree"] >= 2) & (df["gross_n_agree"] >= 2)
    if rule != "roi-recalculat":
        raise ValueError(f"regulă necunoscută: {rule}")

    budget_sources = budget_sources or BUDGET_SOURCES
    gross_sources = gross_sources or GROSS_SOURCES
    lo, hi = ROI_EXTREME[0] * CONFLICT_RATIO ** 2, ROI_EXTREME[1] / CONFLICT_RATIO ** 2
    high = df["roi"] > ROI_EXTREME[1]

    def extreme(roi: pd.Series) -> pd.Series:
        return roi.gt(hi).where(high, roi.lt(lo))      # NaN → False

    def alternatives(sources: dict, chosen: pd.Series) -> pd.DataFrame:
        group = chosen.map(SOURCE_GROUP)
        return pd.DataFrame({n: df[c].where(group != SOURCE_GROUP[n]) for n, c in sources.items()})

    b_alt = alternatives(budget_sources, df["budget_source"])
    g_alt = alternatives(gross_sources, df["gross_source"])
    by_budget = b_alt.rdiv(df["worldwide_gross"], axis=0).apply(extreme).any(axis=1)
    by_gross = g_alt.div(df["budget"], axis=0).apply(extreme).any(axis=1)
    return by_budget & by_gross


# ---------------------------------------------------------------------------
# Curățare
# ---------------------------------------------------------------------------

def count_gross_only(manifest) -> int:
    """Filmele cu încasări (în USD plauzibile, GROSS_RANGE, sau în altă monedă) în cel puțin o
    sursă care nu sunt în `manifest` (filmele cu buget în vreo sursă) și au anul în
    YEAR_MIN..YEAR_MAX. Anul: TMDB release_date, apoi anul din DBpedia, apoi cel din Wikidata."""
    in_manifest = {k for k in manifest if k}
    tmdb = pd.concat([pd.read_parquet(INTERIM / f, columns=["imdb_id", "release_date", "revenue_tmdb"])
                      for f in ("tmdb_movies.parquet", "tmdb_extra.parquet")], ignore_index=True)
    dbp = pd.read_parquet(INTERIM / "dbpedia.parquet", columns=["imdb_id", "year", "worldwide_gross_dbp"])
    wd = pd.read_parquet(INTERIM / "wikidata.parquet",
                         columns=["imdb_id", "year", "gross_wd", "gross_wd_amount", "gross_wd_currency"])
    wp = pd.read_parquet(INTERIM / "wikipedia.parquet",
                         columns=["imdb_id", "gross_wp", "gross_wp_amount", "gross_wp_currency"])

    with_gross = set()
    for src, col in [(tmdb, "revenue_tmdb"), (dbp, "worldwide_gross_dbp"), (wd, "gross_wd"), (wp, "gross_wp")]:
        with_gross |= set(src.loc[_plausible(src[col], *GROSS_RANGE).notna(), "imdb_id"].dropna())
    for src, p in [(wd, "wd"), (wp, "wp")]:
        other = src[f"gross_{p}_currency"].notna() & (src[f"gross_{p}_currency"] != "USD")
        with_gross |= set(src.loc[other & (src[f"gross_{p}_amount"] > 0), "imdb_id"].dropna())
    ids = pd.Index(sorted(with_gross - in_manifest - {""}))

    year = pd.Series(np.nan, index=ids)
    tmdb_year = pd.to_datetime(tmdb["release_date"], errors="coerce").dt.year
    for src_year in [tmdb_year.groupby(tmdb["imdb_id"]).min(),
                     pd.to_numeric(dbp["year"], errors="coerce").groupby(dbp["imdb_id"]).min(),
                     pd.to_numeric(wd["year"], errors="coerce").groupby(wd["imdb_id"]).min()]:
        src_year = src_year.astype(float)
        year = year.fillna(src_year.reindex(ids))
    return int(year.between(YEAR_MIN, YEAR_MAX).sum())


def clean(df: pd.DataFrame, funnel: list, gross_only: int = 0, roi_rule: str = "roi-recalculat",
          budget_choice: str = "majoritate", budget_sources: dict = None,
          gross_sources: dict = None) -> pd.DataFrame:
    budget_sources = budget_sources or BUDGET_SOURCES
    gross_sources = gross_sources or GROSS_SOURCES

    def step(name: str, mask: pd.Series, rule: str) -> pd.DataFrame:
        nonlocal df
        removed = int((~mask).sum())
        df = df[mask]
        funnel.append({"Pas": name, "Ce s-a scos": rule, "Scoase": removed, "Rămân": len(df)})
        return df

    funnel.append({"Pas": "start",
                   "Ce s-a scos": f"filme {YEAR_MIN}–{YEAR_MAX} cu buget sau încasări în cel puțin o sursă",
                   "Scoase": 0, "Rămân": len(df) + gross_only})
    funnel.append({"Pas": "fără buget",
                   "Ce s-a scos": "au încasări, dar nicio sursă nu are buget (nu intră în manifest; "
                                  "fără buget nu există ROI)",
                   "Scoase": gross_only, "Rămân": len(df)})

    df = usd_values(df)
    def other_currency(kind: str) -> pd.Series:
        return pd.concat([df[f"{kind}_{p}_currency"].notna() & (df[f"{kind}_{p}_currency"] != "USD")
                          & (df[f"{kind}_{p}_amount"] > 0) for p in SRC_SHORT.values()], axis=1).any(axis=1)
    budget_other, gross_other = other_currency("budget"), other_currency("gross")

    # Valori absurde per sursă → lipsă (buget < 10.000 USD = eroare)
    n_budget_err = {}
    for col in budget_sources.values():
        raw = pd.to_numeric(df[col], errors="coerce")
        df[col] = _plausible(raw, *BUDGET_RANGE)
        n_budget_err[col] = int((raw.notna() & (raw > 0) & df[col].isna()).sum())
    for col in gross_sources.values():
        df[col] = _plausible(df[col], *GROSS_RANGE)

    df = choose_value(df, budget_sources, "budget", majority=budget_choice == "majoritate")
    df = choose_value(df, gross_sources, "worldwide_gross")
    df = df.rename(columns={f"{p}worldwide_gross_{k}": f"{p}gross_{k}" for p in ("", "_")
                            for k in ("source", "n_sources", "conflict", "n_agree", "n_agree_raw")})

    # Fără curs de schimb, un film cu bugetul și încasările doar în altă monedă are ROI doar
    # dacă amândouă sunt în aceeași monedă
    no_usd = df["budget"].isna() | df["worldwide_gross"].isna()
    native = native_pair(df[no_usd], budget_sources, gross_sources).reindex(df.index)
    df = df.join(native)
    step("buget", df["budget"].notna() | budget_other,
         "niciun buget în USD între 10.000 $ și 500 mil. $ și niciun buget în altă monedă (bugete în USD în afara intervalului, "
         "ignorate: " + ", ".join(f"{k.split('_')[1]} {_n(v)}" for k, v in n_budget_err.items() if v) + ")")
    step("încasări", df["worldwide_gross"].notna() | gross_other.reindex(df.index),
         "nicio sursă nu are încasări mondiale (în nicio monedă)")
    pair = (df["budget"].notna() & df["worldwide_gross"].notna()) | df["roi_currency"].notna()
    step("monede diferite", pair,
         "bugetul și încasările sunt în monede diferite (fără curs de schimb nu se pot compara)")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    step("dată lansare", df["release_date"].notna(),
         "fără dată de lansare în TMDB")
    year = df["release_date"].dt.year
    step("interval ani", year.between(YEAR_MIN, YEAR_MAX), f"lansate înainte de {YEAR_MIN} sau după {YEAR_MAX}")

    is_tv = df["genres"].apply(lambda g: g is not None and "TV Movie" in list(g))
    step("film TV", ~is_tv, "filme TV (n-au rulat în cinema)")
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    step("scurtmetraj", ~runtime.between(1, MIN_RUNTIME - 1), f"scurtmetraje (sub {MIN_RUNTIME} min)")

    usd_pair = df["budget"].notna() & df["worldwide_gross"].notna()
    df["roi"] = np.where(usd_pair, df["worldwide_gross"] / df["budget"], df["gross_native"] / df["budget_native"])
    df["roi_currency"] = df["roi_currency"].where(~usd_pair, "USD")
    extreme = ~df["roi"].between(*ROI_EXTREME)
    confirmed = roi_confirmed(df, roi_rule, budget_sources, gross_sources)
    step("ROI extrem", ~extreme | confirmed, ROI_RULE_TEXT[roi_rule])
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


def prev_history(df: pd.DataFrame, col: str, prefix: str, threshold: float) -> pd.DataFrame:
    """Pentru fiecare film și fiecare persoană din `col`: filmele din dataset ale
    persoanei lansate strict înainte. Statisticile se calculează per persoană și
    apoi se mediază peste persoanele cu istoric:
      {prefix}_prev_films            nr. maxim de filme anterioare
      {prefix}_prev_mean_roi         media ROI (cerută în issue; sensibilă la hituri extreme)
      {prefix}_prev_mean_log_roi     media lui ln(ROI) — robustă la valori extreme
      {prefix}_prev_profitable_rate  ce fracție din filmele anterioare a fost profitabilă"""
    films_by_person = defaultdict(list)
    for people, date, roi in zip(df[col], df["release_date"], df["roi"]):
        for p in _as_list(people):
            films_by_person[p].append((date, roi))

    rows = []
    for people, date in zip(df[col], df["release_date"]):
        counts, means, logs, rates = [], [], [], []
        for p in _as_list(people):
            prior = np.array([r for d, r in films_by_person[p] if d < date])
            counts.append(len(prior))
            if len(prior):
                means.append(prior.mean())
                logs.append(np.log(prior).mean())
                rates.append((prior >= threshold).mean())
        avg = lambda xs: round(float(np.mean(xs)), 4) if xs else np.nan
        rows.append({f"{prefix}_prev_films": max(counts) if counts else 0,
                     f"{prefix}_prev_mean_roi": avg(means),
                     f"{prefix}_prev_mean_log_roi": avg(logs),
                     f"{prefix}_prev_profitable_rate": avg(rates)})
    return pd.DataFrame(rows, index=df.index)


def features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["tmdb_id"] = df["tmdb_id"].astype("Int64")
    out["title"] = df["title"]

    # Ținta
    out["budget"] = df["budget"].round().astype("Int64")
    out["worldwide_gross"] = df["worldwide_gross"].round().astype("Int64")
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

    # 10. Cele mai frecvente keywords (one-hot), fără etichetele de ton (reacția publicului)
    kws = df["keywords"].apply(lambda k: [w.strip().lower() for w in _as_list(k)])
    top = pd.Series([w for ws in kws for w in set(ws) if w not in TONE_KEYWORDS]).value_counts()
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
    out = out.join(prev_history(df, "director_ids", "director", threshold))
    out = out.join(prev_history(df, "cast_top3_ids", "cast", threshold))

    # Sursele și valorile brute (nu intră în model)
    for col in ["budget_source", "budget_n_sources", "budget_n_agree", "budget_conflict",
                "gross_source", "gross_n_sources", "gross_n_agree", "gross_conflict",
                *BUDGET_SOURCES.values(), *GROSS_SOURCES.values()]:
        out[col] = df[col]
    for col in [*BUDGET_SOURCES.values(), *GROSS_SOURCES.values()]:
        out[col] = out[col].round().astype("Int64")
    out["budget_conflict"] = out["budget_conflict"].astype(int)
    out["gross_conflict"] = out["gross_conflict"].astype(int)
    # Suma și moneda fiecărei surse, așa cum sunt scrise; moneda în care s-a calculat ROI-ul
    for kind, sources in (("budget", BUDGET_SOURCES), ("gross", GROSS_SOURCES)):
        for name in sources:
            p = SRC_SHORT[name]
            out[f"{kind}_{p}_amount"] = df[f"{kind}_{p}_amount"].round(2)
            out[f"{kind}_{p}_currency"] = df[f"{kind}_{p}_currency"]
    out["roi_currency"] = df["roi_currency"]

    # Doar pentru experimentul leakage vs. no-leakage
    out["vote_average"] = df["vote_average"]
    out["vote_count"] = df["vote_count"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------

def _n(v) -> str:
    """Număr în format românesc: 15.972"""
    return f"{v:,}".replace(",", ".")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",") + "%"


def _md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_n(v) if isinstance(v, (int, np.integer)) else str(v)
                                       for v in r.values()) + " |")
    return "\n".join(lines)


def _same_share(out: pd.DataFrame, a: str, b: str) -> str:
    """Ce parte din filmele cu valoare în ambele coloane au exact aceeași valoare."""
    both = out[[a, b]].dropna()
    return _pct((both[a] == both[b]).mean()) if len(both) else "–"


def currency_section(out: pd.DataFrame, stats: dict) -> str:
    n = len(out)
    roi_cur = out["roi_currency"].value_counts()
    top = _md_table([{"Monedă": k, "Filme": int(v), "%": _pct(v / n)} for k, v in roi_cur.head(10).items()]
                    + [{"Monedă": "altele", "Filme": int(roi_cur.iloc[10:].sum()),
                        "%": _pct(roi_cur.iloc[10:].sum() / n)}])
    non_us = out.loc[out["roi_currency"] != "USD", ["country_us"]].mean().iloc[0]
    return f"""## Monede

Bugetul și încasările se iau în moneda în care sunt scrise (`budget_<sursă>_amount`, `budget_<sursă>_currency`, la fel pentru încasări). Când bugetul și încasările unui film sunt în aceeași monedă, ROI-ul (un raport) nu are nevoie de curs de schimb: `roi_currency` arată moneda în care s-a calculat.

{top}

- {_n(int((out["roi_currency"] != "USD").sum()))} filme au ROI-ul calculat în altă monedă decât USD; doar {_pct(non_us)} dintre ele sunt (co)producții americane.
- DBpedia marchează orice sumă ca USD, deși uneori a citit „A$2,417,000” sau „MXN$142.3 million”. Când suma DBpedia e chiar prima sumă din infobox-ul Wikipedia, iar aceea e în altă monedă, DBpedia primește moneda din infobox: {_n(stats["dbp_budget_fixed"])} bugete și {_n(stats["dbp_gross_fixed"])} încasări (dintre ele, în dataset: {_n(int(out["budget_dbp_currency"].ne("USD").sum() - out["budget_dbp_currency"].isna().sum()))} bugete).
- Simboluri ambigue („¥” = JPY sau CNY, „kr” = SEK / NOK / DKK, „Rs”, „$”) se lămuresc după țara de producție din TMDB. Dacă tot rămân ambigue, suma e ignorată: {_n(stats["wp_ambiguous_budget"])} bugete și {_n(stats["wp_ambiguous_gross"])} încasări din Wikipedia.
- Un „$” simplu înseamnă USD, cu o excepție: un film fără SUA printre țările de producție, dar dintr-o țară cu dolar propriu (Australia, Canada, Hong Kong, Noua Zeelandă, Mexic ...), primește moneda acelei țări. Limită: Wikidata are aceeași sumă marcată USD la multe dintre aceste filme, dar și acolo valorile sunt adesea importate automat din infobox, deci nu e o confirmare independentă.
"""


def write_report(out: pd.DataFrame, funnel: list, args, stats: dict = None) -> None:
    threshold = args.threshold
    stats = stats or {}
    n = len(out)
    pos = int(out["profitable"].sum())
    t = str(threshold).replace(".", ",")

    def sources(col):
        vc = out[col].value_counts().sort_index()
        return _md_table([{"Surse": k, "Filme": int(v), "%": _pct(v / n)} for k, v in vc.items()])

    missing = _md_table([{"Coloană": f"`{c}`", "Lipsesc": int(out[c].isna().sum()),
                          "%": _pct(out[c].isna().mean())}
                         for c in ["runtime_min", "certification", "sequel_number",
                                   "director_prev_mean_roi", "cast_prev_mean_roi"]])
    corr = _md_table([{"Coloană": f"`{c}`", "Corelație cu `profitable`":
                       f"{out[c].corr(out['profitable']):.3f}".replace(".", ",")}
                      for c in ["director_prev_mean_roi", "director_prev_mean_log_roi",
                                "director_prev_profitable_rate", "cast_prev_mean_roi",
                                "cast_prev_mean_log_roi", "cast_prev_profitable_rate"]])
    budget_rule = "" if args.budget_choice == "prioritate" else f""", dar prin **majoritate**: din fiecare grup independent (tmdb, wikidata, wiki = dbpedia/wikipedia) intră o singură valoare, a sursei celei mai prioritare din grup. Câștigă valoarea cu care sunt de acord (la ±25%) cele mai multe grupuri; la egalitate rămâne ordinea de mai sus. Așa, o greșeală izolată a TMDB nu mai câștigă doar pentru că TMDB e primul (Dragon Ball Super: Broly: TMDB 1 mil. $, Wikidata și Wikipedia 8,5 mil. $). Limită: Wikidata are exact aceeași valoare ca Wikipedia la {_same_share(out, "budget_wd", "budget_wp")} din bugete, deci și „majoritatea” poate fi uneori aceeași informație copiată."""
    final_src = _md_table([{"Sursă": k, "Buget": int((out["budget_source"] == k).sum()),
                            "Încasări": int((out["gross_source"] == k).sum())} for k in BUDGET_SOURCES])

    text = f"""# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold {threshold} --roi-rule {args.roi_rule} --budget-choice {args.budget_choice}`.

**Rezultat:** {_n(n)} filme în `data/processed/movies_dataset.csv`.

## Ce filme am scos

{_md_table(funnel)}

## Ținta

Un film e **profitabil** dacă încasările sunt de cel puțin {t} ori bugetul (`roi = încasări / buget`).

| | Filme | % |
|---|---|---|
| profitabile | {_n(pos)} | {_pct(pos / n)} |
| neprofitabile | {_n(n - pos)} | {_pct((n - pos) / n)} |

Clasele sunt dezechilibrate: un model care spune mereu „neprofitabil” are dreptate în {_pct((n - pos) / n)} din cazuri.

## De unde vin bugetul și încasările

Când mai multe surse au o valoare, o luăm pe cea din sursa mai sigură:
- buget: {" → ".join(BUDGET_SOURCES)}{budget_rule}
- încasări: {" → ".join(GROSS_SOURCES)} (doar ordinea; TMDB raportează sistematic încasări mai mici, de aceea e ultima)

{final_src}

Box Office Mojo și The Numbers nu sunt folosite: interzic colectarea automată (vezi README).

### Cât de sigure sunt valorile

Câte surse au o valoare pentru fiecare film:

**Buget**

{sources("budget_n_sources")}

**Încasări**

{sources("gross_n_sources")}

Sursele nu se pun mereu de acord: la {_n(int(out["budget_conflict"].sum()))} filme ({_pct(out["budget_conflict"].mean())}) bugetele diferă cu peste 25%, iar la {_n(int(out["gross_conflict"].sum()))} ({_pct(out["gross_conflict"].mean())}) încasările.

DBpedia copiază infobox-ul Wikipedia (valori identice la {_same_share(out, "budget_dbp", "budget_wp")} din bugete și {_same_share(out, "worldwide_gross_dbp", "gross_wp")} din încasări), deci nu e o sursă independentă. De aceea `budget_n_agree` și `gross_n_agree` (câte surse confirmă valoarea aleasă, la ±25%) le numără ca **o singură sursă**: tmdb, wikidata și wiki (dbpedia sau wikipedia) — maximum 3.

{currency_section(out, stats)}
## Date lipsă

{missing}

Istoricul regizorului și al actorilor lipsește când e primul lor film din dataset.

## Istoricul ROI: de ce nu ajunge media

Câteva filme au un ROI uriaș ({out.loc[out["roi"].idxmax(), "title"]}: {_n(round(out["roi"].max()))}), deși la jumătate dintre filme ROI-ul e sub {f"{out['roi'].median():.2f}".replace(".", ",")}. Un singur astfel de hit strică media pentru toată cariera unui regizor. De aceea, pe lângă media cerută în issue, am adăugat:

- `*_prev_profitable_rate`: ce parte din filmele anterioare au fost profitabile;
- `*_prev_mean_log_roi`: media pe logaritm, unde un hit contează mult mai puțin.

{corr}

Media simplă aproape nu are legătură cu ținta; variantele noi au.

## Fără informații de după lansare (anti-leakage)

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**.
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- `vote_average` și `vote_count` rămân în tabel doar pentru experimentul cu leakage.
- La antrenare, e mai sigur să împărțim filmele după an (vechi = antrenare, noi = test). La o împărțire aleatoare, istoricul unui film vechi ar putea folosi ROI-ul unui film din test.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=2.5, help="prag ROI pentru `profitable`")
    ap.add_argument("--roi-rule", choices=ROI_RULES, default="roi-recalculat",
                    help="când e confirmat un ROI extrem (sub 0,001 sau peste 100)")
    ap.add_argument("--budget-choice", choices=BUDGET_CHOICES, default="majoritate",
                    help="cum se alege bugetul când sursele nu se pun de acord")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # consola Windows (cp1252)

    manifest = json.loads((INTERIM / "movies_manifest.json").read_text(encoding="utf-8"))
    df = load_sources(manifest)
    funnel: list = []
    stats = {"dbp_budget_fixed": int(df["dbp_budget_currency_fixed"].sum()),
             "dbp_gross_fixed": int(df["dbp_gross_currency_fixed"].sum()),
             "wp_ambiguous_budget": int((df["budget_wp_note"] == "ambiguous").sum()),
             "wp_ambiguous_gross": int((df["gross_wp_note"] == "ambiguous").sum())}
    df = clean(df, funnel, gross_only=count_gross_only(df.index), roi_rule=args.roi_rule,
               budget_choice=args.budget_choice)
    out = features(df, args.threshold)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8", lineterminator="\n")
    write_report(out, funnel, args, stats)

    print(f"Scris {OUT_CSV.relative_to(PROJECT_ROOT)}: {len(out):,} filme, {len(out.columns)} coloane")
    print(f"Scris {REPORT.relative_to(PROJECT_ROOT)}")
    for f in funnel:
        print(f"  {f['Pas']:15s} -{f['Scoase']:>6,}  -> {f['Rămân']:>6,}")
    print(f"  profitable: {out['profitable'].mean():.1%}")


if __name__ == "__main__":
    main()
