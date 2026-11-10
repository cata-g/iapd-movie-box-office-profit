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
from extraction.columns import categories, feature_columns
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

# Filme de streaming cu o lansare în cinema doar simbolică: în afara populației (filme lansate
# în cinema). Se scot doar dacă AU încasări sub 5% din buget (Manchester by the Sea, distribuit
# de Amazon, a rulat cu adevărat în cinema și rămâne).
STREAMING_DISTRIBUTORS = re.compile(
    r"^(?:Netflix|Amazon Studios|Amazon Prime Video|Amazon Prime|Prime Video|Apple TV\+|Disney\+"
    r"(?: Hotstar)?|Hulu|HBO Max|Max|Paramount\+|Peacock)$", re.I)
STREAMING_DIGITAL_DAYS = 30    # lansare digitală (TMDB tip 4) la cel mult 30 de zile după cinema
STREAMING_DIGITAL_FROM = 2005  # înainte nu exista distribuție digitală: tipul 4 e o casetă/DVD marcată greșit
STREAMING_MAX_ROI = 0.05

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
# Aceleași grupuri, după numele distribuitorului principal din infobox (#45); diviziile de
# video (Home Entertainment / Home Video) nu distribuie în cinema și nu contează
MAJOR_DISTRIBUTORS = re.compile(
    r"^(?:walt disney|buena vista|touchstone|warner bros|new line|universal|focus features|paramount"
    r"(?! ?\+)|sony pictures|columbia|tri-?star|20th century|twentieth century|fox searchlight|"
    r"searchlight pictures|lions ?gate|summit entertainment)(?!.*home (?:entertainment|video))", re.I)

CERTIFICATION = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4}

EUROPE = set("AD AL AT BA BE BG BY CH CS CY CZ DD DE DK EE ES FI FR GB GR HR HU IE IS IT "
             "LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SK SM SU UA VA XC XG YU".split())
ASIA = set("AE AF BD BN BT CN HK ID IL IN IQ IR JO JP KG KH KP KR KW KZ LA LB LK MM MN MO "
           "MV MY NP OM PH PK PS QA SA SG SY TH TJ TL TM TR TW UZ VN YE".split())

TOP_KEYWORDS = 30
# Istoricul regizorului/actorilor folosește doar filmele lansate cu cel puțin 6 luni înainte:
# încasările unui film abia lansat nu sunt încă cunoscute (#49)
PREV_MIN_GAP = pd.Timedelta(days=182)

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
# Etichete care se află abia după lansare (#47): scenele de după generic se descoperă la
# vizionare, iar „controversă” / „film interzis” descriu reacția de după lansare
POST_RELEASE_KEYWORDS = {"duringcreditsstinger", "aftercreditsstinger", "post-credits scene",
                         "mid-credits scene", "banned film", "controversy", "cult film", "box office flop"}
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
    wd = _dedupe(pd.read_parquet(INTERIM / "wikidata.parquet")).rename(columns={"year": "year_wd"})
    wp = _dedupe(pd.read_parquet(INTERIM / "wikipedia.parquet"))
    aw = _dedupe(pd.read_parquet(INTERIM / "awards.parquet"))
    rd = pd.read_parquet(INTERIM / "tmdb_release_dates.parquet").drop_duplicates("tmdb_id")

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
    df = df.join(wd[["year_wd", "budget_wd", "gross_wd", "budget_wd_amount", "budget_wd_currency", "gross_wd_amount",
                     "gross_wd_currency", "is_adaptation", "adaptation_type"]])
    df = df.join(wp[["budget_wp", "gross_wp", "budget_wp_amount", "budget_wp_currency", "budget_wp_note",
                     "budget_wp_range", "gross_wp_amount", "gross_wp_currency", "gross_wp_note",
                     "gross_wp_domestic", "gross_wp_rental", "gross_wp_single_market", "gross_wp_worldwide",
                     "based_on_wp", "distributor_wp"]])
    df = df.join(aw)
    df = df.merge(rd[["tmdb_id", "first_theatrical", "first_digital"]], on="tmdb_id", how="left").set_index(df.index)
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


FX_PATH = INTERIM / "fx_rates.parquet"
FX_NOTE = {"pre1960": "fără curs înainte de 1960", "hyper": "hiperinflație (curs schimbat cu peste 50% "
           "față de anul anterior)", "norate": "monedă sau an fără curs World Bank"}


def load_fx() -> dict:
    """(monedă, an) -> (curs = unități pentru 1 USD, hiperinflație), din fx_rates.py."""
    fx = pd.read_parquet(FX_PATH)
    return {(c, int(y)): (r, bool(h)) for c, y, r, h in
            zip(fx["currency"], fx["year"], fx["rate"], fx["hyperinflation"])}


def usd_values(df: pd.DataFrame, fx: dict) -> pd.DataFrame:
    """Valoarea în USD a fiecărei surse (coloanele din BUDGET_SOURCES / GROSS_SOURCES):
      - suma însăși dacă e în USD;
      - pentru Wikipedia, suma în USD scrisă în infobox („£800,000 ($1.4 million)”) are prioritate;
      - altfel suma / cursul mediu World Bank din anul lansării (TMDB; altfel anul din Wikidata).
    Fără curs (înainte de 1960, monedă fără serie) sau în an de hiperinflație, valoarea lipsește;
    motivul rămâne în {kind}_{src}_fx_note. Cursul folosit: {kind}_{src}_fx_rate (1 = USD,
    gol = suma în USD scrisă în infobox)."""
    year = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype(float).fillna(
        pd.to_numeric(df["year_wd"], errors="coerce").astype(float))
    df["fx_year"] = year.astype("Int64")
    for kind, sources in (("budget", BUDGET_SOURCES), ("gross", GROSS_SOURCES)):
        for name, col in sources.items():
            p = SRC_SHORT[name]
            amount, cur = df[f"{kind}_{p}_amount"], df[f"{kind}_{p}_currency"]
            looked = [fx.get((c, int(y))) if isinstance(c, str) and c != "USD" and pd.notna(y) else None
                      for c, y in zip(cur, year)]
            rate = pd.Series([r[0] if r and not r[1] else np.nan for r in looked], index=df.index)
            rate = rate.where(cur != "USD", 1.0)
            note = pd.Series([None if not isinstance(c, str) or c == "USD" or (r and not r[1]) else
                              "hyper" if r else "pre1960" if pd.notna(y) and y < 1960 else "norate"
                              for c, y, r in zip(cur, year, looked)], index=df.index, dtype=object)
            usd = amount / rate
            if name == "wikipedia":
                written = pd.to_numeric(df[col], errors="coerce")       # USD scris în infobox
                rate = rate.where(written.isna())
                note = note.where(written.isna())
                usd = written.fillna(usd)
            df[col] = usd
            df[f"{kind}_{p}_fx_rate"] = rate.where(usd.notna())
            df[f"{kind}_{p}_fx_note"] = note.where(amount > 0)
    return df


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
          gross_sources: dict = None, fx: dict = None, stats: dict = None) -> pd.DataFrame:
    budget_sources = budget_sources or BUDGET_SOURCES
    gross_sources = gross_sources or GROSS_SOURCES
    fx = load_fx() if fx is None else fx
    stats = {} if stats is None else stats

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

    df = usd_values(df, fx)

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

    # Filmele pierdute doar pentru că suma lor nu s-a putut converti (pentru raport)
    for kind, final in (("budget", "budget"), ("gross", "worldwide_gross")):
        notes = df[[f"{kind}_{p}_fx_note" for p in SRC_SHORT.values()]]
        lost = df[final].isna() & notes.notna().any(axis=1)
        first = notes[lost].bfill(axis=1).iloc[:, 0]
        stats[f"fx_lost_{kind}"] = {k: int((first == k).sum()) for k in FX_NOTE}

    step("buget", df["budget"].notna(),
         "niciun buget între 10.000 $ și 500 mil. $ după conversia în USD (bugete în afara intervalului, "
         "ignorate: " + ", ".join(f"{k.split('_')[1]} {_n(v)}" for k, v in n_budget_err.items() if v)
         + "; fără curs: " + _n(sum(stats["fx_lost_budget"].values())) + ")")
    no_gross = df["worldwide_gross"].isna()
    stats["dropped_gross"] = pd.DataFrame({
        "vote_count": pd.to_numeric(df.loc[no_gross, "vote_count"], errors="coerce"),
        "budget": df.loc[no_gross, "budget"],
        "us": df.loc[no_gross, "production_countries"].apply(lambda cs: "US" in _as_list(cs)),
        "english": df.loc[no_gross, "original_language"] == "en",
        "year": df.loc[no_gross, "fx_year"],
    })
    step("încasări", df["worldwide_gross"].notna(),
         "nicio sursă nu are încasări mondiale (după conversia în USD; fără curs: "
         + _n(sum(stats["fx_lost_gross"].values())) + ")")

    # ROI: în USD; când bugetul și încasările alese sunt în aceeași monedă, direct din sumele
    # native (raportul nu are nevoie de curs)
    def chosen(kind: str, field: str) -> pd.Series:
        src = df["budget_source" if kind == "budget" else "gross_source"].map(SRC_SHORT)
        return pd.Series([df.at[i, f"{kind}_{p}_{field}"] for i, p in zip(df.index, src)], index=df.index)

    b_cur, g_cur = chosen("budget", "currency"), chosen("gross", "currency")
    df["budget_fx_rate"], df["gross_fx_rate"] = chosen("budget", "fx_rate"), chosen("gross", "fx_rate")
    same = (b_cur == g_cur) & (b_cur != "USD")
    df["roi"] = (df["worldwide_gross"] / df["budget"]).where(
        ~same, chosen("gross", "amount").astype(float) / chosen("budget", "amount").astype(float))
    df["roi_currency"] = b_cur.where(same, "USD")

    # Calitatea țintei (#47): încasările alese sunt doar SUA/Canada sau „rentals” (partea
    # distribuitorului) după eticheta din infobox. Se aplică și când a câștigat DBpedia cu
    # aceeași sumă (DBpedia copiază infobox-ul).
    from_infobox = (df["gross_source"] == "wikipedia") | (
        (df["gross_source"] == "dbpedia")
        & ((df["gross_dbp_amount"] - df["gross_wp_amount"]).abs() <= DBP_SAME_AMOUNT * df["gross_wp_amount"]))
    df["gross_is_domestic"] = from_infobox & df["gross_wp_domestic"].fillna(False).astype(bool)
    df["gross_is_rental"] = from_infobox & df["gross_wp_rental"].fillna(False).astype(bool)
    # #49: orice piață unică din eticheta infobox-ului (UK, Australia, Japan ... inclusiv SUA/Canada) și,
    # mai slab, încasări în altă monedă fără „worldwide”: de obicei sunt doar cele de acasă
    df["gross_is_single_market"] = from_infobox & df["gross_wp_single_market"].fillna(False).astype(bool)
    labelled_ww = from_infobox & df["gross_wp_worldwide"].fillna(False).astype(bool)
    df["gross_is_local_currency"] = (g_cur != "USD") & ~df["gross_is_single_market"] & ~labelled_ww
    df["_gross_currency"] = g_cur

    # Valori de umplutură (#49): aceeași sursă scrie exact aceeași sumă la buget și la încasări
    # (un lot de filme din TMDB are 10.000 $ / 10.000 $). Egalitatea între surse diferite e
    # posibilă (filme vechi cu „rentals” rotunjite egale cu bugetul) și nu se scoate.
    placeholder = (df["budget_source"] == df["gross_source"]) & np.isclose(df["budget"], df["worldwide_gross"])
    step("valori identice", ~placeholder,
         "aceeași sursă dă exact aceeași sumă pentru buget și încasări (valoare de umplutură, ROI = 1)")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    step("dată lansare", df["release_date"].notna(),
         "fără dată de lansare în TMDB")
    year = df["release_date"].dt.year
    step("interval ani", year.between(YEAR_MIN, YEAR_MAX), f"lansate înainte de {YEAR_MIN} sau după {YEAR_MAX}")

    is_tv = df["genres"].apply(lambda g: g is not None and "TV Movie" in list(g))
    step("film TV", ~is_tv, "filme TV (n-au rulat în cinema)")

    by_distributor = df["distributor_wp"].fillna("").str.strip().str.match(STREAMING_DISTRIBUTORS)
    digital_days = (df["first_digital"] - df["first_theatrical"]).dt.days
    by_digital = (digital_days <= STREAMING_DIGITAL_DAYS) & (df["first_theatrical"].dt.year >= STREAMING_DIGITAL_FROM)
    streaming = (by_distributor | by_digital) & (df["roi"] < STREAMING_MAX_ROI)
    stats["streaming"] = {
        "distribuitor": int((streaming & by_distributor & ~by_digital).sum()),
        "digital": int((streaming & by_digital & ~by_distributor).sum()),
        "ambele": int((streaming & by_distributor & by_digital).sum()),
        "fără date TMDB": int(df["first_theatrical"].isna().sum()),
        "titluri": df.loc[streaming].sort_values("release_date")["title"].head(20).tolist(),
    }
    step("streaming", ~streaming,
         f"lansare de streaming (distribuitor Netflix, Amazon, Apple TV+, Disney+ ... sau lansare digitală "
         f"la cel mult {STREAMING_DIGITAL_DAYS} de zile după cinema) cu încasări sub "
         f"{int(STREAMING_MAX_ROI * 100)}% din buget: lansarea în cinema a fost doar simbolică")
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    step("scurtmetraj", ~runtime.between(1, MIN_RUNTIME - 1), f"scurtmetraje (sub {MIN_RUNTIME} min)")

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


def recent_prior_count(df: pd.DataFrame) -> int:
    """Câte filme au un film anterior al regizorului lansat cu mai puțin de PREV_MIN_GAP înainte."""
    dates = defaultdict(list)
    for people, date in zip(df["director_ids"], df["release_date"]):
        for p in _as_list(people):
            dates[p].append(date)
    return int(sum(any(date - PREV_MIN_GAP < d < date for p in _as_list(people) for d in dates[p])
                   for people, date in zip(df["director_ids"], df["release_date"])))


def prev_history(df: pd.DataFrame, col: str, prefix: str, threshold: float) -> pd.DataFrame:
    """Pentru fiecare film și fiecare persoană din `col`: filmele din dataset ale
    persoanei lansate cu cel puțin PREV_MIN_GAP (6 luni) înainte, când încasările lor erau cunoscute. Statisticile se calculează per persoană și
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
            prior = np.array([r for d, r in films_by_person[p] if d <= date - PREV_MIN_GAP])
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


def franchise_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(is_franchise, sequel_number) știute înainte de lansare: câte filme din aceeași colecție
    TMDB au apărut înainte (0 = primul film sau film independent; gol = poziție necunoscută)."""
    coll = pd.to_numeric(df["collection_id"], errors="coerce").fillna(0)
    pos = pd.to_numeric(df["sequel_number"], errors="coerce")
    prior = (pos - 1).clip(lower=0).where(coll > 0, 0)
    return (prior >= 1).astype(int), prior.astype("Int64")


def keyword_stats(df: pd.DataFrame, threshold: float) -> dict:
    """Cât de mult țin keywords de popularitate (pentru raport)."""
    kws = df["keywords"].apply(lambda k: [w.strip().lower() for w in _as_list(k)])
    n_kw = kws.apply(len)
    votes = np.log1p(pd.to_numeric(df["vote_count"], errors="coerce").fillna(0))
    prof = df["roi"] >= threshold
    stingers = kws.apply(lambda ws: any(w in ("duringcreditsstinger", "aftercreditsstinger") for w in ws))
    return {"corr_votes": float(np.corrcoef(n_kw, votes)[0, 1]),
            "corr_profitable": float(np.corrcoef(n_kw, prof)[0, 1]),
            "none": (int((n_kw == 0).sum()), float(prof[n_kw == 0].mean())),
            "stingers": (int(stingers.sum()), float(prof[stingers].mean()))}


def franchise_stats(df: pd.DataFrame, threshold: float) -> dict:
    """% profitabile după vechea definiție (orice film dintr-o colecție), pentru raport."""
    coll = pd.to_numeric(df["collection_id"], errors="coerce").fillna(0) > 0
    pos = pd.to_numeric(df["sequel_number"], errors="coerce")
    prof = df["roi"] >= threshold
    groups = {"nu e în nicio colecție": ~coll, "primul film dintr-o colecție": coll & (pos == 1),
              "continuare (au apărut deja filme din serie)": coll & (pos >= 2)}
    return {k: (int(m.sum()), float(prof[m].mean())) for k, m in groups.items()}


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

    # 3. Mare studio: producție (companiile TMDB) și distribuție (distribuitorul principal din
    # infobox; lista completă poate avea distribuitori adăugați după lansare: reluări, alte piețe)
    out["major_studio"] = df["company_ids"].apply(
        lambda ids: int(any(int(i) in MAJOR_IDS for i in _as_list(ids))))
    out["major_distributor"] = df["distributor_wp"].fillna("").str.strip().str.match(
        MAJOR_DISTRIBUTORS).astype(int)
    out["major_any"] = (out["major_studio"] | out["major_distributor"]).astype(int)

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

    # 10. Cele mai frecvente keywords (one-hot), fără etichetele de ton (reacția publicului) și
    # fără cele care se află abia după lansare
    kws = df["keywords"].apply(lambda k: [w.strip().lower() for w in _as_list(k)])
    skip = TONE_KEYWORDS | POST_RELEASE_KEYWORDS
    top = pd.Series([w for ws in kws for w in set(ws) if w not in skip]).value_counts()
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

    # 13–14. Franciză și numărul în serie. Colecțiile TMDB apar după ce se face o continuare, iar
    # continuarea se face de obicei pentru că primul film a avut succes: pentru primul film,
    # „e într-o colecție” e informație din viitor (#47). Contează doar filmele din serie lansate
    # ÎNAINTE: sequel_number = poziția în colecție (după data lansării) - 1.
    out["is_franchise"], out["sequel_number"] = franchise_features(df)

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
    out["gross_is_domestic"] = df["gross_is_domestic"].astype(int)
    out["gross_is_rental"] = df["gross_is_rental"].astype(int)
    out["gross_is_single_market"] = df["gross_is_single_market"].astype(int)
    out["gross_is_local_currency"] = df["gross_is_local_currency"].astype(int)
    # Calitatea datelor: bugetul final e mijlocul unui interval din infobox („$10–15 million”)
    wp = pd.to_numeric(df["budget_wp"], errors="coerce")
    out["budget_is_range"] = (df["budget_wp_range"].fillna(False).astype(bool)
                              & (df["budget"] / wp).between(1 / CONFLICT_RATIO, CONFLICT_RATIO)).astype(int)
    out["fx_year"] = df["fx_year"]
    out["budget_fx_rate"] = df["budget_fx_rate"].round(6)
    out["gross_fx_rate"] = df["gross_fx_rate"].round(6)

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
    native = out["roi_currency"] != "USD"
    written = pd.concat([out[f"{k}_wp_amount"].notna() & (out[f"{k}_wp_currency"] != "USD")
                         & (out[f"{k}_source"] == "wikipedia") & out[f"{k}_fx_rate"].isna()
                         for k in ("budget", "gross")], axis=1).any(axis=1)
    roi_cur = out["roi_currency"].value_counts()
    top = _md_table([{"Monedă": k, "Filme": int(v), "%": _pct(v / n)} for k, v in roi_cur.head(10).items()]
                    + [{"Monedă": "altele", "Filme": int(roi_cur.iloc[10:].sum()),
                        "%": _pct(roi_cur.iloc[10:].sum() / n)}])
    lost = _md_table([{"Motiv": FX_NOTE[k], "Buget": stats.get("fx_lost_budget", {}).get(k, 0),
                       "Încasări": stats.get("fx_lost_gross", {}).get(k, 0)} for k in FX_NOTE])
    def chosen_currency(kind: str) -> pd.Series:
        src = out[f"{kind}_source"].map(SRC_SHORT)
        return pd.Series([out.at[i, f"{kind}_{p}_currency"] for i, p in src.items()], index=out.index)
    non_usd = (chosen_currency("budget") != "USD") | (chosen_currency("gross") != "USD")
    us_share = out.loc[non_usd, "country_us"].mean()
    return f"""## Monede

Bugetul și încasările se iau în moneda în care sunt scrise (`budget_<sursă>_amount`, `budget_<sursă>_currency`, la fel pentru încasări) și se convertesc în USD cu cursul mediu al anului lansării (World Bank, indicatorul `PA.NUS.FCRF`, licență CC BY 4.0; `fx_year`, `budget_fx_rate`, `gross_fx_rate`). Abia apoi se aplică regulile de mai sus: intervalele plauzibile, alegerea prin majoritate, toleranța de ±25% și regula pentru ROI extrem.

- {_n(int(non_usd.sum()))} filme ({_pct(non_usd.mean())}) au bugetul sau încasările dintr-o sumă în altă monedă; doar {_pct(us_share)} dintre ele sunt (co)producții americane. La {_n(int(written.sum()))} dintre ele, suma în USD vine chiar din infobox („£800,000 ($1.4 million)”) și are prioritate față de conversia noastră.
- Când bugetul și încasările alese sunt în aceeași monedă, ROI-ul se calculează direct din sumele native (un raport nu are nevoie de curs): {_n(int(native.sum()))} filme. `roi_currency` arată moneda.

**Moneda în care s-a calculat ROI-ul**

{top}

**Sume care nu s-au putut converti** (filme scoase la pașii „buget” / „încasări” doar din acest motiv)

{lost}

Seria World Bank începe în 1960: pentru filmele din 1950–1959 nu folosim alt curs (nu inventăm valori), deci filmele din anii '50 cu sume doar în altă monedă se pierd. Monedele redenominate (Brazilia înainte de 1994, Turcia înainte de 2005) apar în serie în unitatea de azi; anii lor au aproape toți hiperinflație, iar restul cad la verificarea intervalelor plauzibile.

- DBpedia marchează orice sumă ca USD, deși uneori a citit „A$2,417,000” sau „MXN$142.3 million”. Când suma DBpedia e chiar prima sumă din infobox-ul Wikipedia, iar aceea e în altă monedă, DBpedia primește moneda din infobox: {_n(stats.get("dbp_budget_fixed", 0))} bugete și {_n(stats.get("dbp_gross_fixed", 0))} încasări.
- Simboluri ambigue („¥” = JPY sau CNY, „kr” = SEK / NOK / DKK, „Rs”, „$”) se lămuresc după țara de producție din TMDB. Dacă tot rămân ambigue, suma e ignorată: {_n(stats.get("wp_ambiguous_budget", 0))} bugete și {_n(stats.get("wp_ambiguous_gross", 0))} încasări din Wikipedia.
- Un „$” simplu înseamnă USD, cu o excepție: un film fără SUA printre țările de producție, dar dintr-o țară cu dolar propriu (Australia, Canada, Hong Kong, Noua Zeelandă, Mexic ...), primește moneda acelei țări. Limită: Wikidata are aceeași sumă marcată USD la multe dintre aceste filme, dar și acolo valorile sunt adesea importate automat din infobox, deci nu e o confirmare independentă.
- Limită: cursul e media anului lansării; bugetul se cheltuie de obicei cu 1–2 ani înainte, iar încasările vin în anul lansării și după.
"""


def streaming_section(stats: dict) -> str:
    st = stats.get("streaming")
    if not st:
        return ""
    total = st["distribuitor"] + st["digital"] + st["ambele"]
    titles = ", ".join(st["titluri"])
    return f"""## Filme de streaming

Populația proiectului sunt filmele lansate în cinema. Unele filme de streaming au avut doar o lansare simbolică în câteva săli (Hillbilly Elegy: buget 45 mil. $, încasări 38.852 $) și ar apărea ca „neprofitabile” fără să fi încercat să recupereze bugetul din bilete. Pasul „streaming” scoate un film doar dacă **ambele** condiții sunt adevărate:

1. distribuitorul principal din infobox e un serviciu de streaming (Netflix, Amazon Studios / Prime Video, Apple TV+, Disney+, Hulu, HBO Max, Max, Paramount+, Peacock) **sau** TMDB are o lansare digitală (tipul 4) la cel mult {STREAMING_DIGITAL_DAYS} de zile după prima lansare în cinema (tipurile 2 și 3);
2. încasările sunt sub {int(STREAMING_MAX_ROI * 100)}% din buget (ROI < {str(STREAMING_MAX_ROI).replace(".", ",")}).

Criteriul digital se aplică doar filmelor lansate din {STREAMING_DIGITAL_FROM}: înainte nu exista distribuție digitală, iar TMDB are uneori casete sau DVD-uri marcate ca „digital” chiar la data premierei (Eureka, 1983). O lansare digitală *înainte* de cinema (VOD mai întâi, apoi câteva săli) contează și ea.

A doua condiție păstrează filmele de streaming care au rulat cu adevărat în cinema (Manchester by the Sea, distribuit de Amazon Studios).

| Criteriul îndeplinit | Filme scoase |
|---|---|
| doar distribuitorul | {_n(st["distribuitor"])} |
| doar lansarea digitală | {_n(st["digital"])} |
| amândouă | {_n(st["ambele"])} |
| **total** | **{_n(total)}** |

Primele {len(st["titluri"])} (după data lansării): {titles}.

Datele de lansare pe tipuri vin din răspunsurile TMDB deja descărcate (`python extraction/tmdb.py --release-dates`). Lipsesc pentru {_n(st["fără date TMDB"])} filme (detaliile TMDB adăugate la manifest nu sunt în arhiva din Releases); la ele contează doar distribuitorul.
"""


def studio_section(out: pd.DataFrame) -> str:
    rows = []
    for col, what in (("major_studio", "producție: o companie TMDB e un mare studio"),
                      ("major_distributor", "distribuitorul principal din infobox e un mare studio"),
                      ("major_any", "oricare din cele două")):
        yes = out[col] == 1
        rows.append({"Coloană": f"`{col}`", "Ce înseamnă": what, "Filme cu 1": int(yes.sum()),
                     "% profitabile (1)": _pct(out.loc[yes, "profitable"].mean()),
                     "% profitabile (0)": _pct(out.loc[~yes, "profitable"].mean())})
    only_dist = int(((out["major_distributor"] == 1) & (out["major_studio"] == 0)).sum())
    only_prod = int(((out["major_distributor"] == 0) & (out["major_studio"] == 1)).sum())
    return f"""## Mare studio: producție sau distribuție

{_md_table(rows)}

Față de `major_studio`, {_n(only_dist)} filme trec în clasa „mare studio” doar prin distribuitor (de exemplu filme independente cumpărate de Universal sau Sony Pictures Classics), iar {_n(only_prod)} au un mare studio printre producători, dar alt distribuitor principal. `major_any` le cuprinde pe toate. Contează doar distribuitorul principal: lista completă din infobox poate avea distribuitori adăugați după lansare (reluări, alte piețe).

`budget_is_range` = 1 la {_n(int(out["budget_is_range"].sum()))} filme ({_pct(out["budget_is_range"].mean())}): bugetul final e mijlocul unui interval din infobox („$10–15 million”). E o coloană despre calitatea datelor, nu o caracteristică pentru model.
"""


def meta_section(out: pd.DataFrame) -> str:
    def by(col: str, label: str) -> str:
        g = out.groupby(col)["profitable"].agg(["size", "mean"])
        return _md_table([{label: k, "Filme": int(r["size"]), "% profitabile": _pct(r["mean"])}
                          for k, r in g.iterrows()])

    meta = [c for c in out.columns if categories(c) == ["META"]]
    numeric = [c for c in meta if pd.api.types.is_numeric_dtype(out[c]) and out[c].nunique() > 1]
    corr = sorted(((c, out[c].astype(float).corr(out["profitable"])) for c in numeric),
                  key=lambda kv: -abs(kv[1]))
    corr_table = _md_table([{"Coloană": f"`{c}`", "Corelație cu `profitable`": f"{v:.3f}".replace(".", ",")}
                            for c, v in corr])
    n_features = len(feature_columns(out))
    return f"""## De ce excludem coloanele META

`extraction/columns.py` împarte coloanele în TARGETS, IDS, META, LEAKAGE și FEATURES; în model intră doar cele {n_features} din FEATURES (`feature_columns(df)`), iar `extraction/test_columns.py` verifică asta pe CSV. META sunt cele {len(meta)} coloane despre **date**, nu despre film: sursa valorii, câte surse au o valoare și câte sunt de acord, sumele brute, monedele și cursurile. Ele prezic ținta, pentru că un film cu încasări raportate peste tot e de obicei un film cunoscut, care a mers bine:

**% profitabile după numărul de surse cu încasări**

{by("gross_n_sources", "Surse cu încasări")}

**% profitabile după sursa bugetului ales**

{by("budget_source", "Sursa bugetului")}

**% profitabile după sursa încasărilor alese**

{by("gross_source", "Sursa încasărilor")}

**Corelația coloanelor META numerice cu `profitable`**

{corr_table}

La predicție, pentru un film care nu a fost încă lansat, nu știm câte surse îi vor raporta încasările. Un model care le folosește ar învăța „filmul e documentat bine”, nu „filmul va fi profitabil”.
"""


def dropped_section(out: pd.DataFrame, stats: dict) -> str:
    dropped = stats.get("dropped_gross")
    if dropped is None:
        return ""
    final = pd.DataFrame({"vote_count": out["vote_count"], "budget": out["budget"],
                          "us": out["country_us"] == 1, "english": out["is_english"] == 1,
                          "year": out["release_year"]})

    def profile(d: pd.DataFrame) -> dict:
        decade = (d["year"].dropna() // 10 * 10).astype(int).value_counts(normalize=True)
        return {"Filme": _n(len(d)), "Voturi TMDB (mediană)": _n(int(d["vote_count"].median())),
                "Buget (mediană)": f"{d['budget'].median() / 1e6:.1f}".replace(".", ",") + " mil. $",
                "Producții SUA": _pct(d["us"].mean()), "În limba engleză": _pct(d["english"].mean()),
                **{f"{y}–{min(y + 9, YEAR_MAX)}": _pct(decade.get(y, 0)) for y in range(YEAR_MIN, YEAR_MAX + 1, 10)}}

    a, b = profile(dropped), profile(final)
    table = "\n".join(["| | Scoase la „încasări” | Dataset final |", "|---|---|---|"]
                      + [f"| {k} | {a[k]} | {b[k]} |" for k in a])
    return f"""## Cine lipsește: filmele fără încasări

Cea mai mare pierdere din pâlnie e pasul „încasări”: filme cu buget, dar fără încasări mondiale în nicio sursă. Nu sunt o mostră la întâmplare:

{table}

Filmele scoase sunt mult mai mici și mai puțin cunoscute (mai puține voturi, buget mai mic, mai puține producții americane). Încasările lor nu apar nicăieri tocmai pentru că au fost mici, deci ele sunt, foarte probabil, mai des neprofitabile. Rata de {_pct(out["profitable"].mean())} filme profitabile e deci **supraestimată** față de toate filmele lansate în cinema: descrie filmele destul de cunoscute ca să le găsim încasările, iar modelul va învăța pe această populație.
"""


def target_quality_section(out: pd.DataFrame, stats: dict) -> str:
    dom, ren = out["gross_is_domestic"] == 1, out["gross_is_rental"] == 1
    single, local = out["gross_is_single_market"] == 1, out["gross_is_local_currency"] == 1
    partial = single | ren
    decade = (out["release_year"] // 10 * 10).astype(int)
    rows = []
    for d, g in out.groupby(decade):
        e = partial[g.index]
        rows.append({"Deceniu": f"{d}–{min(d + 9, YEAR_MAX)}", "Filme": int(len(g)),
                     "O singură piață": int(single[g.index].sum()), "Rentals": int(ren[g.index].sum()),
                     "% din deceniu": _pct(e.mean()),
                     "% profitabile (cu ele / fără ele)": f"{_pct(g['profitable'].mean())} / "
                                                          f"{_pct(g.loc[~e, 'profitable'].mean())}"})
    cur = stats.get("gross_currency", pd.Series("USD", index=out.index))
    group = np.where(cur == "USD", "USD", np.where(cur == "INR", "INR", "alte monede"))
    by_cur = []
    for k in ("USD", "INR", "alte monede"):
        m = pd.Series(group == k, index=out.index)
        row = {"Moneda încasărilor": k}
        for label, mask in (("o singură piață / rentals", m & (single | ren)),
                            ("monedă locală fără „worldwide”", m & local & ~(single | ren)),
                            ("restul", m & ~(single | ren | local))):
            row[label] = (f"{_n(int(mask.sum()))} filme, {_pct(out.loc[mask, 'profitable'].mean())}"
                          if mask.sum() >= 30 else f"{_n(int(mask.sum()))} filme")
        by_cur.append(row)
    return f"""## Calitatea țintei: încasări parțiale

Ținta presupune încasări **mondiale**. La unele filme, singura cifră din infobox e alta (coloane META, nu intră în model):
- `gross_is_single_market` = 1: încasările unei singure piețe, după eticheta din infobox („$12 million (U.S.)”, „£49,981 (UK)”, „A$3.2 million (Australia)”): {_n(int(single.sum()))} filme, {_pct(out.loc[single, "profitable"].mean())} profitabile. `gross_is_domestic` = 1 e cazul SUA/Canada ({_n(int(dom.sum()))} filme).
- `gross_is_rental` = 1: „rentals”, adică partea care revine distribuitorului, cam jumătate din încasări: {_n(int(ren.sum()))} filme, {_pct(out.loc[ren, "profitable"].mean())} profitabile.
- `gross_is_local_currency` = 1 (mai slab): încasări în altă monedă decât USD, fără etichetă de piață și fără „worldwide”. De obicei sunt doar cele de acasă, dar nu e sigur: {_n(int(local.sum()))} filme, {_pct(out.loc[local, "profitable"].mean())} profitabile.

La aceste filme, ROI-ul e **subestimat**, deci unele sunt etichetate greșit ca neprofitabile. Încasările parțiale (o singură piață sau rentals) sunt concentrate în filmele vechi:

{_md_table(rows)}

Sumele în moneda locală afectează filmele din afara SUA. Fără o corecție, un model ar învăța că filmele non-americane eșuează, iar asta ar veni din date, nu din realitate. Tabelul arată filmele și % profitabile pentru fiecare categorie (procentul apare de la 30 de filme în sus):

{_md_table(by_cur)}

Filmele cu încasări în alte monede (fără USD și INR) sunt profitabile mult mai rar decât filmele în USD fără semnale, ceea ce confirmă că suma lor e de obicei doar cea de acasă. Filmele indiene (INR) merg în sens invers. Infobox-urile lor dau de regulă încasările mondiale, adesea anunțate de producători și cu taxe incluse, deci probabil umflate. Același semnal (monedă locală fără „worldwide”) are așadar sensuri diferite pentru cele două grupuri.

`extraction/sensitivity.py` arată ținta și fără aceste filme. Pentru antrenare, varianta cea mai sigură e să le scoatem sau să le dăm o pondere mai mică, iar concluziile despre țară sau limbă trebuie verificate și fără ele.
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
    kw = stats.get("keywords", {"corr_votes": 0, "corr_profitable": 0, "none": (0, 0), "stingers": (0, 0)})
    franchise_table = "\n".join(
        ["  | Vechea definiție | Filme | % profitabile |", "  |---|---|---|"]
        + [f"  | {k} | {_n(v[0])} | {_pct(v[1])} |" for k, v in stats.get("franchise", {}).items()])
    final_src = _md_table([{"Sursă": k, "Buget": int((out["budget_source"] == k).sum()),
                            "Încasări": int((out["gross_source"] == k).sum())} for k in BUDGET_SOURCES])

    text = f"""# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold {threshold} --roi-rule {args.roi_rule} --budget-choice {args.budget_choice}`.

**Rezultat:** {_n(n)} filme în `data/processed/movies_dataset.csv`.

## Ce filme am scos

{_md_table(funnel)}

{streaming_section(stats)}
{dropped_section(out, stats)}
{target_quality_section(out, stats)}
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
{studio_section(out)}
{meta_section(out)}
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

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**. Pentru istoricul ROI, filmul anterior trebuie lansat cu cel puțin 6 luni înainte (#49): încasările unui film abia lansat nu sunt încă cunoscute. Înainte de această regulă, {_n(stats.get("prev_recent", 0))} filme foloseau ROI-ul unui film al regizorului lansat cu mai puțin de 6 luni în urmă.
- **Franciza** (#47): colecțiile TMDB se creează după ce apare o continuare, iar continuarea se face de obicei pentru că primul film a avut succes. Cu vechea definiție (`is_franchise` = filmul e într-o colecție), primul film al unei serii ar fi purtat informație din viitor:

{franchise_table}

  Acum `is_franchise` = 1 doar pentru continuări, iar `sequel_number` = câte filme din serie au apărut înainte (0 = primul film sau film independent).
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- **Keywords de după lansare** (#47): `duringcreditsstinger` / `aftercreditsstinger` (scenă după generic) se află abia la vizionare și anunțau succesul ({_n(kw["stingers"][0])} filme, {_pct(kw["stingers"][1])} profitabile). Le-am scos, împreună cu `controversy`, `banned film` și altele asemănătoare.
- **Limită rămasă:** keywords sunt adăugate de utilizatorii TMDB, după lansare. Filmele populare primesc mai multe: numărul de keywords se corelează {f"{kw['corr_votes']:.2f}".replace(".", ",")} cu log(voturi) și {f"{kw['corr_profitable']:.2f}".replace(".", ",")} cu `profitable`, iar cele {_n(kw["none"][0])} filme fără niciun keyword au doar {_pct(kw["none"][1])} profitabile. Chiar dacă fiecare keyword descrie conținutul, un vector `kw_*` plin de zerouri spune „film puțin cunoscut”. În experimentul cu leakage, merită comparat modelul cu și fără `kw_*`.
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
               budget_choice=args.budget_choice, stats=stats)
    stats["franchise"] = franchise_stats(df, args.threshold)
    stats["prev_recent"] = recent_prior_count(df)
    stats["gross_currency"] = df["_gross_currency"].reset_index(drop=True)
    stats["keywords"] = keyword_stats(df, args.threshold)
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
