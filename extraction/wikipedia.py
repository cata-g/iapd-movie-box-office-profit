"""
Wikipedia extraction: `| budget =`, `| gross =`, `| distributor =` and `| based_on =`
parsed from the infobox of every English Wikipedia film article (from wikidata.parquet).
has_budget_elsewhere marks films that already have a USD budget from another source;
the others are the "new budgets" Wikipedia adds.

Run:
  python extraction/wikipedia.py          # full run -> data/interim/wikipedia.parquet
  python extraction/wikipedia.py --test   # a few known articles, printed only

Needs data/interim/wikidata.parquet (#4, gives imdb_id <-> enwiki_title).
"Another source" = Wikidata, DBpedia and TMDB, whichever parquet files exist.

Section 0 (lead + infobox) is fetched for 50 articles per API call
(action=query&prop=revisions&rvsection=0). Amounts are kept only in USD:
`$15 million`, `US$2.5 million`, `$1,200,000`; a range `$10–15 million` gives the
midpoint and budget_wp_range = True. A value whose first amount is in another
currency (`£`, `€`, `{{INR|...}}`, crore) is ignored.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import BUDGET_RANGE, GROSS_RANGE, PROJECT_ROOT, _get_logger, fetch

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
BATCH = 50  # max pages per API call when fetching content

INTERIM = PROJECT_ROOT / "data" / "interim"
OUT_PATH = INTERIM / "wikipedia.parquet"
OTHER_SOURCES = {  # parquet -> budget column
    "wikidata.parquet": "budget_wd",
    "dbpedia.parquet": "budget_dbp",
    "tmdb_movies.parquet": "budget_tmdb",
}

TEST_FILMS = ["Inception", "Paranormal Activity", "The Blair Witch Project", "Mad Max: Fury Road",
              "3 Idiots", "El Mariachi", "Avengers: Endgame", "The Full Monty"]

log = _get_logger("wikipedia")


# ---------------------------------------------------------------------------
# Infobox parsing
# ---------------------------------------------------------------------------
def infobox_field(wikitext: str, name: str) -> str | None:
    """Raw value of `| name = ...` (plus continuation lines), or None."""
    lines = wikitext.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"\s*\|\s*{name}\s*=(.*)$", line)
        if m:
            value = [m.group(1)]
            for nxt in lines[i + 1:]:
                if re.match(r"\s*(\||\}\})", nxt):
                    break
                value.append(nxt)
            value = " ".join(value).strip()
            return value or None
    return None


NON_USD = (r"[£€₹¥₩₱₽¤]|\b(?:crore|lakh|Rs\.?|RMB|yen|rupees?|won|francs?|marks?|pesos?)\b"
           r"|\b(?!USD)[A-Z]{3}\s?\d|(?<=[A-Za-z])\$")                     # ISO code before a number: MYR 2.8
CUR = r"(?:(?<![A-Za-z])US\$|(?<![A-Za-z])USD\s?|(?<![A-Za-z$])\$)"
NUM = r"\d[\d,]*(?:\.\d+)?"
UNIT = r"(?:\s*(million|billion|mil\b\.?|m\b|bn\b))?"
SEP = r"\s*(?:–|—|-|to)\s*"
MONEY = re.compile(rf"{CUR}\s*({NUM}){UNIT}(?:{SEP}{CUR}?\s*({NUM}){UNIT})?", re.IGNORECASE)
MULT = {"million": 1e6, "mil": 1e6, "mil.": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}


def clean(raw: str) -> str:
    s = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", " ; ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&nbsp;", " ").replace("&ndash;", "–").replace("&mdash;", "—")
    s = re.sub(r"\{\{\s*(?:nbsp|nowrap\s*\|)\s*\}?\}?", " ", s, flags=re.I)
    s = re.sub(r"\{\{\s*(?:US\$|USD)\s*\|\s*([^|}]+)[^}]*\}\}", r"US$\1", s, flags=re.I)  # {{US$|15 million}}
    s = re.sub(r"\{\{\s*(?:US\$|USD)\s*\}\}\s*", "US$", s, flags=re.I)                 # {{USD}}15 million
    s = re.sub(r"\{\{\s*([A-Z]{3}|£|€|¥|₹)\s*(\||\}\})", "¤ ", s)                      # {{INR|55 crore}}, {{INR}}30
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)                          # [[a|b]] -> b
    for _ in range(3):
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)                                         # other templates
    s = s.replace("{{", "").replace("}}", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_money(raw: str | None) -> tuple[float | None, bool, bool]:
    """-> (USD value, is_range, has_other_currency). Value is the midpoint for a range."""
    if not raw:
        return None, False, False
    s = clean(raw)
    m = MONEY.search(s)
    other_before = re.search(NON_USD, s[:m.start()] if m else s, re.IGNORECASE) is not None
    if m is None or other_before:
        return None, False, other_before
    n1, u1, n2, u2 = m.groups()
    u1 = (u1 or u2 or "").lower()
    u2 = (u2 or u1 or "").lower()
    v1 = float(n1.replace(",", "")) * MULT.get(u1, 1)
    if n2 is None:
        return v1, False, False
    v2 = float(n2.replace(",", "")) * MULT.get(u2, 1)
    if v2 < v1:  # "$5–6" where the unit belongs to both but got lost, or a typo
        return None, False, False
    return (v1 + v2) / 2, True, False


# ---------------------------------------------------------------------------
# Wikipedia API
# ---------------------------------------------------------------------------
def fetch_section0(titles: list[str]) -> dict[str, str]:
    """title -> wikitext of section 0 (follows redirects)."""
    result = {}
    titles = sorted(set(titles))  # stable batches -> cache hits on rerun
    for i in range(0, len(titles), BATCH):
        batch = titles[i:i + BATCH]
        data = fetch(WIKIPEDIA_API, source="wikipedia", params={
            "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
            "rvsection": 0, "redirects": 1, "titles": "|".join(batch),
            "format": "json", "formatversion": 2,
        })
        if data is None:
            log.warning("Batch %d failed", i)
            continue
        q = data.get("query", {})
        renamed = {r["from"]: r["to"] for r in q.get("normalized", []) + q.get("redirects", [])}
        text_by_page = {
            p["title"]: p["revisions"][0]["slots"]["main"]["content"]
            for p in q.get("pages", []) if p.get("revisions")
        }
        for t in batch:
            page = t
            while page in renamed:
                page = renamed[page]
            if page in text_by_page:
                result[t] = text_by_page[page]
        print(f"\rArticole: {min(i + BATCH, len(titles))}/{len(titles)}", end="", flush=True)
    print()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def films_with_articles() -> pd.DataFrame:
    """All films with an enwiki article; has_budget_elsewhere = USD budget from another source."""
    wd = pd.read_parquet(INTERIM / "wikidata.parquet")
    films = wd[wd["enwiki_title"].notna()][["imdb_id", "wikidata_id", "enwiki_title"]]
    has_budget = set()
    for file, col in OTHER_SOURCES.items():
        path = INTERIM / file
        if path.exists():
            src = pd.read_parquet(path, columns=["imdb_id", col])
            ids = set(src.loc[src[col] > 0, "imdb_id"])
            log.info("%s: %d films with %s", file, len(ids), col)
            has_budget |= ids
        else:
            log.info("%s missing, skipped", file)
    return films.assign(has_budget_elsewhere=films["imdb_id"].isin(has_budget)).reset_index(drop=True)


LIST_TEMPLATES = re.compile(r"\{\{\s*(?:plain\s*list|ubl|unbulleted\s*list|flatlist|hlist)\s*\|", re.I)
US_MARKERS = re.compile(r"\b(?:United States|U\.S\.|US|USA|North America|worldwide|international)\b", re.I)


def parse_names(raw: str | None) -> list[str]:
    """`[[Warner Bros. Pictures]]`, `{{Plainlist| * A (US) * B (International)}}` -> names in order.
    The first name is the one marked as US / worldwide distributor if any, else the first listed."""
    if not raw:
        return []
    s = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\{\{\s*(?:efn|sfn|refn|citation needed|cn)[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*\}\}", "", s, flags=re.I)
    s = re.sub(r"\{\{\s*(?:ill|interlanguage link)\s*\|\s*([^|}]+)[^}]*\}\}", r"\1", s, flags=re.I)  # {{ill|Name|de}}
    s = LIST_TEMPLATES.sub("|", s).replace("}}", "").replace("{{", "")
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    items = re.split(r"<br\s*/?>|\*|\||\n|;", s)
    names, us_first = [], None
    for item in items:
        item = re.sub(r"<[^>]+>|''+", "", item).strip(" ,")
        name = re.sub(r"\s*\(.*?\)\s*", " ", item).strip(" ,")
        if not name or len(name) > 80:
            continue
        if us_first is None and US_MARKERS.search(item) and item != name:
            us_first = name
        if name not in names:
            names.append(name)
    if us_first:
        names.remove(us_first)
        names.insert(0, us_first)
    return names


def build(films: pd.DataFrame) -> pd.DataFrame:
    # Two separate lists keep the cached batches of the first run (candidates only) valid
    texts = {}
    if "has_budget_elsewhere" in films:
        texts |= fetch_section0(films.loc[~films["has_budget_elsewhere"], "enwiki_title"].tolist())
        texts |= fetch_section0(films.loc[films["has_budget_elsewhere"], "enwiki_title"].tolist())
    else:
        texts = fetch_section0(films["enwiki_title"].tolist())
    rows = []
    for f in films.itertuples():
        text = texts.get(f.enwiki_title, "")
        # A field holding only an editor comment counts as empty
        budget_raw, gross_raw, distributor_raw, based_on_raw = (
            v if v and re.sub(r"<!--.*?-->", "", v, flags=re.S).strip() else None
            for v in (infobox_field(text, "budget"), infobox_field(text, "gross"),
                      infobox_field(text, "distributor"), infobox_field(text, r"based[ _]on")))
        budget, budget_range, budget_other = parse_money(budget_raw)
        gross, gross_range, gross_other = parse_money(gross_raw)
        distributors = parse_names(distributor_raw)
        rows.append({
            "imdb_id": f.imdb_id, "wikidata_id": f.wikidata_id, "enwiki_title": f.enwiki_title,
            "has_budget_elsewhere": getattr(f, "has_budget_elsewhere", False),
            "budget_wp": budget, "gross_wp": gross,
            "budget_wp_raw": budget_raw, "budget_wp_range": budget_range,
            "gross_wp_raw": gross_raw, "gross_wp_range": gross_range,
            "budget_wp_other_currency": budget_other, "gross_wp_other_currency": gross_other,
            "distributor_wp": distributors[0] if distributors else None,
            "distributors_wp": "|".join(distributors) or None,
            # {{Based on|work|author}} would be dropped by clean() like any template: unwrap it first
            "based_on_wp": clean(re.sub(r"\{\{\s*based on\s*\|", "", based_on_raw, flags=re.I)) or None
                           if based_on_raw else None,
            "article_found": f.enwiki_title in texts,
        })
    df = pd.DataFrame(rows)
    for col, (lo, hi) in [("budget_wp", BUDGET_RANGE), ("gross_wp", GROSS_RANGE)]:
        bad = df[col].notna() & ~df[col].between(lo, hi)
        log.info("%s: %d values outside [%d, %d] -> NaN", col, bad.sum(), lo, hi)
        df[f"{col}_implausible"] = bad
        df.loc[bad, col] = float("nan")
    return df


def report(df: pd.DataFrame) -> str:
    new = df[~df["has_budget_elsewhere"]]
    return "\n".join([
        f"Filme cu articol en.wikipedia: {len(df)} (găsite: {df['article_found'].sum()})",
        f"Bugete în USD: {df['budget_wp'].notna().sum()}; încasări în USD: {df['gross_wp'].notna().sum()}; "
        f"ambele: {(df['budget_wp'].notna() & df['gross_wp'].notna()).sum()}",
        f"Candidați (fără buget USD din altă sursă): {len(new)}",
        f"  cu câmp budget în infobox: {new['budget_wp_raw'].notna().sum()}",
        f"  bugete noi în USD adăugate de Wikipedia: {new['budget_wp'].notna().sum()} "
        f"(din care intervale: {(new['budget_wp'].notna() & new['budget_wp_range']).sum()})",
        f"  buget doar în altă monedă: {new['budget_wp_other_currency'].sum()}",
        f"Cu distribuitor în infobox: {df['distributor_wp'].notna().sum()} "
        f"(top: {', '.join(f'{k}={v}' for k, v in df['distributor_wp'].value_counts().head(5).items())})",
        f"Cu câmp based_on în infobox: {df['based_on_wp'].notna().sum()}",
        f"Valori neplauzibile șterse: {df['budget_wp_implausible'].sum()} bugete, {df['gross_wp_implausible'].sum()} încasări",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="only a few known articles, no file written")
    args = parser.parse_args()

    if args.test:
        films = pd.DataFrame({"imdb_id": None, "wikidata_id": None, "enwiki_title": TEST_FILMS})
        df = build(films)
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 45):
            print(df[["enwiki_title", "budget_wp", "budget_wp_range", "gross_wp", "budget_wp_other_currency", "budget_wp_raw"]])
            print(df[["enwiki_title", "distributor_wp", "distributors_wp", "based_on_wp"]])
    else:
        df = build(films_with_articles())
        df.to_parquet(OUT_PATH, index=False)
        print(f"Scris: {OUT_PATH}\n")
        print(report(df))
