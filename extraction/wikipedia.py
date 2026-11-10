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
(action=query&prop=revisions&rvsection=0). The first amount of each field is kept in
its own currency (`budget_wp_amount`, `budget_wp_currency`): `$15 million`, `£3 million`,
`{{INR|55 crore}}`, `A$2,417,000`, `MYR 35.88 million`. `budget_wp` is the amount in USD
written in the infobox (the amount itself if it is in USD, else a USD amount in brackets:
`£800,000 ($1.4 million)`); otherwise it stays empty and integrate.py converts it.
A range `$10–15 million` gives the midpoint and budget_wp_range = True.
Ambiguous symbols (`¥`, `kr`, `Rs`, a `$` of a non-US film from a dollar country) are
resolved with the TMDB production countries; if still ambiguous the amount is ignored
(budget_wp_note = "ambiguous").
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

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


# ---------------------------------------------------------------------------
# Monede
# ---------------------------------------------------------------------------
# Codurile ISO recunoscute (și cele istorice: DEM, FRF, ITL ... înainte de euro)
ISO_CODES = set("""
USD EUR GBP JPY CNY INR KRW RUB CAD AUD HKD TWD NZD SGD BRL MXN ARS CLP COP PEN VES DOP JMD
CHF SEK NOK DKK ISK PLN CZK HUF RON BGN HRK RSD TRY UAH KZT GEL ILS EGP NGN ZAR MAD IRR PKR
BDT LKR NPR MVR MYR IDR THB PHP VND KES GHS ETB TND DZD LBP JOD AED SAR QAR KWD
DEM FRF ITL ESP NLG BEF ATS PTE IEP FIM GRD LUF LTL LVL EEK SKK SIT CSK SUR
""".split())

# Marcaje înainte de număr (cele mai lungi primele); "?" = ambiguu, se rezolvă după țară
PREFIXES = {
    "US$": "USD", "USD$": "USD", "U.S.$": "USD",
    "A$": "AUD", "AU$": "AUD", "AUD$": "AUD", "AUS$": "AUD",
    "C$": "CAD", "CA$": "CAD", "CAD$": "CAD", "Can$": "CAD", "CAN$": "CAD",
    "HK$": "HKD", "HKD$": "HKD", "NT$": "TWD", "NTD$": "TWD", "NZ$": "NZD", "S$": "SGD", "SG$": "SGD",
    "R$": "BRL", "MX$": "MXN", "MXN$": "MXN", "Mex$": "MXN", "RD$": "DOP", "JA$": "JMD", "J$": "JMD",
    "CN¥": "CNY", "JP¥": "JPY", "RMB": "CNY", "IR£": "IEP", "£E": "EGP", "E£": "EGP",
    "€": "EUR", "₹": "INR", "₩": "KRW", "₽": "RUB", "₱": "PHP", "৳": "BDT", "रु": "NPR", "NRs": "NPR",
    "₺": "TRY", "฿": "THB", "₦": "NGN", "₪": "ILS", "₫": "VND", "₣": "FRANC?", "₤": "LIRA?",
    "Rs.": "RUPEE?", "Rs": "RUPEE?", "RM": "MYR", "Rp": "IDR", "DM": "DEM", "FF": "FRF",
    "DKr": "DKK", "SKr": "SEK", "NKr": "NOK", "kr": "KRONA?", "Kr": "KRONA?", "zł": "PLN",
    "£": "POUND?", "¥": "YEN?", "$": "DOLLAR?",
}
# Marcaje după număr („4 million kr”, „60 million baht”, „415,000,000 ITL”)
SUFFIXES = {
    "yen": "JPY", "won": "KRW", "yuan": "CNY", "renminbi": "CNY", "RMB": "CNY",
    "rupees": "RUPEE?", "rupee": "RUPEE?", "lire": "ITL", "lira": "LIRA?",
    "francs": "FRANC?", "franc": "FRANC?", "Deutsche Marks": "DEM", "Deutsche Mark": "DEM", "DM": "DEM",
    "marks": "MARK?", "markkaa": "FIM", "markka": "FIM", "pesetas": "ESP", "escudos": "PTE",
    "drachmas": "GRD", "guilders": "NLG", "kronor": "SEK", "kroner": "KRONE?", "krona": "KRONA?",
    "kr": "KRONA?", "pesos": "PESO?", "rubles": "RUB", "roubles": "RUB", "Rbls": "RUB", "baht": "THB",
    "ringgit": "MYR", "rupiah": "IDR", "pounds": "POUND?", "euros": "EUR", "euro": "EUR",
    "naira": "NGN", "lei": "RON", "forint": "HUF", "Ft": "HUF", "Kč": "CZK", "KČs": "CSK", "Kčs": "CSK",
    "złoty": "PLN", "zł": "PLN", "zl": "PLN", "dirhams": "MAD", "shekels": "ILS", "NIS": "ILS",
    "đồng": "VND", "dong": "VND", "taka": "BDT", "tenge": "KZT", "rand": "ZAR", "hryvnias": "UAH",
    "L.E": "EGP", "LE": "EGP", "dollars": "DOLLAR?", "US dollars": "USD",
}
# Marcaj ambiguu -> {țara de producție (TMDB): monedă}
AMBIGUOUS = {
    "DOLLAR?": {"AU": "AUD", "CA": "CAD", "NZ": "NZD", "HK": "HKD", "SG": "SGD", "TW": "TWD",
                "MX": "MXN", "AR": "ARS", "CL": "CLP", "CO": "COP", "JM": "JMD"},
    "POUND?": {"GB": "GBP", "IE": "IEP", "EG": "EGP"},
    "YEN?": {"JP": "JPY", "CN": "CNY"},
    "KRONA?": {"SE": "SEK", "NO": "NOK", "DK": "DKK", "IS": "ISK"},
    "KRONE?": {"NO": "NOK", "DK": "DKK"},
    "RUPEE?": {"IN": "INR", "PK": "PKR", "LK": "LKR", "NP": "NPR"},
    "PESO?": {"MX": "MXN", "AR": "ARS", "PH": "PHP", "CL": "CLP", "CO": "COP", "DO": "DOP"},
    "FRANC?": {"FR": "FRF", "BE": "BEF", "CH": "CHF", "LU": "LUF"},
    "MARK?": {"DE": "DEM", "FI": "FIM"},
    "LIRA?": {"IT": "ITL", "TR": "TRY"},
}
# Fără țară care să-l lămurească, „$” e dolar american și „£” liră sterlină (convenția en.wikipedia)
AMBIGUOUS_DEFAULT = {"DOLLAR?": ("US", "USD"), "POUND?": ("GB", "GBP")}

# Șabloane de monedă cu nume: {{Philippine peso|5 million}}, {{Indian Rupee}}10 crore
NAMED_TEMPLATES = {"indian rupee": "INR", "philippine peso": "PHP", "sk won": "KRW", "turkish lira": "TRY",
                   "nepalese rupee": "NPR", "euro": "EUR", "pakistani rupee": "PKR", "sri lankan rupee": "LKR",
                   "bangladeshi taka": "BDT", "russian ruble": "RUB", "yen": "JPY", "renminbi": "CNY"}
SYMBOL_TEMPLATES = {"£": "£", "€": "€", "¥": "¥", "₹": "₹", "A$": "A$", "HK$": "HK$", "NZ$": "NZ$",
                    "CN¥": "CN¥", "JP¥": "JP¥", "NT$": "NT$", "S$": "S$", "R$": "R$", "C$": "C$"}
# {{INRConvert|3.6|c}}: al doilea argument e unitatea
CONVERT_UNITS = {"k": "thousand", "l": "lakh", "c": "crore", "m": "million", "b": "billion", "lc": "lakh crore"}

NUM = r"\d{1,3}(?:(?:[   ]\d{3})+(?!\d)|(?:\.\d{3}){2,}(?:,\d+)?(?!\d))|\d[\d,]*(?:\.\d+)?"
UNIT = r"(?:\s*((?i:million|billion|mil\b\.?|m\b|bn\b|crores?|lakhs?|lacs?|thousand)))?"
SEP = r"\s*(?:–|—|-|to)\s*"
MULT = {"million": 1e6, "mil": 1e6, "mil.": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "thousand": 1e3,
        "crore": 1e7, "crores": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5}


def _alt(markers) -> str:
    """Alternare regex; marcajele care încep cu literă nu pot continua un cuvânt (CAD, nu „aCAD”)."""
    out = []
    for m in sorted(markers, key=len, reverse=True):
        e = re.escape(m)
        if len(m) > 1 and m.endswith("$"):          # „CAD $1.5 million”, „HK $2,393,396”
            e = re.escape(m[:-1]) + r"\s?\$"
        out.append(rf"(?<![A-Za-z]){e}" if m[0].isalpha() else rf"(?<![A-Za-z$]){e}" if m == "$" else e)
    return "|".join(out)


PRE = rf"(?:{_alt([*PREFIXES, *(c + '$' for c in ISO_CODES), *ISO_CODES])})"
SUF = rf"(?:{_alt([*SUFFIXES, *ISO_CODES])})(?![A-Za-z])"
MONEY_ANY = re.compile(
    rf"(?P<pre>{PRE})?\s*(?P<n1>{NUM})(?P<u1>{UNIT})(?:{SEP}(?:{PRE})?\s*(?P<n2>{NUM})(?P<u2>{UNIT}))?"
    rf"(?:\s*(?P<suf>{SUF}))?")
WORLDWIDE = re.compile(r"\b(?:worldwide|world-wide|global|total)\b", re.I)
# Un $ precedat de literă care nu e marcaj cunoscut („ZW$5 million”) = monedă necunoscută
UNKNOWN_DOLLAR = re.compile(r"[A-Za-z]{1,3}\$\s*\d")


class Money(NamedTuple):
    amount: float | None          # prima sumă din câmp (mijlocul unui interval), în moneda ei
    currency: str | None          # cod ISO; None dacă moneda e ambiguă sau necunoscută
    is_range: bool = False
    usd: float | None = None      # suma în USD scrisă în infobox (aceeași sumă sau una în paranteză)
    note: str = ""                # "ambiguous" / "unknown" când suma e ignorată


def _code(marker: str | None, table: dict) -> str | None:
    if marker is None:
        return None
    marker = re.sub(r"\s+\$$", "$", marker)
    if marker in table:
        return table[marker]
    up = marker.rstrip("$").upper()
    if up in ISO_CODES:
        return up
    return next((v for k, v in table.items() if k.lower() == marker.lower()), None)


def resolve_currency(code: str, countries=()) -> str | None:
    """Rezolvă un marcaj ambiguu („$”, „¥”, „kr”, „Rs” ...) după țările de producție TMDB.
    None = tot ambiguu (mai multe candidate sau nicio țară potrivită)."""
    if code not in AMBIGUOUS:
        return code
    countries = set(countries or ())
    found = {AMBIGUOUS[code][c] for c in countries if c in AMBIGUOUS[code]}
    if code in AMBIGUOUS_DEFAULT:
        home, default = AMBIGUOUS_DEFAULT[code]
        if home in countries or not found:
            return default
    return found.pop() if len(found) == 1 else None


def _num(t: str) -> float:
    t = t.replace(" ", " ").replace(" ", " ")
    if re.fullmatch(r"\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?", t):      # 21.074.048,58
        return float(t.replace(".", "").replace(",", "."))
    return float(t.replace(",", "").replace(" ", ""))


def _value(m: re.Match) -> tuple[float | None, bool]:
    n1, u1, n2, u2 = m.group("n1", "u1", "n2", "u2")
    u1, u2 = (u1 or "").strip().lower(), (u2 or "").strip().lower()
    u1, u2 = u1 or u2, u2 or u1
    v1 = _num(n1) * MULT.get(u1, 1)
    if n2 is None:
        return v1, False
    v2 = _num(n2) * MULT.get(u2, 1)
    if v2 < v1:  # "$5–6" where the unit belongs to both but got lost, or a typo
        return None, False
    return (v1 + v2) / 2, True


def _first_money(s: str):
    """Prima sumă care are un marcaj de monedă (înainte sau după număr)."""
    for m in MONEY_ANY.finditer(s):
        if m.group("pre") or m.group("suf"):
            return m
    return None


def _explicit_usd(rest: str) -> float | None:
    """Suma în USD dintr-o paranteză care urmează: „£800,000 ($1.4 million)”."""
    for p in re.finditer(r"\(([^()]*)\)", rest[:120]):
        m = _first_money(p.group(1))
        if m is None:
            continue
        code = _code(m.group("pre") or m.group("suf"), {**PREFIXES, **SUFFIXES})
        if code in ("USD", "DOLLAR?"):   # în paranteza de conversie, „$” e dolarul american
            return _value(m)[0]
    return None


def parse_money(raw: str | None, countries=()) -> Money:
    """Prima sumă din câmp, în moneda ei: Money(amount, currency, is_range, usd, note).
    `countries` (codurile ISO ale țărilor de producție TMDB) lămuresc simbolurile ambigue:
    „¥” = JPY / CNY, „kr” = SEK / NOK / DKK, „$” al unui film fără SUA, dar din Australia = AUD.
    Dacă tot rămâne ambiguu, suma e ignorată (note = "ambiguous")."""
    if not raw:
        return Money(None, None)
    s = clean(raw)
    # Într-o listă („JPY 1.5 billion (Japan) ; US$14.7 million (worldwide)”), elementul marcat
    # worldwide / total e cel bun; altfel, primul
    for part in s.split(" ; "):
        if WORLDWIDE.search(part) and _first_money(part):
            s = part
            break
    m = _first_money(s)
    if m is None:
        return Money(None, None, note="unknown" if UNKNOWN_DOLLAR.search(s) else "")
    marker = m.group("pre") or m.group("suf")
    if m.group("pre") and m.group("pre") == "$" and UNKNOWN_DOLLAR.match(s[max(0, m.start() - 3):m.start() + 2]):
        return Money(None, None, note="unknown")
    code = _code(marker, PREFIXES if m.group("pre") else SUFFIXES)
    if code is None:
        return Money(None, None, note="unknown")
    value, is_range = _value(m)
    if value is None:
        return Money(None, None)
    currency = resolve_currency(code, countries)
    if currency is None:
        return Money(value, None, is_range, note="ambiguous")
    usd = value if currency == "USD" else _explicit_usd(s[m.end():])
    return Money(value, currency, is_range, usd)


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
    # Listele ({{Plainlist| * A * B}}) se despachetează: rămân elementele, separate prin „;”
    s = re.sub(r"\{\{\s*(?:plain\s*list|ubl|unbulleted\s*list|flatlist|hlist)\s*\|", " ; ", s, flags=re.I)
    s = re.sub(r"(?:^|\s)\*\s*", " ; ", s)
    # Șabloanele de monedă păstrează codul: {{INR|55 crore}} -> "INR 55 crore", {{INR}}30 -> "INR 30"
    s = re.sub(r"\{\{\s*(?:To USD|Inflation|inflation/fn|Inflation-year)[^{}]*\}\}", "", s, flags=re.I)
    s = re.sub(r"\{\{\s*format price\s*\|\s*([^|}]*)[^{}]*\}\}", r"\1", s, flags=re.I)
    s = re.sub(r"\|\s*(?:link|lk|year|round|long|passthrough|sigfig|nolink|lakh|r|fmt)\s*=[^|}]*", "", s, flags=re.I)
    s = re.sub(r"\{\{\s*([A-Z]{3})\s*Convert\s*\|\s*([^|}]+)(?:\|\s*([a-zA-Z]{1,2})\s*)?[^{}]*\}\}",
               lambda m: f"{m[1].upper()} {m[2]} {CONVERT_UNITS.get((m[3] or '').lower(), '')} ", s, flags=re.I)
    s = re.sub(r"\{\{\s*currency\s*\|\s*([^|}]+)\|\s*([A-Za-z]{3})\s*[^{}]*\}\}",
               lambda m: f"{m[2].upper()} {m[1]} ", s, flags=re.I)
    s = re.sub(r"\{\{\s*([A-Za-z][A-Za-z ]+?)\s*(?:\|\s*([^|{}]*))?(?:\|[^{}]*)?\}\}",
               lambda m: (f"{NAMED_TEMPLATES[m[1].strip().lower()]} {m[2] or ''} "
                          if m[1].strip().lower() in NAMED_TEMPLATES else
                          f"{m[1].strip().upper()} {m[2] or ''} " if m[1].strip().upper() in ISO_CODES
                          else m[0]), s)
    s = re.sub(r"\{\{\s*(" + "|".join(re.escape(k) for k in sorted(SYMBOL_TEMPLATES, key=len, reverse=True))
               + r")\s*(?:\|\s*([^|{}]*))?(?:\|[^{}]*)?\}\}", lambda m: f"{m[1]}{m[2] or ''} ", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)                          # [[a|b]] -> b
    for _ in range(3):
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)                                         # other templates
    s = s.replace("{{", "").replace("}}", "")
    return re.sub(r"\s+", " ", s).strip()


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
def production_countries() -> dict[str, list[str]]:
    """imdb_id -> țările de producție din TMDB (pentru simbolurile ambigue de monedă)."""
    frames = [pd.read_parquet(INTERIM / f, columns=["imdb_id", "production_countries"])
              for f in ("tmdb_movies.parquet", "tmdb_extra.parquet") if (INTERIM / f).exists()]
    if not frames:
        return {}
    tmdb = pd.concat(frames).dropna(subset=["imdb_id"]).drop_duplicates("imdb_id")
    return {i: list(c) for i, c in zip(tmdb["imdb_id"], tmdb["production_countries"]) if c is not None}


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


def _first_run_split(films: pd.DataFrame) -> pd.Series:
    """Împărțirea din prima rulare (#4), făcută înainte să existe tmdb_movies.parquet: buget USD
    doar în Wikidata sau DBpedia. Loturile de 50 de titluri rămân aceleași -> cache valid."""
    has_budget = set()
    for file in ("wikidata.parquet", "dbpedia.parquet"):
        col = OTHER_SOURCES[file]
        if (INTERIM / file).exists():
            src = pd.read_parquet(INTERIM / file, columns=["imdb_id", col])
            has_budget |= set(src.loc[src[col] > 0, "imdb_id"])
    return films["imdb_id"].isin(has_budget)


def build(films: pd.DataFrame) -> pd.DataFrame:
    # Two separate lists keep the cached batches of the first run (candidates only) valid
    texts = {}
    if "has_budget_elsewhere" in films:
        split = _first_run_split(films)
        texts |= fetch_section0(films.loc[~split, "enwiki_title"].tolist())
        texts |= fetch_section0(films.loc[split, "enwiki_title"].tolist())
    else:
        texts = fetch_section0(films["enwiki_title"].tolist())
    countries = production_countries()
    rows = []
    for f in films.itertuples():
        text = texts.get(f.enwiki_title, "")
        film_countries = countries.get(f.imdb_id, [])
        # A field holding only an editor comment counts as empty
        budget_raw, gross_raw, distributor_raw, based_on_raw = (
            v if v and re.sub(r"<!--.*?-->", "", v, flags=re.S).strip() else None
            for v in (infobox_field(text, "budget"), infobox_field(text, "gross"),
                      infobox_field(text, "distributor"), infobox_field(text, r"based[ _]on")))
        budget = parse_money(budget_raw, film_countries)
        gross = parse_money(gross_raw, film_countries)
        distributors = parse_names(distributor_raw)
        rows.append({
            "imdb_id": f.imdb_id, "wikidata_id": f.wikidata_id, "enwiki_title": f.enwiki_title,
            "has_budget_elsewhere": getattr(f, "has_budget_elsewhere", False),
            "budget_wp": budget.usd, "gross_wp": gross.usd,
            "budget_wp_raw": budget_raw, "budget_wp_range": budget.is_range,
            "gross_wp_raw": gross_raw, "gross_wp_range": gross.is_range,
            "budget_wp_amount": budget.amount, "budget_wp_currency": budget.currency,
            "gross_wp_amount": gross.amount, "gross_wp_currency": gross.currency,
            "budget_wp_note": budget.note or None, "gross_wp_note": gross.note or None,
            "budget_wp_other_currency": budget.currency not in (None, "USD"),
            "gross_wp_other_currency": gross.currency not in (None, "USD"),
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
        f"Bugete în orice monedă: {df['budget_wp_currency'].notna().sum()} "
        f"(top: {', '.join(f'{k}={v}' for k, v in df['budget_wp_currency'].value_counts().head(8).items())}); "
        f"încasări: {df['gross_wp_currency'].notna().sum()}",
        f"Ignorate: simbol ambiguu {(df['budget_wp_note'] == 'ambiguous').sum()} bugete / "
        f"{(df['gross_wp_note'] == 'ambiguous').sum()} încasări; monedă necunoscută "
        f"{(df['budget_wp_note'] == 'unknown').sum()} / {(df['gross_wp_note'] == 'unknown').sum()}",
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
            print(df[["enwiki_title", "budget_wp", "budget_wp_amount", "budget_wp_currency", "budget_wp_range",
                      "gross_wp", "gross_wp_amount", "gross_wp_currency", "budget_wp_raw"]])
            print(df[["enwiki_title", "distributor_wp", "distributors_wp", "based_on_wp"]])
    else:
        df = build(films_with_articles())
        df.to_parquet(OUT_PATH, index=False)
        print(f"Scris: {OUT_PATH}\n")
        print(report(df))
