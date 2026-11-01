"""
TMDB extraction — discovers cinema releases 1950-2024 and fetches details.

Usage:
    python extraction/tmdb.py

Output:
    data/interim/tmdb_movies.parquet
"""

import calendar
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.utils import fetch, RAW_DIR, LOGS_DIR

load_dotenv()

TMDB_KEY = os.getenv("TMDB_API_KEY")
BASE = "https://api.themoviedb.org/3"
SOURCE = "tmdb"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

DISCOVERED_FILE = RAW_DIR / "tmdb" / "discovered_ids.json"
OUTPUT_FILE = INTERIM_DIR / "tmdb_movies.parquet"

YEARS = range(1950, 2025)

logger = logging.getLogger(f"iapd.{SOURCE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(**kwargs):
    """Build params dict with api_key included."""
    return {"api_key": TMDB_KEY, **{k: v for k, v in kwargs.items() if v is not None}}


def _discover_page(page, year=None, date_gte=None, date_lte=None):
    params = {
        "api_key": TMDB_KEY,
        "with_release_type": "2|3",
        "vote_count.gte": 5,
        "sort_by": "primary_release_date.asc",
        "page": page,
    }
    if date_gte:
        params["primary_release_date.gte"] = date_gte
        params["primary_release_date.lte"] = date_lte
    else:
        params["primary_release_year"] = year
    return fetch(f"{BASE}/discover/movie", SOURCE, params=params)


def _collect_ids_from_pages(first_resp, page_fetcher, max_pages=500):
    ids = [r["id"] for r in first_resp.get("results", [])]
    total = min(first_resp.get("total_pages", 1), max_pages)
    for page in range(2, total + 1):
        resp = page_fetcher(page)
        if resp:
            ids.extend(r["id"] for r in resp.get("results", []))
    return ids


# ---------------------------------------------------------------------------
# Phase 1 — Discover
# ---------------------------------------------------------------------------

def discover_year(year):
    first = _discover_page(1, year=year)
    if not first:
        return []

    if first.get("total_pages", 1) <= 500:
        return _collect_ids_from_pages(
            first,
            lambda p: _discover_page(p, year=year),
        )

    # Year has >10 000 films — split by month
    ids = []
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]
        gte = f"{year}-{month:02d}-01"
        lte = f"{year}-{month:02d}-{last_day:02d}"
        first_m = _discover_page(1, date_gte=gte, date_lte=lte)
        if not first_m:
            continue
        ids.extend(_collect_ids_from_pages(
            first_m,
            lambda p, g=gte, l=lte: _discover_page(p, date_gte=g, date_lte=l),
        ))
    return ids


def get_all_candidates():
    """Discover all candidates 1950-2024, resumable via discovered_ids.json."""
    (RAW_DIR / "tmdb").mkdir(parents=True, exist_ok=True)
    discovered = json.loads(DISCOVERED_FILE.read_text()) if DISCOVERED_FILE.exists() else {}

    for year in tqdm(YEARS, desc="Discovering years"):
        if str(year) in discovered:
            continue
        ids = discover_year(year)
        discovered[str(year)] = ids
        DISCOVERED_FILE.write_text(json.dumps(discovered))
        logger.info("Year %d: %d films", year, len(ids))

    all_ids = {id_ for ids in discovered.values() for id_ in ids}
    logger.info("Total unique candidates: %d", len(all_ids))
    return list(all_ids)


# ---------------------------------------------------------------------------
# Phase 2 — Details
# ---------------------------------------------------------------------------

def fetch_movie_details(tmdb_id):
    return fetch(
        f"{BASE}/movie/{tmdb_id}",
        SOURCE,
        params=_p(append_to_response="credits,release_dates,keywords"),
    )


def _extract_release_date(release_dates_data):
    """Earliest theatrical date: type 3 preferred, fallback type 2."""
    if not release_dates_data:
        return None
    candidates = []
    for country in release_dates_data.get("results", []):
        for rd in country.get("release_dates", []):
            rt = rd.get("type")
            d = (rd.get("release_date") or "")[:10]
            if rt in (2, 3) and d:
                candidates.append((rt, d))
    if not candidates:
        return None
    type3 = sorted(d for rt, d in candidates if rt == 3)
    if type3:
        return type3[0]
    return sorted(d for rt, d in candidates if rt == 2)[0]


def _extract_cert_us(release_dates_data):
    if not release_dates_data:
        return None
    for country in release_dates_data.get("results", []):
        if country.get("iso_3166_1") == "US":
            for rd in country.get("release_dates", []):
                cert = rd.get("certification", "")
                if cert:
                    return cert
    return None


def _parse_movie(data):
    if not data or not data.get("id"):
        return None
    companies = data.get("production_companies") or []
    collection = data.get("belongs_to_collection") or {}
    credits = data.get("credits") or {}
    release_dates = data.get("release_dates") or {}
    keywords_data = data.get("keywords") or {}

    directors = [
        c["id"] for c in credits.get("crew", [])
        if c.get("department") == "Directing" and c.get("job") == "Director"
    ]
    cast_top3 = [
        c["id"] for c in sorted(credits.get("cast", []), key=lambda x: x.get("order", 9999))[:3]
    ]

    return {
        "tmdb_id": data["id"],
        "imdb_id": data.get("imdb_id"),
        "title": data.get("title"),
        "original_title": data.get("original_title"),
        "release_date": _extract_release_date(release_dates),
        "budget_tmdb": data.get("budget") or 0,
        "revenue_tmdb": data.get("revenue") or 0,
        "runtime": data.get("runtime"),
        "genres": [g["name"] for g in (data.get("genres") or [])],
        "original_language": data.get("original_language"),
        "production_countries": [c["iso_3166_1"] for c in (data.get("production_countries") or [])],
        "company_ids": [c["id"] for c in companies],
        "company_names": [c["name"] for c in companies],
        "collection_id": collection.get("id") or 0,
        "certification_us": _extract_cert_us(release_dates),
        "director_ids": directors,
        "cast_top3_ids": cast_top3,
        "keywords": [k["name"] for k in (keywords_data.get("keywords") or [])],
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
    }


# ---------------------------------------------------------------------------
# Phase 3 — Collections → sequel_number
# ---------------------------------------------------------------------------

def fetch_collection(collection_id):
    return fetch(f"{BASE}/collection/{collection_id}", SOURCE, params=_p())


def _compute_sequel_numbers(rows):
    coll_ids = {r["collection_id"] for r in rows if r.get("collection_id")}
    sequel_map = {}  # tmdb_id → sequel_number

    for cid in tqdm(coll_ids, desc="Collections"):
        data = fetch_collection(cid)
        if not data:
            continue
        parts = sorted(
            [p for p in data.get("parts", []) if p.get("release_date")],
            key=lambda p: p["release_date"],
        )
        for i, part in enumerate(parts, start=1):
            sequel_map[part["id"]] = i

    for r in rows:
        r["sequel_number"] = sequel_map.get(r["tmdb_id"], 0)
    return rows


# ---------------------------------------------------------------------------
# Reusable search helpers (used by issues #3, #4, #5)
# ---------------------------------------------------------------------------

def find_by_imdb(imdb_id):
    """Return first TMDB movie result for a given IMDb ID, or None."""
    resp = fetch(
        f"{BASE}/find/{imdb_id}",
        SOURCE,
        params=_p(external_source="imdb_id"),
    )
    if resp:
        results = resp.get("movie_results") or []
        return results[0] if results else None
    return None


def search_title_year(title, year):
    """Return first TMDB search result for title + year, or None."""
    resp = fetch(
        f"{BASE}/search/movie",
        SOURCE,
        params=_p(query=title, year=year),
    )
    if resp:
        results = resp.get("results") or []
        return results[0] if results else None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset():
    if not TMDB_KEY:
        raise RuntimeError("TMDB_API_KEY not set in .env")

    # Phase 1
    all_ids = get_all_candidates()

    # Phase 2
    rows = []
    for tmdb_id in tqdm(all_ids, desc="Fetching details"):
        data = fetch_movie_details(tmdb_id)
        row = _parse_movie(data)
        if row:
            rows.append(row)
    logger.info("Parsed %d / %d", len(rows), len(all_ids))

    # Phase 3
    rows = _compute_sequel_numbers(rows)

    # Phase 4 — save
    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Saved → %s", OUTPUT_FILE)

    # Stats (for issue comment)
    df["decade"] = (
        pd.to_datetime(df["release_date"], errors="coerce").dt.year // 10 * 10
    ).astype("Int64")
    print(f"\nTotal films:      {len(df)}")
    print(f"With imdb_id:     {df['imdb_id'].notna().sum()}")
    print(f"With budget > 0:  {(df['budget_tmdb'] > 0).sum()}")
    print("\nPer decade:")
    print(df.groupby("decade").size().to_string())

    return df


if __name__ == "__main__":
    build_dataset()
