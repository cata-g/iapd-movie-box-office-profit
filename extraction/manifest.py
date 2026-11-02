"""
Issue #5 — Manifest + decizia la 10.000 de filme

Reuneste TMDB, DBpedia, Wikidata, Wikipedia pe imdb_id, rezolva tmdb_id-urile
lipsa via find_by_imdb(), si scrie data/interim/movies_manifest.json.

Usage:
    python extraction/manifest.py
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.utils import fetch_async, PROJECT_ROOT
from extraction.tmdb import BASE, _p, _parse_movie

INTERIM = PROJECT_ROOT / "data" / "interim"
MANIFEST_FILE = INTERIM / "movies_manifest.json"


# ---------------------------------------------------------------------------
# Helpers pentru utils.py (load_manifest / mark) — definiti si aici local
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mark(imdb_id: str, source: str, status: str) -> None:
    manifest = load_manifest()
    if imdb_id in manifest:
        manifest[imdb_id]["status"][source] = status
        _save_manifest(manifest)


# ---------------------------------------------------------------------------
# Pasul 1 — reuniunea filmelor cu budget
# ---------------------------------------------------------------------------

def build_budget_union() -> pd.DataFrame:
    tmdb = pd.read_parquet(INTERIM / "tmdb_movies.parquet")
    dbp  = pd.read_parquet(INTERIM / "dbpedia.parquet")
    wiki = pd.read_parquet(INTERIM / "wikidata.parquet")
    wp   = pd.read_parquet(INTERIM / "wikipedia.parquet")

    # Filme cu budget valid per sursa
    s_tmdb = set(tmdb[tmdb["budget_tmdb"] > 0]["imdb_id"].dropna())
    s_dbp  = set(dbp[dbp["budget_dbp"] > 0]["imdb_id"].dropna())
    s_wd   = set(wiki[wiki["budget_wd"] > 0]["imdb_id"].dropna())
    s_wp   = set(wp[wp["budget_wp"] > 0]["imdb_id"].dropna())
    all_ids = s_tmdb | s_dbp | s_wd | s_wp

    print(f"TMDB:      {len(s_tmdb):>6}")
    print(f"DBpedia:   {len(s_dbp):>6}")
    print(f"Wikidata:  {len(s_wd):>6}")
    print(f"Wikipedia: {len(s_wp):>6}")
    print(f"UNION:     {len(all_ids):>6}")

    # Indexuri pentru lookup rapid (keep first in caz de duplicate)
    tmdb_idx  = tmdb.drop_duplicates("imdb_id").set_index("imdb_id")
    dbp_idx   = dbp.drop_duplicates("imdb_id").set_index("imdb_id")
    wiki_idx  = wiki.drop_duplicates("imdb_id").set_index("imdb_id")
    wp_idx    = wp.drop_duplicates("imdb_id").set_index("imdb_id")

    rows = []
    for iid in all_ids:
        t = tmdb_idx.loc[iid] if iid in tmdb_idx.index else None
        d = dbp_idx.loc[iid]  if iid in dbp_idx.index  else None
        w = wiki_idx.loc[iid] if iid in wiki_idx.index else None
        p = wp_idx.loc[iid]   if iid in wp_idx.index   else None

        # tmdb_id: TMDB parquet > DBpedia > Wikidata > None
        tmdb_id = None
        if t is not None:
            tmdb_id = int(t["tmdb_id"]) if pd.notna(t["tmdb_id"]) else None
        if tmdb_id is None and d is not None:
            tmdb_id = int(d["tmdb_id"]) if pd.notna(d.get("tmdb_id")) else None
        if tmdb_id is None and w is not None:
            tmdb_id = int(w["tmdb_id"]) if pd.notna(w.get("tmdb_id")) else None

        # title / year: TMDB > DBpedia > Wikidata > Wikipedia
        title = None
        year  = None
        if t is not None:
            title = t.get("title")
            rd = t.get("release_date")
            year = int(str(rd)[:4]) if rd and pd.notna(rd) and str(rd)[:4].isdigit() else None
        if not title and d is not None:
            title = d.get("dbp_title")
        if not title and w is not None:
            title = w.get("enwiki_title")
        if not title and p is not None:
            title = p.get("enwiki_title")
        if year is None:
            for src in [d, w]:
                if src is not None:
                    y = src.get("year")
                    if y is not None and pd.notna(y):
                        year = int(y)
                        break

        budget_sources = {
            "tmdb":      int(t["budget_tmdb"]) if t is not None and t["budget_tmdb"] > 0 else None,
            "dbp":       int(d["budget_dbp"])  if d is not None and d["budget_dbp"] > 0  else None,
            "wikidata":  int(w["budget_wd"])   if w is not None and w["budget_wd"] > 0   else None,
            "wikipedia": int(p["budget_wp"])   if p is not None and p["budget_wp"] > 0   else None,
        }

        rows.append({
            "imdb_id":        iid,
            "tmdb_id":        tmdb_id,
            "title":          title,
            "year":           year,
            "budget_sources": budget_sources,
            "in_tmdb":        t is not None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pasul 2 — rezolva tmdb_id-urile lipsa via find_by_imdb (async)
# ---------------------------------------------------------------------------

async def _find_imdb_async(session, imdb_id, pbar):
    data = await fetch_async(
        session,
        f"{BASE}/find/{imdb_id}",
        "tmdb",
        params=_p(external_source="imdb_id"),
    )
    pbar.update(1)
    if data:
        results = data.get("movie_results") or []
        if results:
            return imdb_id, results[0].get("id")
    return imdb_id, None


async def _fetch_details_async(session, tmdb_id, pbar):
    data = await fetch_async(
        session,
        f"{BASE}/movie/{tmdb_id}",
        "tmdb",
        params=_p(append_to_response="credits,release_dates,keywords"),
    )
    pbar.update(1)
    return tmdb_id, _parse_movie(data)


async def resolve_missing_async(df: pd.DataFrame) -> pd.DataFrame:
    missing_mask = df["tmdb_id"].isna()
    missing_ids  = df.loc[missing_mask, "imdb_id"].tolist()
    print(f"\nFilme fara tmdb_id: {len(missing_ids)} - cauta via find_by_imdb...")

    found = {}
    if missing_ids:
        with tqdm(total=len(missing_ids), desc="find_by_imdb") as pbar:
            async with aiohttp.ClientSession() as session:
                tasks = [_find_imdb_async(session, iid, pbar) for iid in missing_ids]
                for fut in asyncio.as_completed(tasks):
                    iid, tid = await fut
                    if tid:
                        found[iid] = tid

    print(f"Gasite: {len(found)} / {len(missing_ids)}")
    for iid, tid in found.items():
        df.loc[df["imdb_id"] == iid, "tmdb_id"] = tid

    # Filme cu tmdb_id dar fara detalii TMDB (nu erau în tmdb_movies.parquet)
    needs_details = df[~df["in_tmdb"] & df["tmdb_id"].notna()].copy()
    print(f"Filme extra care necesita detalii TMDB: {len(needs_details)}")

    if not needs_details.empty:
        extra_rows = {}
        with tqdm(total=len(needs_details), desc="Fetch detalii extra") as pbar:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    _fetch_details_async(session, int(row["tmdb_id"]), pbar)
                    for _, row in needs_details.iterrows()
                ]
                for fut in asyncio.as_completed(tasks):
                    tid, parsed = await fut
                    if parsed:
                        extra_rows[parsed["tmdb_id"]] = parsed

        if extra_rows:
            extra_df = pd.DataFrame(extra_rows.values())
            out_path = INTERIM / "tmdb_extra.parquet"
            extra_df.to_parquet(out_path, index=False)
            print(f"Salvat {len(extra_df)} filme extra -> {out_path}")

    return df


# ---------------------------------------------------------------------------
# Pasul 3 — scrie manifestul
# ---------------------------------------------------------------------------

def write_manifest(df: pd.DataFrame) -> dict:
    manifest = {}
    for _, row in df.iterrows():
        iid = row["imdb_id"]
        manifest[iid] = {
            "imdb_id":        iid,
            "tmdb_id":        int(row["tmdb_id"]) if pd.notna(row["tmdb_id"]) else None,
            "title":          row["title"],
            "year":           int(row["year"]) if pd.notna(row["year"]) else None,
            "budget_sources": row["budget_sources"],
            "status": {
                "tmdb_details":   "done" if row["in_tmdb"] else ("done" if pd.notna(row["tmdb_id"]) else "missing"),
                "box_office_mojo": "pending",
            },
        }
    _save_manifest(manifest)
    print(f"\nManifest scris: {MANIFEST_FILE} ({len(manifest)} filme)")
    return manifest


# ---------------------------------------------------------------------------
# Pasul 4 — numărătoarea pentru comentariul din issue
# ---------------------------------------------------------------------------

def print_stats(df: pd.DataFrame, manifest: dict) -> str:
    total = len(manifest)
    n_sources = df["budget_sources"].apply(lambda d: sum(1 for v in d.values() if v))
    tmdb = pd.read_parquet(INTERIM / "tmdb_movies.parquet").set_index("imdb_id")

    # Distributie pe decade
    decade_counts = df.dropna(subset=["year"]).copy()
    decade_counts["decade"] = (decade_counts["year"] // 10 * 10).astype(int)
    by_decade = decade_counts.groupby("decade").size()

    # Distributie pe tara (din TMDB)
    country_counts = {}
    for iid in df["imdb_id"]:
        if iid in tmdb.index:
            countries = tmdb.loc[iid, "production_countries"]
            if isinstance(countries, list) and countries:
                country_counts[countries[0]] = country_counts.get(countries[0], 0) + 1
    top_countries = sorted(country_counts.items(), key=lambda x: -x[1])[:10]

    lines = [
        f"## Numaratoare manifest — issue #5\n",
        f"**Total filme cu budget:** {total:,}",
        f"**Decizia:** {'[OK] CONTINUAM cu profit ca tinta (>= 10.000)' if total >= 10000 else '[!] VARIANTA DE REZERVA (încasări)'}",
        f"\n### Budget per numar de surse",
    ]
    for n in sorted(n_sources.unique()):
        lines.append(f"- {n} surse: {(n_sources == n).sum():,} filme")

    lines += ["\n### Contribuție per sursa fata de TMDB"]
    s_tmdb = set(df[df["budget_sources"].apply(lambda d: d.get("tmdb") is not None)]["imdb_id"])
    for src, key in [("DBpedia", "dbp"), ("Wikidata", "wikidata"), ("Wikipedia", "wikipedia")]:
        s = set(df[df["budget_sources"].apply(lambda d: d.get(key) is not None)]["imdb_id"])
        extra = s - s_tmdb
        lines.append(f"- {src}: {len(s):,} filme, **+{len(extra):,}** fata de TMDB")

    lines += ["\n### Distribuție pe decenii"]
    for decade, count in by_decade.items():
        lines.append(f"- {decade}s: {count:,}")

    lines += ["\n### Top 10 tari de productie"]
    for country, count in top_countries:
        lines.append(f"- {country}: {count:,}")

    body = "\n".join(lines)
    print("\n" + body)
    return body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main_async(df):
    return await resolve_missing_async(df)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=== Pasul 1: Reuniunea surselor ===")
    df = build_budget_union()

    print("\n=== Pasul 2: Rezolvare tmdb_id lipsa ===")
    df = asyncio.run(_main_async(df))

    print("\n=== Pasul 3: Scriere manifest ===")
    manifest = write_manifest(df)

    print("\n=== Pasul 4: Statistici ===")
    stats_body = print_stats(df, manifest)

    # Postează comentariu în issue #5
    import subprocess
    result = subprocess.run(
        ["gh", "issue", "comment", "5", "--body", stats_body],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("\nComentar postat în issue #5.")
    else:
        print(f"\nEroare la postarea comentariului: {result.stderr}")


if __name__ == "__main__":
    main()
