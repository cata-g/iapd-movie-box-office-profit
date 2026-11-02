# IAPD – Movie Box Office Profit Prediction

Proiect de master la Politehnica București (IAPD – Învățare automată și procesarea datelor).

**Scop:** predicția profitabilității filmelor lansate în cinema folosind exclusiv date disponibile înainte de lansare (buget, gen, regizor, actori etc.).

Dataset propriu construit din TMDB, DBpedia, Wikidata, Wikipedia și Box Office Mojo — 10.000+ filme cu buget și încasări mondiale în USD.

## Instalare

```bash
pip install -r requirements.txt
cp .env.example .env
# Editează .env și adaugă cheia TMDB_API_KEY
```

## Structura

```
extraction/   — crawlere per sursă + utils.py (cache, retry, rate limit)
data/raw/     — cache brut per sursă (HTML/JSON descărcat)
data/interim/ — tabele curate per sursă + manifest
data/processed/ — dataset final
docs/         — documentație
```

## Date brute

Răspunsurile HTTP brute (cache-ul din `data/raw/<sursa>/`, câte un JSON per cerere) sunt păstrate ca arhive zip în `data/raw/`. Dezarhivează în rădăcina repo-ului; scripturile le refolosesc fără să descarce din nou. Arhivele prea mari pentru git se publică în [Releases](https://github.com/cata-g/iapd-movie-box-office-profit/releases).

| Sursă | Arhivă | Descărcat | Tabel rezultat |
|---|---|---|---|
| DBpedia (+ Wikipedia/Wikidata pentru `imdb_id`) | [data/raw/raw_dbpedia_2026-11-01.zip](data/raw/raw_dbpedia_2026-11-01.zip) | 2026-11-01 | `data/interim/dbpedia.parquet` |
| Wikidata | [data/raw/raw_wikidata_2026-11-02.zip](data/raw/raw_wikidata_2026-11-02.zip) | 2026-11-02 | `data/interim/wikidata.parquet` |
| Wikipedia (86 MB) | [Release: raw_wikipedia_2026-11-02.zip](https://github.com/cata-g/iapd-movie-box-office-profit/releases/download/raw-data-wikipedia-2026-11-02/raw_wikipedia_2026-11-02.zip) | 2026-11-02 | `data/interim/wikipedia.parquet` |

## Extracție DBpedia (#3)

```bash
python extraction/dbpedia.py --test   # 5 filme cunoscute, doar afișare
python extraction/dbpedia.py          # rulare completă (~9 min fără cache, câteva secunde cu cache)
```

Buget și încasări mondiale în USD din infobox-urile Wikipedia (via DBpedia), filme lansate 1950–2024, un rând per `imdb_id`. Rândurile excluse sunt în `data/interim/dbpedia_unmatched.csv` cu motivul.

## Extracție Wikidata + Wikipedia (#4)

```bash
python extraction/wikidata.py --test     # 5 filme cunoscute, doar afișare
python extraction/wikidata.py            # toate filmele cu imdb_id, 1950–2024 (~1 h fără cache)
python extraction/test_wikipedia.py      # test pentru parserul de sume din infobox
python extraction/wikipedia.py --test    # câteva articole cunoscute
python extraction/wikipedia.py           # după wikidata.py (și dbpedia.py, tmdb dacă există), ~45 min fără cache
```

- `wikidata.parquet`: cost (P2130) și box office mondial (P2142) doar în USD, `is_adaptation` (P144) și `adaptation_type` (book, comics, tv, video_game, stage, film, true_story, franchise, other), titlul articolului en.wikipedia, `tmdb_id`.
- `wikipedia.parquet`: `| budget =`, `| gross =`, `| distributor =` și `| based_on =` din infobox, pentru toate filmele cu articol. `has_budget_elsewhere = False` marchează bugetele noi (fără buget USD din altă sursă). Intervalele (`$10–15 million`) dau mijlocul, cu `budget_wp_range = True`. `distributor_wp` e distribuitorul principal (cel marcat SUA/internațional, altfel primul din listă).

## Test utilitare

```bash
python extraction/test_utils.py
```

Verifică că al doilea apel pe același URL vine din cache (fără request HTTP) și că un 404 returnează `None` fără excepție.
