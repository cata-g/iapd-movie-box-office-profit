# IAPD – Movie Box Office Profit Prediction

Proiect de master la Politehnica București (IAPD – Învățare automată și procesarea datelor).

**Scop:** predicția profitabilității filmelor lansate în cinema folosind exclusiv date disponibile înainte de lansare (buget, gen, regizor, actori etc.).

Dataset propriu construit din TMDB, DBpedia, Wikidata și Wikipedia — 10.000+ filme cu buget și încasări mondiale în USD.

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
| Wikidata (+ premii) | [data/raw/raw_wikidata_2026-11-03.zip](data/raw/raw_wikidata_2026-11-03.zip) | 2026-11-03 | `data/interim/wikidata.parquet`, `data/interim/awards.parquet` |
| Wikipedia (86 MB) | [Release: raw_wikipedia_2026-11-02.zip](https://github.com/cata-g/iapd-movie-box-office-profit/releases/download/raw-data-wikipedia-2026-11-02/raw_wikipedia_2026-11-02.zip) | 2026-11-02 | `data/interim/wikipedia.parquet` |

## Extracție DBpedia (#3)

```bash
python extraction/dbpedia.py --test   # 5 filme cunoscute, doar afișare
python extraction/dbpedia.py          # rulare completă (~9 min fără cache, câteva secunde cu cache)
```

Buget și încasări mondiale în USD din infobox-urile Wikipedia (via DBpedia), filme lansate 1950–2024, un rând per `imdb_id`. Rândurile excluse sunt în `data/interim/dbpedia_unmatched.parquet` cu motivul.

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

## Premii ale regizorului și actorilor (#35)

```bash
python extraction/awards.py --test   # 5 filme cunoscute, doar afișare
python extraction/awards.py          # toate filmele din manifest (~5 min fără cache)
```

`awards.parquet`, un rând pentru fiecare film din manifest: nominalizări și premii Oscar, Globul de Aur și BAFTA ale regizorului (`director_*`) și ale primilor 3 actori (`cast_*`), numărate **doar înainte de lansare**. Persoanele din TMDB sunt găsite în Wikidata prin P4985 (TMDb person ID). O dată cunoscută doar la nivel de an e tratată ca 31 decembrie (anti-leakage). `*_found_wd` / `*_n` = câte persoane au fost găsite în Wikidata din câte are filmul.

## Integrare și curățare (#27)

```bash
python extraction/integrate.py                 # prag profitable = 2.5
python extraction/integrate.py --threshold 3   # alt prag
```

Citește doar `data/interim/` (fără cereri HTTP, câteva secunde) și scrie `data/processed/movies_dataset.csv` (un rând per film) și [docs/integration_report.md](docs/integration_report.md) (pâlnia de curățare, conflicte, echilibrul claselor). Rulat de două ori, produce același fișier.

### Dicționar de date

**Identificare (nu intră în model):** `imdb_id`, `tmdb_id`, `title`.

**Ținta (nu intră în model):**

| Coloană | Descriere |
|---|---|
| `budget` | buget final în USD; prioritate TMDB → DBpedia → Wikidata → Wikipedia (e și caracteristică de intrare) |
| `worldwide_gross` | încasări mondiale în USD; prioritate Wikipedia → DBpedia → Wikidata → TMDB |
| `roi` | `worldwide_gross / budget` |
| `profitable` | 1 dacă `roi >= prag` (implicit 2.5) |

**Caracteristici (cunoscute înainte de lansare):**

| Coloană | Descriere |
|---|---|
| `director_award_{noms,wins}_before`, `cast_award_{noms,wins}_before` | nominalizări / premii Oscar + Globul de Aur + BAFTA ale regizorului și primilor 3 actori, strict înainte de lansare |
| `major_studio` | 1 dacă o companie de producție e Disney, Warner Bros., Universal, Paramount, Sony/Columbia, 20th Century sau Lionsgate (inclusiv diviziile de film, după ID-ul TMDB) |
| `release_date`, `release_year`, `release_month`, `release_weekday` | data lansării din TMDB; `release_weekday` 0 = luni |
| `genres`, `genre_*` | genurile TMDB (separate prin `\|`) și one-hot |
| `runtime_min` | durata în minute (gol = necunoscută) |
| `certification` | G=0, PG=1, PG-13=2, R=3, NC-17=4 (gol = lipsă / NR) |
| `kw_*` | one-hot pentru cele mai frecvente 30 de keywords TMDB din dataset |
| `is_english`, `original_language` | limba originală: `en`, top 5 alte limbi sau `other` |
| `country_{us,europe,asia,other}` | grupul țărilor de producție (un film poate avea mai multe) |
| `is_franchise`, `sequel_number` | face parte dintr-o colecție TMDB; al câtelea film din colecție (0 = nu e în colecție) |
| `is_adaptation`, `adaptation_type` | ecranizare (Wikidata P144, keywords „based on …” sau `based on` din infobox); tipul sursei din Wikidata, `other` dacă e ecranizare fără tip cunoscut, `none` altfel |
| `director_prev_films`, `director_prev_mean_roi` | filme anterioare din dataset ale regizorului și ROI-ul lor mediu (gol = fără istoric) |
| `cast_prev_films`, `cast_prev_mean_roi` | același calcul pentru primii 3 actori (maxim de filme, media ROI-urilor medii) |

**Surse (nu intră în model):** `budget_source` / `gross_source` (sursa valorii finale), `*_n_sources` (câte surse au valoare), `*_n_agree` (câte surse sunt la ±25% de valoarea finală), `*_conflict` (max/min > 1,25), valorile brute `budget_tmdb`, `budget_dbp`, `budget_wd`, `budget_wp`, `gross_wp`, `worldwide_gross_dbp`, `gross_wd`, `revenue_tmdb`.

**Doar pentru experimentul leakage vs. no-leakage:** `vote_average`, `vote_count`.

## Surse excluse

| Sursă | Motiv |
|---|---|
| The Numbers | lista de bugete nu mai e publică; Terms of Service interzic colectarea automată pentru seturi de date ML |
| Box Office Mojo (#6) | `robots.txt`: `Disallow: /` pentru orice crawler nelistat; Conditions of Use IMDb interzic scraping-ul fără permisiune scrisă |

Valorile lor ajung oricum indirect în date: infobox-urile Wikipedia le citează ca sursă.

## Test utilitare

```bash
python extraction/test_utils.py
```

Verifică că al doilea apel pe același URL vine din cache (fără request HTTP) și că un 404 returnează `None` fără excepție.
