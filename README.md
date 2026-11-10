# IAPD – Movie Box Office Profit Prediction

Proiect de master la Politehnica București (IAPD – Învățare automată și procesarea datelor).

**Scop:** predicția profitabilității filmelor lansate în cinema folosind exclusiv date disponibile înainte de lansare (buget, gen, regizor, actori etc.).

Dataset propriu construit din TMDB, DBpedia, Wikidata și Wikipedia — **17.911 filme** cu buget și încasări mondiale (convertite în USD), 32,2% profitabile. Detalii: [raportul de integrare](docs/integration_report.md) și [testul de sensibilitate](docs/sensitivity_report.md).

**Monede.** Bugetul și încasările se citesc în moneda în care sunt scrise (`$`, `£`, `€`, `₹`, `A$`, `{{INR|55 crore}}`, `MYR 35.88 million` ...) și se convertesc în USD cu cursul mediu World Bank din anul lansării; abia apoi se compară sursele. O sumă în USD scrisă explicit în infobox are prioritate; dacă bugetul și încasările sunt în aceeași monedă, ROI-ul se calculează direct din ele. Sumele fără curs (înainte de 1960, ani de hiperinflație) și simbolurile ambigue nerezolvate se ignoră și se numără în raport.

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
| TMDB (601 MB) | [Release: raw_tmdb_2026-11-02.zip](https://github.com/cata-g/iapd-movie-box-office-profit/releases/download/raw-data-tmdb-2026-11-02/raw_tmdb_2026-11-02.zip) | 2026-11-02 | `data/interim/tmdb_movies.parquet`, `data/interim/tmdb_release_dates.parquet` |
| World Bank, cursuri de schimb (CC BY 4.0) | [data/raw/raw_worldbank_2026-11-10.zip](data/raw/raw_worldbank_2026-11-10.zip) | 2026-11-10 | `data/interim/fx_rates.parquet` |

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

- `wikidata.parquet`: cost (P2130) și box office mondial (P2142) în USD (`budget_wd`, `gross_wd`) și în moneda lor (`budget_wd_amount`, `budget_wd_currency`, cod ISO), `is_adaptation` (P144) și `adaptation_type` (book, comics, tv, video_game, stage, film, true_story, franchise, other), titlul articolului en.wikipedia, `tmdb_id`.
- `wikipedia.parquet`: `| budget =`, `| gross =`, `| distributor =` și `| based_on =` din infobox, pentru toate filmele cu articol. Prima sumă din câmp (sau cea marcată „worldwide” dintr-o listă) se păstrează în moneda ei: `budget_wp_amount`, `budget_wp_currency`; `budget_wp` e suma în USD scrisă în infobox (ea însăși sau cea din paranteză: `£800,000 ($1.4 million)`). Simbolurile ambigue (`¥`, `kr`, `Rs`, `$` la un film fără SUA dintr-o țară cu dolar propriu) se lămuresc după țara de producție TMDB; altfel `budget_wp_note = ambiguous` și suma se ignoră. `has_budget_elsewhere = False` marchează bugetele noi (fără buget USD din altă sursă). Intervalele (`$10–15 million`) dau mijlocul, cu `budget_wp_range = True`. `distributor_wp` e distribuitorul principal (cel marcat SUA/internațional, altfel primul din listă).

## Premii ale regizorului și actorilor (#35)

```bash
python extraction/awards.py --test   # 5 filme cunoscute, doar afișare
python extraction/awards.py          # toate filmele din manifest (~5 min fără cache)
```

`awards.parquet`, un rând pentru fiecare film din manifest și pentru filmele cu buget doar în altă monedă (persoanele care apar doar în acestea nu sunt căutate în Wikidata: `*_not_looked_up`): nominalizări și premii Oscar, Globul de Aur și BAFTA ale regizorului (`director_*`) și ale primilor 3 actori (`cast_*`), numărate **doar înainte de lansare**. Persoanele din TMDB sunt găsite în Wikidata prin P4985 (TMDb person ID). O dată cunoscută doar la nivel de an e tratată ca 31 decembrie (anti-leakage). `*_found_wd` / `*_n` = câte persoane au fost găsite în Wikidata din câte are filmul.

## Cursuri de schimb și date de lansare (#45)

```bash
python extraction/fx_rates.py                 # World Bank PA.NUS.FCRF -> data/interim/fx_rates.parquet
python extraction/tmdb.py --release-dates     # din cache-ul TMDB -> data/interim/tmdb_release_dates.parquet
```

`fx_rates.parquet`: unități de monedă locală pentru 1 USD, media anului, 1960–2024 ([World Bank, CC BY 4.0](https://data.worldbank.org/indicator/PA.NUS.FCRF)). Zona euro: moneda națională (DEM, FRF, ITL ...) până la euro, EUR din 1999. `hyperinflation = True` când cursul s-a schimbat cu peste 50% față de anul anterior; acei ani nu se folosesc. `tmdb_release_dates.parquet`: prima lansare în cinema (tipurile 2–3) și prima lansare digitală (tipul 4) per film, citite din răspunsurile TMDB deja descărcate.

## Integrare și curățare (#27, #45)

```bash
python extraction/integrate.py                              # prag profitable = 2.5
python extraction/integrate.py --threshold 3                # alt prag
python extraction/integrate.py --roi-rule vechi             # regula veche pentru ROI extrem
python extraction/integrate.py --budget-choice prioritate   # bugetul doar după prioritate
python extraction/sensitivity.py                            # -> docs/sensitivity_report.md
```

Opțiuni:

| Opțiune | Valori | Ce face |
|---|---|---|
| `--threshold` | număr, implicit `2.5` | pragul ROI pentru `profitable` |
| `--roi-rule` | `roi-recalculat` (implicit), `surse-independente`, `vechi` | când se păstrează un ROI extrem (sub 0,001 sau peste 100). `roi-recalculat`: ROI-ul rămâne extrem și cu bugetul dintr-o sursă independentă (alt grup decât sursa aleasă), și cu încasările dintr-o sursă independentă, cu toleranța de ±25%. `surse-independente`: 2 surse confirmă bugetul și 2 încasările, DBpedia și Wikipedia contând o singură dată. `vechi`: 2 surse oarecare (comportamentul de la #27) |
| `--budget-choice` | `majoritate` (implicit), `prioritate` | `majoritate`: din fiecare grup independent (tmdb, wikidata, wiki = dbpedia/wikipedia) intră o valoare; câștigă cea cu care sunt de acord cele mai multe grupuri, la egalitate ordinea TMDB → DBpedia → Wikidata → Wikipedia. `prioritate`: doar ordinea. Încasările se aleg mereu după ordinea Wikipedia → DBpedia → Wikidata → TMDB (TMDB raportează sistematic mai puțin) |

`sensitivity.py` refolosește aceleași funcții și schimbă pe rând câte o decizie (alegerea bugetului, a încasărilor, `--roi-rule`, pragul 2 / 2,5 / 3), cu numărul de filme, % profitabile și câte etichete se schimbă.

Citește doar `data/interim/` (fără cereri HTTP, câteva secunde) și scrie `data/processed/movies_dataset.csv` (un rând per film) și [docs/integration_report.md](docs/integration_report.md): pâlnia de curățare (inclusiv filmele fără buget și cele de streaming), monedele, profilul filmelor scoase, marile studiouri și de ce excludem coloanele META.

Filmele de streaming cu o lansare doar simbolică în cinema ies din populație (pasul „streaming”): distribuitor de streaming sau lansare digitală la cel mult 30 de zile după cinema, **și** încasări sub 5% din buget.

**Ce intră în model:** `extraction/columns.py` împarte coloanele în TARGETS, IDS, META, LEAKAGE și FEATURES; `feature_columns(df)` dă caracteristicile, iar `python extraction/test_columns.py` verifică că nicio coloană META / LEAKAGE / TARGETS nu e printre ele și că fiecare coloană are exact o categorie. Rulat de două ori, produce același fișier.

### Dicționar de date

**Identificare (nu intră în model):** `imdb_id`, `tmdb_id`, `title`.

**Ținta (nu intră în model):**

| Coloană | Descriere |
|---|---|
| `budget` | buget final în USD (convertit); majoritate între sursele independente, apoi prioritatea TMDB → DBpedia → Wikidata → Wikipedia (e și caracteristică de intrare) |
| `worldwide_gross` | încasări mondiale în USD (convertite); prioritate Wikipedia → DBpedia → Wikidata → TMDB |
| `roi` | `worldwide_gross / budget`; direct din sumele native când bugetul și încasările sunt în aceeași monedă (`roi_currency`) |
| `profitable` | 1 dacă `roi >= prag` (implicit 2.5) |

**Caracteristici (cunoscute înainte de lansare):**

| Coloană | Descriere |
|---|---|
| `director_award_{noms,wins}_before`, `cast_award_{noms,wins}_before` | nominalizări / premii Oscar + Globul de Aur + BAFTA ale regizorului și primilor 3 actori, strict înainte de lansare |
| `major_studio` | 1 dacă o companie de producție e Disney, Warner Bros., Universal, Paramount, Sony/Columbia, 20th Century sau Lionsgate (inclusiv diviziile de film, după ID-ul TMDB) |
| `major_distributor` | 1 dacă distribuitorul principal din infobox e unul dintre aceleași studiouri sau o divizie (Buena Vista, New Line, Focus Features, TriStar, Summit ...), după nume |
| `major_any` | `major_studio` sau `major_distributor` |
| `release_date`, `release_year`, `release_month`, `release_weekday` | data lansării din TMDB; `release_weekday` 0 = luni |
| `genres`, `genre_*` | genurile TMDB (separate prin `\|`) și one-hot |
| `runtime_min` | durata în minute (gol = necunoscută) |
| `certification` | G=0, PG=1, PG-13=2, R=3, NC-17=4 (gol = lipsă / NR) |
| `kw_*` | one-hot pentru cele mai frecvente 30 de keywords TMDB din dataset, fără etichetele de ton (`amused`, `suspenseful`, … = reacția publicului, leakage) și fără cele care se află abia după lansare (`duringcreditsstinger`, `aftercreditsstinger`, `controversy` …, #47). Atenție: filmele populare au mai multe keywords, deci `kw_*` poartă și un semnal de popularitate (vezi raportul) |
| `is_english`, `original_language` | limba originală: `en`, top 5 alte limbi sau `other` |
| `country_{us,europe,asia,other}` | grupul țărilor de producție (un film poate avea mai multe) |
| `is_franchise`, `sequel_number` | continuare: au apărut deja filme din aceeași colecție TMDB; câte (0 = primul film al seriei sau film independent). Primul film al unei serii nu contează ca franciză: colecția apare abia după ce succesul lui aduce o continuare (leakage, #47) |
| `is_adaptation`, `adaptation_type` | ecranizare (Wikidata P144, keywords „based on …” sau `based on` din infobox); tipul sursei din Wikidata, `other` dacă e ecranizare fără tip cunoscut, `none` altfel |
| `director_prev_films` | nr. de filme anterioare din dataset ale regizorului, lansate cu cel puțin 6 luni înainte (încasările lor erau deja cunoscute, #49) |
| `director_prev_mean_roi` | ROI-ul mediu al acelor filme (gol = fără istoric); sensibil la un singur hit extrem |
| `director_prev_mean_log_roi` | media lui ln(ROI) a acelor filme; variantă robustă (0 = buget recuperat exact) |
| `director_prev_profitable_rate` | fracția acelor filme cu ROI ≥ prag; variantă robustă |
| `cast_prev_*` | aceleași patru coloane pentru primii 3 actori (maxim de filme; celelalte calculate per actor și mediate) |

**Surse, monede și calitatea datelor (META, nu intră în model):** `budget_source` / `gross_source` (sursa valorii finale), `*_n_sources` (câte surse au valoare), `*_n_agree` (câte surse independente sunt la ±25% de valoarea finală; DBpedia și Wikipedia contează o dată), `*_conflict` (max/min > 1,25), valorile brute în USD `budget_tmdb`, `budget_dbp`, `budget_wd`, `budget_wp`, `gross_wp`, `worldwide_gross_dbp`, `gross_wd`, `revenue_tmdb`, sumele în moneda lor `budget_<src>_amount` / `budget_<src>_currency` și `gross_<src>_amount` / `gross_<src>_currency` (`src` = tmdb, dbp, wd, wp), `roi_currency`, `fx_year`, `budget_fx_rate`, `gross_fx_rate` (cursul folosit; 1 = USD, gol = suma în USD din infobox), `budget_is_range` (bugetul final e mijlocul unui interval din infobox), `gross_is_domestic` / `gross_is_rental` (încasările alese sunt doar SUA/Canada, respectiv „rentals”: ROI subestimat, mai ales la filmele vechi, #47), `gross_is_single_market` (eticheta din infobox arată o singură piață, orice țară) și `gross_is_local_currency` (încasări în altă monedă, fără „worldwide”: de obicei doar cele de acasă; la filmele indiene, de obicei mondiale, #49). Ele prezic ținta (vezi raportul), deci ar fi leakage.

**Doar pentru experimentul leakage vs. no-leakage:** `vote_average`, `vote_count`.

## Surse excluse

| Sursă | Motiv |
|---|---|
| The Numbers | lista de bugete nu mai e publică; Terms of Service interzic colectarea automată pentru seturi de date ML |
| Box Office Mojo (#6) | `robots.txt`: `Disallow: /` pentru orice crawler nelistat; Conditions of Use IMDb interzic scraping-ul fără permisiune scrisă |

Valorile lor ajung oricum indirect în date: infobox-urile Wikipedia le citează ca sursă.

## Teste

```bash
python extraction/test_utils.py       # cache și 404
python extraction/test_wikipedia.py   # sumele din infobox, în orice monedă
python extraction/test_columns.py     # coloanele excluse din model (după integrate.py)
```

`test_utils.py` verifică că al doilea apel pe același URL vine din cache (fără request HTTP) și că un 404 returnează `None` fără excepție. `test_wikipedia.py` verifică parserul de sume (`A$2,417,000` → AUD, `{{INR|55 crore}}` → 5,5·10⁸ INR, `£800,000 ($1.4 million)` → GBP + USD scris explicit, `¥` rezolvat după țară).
