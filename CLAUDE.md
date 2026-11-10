# IAPD – Movie Box Office Profit Prediction

## Scopul proiectului
Predicția profitabilității filmelor lansate în cinema folosind **doar date disponibile înainte de lansare**.
- Ținta binară: `profitable = (worldwide_gross / budget) >= 2.5` (prag configurabil)
- Ținta continuă: `roi = worldwide_gross / budget`
- Repo: https://github.com/cata-g/iapd-movie-box-office-profit

## Faza 1 – Dataset propriu (17.911 filme; pragul cerut era 10.000)
Dataset cu buget ȘI încasări mondiale (convertite în USD) + 18 caracteristici fără leakage. Cifrele curente: `docs/integration_report.md`.
Dataseturile gata (Kaggle etc.) **NU sunt permise** — totul se extrage de noi.

### Ordinea issue-urilor
1. **#1** Setup repo + utilitare comune ← *ești aici*
2. **#2** TMDB, **#3** DBpedia, **#4** Wikidata + Wikipedia (independente)
3. **#5** Manifest + decizia la 10.000 (postează numărătoarea ca comentariu, oprește-te dacă < 10k)
4. ~~**#6** Box Office Mojo~~ — închis, nu se folosește (vezi mai jos)
5. **#27** Integrare și curățare → `data/processed/movies_dataset.parquet`

### Surse pentru buget (resursă rară — reuniune)
TMDB → DBpedia → Wikidata → Wikipedia

### Surse pentru încasări mondiale
Wikipedia → DBpedia → Wikidata → TMDB

> The Numbers a fost scos (nov. 2026): lista `/movie/budgets` nu mai e publică, iar Terms of Service interzic colectarea automată pentru seturi de date ML fără acord scris.
>
> Box Office Mojo a fost scos (nov. 2026): `robots.txt` are `User-agent: * / Disallow: /`, iar Conditions of Use IMDb interzic colectarea automată fără permisiune scrisă. Încasările vin din cele 4 surse deschise; infobox-urile Wikipedia citează deja BOM / The Numbers. Caracteristica `opening_theaters` cade (nu există în nicio sursă deschisă).
>
> Regulă: o sursă care interzice colectarea automată (ToS sau robots.txt) nu intră în proiect, iar blocările nu se ocolesc (fără proxy/browser headless).

## Structura repo
```
extraction/        # un script per sursă + utils.py
data/raw/<sursa>/  # cache brut HTML/JSON
data/interim/      # tabel curat per sursă + manifest
data/processed/    # dataset final
docs/
logs/
```

## Reguli tehnice
- **Python 3.11+**. Dependențe în `requirements.txt`.
- **Toate requesturile HTTP** trec prin `extraction/utils.py`:
  - Cache pe disc în `data/raw/<sursa>/` (sha256 al url+params)
  - Rate limit: TMDB 20 req/s, DBpedia 1 req/s, Wikipedia 5 req/s
  - Retry 3× cu backoff exponențial pe 429/5xx; 404 → `None` fără retry
  - User-Agent: `iapd-student-project (contact: barabas.catalin@gmail.com)`
  - Logging în `logs/<sursa>.log`
- **Crawlerele pot fi oprite și repornite** fără să descarce din nou (cache + status manifest)
- **`imdb_id`** e cheia de unire între toate sursele
- Sumele se citesc **în moneda în care sunt scrise** (suma + cod ISO per sursă: `budget_<src>_amount`, `budget_<src>_currency`) și se convertesc în USD cu cursul mediu World Bank (`PA.NUS.FCRF`, `data/interim/fx_rates.parquet`) din anul lansării; abia apoi se compară sursele. O sumă în USD scrisă explicit în infobox are prioritate față de conversie; dacă bugetul și încasările alese sunt în aceeași monedă, ROI-ul se calculează direct din sumele native. Fără curs (înainte de 1960, monedă fără serie) sau în ani de hiperinflație (> 50% față de anul anterior), valoarea lipsește și se numără în raport; simbolurile ambigue (`¥`, `kr`, `$` ...) se rezolvă după țara de producție TMDB sau se ignoră, nu se ghicesc
- Valorile brute per sursă se păstrează (`budget_tmdb`, `budget_tn`, ...); valoarea finală și sursa ei sunt coloane separate (`budget`, `budget_source`)
- Cheia TMDB în `.env` (`TMDB_API_KEY`), niciodată în git

## Anti-leakage (important pentru notă!)
- **Fără** rating, voturi, popularitate sau încasări ca features de intrare
- Istoricul regizorului/actorilor: calculat **doar din filmele lansate înainte** de filmul curent
- `vote_average` și `vote_count` se păstrează **separat**, doar pentru experimentul leakage vs. no-leakage

## Cum lucrăm
- Un branch per issue: `issue-<nr>-<slug>`
- Un PR per issue cu `Closes #<nr>` în descriere; **nu push direct pe `main`**
- Înainte de PR: verifică fiecare bifă din „Gata când" a issue-ului
- Când un issue cere numărătoare/raport: `gh issue comment <nr> --body ...`
- Testează orice crawler pe 3–5 filme cunoscute înainte de rulare completă
  - Inception: `tt1375666`
  - Paranormal Activity: `tt1179904`

## Instalare
```bash
pip install -r requirements.txt
cp .env.example .env
# Editează .env și pune TMDB_API_KEY
```

## Test cache (issue #1)
```bash
python extraction/test_utils.py
```
