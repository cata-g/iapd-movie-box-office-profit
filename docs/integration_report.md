# Raport de integrare (#27)

Generat de `python extraction/integrate.py --threshold 2.5`.

**Rezultat:** 15,972 filme în `data/processed/movies_dataset.csv`, 99 coloane.

## Pâlnia de curățare

| pas | regula | eliminate | raman |
|---|---|---|---|
| manifest | filme cu buget > 0 în cel puțin o sursă, cu imdb_id | 0 | 27,657 |
| buget | fără buget plauzibil (10,000–500,000,000 USD) în nicio sursă; valori per sursă invalidate: `budget_tmdb` 1,056, `budget_dbp` 0, `budget_wd` 0, `budget_wp` 0 | 972 | 26,685 |
| încasări | fără încasări mondiale plauzibile (1,000–3,000,000,000 USD) în nicio sursă | 10,378 | 16,307 |
| dată lansare | fără dată completă de lansare în TMDB (necesară pentru lună, zi și istoricul anterior) | 156 | 16,151 |
| interval ani | lansat în afara 1950–2024 | 63 | 16,088 |
| film TV | gen TMDB „TV Movie” (fără lansare în cinema) | 24 | 16,064 |
| scurtmetraj | durată între 1 și 39 min | 22 | 16,042 |
| ROI extrem | ROI < 0.001 sau > 100 fără confirmare (≥ 2 surse care dau, la ±25%, aceeași valoare pentru buget și pentru încasări) | 70 | 15,972 |

Ordinea de prioritate:
- buget: tmdb → dbpedia → wikidata → wikipedia;
- încasări mondiale: wikipedia → dbpedia → wikidata → tmdb.

Box Office Mojo și The Numbers sunt excluse (vezi README, „Surse excluse”).

## Ținta

`roi = worldwide_gross / budget`, `profitable = roi >= 2.5`.

| Clasă | Filme | % |
|---|---|---|
| profitable = 1 | 5,140 | 32.2 |
| profitable = 0 | 10,832 | 67.8 |

ROI median: 1.31.

## Conflicte între surse

Conflict = valoarea maximă depășește valoarea minimă cu peste 25%.

| | Filme cu conflict | % |
|---|---|---|
| buget | 1,175 | 7.4 |
| încasări | 1,738 | 10.9 |

### Câte surse confirmă bugetul

| valoare | filme | % |
|---|---|---|
| 1 | 3,934 | 24.6 |
| 2 | 3,516 | 22.0 |
| 3 | 5,641 | 35.3 |
| 4 | 2,881 | 18.0 |

### Câte surse confirmă încasările

| valoare | filme | % |
|---|---|---|
| 1 | 2,698 | 16.9 |
| 2 | 3,480 | 21.8 |
| 3 | 7,106 | 44.5 |
| 4 | 2,688 | 16.8 |

DBpedia și Wikipedia citesc același infobox, deci nu sunt surse independente. Pentru robustețe,
modelul se poate evalua separat doar pe filmele cu `gross_n_sources >= 2`
(13,274 filme).

### Sursa valorii finale

| sursă | buget | încasări |
|---|---|---|
| tmdb | 12,283 | 2,085 |
| dbpedia | 3,252 | 472 |
| wikidata | 138 | 189 |
| wikipedia | 299 | 13,226 |

## Valori lipsă în caracteristici

| coloană | lipsă | % |
|---|---|---|
| `runtime_min` | 19 | 0.1 |
| `certification` | 5,655 | 35.4 |
| `sequel_number` | 12 | 0.1 |
| `director_prev_mean_roi` | 6,693 | 41.9 |
| `cast_prev_mean_roi` | 2,510 | 15.7 |

`director_prev_mean_roi` și `cast_prev_mean_roi` lipsesc când persoana nu are niciun film
anterior în dataset (debut sau filme fără buget/încasări cunoscute).

## Anti-leakage

- Premiile și istoricul regizorului/actorilor folosesc doar filme lansate **strict înainte**.
- `vote_average` și `vote_count` sunt păstrate doar pentru experimentul leakage vs. no-leakage.
- Istoricul ROI e calculat pe tot datasetul: la o împărțire train/test aleatoare, un film din
  train poate folosi ROI-ul unui film anterior din test. Împărțirea temporală (după an) evită asta.
