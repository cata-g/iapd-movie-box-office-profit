# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold 2.5`.

**Rezultat:** 15.972 filme în `data/processed/movies_dataset.csv`.

## Ce filme am scos

| Pas | Ce s-a scos | Scoase | Rămân |
|---|---|---|---|
| start | toate filmele din manifest | 0 | 27.657 |
| buget | niciun buget între 10.000 $ și 500 mil. $ (bugete prea mici ignorate: tmdb 1.056) | 972 | 26.685 |
| încasări | nicio sursă nu are încasări mondiale | 10.378 | 16.307 |
| dată lansare | fără dată de lansare în TMDB | 156 | 16.151 |
| interval ani | lansate înainte de 1950 sau după 2024 | 63 | 16.088 |
| film TV | filme TV (n-au rulat în cinema) | 24 | 16.064 |
| scurtmetraj | scurtmetraje (sub 40 min) | 22 | 16.042 |
| ROI extrem | ROI sub 0,001 sau peste 100 pe care nu-l confirmă 2 surse (probabil o greșeală) | 70 | 15.972 |

## Ținta

Un film e **profitabil** dacă încasările sunt de cel puțin 2,5 ori bugetul (`roi = încasări / buget`).

| | Filme | % |
|---|---|---|
| profitabile | 5.140 | 32,2% |
| neprofitabile | 10.832 | 67,8% |

Clasele sunt dezechilibrate: un model care spune mereu „neprofitabil” are dreptate în 67,8% din cazuri.

## De unde vin bugetul și încasările

Când mai multe surse au o valoare, o luăm pe cea din sursa mai sigură:
- buget: tmdb → dbpedia → wikidata → wikipedia
- încasări: wikipedia → dbpedia → wikidata → tmdb

| Sursă | Buget | Încasări |
|---|---|---|
| tmdb | 12.283 | 2.085 |
| dbpedia | 3.252 | 472 |
| wikidata | 138 | 189 |
| wikipedia | 299 | 13.226 |

Box Office Mojo și The Numbers nu sunt folosite: interzic colectarea automată (vezi README).

### Cât de sigure sunt valorile

Câte surse au o valoare pentru fiecare film:

**Buget**

| Surse | Filme | % |
|---|---|---|
| 1 | 3.934 | 24,6% |
| 2 | 3.516 | 22,0% |
| 3 | 5.641 | 35,3% |
| 4 | 2.881 | 18,0% |

**Încasări**

| Surse | Filme | % |
|---|---|---|
| 1 | 2.698 | 16,9% |
| 2 | 3.480 | 21,8% |
| 3 | 7.106 | 44,5% |
| 4 | 2.688 | 16,8% |

Sursele nu se pun mereu de acord: la 1.175 filme (7,4%) bugetele diferă cu peste 25%, iar la 1.738 (10,9%) încasările.

Atenție: DBpedia copiază datele din Wikipedia, deci „2 surse” poate fi de fapt una singură.

## Date lipsă

| Coloană | Lipsesc | % |
|---|---|---|
| `runtime_min` | 19 | 0,1% |
| `certification` | 5.655 | 35,4% |
| `sequel_number` | 12 | 0,1% |
| `director_prev_mean_roi` | 6.693 | 41,9% |
| `cast_prev_mean_roi` | 2.510 | 15,7% |

Istoricul regizorului și al actorilor lipsește când e primul lor film din dataset.

## Istoricul ROI: de ce nu ajunge media

Câteva filme au un ROI uriaș (Paranormal Activity: 12.947), deși la jumătate dintre filme ROI-ul e sub 1,31. Un singur astfel de hit strică media pentru toată cariera unui regizor. De aceea, pe lângă media cerută în issue, am adăugat:

- `*_prev_profitable_rate`: ce parte din filmele anterioare au fost profitabile;
- `*_prev_mean_log_roi`: media pe logaritm, unde un hit contează mult mai puțin.

| Coloană | Corelație cu `profitable` |
|---|---|
| `director_prev_mean_roi` | 0,003 |
| `director_prev_mean_log_roi` | 0,194 |
| `director_prev_profitable_rate` | 0,187 |
| `cast_prev_mean_roi` | 0,034 |
| `cast_prev_mean_log_roi` | 0,173 |
| `cast_prev_profitable_rate` | 0,164 |

Media simplă aproape nu are legătură cu ținta; variantele noi au.

## Fără informații de după lansare (anti-leakage)

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**.
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- `vote_average` și `vote_count` rămân în tabel doar pentru experimentul cu leakage.
- La antrenare, e mai sigur să împărțim filmele după an (vechi = antrenare, noi = test). La o împărțire aleatoare, istoricul unui film vechi ar putea folosi ROI-ul unui film din test.
