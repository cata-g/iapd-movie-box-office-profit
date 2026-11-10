# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold 2.5 --roi-rule roi-recalculat --budget-choice majoritate`.

**Rezultat:** 15.960 filme în `data/processed/movies_dataset.csv`.

## Ce filme am scos

| Pas | Ce s-a scos | Scoase | Rămân |
|---|---|---|---|
| start | filme 1950–2024 cu buget sau încasări în cel puțin o sursă | 0 | 40.517 |
| fără buget | au încasări, dar nicio sursă nu are buget (nu intră în manifest; fără buget nu există ROI) | 12.860 | 27.657 |
| buget | niciun buget între 10.000 $ și 500 mil. $ (bugete prea mici ignorate: tmdb 1.056) | 972 | 26.685 |
| încasări | nicio sursă nu are încasări mondiale | 10.378 | 16.307 |
| dată lansare | fără dată de lansare în TMDB | 156 | 16.151 |
| interval ani | lansate înainte de 1950 sau după 2024 | 63 | 16.088 |
| film TV | filme TV (n-au rulat în cinema) | 24 | 16.064 |
| scurtmetraj | scurtmetraje (sub 40 min) | 22 | 16.042 |
| ROI extrem | ROI sub 0,001 sau peste 100 care nu rămâne extrem când bugetul, apoi încasările sunt înlocuite cu valoarea unei surse independente (probabil o greșeală) | 82 | 15.960 |

## Ținta

Un film e **profitabil** dacă încasările sunt de cel puțin 2,5 ori bugetul (`roi = încasări / buget`).

| | Filme | % |
|---|---|---|
| profitabile | 5.135 | 32,2% |
| neprofitabile | 10.825 | 67,8% |

Clasele sunt dezechilibrate: un model care spune mereu „neprofitabil” are dreptate în 67,8% din cazuri.

## De unde vin bugetul și încasările

Când mai multe surse au o valoare, o luăm pe cea din sursa mai sigură:
- buget: tmdb → dbpedia → wikidata → wikipedia, dar prin **majoritate**: din fiecare grup independent (tmdb, wikidata, wiki = dbpedia/wikipedia) intră o singură valoare, a sursei celei mai prioritare din grup. Câștigă valoarea cu care sunt de acord (la ±25%) cele mai multe grupuri; la egalitate rămâne ordinea de mai sus. Așa, o greșeală izolată a TMDB nu mai câștigă doar pentru că TMDB e primul (Dragon Ball Super: Broly: TMDB 1 mil. $, Wikidata și Wikipedia 8,5 mil. $). Limită: Wikidata are exact aceeași valoare ca Wikipedia la 76,0% din bugete, deci și „majoritatea” poate fi uneori aceeași informație copiată.
- încasări: wikipedia → dbpedia → wikidata → tmdb (doar ordinea; TMDB raportează sistematic încasări mai mici, de aceea e ultima)

| Sursă | Buget | Încasări |
|---|---|---|
| tmdb | 12.107 | 2.085 |
| dbpedia | 3.370 | 473 |
| wikidata | 165 | 189 |
| wikipedia | 318 | 13.213 |

Box Office Mojo și The Numbers nu sunt folosite: interzic colectarea automată (vezi README).

### Cât de sigure sunt valorile

Câte surse au o valoare pentru fiecare film:

**Buget**

| Surse | Filme | % |
|---|---|---|
| 1 | 3.934 | 24,6% |
| 2 | 3.504 | 22,0% |
| 3 | 5.639 | 35,3% |
| 4 | 2.883 | 18,1% |

**Încasări**

| Surse | Filme | % |
|---|---|---|
| 1 | 2.698 | 16,9% |
| 2 | 3.464 | 21,7% |
| 3 | 7.106 | 44,5% |
| 4 | 2.692 | 16,9% |

Sursele nu se pun mereu de acord: la 1.181 filme (7,4%) bugetele diferă cu peste 25%, iar la 1.739 (10,9%) încasările.

DBpedia copiază infobox-ul Wikipedia (valori identice la 97,4% din bugete și 91,8% din încasări), deci nu e o sursă independentă. De aceea `budget_n_agree` și `gross_n_agree` (câte surse confirmă valoarea aleasă, la ±25%) le numără ca **o singură sursă**: tmdb, wikidata și wiki (dbpedia sau wikipedia) — maximum 3.

## Date lipsă

| Coloană | Lipsesc | % |
|---|---|---|
| `runtime_min` | 19 | 0,1% |
| `certification` | 5.647 | 35,4% |
| `sequel_number` | 13 | 0,1% |
| `director_prev_mean_roi` | 6.690 | 41,9% |
| `cast_prev_mean_roi` | 2.509 | 15,7% |

Istoricul regizorului și al actorilor lipsește când e primul lor film din dataset.

## Istoricul ROI: de ce nu ajunge media

Câteva filme au un ROI uriaș (Paranormal Activity: 12.947), deși la jumătate dintre filme ROI-ul e sub 1,32. Un singur astfel de hit strică media pentru toată cariera unui regizor. De aceea, pe lângă media cerută în issue, am adăugat:

- `*_prev_profitable_rate`: ce parte din filmele anterioare au fost profitabile;
- `*_prev_mean_log_roi`: media pe logaritm, unde un hit contează mult mai puțin.

| Coloană | Corelație cu `profitable` |
|---|---|
| `director_prev_mean_roi` | 0,002 |
| `director_prev_mean_log_roi` | 0,192 |
| `director_prev_profitable_rate` | 0,185 |
| `cast_prev_mean_roi` | 0,034 |
| `cast_prev_mean_log_roi` | 0,174 |
| `cast_prev_profitable_rate` | 0,163 |

Media simplă aproape nu are legătură cu ținta; variantele noi au.

## Fără informații de după lansare (anti-leakage)

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**.
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- `vote_average` și `vote_count` rămân în tabel doar pentru experimentul cu leakage.
- La antrenare, e mai sigur să împărțim filmele după an (vechi = antrenare, noi = test). La o împărțire aleatoare, istoricul unui film vechi ar putea folosi ROI-ul unui film din test.
