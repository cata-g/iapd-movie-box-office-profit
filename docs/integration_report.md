# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold 2.5 --roi-rule roi-recalculat --budget-choice majoritate`.

**Rezultat:** 17.546 filme în `data/processed/movies_dataset.csv`.

## Ce filme am scos

| Pas | Ce s-a scos | Scoase | Rămân |
|---|---|---|---|
| start | filme 1950–2024 cu buget sau încasări în cel puțin o sursă | 0 | 47.565 |
| fără buget | au încasări, dar nicio sursă nu are buget (nu intră în manifest; fără buget nu există ROI) | 14.420 | 33.145 |
| buget | niciun buget în USD între 10.000 $ și 500 mil. $ și niciun buget în altă monedă (bugete în USD în afara intervalului, ignorate: tmdb 1.056, wd 224, wp 237) | 1.285 | 31.860 |
| încasări | nicio sursă nu are încasări mondiale (în nicio monedă) | 12.681 | 19.179 |
| monede diferite | bugetul și încasările sunt în monede diferite (fără curs de schimb nu se pot compara) | 724 | 18.455 |
| dată lansare | fără dată de lansare în TMDB | 684 | 17.771 |
| interval ani | lansate înainte de 1950 sau după 2024 | 63 | 17.708 |
| film TV | filme TV (n-au rulat în cinema) | 24 | 17.684 |
| scurtmetraj | scurtmetraje (sub 40 min) | 24 | 17.660 |
| ROI extrem | ROI sub 0,001 sau peste 100 care nu rămâne extrem când bugetul, apoi încasările sunt înlocuite cu valoarea unei surse independente (probabil o greșeală) | 114 | 17.546 |

## Ținta

Un film e **profitabil** dacă încasările sunt de cel puțin 2,5 ori bugetul (`roi = încasări / buget`).

| | Filme | % |
|---|---|---|
| profitabile | 5.689 | 32,4% |
| neprofitabile | 11.857 | 67,6% |

Clasele sunt dezechilibrate: un model care spune mereu „neprofitabil” are dreptate în 67,6% din cazuri.

## De unde vin bugetul și încasările

Când mai multe surse au o valoare, o luăm pe cea din sursa mai sigură:
- buget: tmdb → dbpedia → wikidata → wikipedia, dar prin **majoritate**: din fiecare grup independent (tmdb, wikidata, wiki = dbpedia/wikipedia) intră o singură valoare, a sursei celei mai prioritare din grup. Câștigă valoarea cu care sunt de acord (la ±25%) cele mai multe grupuri; la egalitate rămâne ordinea de mai sus. Așa, o greșeală izolată a TMDB nu mai câștigă doar pentru că TMDB e primul (Dragon Ball Super: Broly: TMDB 1 mil. $, Wikidata și Wikipedia 8,5 mil. $). Limită: Wikidata are exact aceeași valoare ca Wikipedia la 75,9% din bugete, deci și „majoritatea” poate fi uneori aceeași informație copiată.
- încasări: wikipedia → dbpedia → wikidata → tmdb (doar ordinea; TMDB raportează sistematic încasări mai mici, de aceea e ultima)

| Sursă | Buget | Încasări |
|---|---|---|
| tmdb | 12.211 | 2.446 |
| dbpedia | 2.942 | 149 |
| wikidata | 191 | 242 |
| wikipedia | 361 | 12.884 |

Box Office Mojo și The Numbers nu sunt folosite: interzic colectarea automată (vezi README).

### Cât de sigure sunt valorile

Câte surse au o valoare pentru fiecare film:

**Buget**

| Surse | Filme | % |
|---|---|---|
| 0 | 1.841 | 10,5% |
| 1 | 4.012 | 22,9% |
| 2 | 3.324 | 18,9% |
| 3 | 5.518 | 31,4% |
| 4 | 2.851 | 16,2% |

**Încasări**

| Surse | Filme | % |
|---|---|---|
| 0 | 1.825 | 10,4% |
| 1 | 2.891 | 16,5% |
| 2 | 3.278 | 18,7% |
| 3 | 6.905 | 39,4% |
| 4 | 2.647 | 15,1% |

Sursele nu se pun mereu de acord: la 1.111 filme (6,3%) bugetele diferă cu peste 25%, iar la 1.618 (9,2%) încasările.

DBpedia copiază infobox-ul Wikipedia (valori identice la 97,4% din bugete și 91,9% din încasări), deci nu e o sursă independentă. De aceea `budget_n_agree` și `gross_n_agree` (câte surse confirmă valoarea aleasă, la ±25%) le numără ca **o singură sursă**: tmdb, wikidata și wiki (dbpedia sau wikipedia) — maximum 3.

## Monede

Bugetul și încasările se iau în moneda în care sunt scrise (`budget_<sursă>_amount`, `budget_<sursă>_currency`, la fel pentru încasări). Când bugetul și încasările unui film sunt în aceeași monedă, ROI-ul (un raport) nu are nevoie de curs de schimb: `roi_currency` arată moneda în care s-a calculat.

| Monedă | Filme | % |
|---|---|---|
| USD | 15.499 | 88,3% |
| INR | 1.082 | 6,2% |
| AUD | 221 | 1,3% |
| GBP | 127 | 0,7% |
| CAD | 111 | 0,6% |
| EUR | 94 | 0,5% |
| BRL | 55 | 0,3% |
| RUB | 47 | 0,3% |
| HKD | 43 | 0,2% |
| JPY | 33 | 0,2% |
| altele | 234 | 1,3% |

- 2.047 filme au ROI-ul calculat în altă monedă decât USD; doar 1,1% dintre ele sunt (co)producții americane.
- DBpedia marchează orice sumă ca USD, deși uneori a citit „A$2,417,000” sau „MXN$142.3 million”. Când suma DBpedia e chiar prima sumă din infobox-ul Wikipedia, iar aceea e în altă monedă, DBpedia primește moneda din infobox: 1.456 bugete și 855 încasări (dintre ele, în dataset: 718 bugete).
- Simboluri ambigue („¥” = JPY sau CNY, „kr” = SEK / NOK / DKK, „Rs”, „$”) se lămuresc după țara de producție din TMDB. Dacă tot rămân ambigue, suma e ignorată: 37 bugete și 17 încasări din Wikipedia.
- Un „$” simplu înseamnă USD, cu o excepție: un film fără SUA printre țările de producție, dar dintr-o țară cu dolar propriu (Australia, Canada, Hong Kong, Noua Zeelandă, Mexic ...), primește moneda acelei țări. Limită: Wikidata are aceeași sumă marcată USD la multe dintre aceste filme, dar și acolo valorile sunt adesea importate automat din infobox, deci nu e o confirmare independentă.

## Date lipsă

| Coloană | Lipsesc | % |
|---|---|---|
| `runtime_min` | 33 | 0,2% |
| `certification` | 7.122 | 40,6% |
| `sequel_number` | 13 | 0,1% |
| `director_prev_mean_roi` | 7.318 | 41,7% |
| `cast_prev_mean_roi` | 2.733 | 15,6% |

Istoricul regizorului și al actorilor lipsește când e primul lor film din dataset.

## Istoricul ROI: de ce nu ajunge media

Câteva filme au un ROI uriaș (Paranormal Activity: 12.947), deși la jumătate dintre filme ROI-ul e sub 1,36. Un singur astfel de hit strică media pentru toată cariera unui regizor. De aceea, pe lângă media cerută în issue, am adăugat:

- `*_prev_profitable_rate`: ce parte din filmele anterioare au fost profitabile;
- `*_prev_mean_log_roi`: media pe logaritm, unde un hit contează mult mai puțin.

| Coloană | Corelație cu `profitable` |
|---|---|
| `director_prev_mean_roi` | 0,002 |
| `director_prev_mean_log_roi` | 0,190 |
| `director_prev_profitable_rate` | 0,179 |
| `cast_prev_mean_roi` | 0,033 |
| `cast_prev_mean_log_roi` | 0,180 |
| `cast_prev_profitable_rate` | 0,173 |

Media simplă aproape nu are legătură cu ținta; variantele noi au.

## Fără informații de după lansare (anti-leakage)

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**.
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- `vote_average` și `vote_count` rămân în tabel doar pentru experimentul cu leakage.
- La antrenare, e mai sigur să împărțim filmele după an (vechi = antrenare, noi = test). La o împărțire aleatoare, istoricul unui film vechi ar putea folosi ROI-ul unui film din test.
