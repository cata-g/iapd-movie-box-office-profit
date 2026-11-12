# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold 2.5 --roi-rule roi-recalculat --budget-choice majoritate`.

**Rezultat:** 18.056 filme în `data/processed/movies_dataset.parquet`.

## Ce filme am scos

| Pas | Ce s-a scos | Scoase | Rămân |
|---|---|---|---|
| start | filme 1950–2024 cu buget sau încasări în cel puțin o sursă | 0 | 47.568 |
| fără buget | au încasări, dar nicio sursă nu are buget (nu intră în manifest; fără buget nu există ROI) | 14.419 | 33.149 |
| buget | niciun buget între 10.000 $ și 500 mil. $ după conversia în USD (bugete în afara intervalului, ignorate: tmdb 1.056, dbp 26, wd 426, wp 420; fără curs: 335) | 1.961 | 31.188 |
| încasări | nicio sursă nu are încasări mondiale (după conversia în USD; fără curs: 96) | 12.200 | 18.988 |
| valori identice | aceeași sursă dă exact aceeași sumă pentru buget și încasări (valoare de umplutură, ROI = 1) | 58 | 18.930 |
| dată lansare | fără dată de lansare în TMDB | 680 | 18.250 |
| interval ani | lansate înainte de 1950 sau după 2024 | 61 | 18.189 |
| film TV | filme TV (n-au rulat în cinema) | 26 | 18.163 |
| scurtmetraj | scurtmetraje (sub 40 min) | 20 | 18.143 |
| ROI extrem | ROI sub 0,001 sau peste 100 care nu rămâne extrem când bugetul, apoi încasările sunt înlocuite cu valoarea unei surse independente (probabil o greșeală) | 87 | 18.056 |

## Filme de streaming: o caracteristică, nu un filtru

Până la #52, pasul „streaming” scotea un film dacă avea un semnal de streaming (distribuitor de streaming sau lansare digitală TMDB la cel mult 30 de zile după cinema) **și** ROI < 0,05. Filtrul avea două defecte, așa că nu mai filtrăm:

1. **Alegea filmele după rezultat.** Condiția ROI < 0,05 este chiar ținta pe care modelul trebuie s-o prezică: dintre filmele cu semnal de streaming ieșeau doar cele cu cel mai prost rezultat, ceea ce umfla artificial rata de filme profitabile.
2. **Criteriul digital nu detecta filmele de streaming.** Prindea 2.262 filme (12,5% din filmele rămase în acel punct), printre ele Interstellar, Deadpool, Avengers: Age of Ultron, Black Panther, iar rata lor de profitabile (32,4%) era practic cea a tuturor filmelor (32,0%). Condiția pe ROI făcea aproape toată filtrarea: 162 din cele 199 filme scoase erau prinse doar de criteriul digital.

Acum distribuitorul de streaming e o **caracteristică**, `streaming_distributor`: 1 dacă distribuitorul principal din infobox e Netflix, Amazon Studios / Prime Video, Apple TV+, Disney+, Hulu, HBO Max, Max, Paramount+ sau Peacock. Se știe înainte de lansare. 95 filme au `streaming_distributor` = 1, cu 5,3% profitabile, față de 32,0% la restul.

Dintre cele 199 filme pe care le scotea filtrul vechi, 193 sunt din nou în dataset, toate neprofitabile (firesc: aveau ROI < 0,05); de aceea rata de profitabile scade puțin. Celelalte 6 ies în continuare la pașii următori: 6 la „ROI extrem” (ROI sub 0,001 pe care nu-l confirmă o sursă independentă): The Last Rifleman, Society of the Snow, Line of Descent, Hillbilly Elegy, Semper Fi, The Fanatic. `extraction/sensitivity.py` arată ținta și cu filtrul vechi.

## Cine lipsește: filmele fără încasări

Cea mai mare pierdere din pâlnie e pasul „încasări”: filme cu buget, dar fără încasări mondiale în nicio sursă. Nu sunt o mostră la întâmplare:

| | Scoase la „încasări” | Dataset final |
|---|---|---|
| Filme | 12.200 | 18.056 |
| Voturi TMDB (mediană) | 18 | 192 |
| Buget (mediană) | 0,8 mil. $ | 6,0 mil. $ |
| Producții SUA | 40,8% | 56,9% |
| În limba engleză | 60,6% | 65,1% |
| 1950–1959 | 2,6% | 4,3% |
| 1960–1969 | 5,1% | 3,3% |
| 1970–1979 | 7,9% | 4,8% |
| 1980–1989 | 7,7% | 9,8% |
| 1990–1999 | 8,0% | 13,4% |
| 2000–2009 | 22,8% | 24,0% |
| 2010–2019 | 37,2% | 31,2% |
| 2020–2024 | 8,5% | 9,1% |

Filmele scoase sunt mult mai mici și mai puțin cunoscute (mai puține voturi, buget mai mic, mai puține producții americane). Încasările lor nu apar nicăieri tocmai pentru că au fost mici, deci ele sunt, foarte probabil, mai des neprofitabile. Rata de 31,9% filme profitabile e deci **supraestimată** față de toate filmele lansate în cinema: descrie filmele destul de cunoscute ca să le găsim încasările, iar modelul va învăța pe această populație.

## Calitatea țintei: încasări parțiale

Ținta presupune încasări **mondiale**. La unele filme, singura cifră din infobox e alta (coloane META, nu intră în model):
- `gross_is_single_market` = 1: încasările unei singure piețe, după eticheta din infobox („$12 million (U.S.)”, „£49,981 (UK)”, „A$3.2 million (Australia)”): 1.420 filme, 22,0% profitabile. `gross_is_domestic` = 1 e cazul SUA/Canada (1.039 filme).
- `gross_is_rental` = 1: „rentals”, adică partea care revine distribuitorului, cam jumătate din încasări: 456 filme, 26,5% profitabile.
- `gross_is_local_currency` = 1 (mai slab): încasări în altă monedă decât USD, fără etichetă de piață și fără „worldwide”. De obicei sunt doar cele de acasă, dar nu e sigur: 2.807 filme, 35,0% profitabile.

La aceste filme, ROI-ul e **subestimat**, deci unele sunt etichetate greșit ca neprofitabile. Încasările parțiale (o singură piață sau rentals) sunt concentrate în filmele vechi:

| Deceniu | Filme | O singură piață | Rentals | % din deceniu | % profitabile (cu ele / fără ele) |
|---|---|---|---|---|---|
| 1950–1959 | 774 | 223 | 192 | 33,5% | 33,6% / 35,9% |
| 1960–1969 | 592 | 200 | 141 | 40,5% | 42,2% / 54,3% |
| 1970–1979 | 860 | 187 | 109 | 26,6% | 56,9% / 63,7% |
| 1980–1989 | 1.777 | 329 | 12 | 19,1% | 33,4% / 36,0% |
| 1990–1999 | 2.424 | 285 | 0 | 11,8% | 30,0% / 32,8% |
| 2000–2009 | 4.341 | 77 | 0 | 1,8% | 28,0% / 28,2% |
| 2010–2019 | 5.636 | 105 | 2 | 1,9% | 32,4% / 32,5% |
| 2020–2024 | 1.652 | 14 | 0 | 0,8% | 23,9% / 23,7% |

Sumele în moneda locală afectează filmele din afara SUA. Fără o corecție, un model ar învăța că filmele non-americane eșuează, iar asta ar veni din date, nu din realitate. Tabelul arată filmele și % profitabile pentru fiecare categorie (procentul apare de la 30 de filme în sus):

| Moneda încasărilor | o singură piață / rentals | monedă locală fără „worldwide” | restul |
|---|---|---|---|
| USD | 1.161 filme, 24,1% | 0 filme | 13.676 filme, 32,1% |
| INR | 17 filme | 1.569 filme, 43,3% | 7 filme |
| alte monede | 372 filme, 21,5% | 1.234 filme, 24,3% | 20 filme |

Filmele cu încasări în alte monede (fără USD și INR) sunt profitabile mult mai rar decât filmele în USD fără semnale, ceea ce confirmă că suma lor e de obicei doar cea de acasă. Filmele indiene (INR) merg în sens invers. Infobox-urile lor dau de regulă încasările mondiale, adesea anunțate de producători și cu taxe incluse, deci probabil umflate. Același semnal (monedă locală fără „worldwide”) are așadar sensuri diferite pentru cele două grupuri.

`extraction/sensitivity.py` arată ținta și fără aceste filme. Pentru antrenare, varianta cea mai sigură e să le scoatem sau să le dăm o pondere mai mică, iar concluziile despre țară sau limbă trebuie verificate și fără ele.

## Ținta

Un film e **profitabil** dacă încasările sunt de cel puțin 2,5 ori bugetul (`roi = încasări / buget`).

| | Filme | % |
|---|---|---|
| profitabile | 5.752 | 31,9% |
| neprofitabile | 12.304 | 68,1% |

Clasele sunt dezechilibrate: un model care spune mereu „neprofitabil” are dreptate în 68,1% din cazuri.

## De unde vin bugetul și încasările

Când mai multe surse au o valoare, o luăm pe cea din sursa mai sigură:
- buget: tmdb → dbpedia → wikidata → wikipedia, dar prin **majoritate**: din fiecare grup independent (tmdb, wikidata, wiki = dbpedia/wikipedia) intră o singură valoare, a sursei celei mai prioritare din grup. Câștigă valoarea cu care sunt de acord (la ±25%) cele mai multe grupuri; la egalitate rămâne ordinea de mai sus. Așa, o greșeală izolată a TMDB nu mai câștigă doar pentru că TMDB e primul (Dragon Ball Super: Broly: TMDB 1 mil. $, Wikidata și Wikipedia 8,5 mil. $). Când toate cele 3 grupuri se contrazic (niciunul la ±25% de altul), câștigă valoarea din mijloc, nu sursa prioritară (#52): 24 filme au bugetul ales prin mediană (3 Idiots: TMDB 50 mil. $, Wikipedia 11,4 mil. $, Wikidata 7,2 mil. $ → 11,4 mil. $). Limită: Wikidata are exact aceeași valoare ca Wikipedia la 70,7% din bugete, deci și „majoritatea” poate fi uneori aceeași informație copiată.
- încasări: wikipedia → dbpedia → wikidata → tmdb (doar ordinea; TMDB raportează sistematic încasări mai mici, de aceea e ultima)

| Sursă | Buget | Încasări |
|---|---|---|
| tmdb | 12.126 | 1.438 |
| dbpedia | 3.407 | 48 |
| wikidata | 493 | 248 |
| wikipedia | 2.030 | 16.322 |

Box Office Mojo și The Numbers nu sunt folosite: interzic colectarea automată (vezi README).

### Cât de sigure sunt valorile

Câte surse au o valoare pentru fiecare film:

**Buget**

| Surse | Filme | % |
|---|---|---|
| 1 | 4.370 | 24,2% |
| 2 | 4.630 | 25,6% |
| 3 | 6.006 | 33,3% |
| 4 | 3.050 | 16,9% |

**Încasări**

| Surse | Filme | % |
|---|---|---|
| 1 | 3.271 | 18,1% |
| 2 | 4.625 | 25,6% |
| 3 | 7.375 | 40,8% |
| 4 | 2.785 | 15,4% |

Sursele nu se pun mereu de acord: la 1.845 filme (10,2%) bugetele diferă cu peste 25%, iar la 2.351 (13,0%) încasările.

DBpedia copiază infobox-ul Wikipedia (valori identice la 96,6% din bugete și 91,5% din încasări), deci nu e o sursă independentă. De aceea `budget_n_agree` și `gross_n_agree` (câte surse confirmă valoarea aleasă, la ±25%) le numără ca **o singură sursă**: tmdb, wikidata și wiki (dbpedia sau wikipedia) — maximum 3.

## Monede

Bugetul și încasările se iau în moneda în care sunt scrise (`budget_<sursă>_amount`, `budget_<sursă>_currency`, la fel pentru încasări) și se convertesc în USD cu cursul mediu al anului lansării (World Bank, indicatorul `PA.NUS.FCRF`, licență CC BY 4.0; `fx_year`, `budget_fx_rate`, `gross_fx_rate`). Abia apoi se aplică regulile de mai sus: intervalele plauzibile, alegerea prin majoritate, toleranța de ±25% și regula pentru ROI extrem.

- 3.819 filme (21,2%) au bugetul sau încasările dintr-o sumă în altă monedă; doar 2,4% dintre ele sunt (co)producții americane. La 144 dintre ele, suma în USD vine chiar din infobox („£800,000 ($1.4 million)”) și are prioritate față de conversia noastră.
- Când bugetul și încasările alese sunt în aceeași monedă, ROI-ul se calculează direct din sumele native (un raport nu are nevoie de curs): 1.832 filme. `roi_currency` arată moneda.

**Moneda în care s-a calculat ROI-ul**

| Monedă | Filme | % |
|---|---|---|
| USD | 16.224 | 89,9% |
| INR | 1.030 | 5,7% |
| AUD | 184 | 1,0% |
| GBP | 95 | 0,5% |
| CAD | 94 | 0,5% |
| EUR | 70 | 0,4% |
| BRL | 55 | 0,3% |
| HKD | 40 | 0,2% |
| RUB | 40 | 0,2% |
| JPY | 32 | 0,2% |
| altele | 192 | 1,1% |

**Sume care nu s-au putut converti** (filme scoase la pașii „buget” / „încasări” doar din acest motiv)

| Motiv | Buget | Încasări |
|---|---|---|
| fără curs înainte de 1960 | 204 | 51 |
| hiperinflație (curs schimbat cu peste 50% față de anul anterior) | 16 | 6 |
| monedă sau an fără curs World Bank | 115 | 39 |

Seria World Bank începe în 1960: pentru filmele din 1950–1959 nu folosim alt curs (nu inventăm valori), deci filmele din anii '50 cu sume doar în altă monedă se pierd. Monedele redenominate (Brazilia înainte de 1994, Turcia înainte de 2005) apar în serie în unitatea de azi; anii lor au aproape toți hiperinflație, iar restul cad la verificarea intervalelor plauzibile.

- DBpedia marchează orice sumă ca USD, deși uneori a citit „A$2,417,000” sau „MXN$142.3 million”. Când suma DBpedia e chiar prima sumă din infobox-ul Wikipedia, iar aceea e în altă monedă, DBpedia primește moneda din infobox: 1.457 bugete și 856 încasări.
- Virgula zecimală europeană (#52): „€13,7 million” înseamnă 13,7 milioane, nu 137. O virgulă urmată de 1–2 cifre și de o unitate e zecimală; urmată de 3 cifre rămâne separator de mii („1,250 million”). În dataset, 26 bugete și 22 încasări din infobox sunt citite acum corect (32 dintre ele sunt chiar valoarea folosită). DBpedia nu are suma greșită la niciun film (de obicei nu are deloc aceste sume); dacă ar avea-o, ar primi valoarea corectată din infobox.
- Simboluri ambigue („¥” = JPY sau CNY, „kr” = SEK / NOK / DKK, „Rs”, „$”) se lămuresc după țara de producție din TMDB. Dacă tot rămân ambigue, suma e ignorată: 37 bugete și 17 încasări din Wikipedia.
- Un „$” simplu înseamnă USD, cu o excepție: un film fără SUA printre țările de producție, dar dintr-o țară cu dolar propriu (Australia, Canada, Hong Kong, Noua Zeelandă, Mexic ...), primește moneda acelei țări. Limită: Wikidata are aceeași sumă marcată USD la multe dintre aceste filme, dar și acolo valorile sunt adesea importate automat din infobox, deci nu e o confirmare independentă.
- Limită: cursul e media anului lansării; bugetul se cheltuie de obicei cu 1–2 ani înainte, iar încasările vin în anul lansării și după.

## Mare studio: producție sau distribuție

| Coloană | Ce înseamnă | Filme cu 1 | % profitabile (1) | % profitabile (0) |
|---|---|---|---|---|
| `major_studio` | producție: o companie TMDB e un mare studio | 4.374 | 41,3% | 28,8% |
| `major_distributor` | distribuitorul principal din infobox e un mare studio | 6.508 | 39,1% | 27,8% |
| `major_any` | oricare din cele două | 6.738 | 38,8% | 27,7% |

Față de `major_studio`, 2.364 filme trec în clasa „mare studio” doar prin distribuitor (de exemplu filme independente cumpărate de Universal sau Sony Pictures Classics), iar 230 au un mare studio printre producători, dar alt distribuitor principal. `major_any` le cuprinde pe toate. Contează doar distribuitorul principal: lista completă din infobox poate avea distribuitori adăugați după lansare (reluări, alte piețe).

`budget_is_range` = 1 la 899 filme (5,0%): bugetul final e mijlocul unui interval din infobox („$10–15 million”). E o coloană despre calitatea datelor, nu o caracteristică pentru model.

## De ce excludem coloanele META

`extraction/columns.py` împarte coloanele în TARGETS, IDS, META, LEAKAGE și FEATURES; în model intră doar cele 50 din FEATURES (`feature_columns(df)`), iar `extraction/test_columns.py` verifică asta pe dataset-ul final. META sunt cele 41 coloane despre **date**, nu despre film: sursa valorii, câte surse au o valoare și câte sunt de acord, sumele brute, monedele și cursurile. Ele prezic ținta, pentru că un film cu încasări raportate peste tot e de obicei un film cunoscut, care a mers bine:

**% profitabile după numărul de surse cu încasări**

| Surse cu încasări | Filme | % profitabile |
|---|---|---|
| 1 | 3.271 | 26,8% |
| 2 | 4.625 | 23,4% |
| 3 | 7.375 | 29,7% |
| 4 | 2.785 | 57,4% |

**% profitabile după sursa bugetului ales**

| Sursa bugetului | Filme | % profitabile |
|---|---|---|
| dbpedia | 3.407 | 19,7% |
| tmdb | 12.126 | 35,6% |
| wikidata | 493 | 26,2% |
| wikipedia | 2.030 | 31,1% |

**% profitabile după sursa încasărilor alese**

| Sursa încasărilor | Filme | % profitabile |
|---|---|---|
| dbpedia | 48 | 33,3% |
| tmdb | 1.438 | 19,1% |
| wikidata | 248 | 21,4% |
| wikipedia | 16.322 | 33,1% |

**Corelația coloanelor META numerice cu `profitable`**

| Coloană | Corelație cu `profitable` |
|---|---|
| `gross_wd` | 0,372 |
| `worldwide_gross_dbp` | 0,352 |
| `gross_dbp_amount` | 0,351 |
| `revenue_tmdb` | 0,328 |
| `gross_tmdb_amount` | 0,328 |
| `gross_wp` | 0,319 |
| `gross_n_agree` | 0,224 |
| `budget_n_agree` | 0,190 |
| `gross_n_sources` | 0,178 |
| `budget_n_sources` | 0,162 |
| `gross_wd_amount` | 0,089 |
| `budget_wd` | 0,085 |
| `fx_year` | -0,079 |
| `budget_wp` | 0,065 |
| `gross_is_single_market` | -0,062 |
| `budget_dbp` | 0,061 |
| `budget_is_range` | 0,057 |
| `budget_dbp_amount` | 0,055 |
| `gross_is_domestic` | -0,048 |
| `budget_tmdb_amount` | 0,047 |
| `budget_tmdb` | 0,046 |
| `budget_conflict` | 0,044 |
| `gross_fx_rate` | 0,041 |
| `gross_conflict` | 0,031 |
| `gross_is_local_currency` | 0,029 |
| `gross_wp_amount` | 0,024 |
| `budget_fx_rate` | 0,021 |
| `gross_is_rental` | -0,018 |
| `budget_wp_amount` | 0,005 |
| `budget_wd_amount` | -0,003 |

La predicție, pentru un film care nu a fost încă lansat, nu știm câte surse îi vor raporta încasările. Un model care le folosește ar învăța „filmul e documentat bine”, nu „filmul va fi profitabil”.

## Date lipsă

| Coloană | Lipsesc | % |
|---|---|---|
| `runtime_min` | 33 | 0,2% |
| `certification` | 7.507 | 41,6% |
| `sequel_number` | 13 | 0,1% |
| `director_prev_mean_roi` | 7.576 | 42,0% |
| `cast_prev_mean_roi` | 2.939 | 16,3% |

Istoricul regizorului și al actorilor lipsește când e primul lor film din dataset.

## Istoricul ROI: de ce nu ajunge media

Câteva filme au un ROI uriaș (Paranormal Activity: 12.947), deși la jumătate dintre filme ROI-ul e sub 1,33. Un singur astfel de hit strică media pentru toată cariera unui regizor. De aceea, pe lângă media cerută în issue, am adăugat:

- `*_prev_profitable_rate`: ce parte din filmele anterioare au fost profitabile;
- `*_prev_mean_log_roi`: media pe logaritm, unde un hit contează mult mai puțin.

| Coloană | Corelație cu `profitable` |
|---|---|
| `director_prev_mean_roi` | 0,003 |
| `director_prev_mean_log_roi` | 0,202 |
| `director_prev_profitable_rate` | 0,188 |
| `cast_prev_mean_roi` | 0,034 |
| `cast_prev_mean_log_roi` | 0,189 |
| `cast_prev_profitable_rate` | 0,180 |

Media simplă aproape nu are legătură cu ținta; variantele noi au.

## Limite de care trebuie să țină cont modelarea

- **Selecție după popularitate.** Lista de bază din TMDB cere cel puțin 5 voturi, adunate după lansare: 17.819 din 18.056 filme au intrat așa. Dataset-ul conține deci filme care „au fost observate”, iar profitabilitatea crește cu popularitatea: 22,1% profitabile sub 20 de voturi, 54,9% peste 1.000. Fără descărcări noi nu se poate corecta. Concluziile sunt valabile pentru filmele cunoscute, nu pentru toată producția.
- **COVID (2020–2021).** Piața cinematografelor s-a prăbușit: 17,9% profitabile în 2020–2021, față de 33,8% în 2017–2019 și 26,9% în 2022–2024. La o împărțire după an, setul de test ar conține tocmai acești ani atipici. Recomandare: un indicator pentru 2020–2021 sau acești ani raportați separat.

## Fără informații de după lansare (anti-leakage)

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**. Pentru istoricul ROI, filmul anterior trebuie lansat cu cel puțin 6 luni înainte (#49): încasările unui film abia lansat nu sunt încă cunoscute. Înainte de această regulă, 489 filme foloseau ROI-ul unui film al regizorului lansat cu mai puțin de 6 luni în urmă.
- **Franciza** (#47): colecțiile TMDB se creează după ce apare o continuare, iar continuarea se face de obicei pentru că primul film a avut succes. Cu vechea definiție (`is_franchise` = filmul e într-o colecție), primul film al unei serii ar fi purtat informație din viitor:

  | Vechea definiție | Filme | % profitabile |
  |---|---|---|
  | nu e în nicio colecție | 14.774 | 25,8% |
  | primul film dintr-o colecție | 1.659 | 65,3% |
  | continuare (au apărut deja filme din serie) | 1.610 | 52,9% |

  Acum `is_franchise` = 1 doar pentru continuări, iar `sequel_number` = câte filme din serie au apărut înainte (0 = primul film sau film independent).
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- **Keywords de după lansare** (#47): `duringcreditsstinger` / `aftercreditsstinger` (scenă după generic) se află abia la vizionare și anunțau succesul (779 filme, 54,7% profitabile). Le-am scos, împreună cu `controversy`, `banned film` și altele asemănătoare.
- **Keywords excluse din model** (#54): keywords sunt adăugate de utilizatorii TMDB, mai ales după lansare, iar filmele populare primesc mai multe: numărul de keywords se corelează 0,62 cu log(voturi) și 0,22 cu `profitable`, iar cele 2.613 filme fără niciun keyword au doar 23,6% profitabile. Chiar dacă fiecare keyword descrie conținutul, un vector `kw_*` plin de zerouri spune „film puțin cunoscut”. De aceea coloanele `kw_*` sunt în categoria LEAKAGE din `extraction/columns.py`, lângă `vote_average` și `vote_count`: rămân în tabel doar pentru experimentul leakage vs. no-leakage și nu intră în `feature_columns(df)`.
- `vote_average`, `vote_count` și `kw_*` rămân în tabel doar pentru experimentul cu leakage.
- La antrenare, e mai sigur să împărțim filmele după an (vechi = antrenare, noi = test). La o împărțire aleatoare, istoricul unui film vechi ar putea folosi ROI-ul unui film din test.
