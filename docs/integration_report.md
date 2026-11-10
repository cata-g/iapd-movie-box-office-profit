# Raport de integrare (#27)

Generat cu `python extraction/integrate.py --threshold 2.5 --roi-rule roi-recalculat --budget-choice majoritate`.

**Rezultat:** 17.861 filme în `data/processed/movies_dataset.csv`.

## Ce filme am scos

| Pas | Ce s-a scos | Scoase | Rămân |
|---|---|---|---|
| start | filme 1950–2024 cu buget sau încasări în cel puțin o sursă | 0 | 47.565 |
| fără buget | au încasări, dar nicio sursă nu are buget (nu intră în manifest; fără buget nu există ROI) | 14.420 | 33.145 |
| buget | niciun buget între 10.000 $ și 500 mil. $ după conversia în USD (bugete în afara intervalului, ignorate: tmdb 1.056, dbp 26, wd 426, wp 418; fără curs: 334) | 1.961 | 31.184 |
| încasări | nicio sursă nu are încasări mondiale (după conversia în USD; fără curs: 95) | 12.199 | 18.985 |
| valori identice | aceeași sursă dă exact aceeași sumă pentru buget și încasări (valoare de umplutură, ROI = 1) | 58 | 18.927 |
| dată lansare | fără dată de lansare în TMDB | 680 | 18.247 |
| interval ani | lansate înainte de 1950 sau după 2024 | 61 | 18.186 |
| film TV | filme TV (n-au rulat în cinema) | 25 | 18.161 |
| streaming | lansare de streaming (distribuitor Netflix, Amazon, Apple TV+, Disney+ ... sau lansare digitală la cel mult 30 de zile după cinema) cu încasări sub 5% din buget: lansarea în cinema a fost doar simbolică | 199 | 17.962 |
| scurtmetraj | scurtmetraje (sub 40 min) | 20 | 17.942 |
| ROI extrem | ROI sub 0,001 sau peste 100 care nu rămâne extrem când bugetul, apoi încasările sunt înlocuite cu valoarea unei surse independente (probabil o greșeală) | 81 | 17.861 |

## Filme de streaming

Populația proiectului sunt filmele lansate în cinema. Unele filme de streaming au avut doar o lansare simbolică în câteva săli (Hillbilly Elegy: buget 45 mil. $, încasări 38.852 $) și ar apărea ca „neprofitabile” fără să fi încercat să recupereze bugetul din bilete. Pasul „streaming” scoate un film doar dacă **ambele** condiții sunt adevărate:

1. distribuitorul principal din infobox e un serviciu de streaming (Netflix, Amazon Studios / Prime Video, Apple TV+, Disney+, Hulu, HBO Max, Max, Paramount+, Peacock) **sau** TMDB are o lansare digitală (tipul 4) la cel mult 30 de zile după prima lansare în cinema (tipurile 2 și 3);
2. încasările sunt sub 5% din buget (ROI < 0,05).

Criteriul digital se aplică doar filmelor lansate din 2005: înainte nu exista distribuție digitală, iar TMDB are uneori casete sau DVD-uri marcate ca „digital” chiar la data premierei (Eureka, 1983). O lansare digitală *înainte* de cinema (VOD mai întâi, apoi câteva săli) contează și ea.

A doua condiție păstrează filmele de streaming care au rulat cu adevărat în cinema (Manchester by the Sea, distribuit de Amazon Studios).

| Criteriul îndeplinit | Filme scoase |
|---|---|
| doar distribuitorul | 4 |
| doar lansarea digitală | 162 |
| amândouă | 33 |
| **total** | **199** |

Primele 20 (după data lansării): D.E.B.S., Edmond, Pistol Whipped, Sleep Dealer, I Come with the Rain, The Yellow Handkerchief, Girl Clock!, The Chaperone, The Magic of Belle Isle, Spiders, Blinder, Struck by Lightning, The Congress, The Last Days on Mars, A Thousand Times Good Night, Lost Place, A Touch of Sin, Jealousy, The Decent One, Better Living Through Chemistry.

Datele de lansare pe tipuri vin din răspunsurile TMDB deja descărcate (`python extraction/tmdb.py --release-dates`). Lipsesc pentru 242 filme (detaliile TMDB adăugate la manifest nu sunt în arhiva din Releases); la ele contează doar distribuitorul.

## Cine lipsește: filmele fără încasări

Cea mai mare pierdere din pâlnie e pasul „încasări”: filme cu buget, dar fără încasări mondiale în nicio sursă. Nu sunt o mostră la întâmplare:

| | Scoase la „încasări” | Dataset final |
|---|---|---|
| Filme | 12.199 | 17.861 |
| Voturi TMDB (mediană) | 18 | 190 |
| Buget (mediană) | 0,8 mil. $ | 6,0 mil. $ |
| Producții SUA | 40,8% | 56,7% |
| În limba engleză | 60,6% | 64,9% |
| 1950–1959 | 2,6% | 4,3% |
| 1960–1969 | 5,1% | 3,3% |
| 1970–1979 | 7,9% | 4,8% |
| 1980–1989 | 7,7% | 9,9% |
| 1990–1999 | 8,0% | 13,6% |
| 2000–2009 | 22,8% | 24,3% |
| 2010–2019 | 37,2% | 31,0% |
| 2020–2024 | 8,5% | 8,7% |

Filmele scoase sunt mult mai mici și mai puțin cunoscute (mai puține voturi, buget mai mic, mai puține producții americane). Încasările lor nu apar nicăieri tocmai pentru că au fost mici, deci ele sunt, foarte probabil, mai des neprofitabile. Rata de 32,3% filme profitabile e deci **supraestimată** față de toate filmele lansate în cinema: descrie filmele destul de cunoscute ca să le găsim încasările, iar modelul va învăța pe această populație.

## Calitatea țintei: încasări parțiale

Ținta presupune încasări **mondiale**. La unele filme, singura cifră din infobox e alta (coloane META, nu intră în model):
- `gross_is_single_market` = 1: încasările unei singure piețe, după eticheta din infobox („$12 million (U.S.)”, „£49,981 (UK)”, „A$3.2 million (Australia)”): 1.414 filme, 22,1% profitabile. `gross_is_domestic` = 1 e cazul SUA/Canada (1.036 filme).
- `gross_is_rental` = 1: „rentals”, adică partea care revine distribuitorului, cam jumătate din încasări: 456 filme, 26,5% profitabile.
- `gross_is_local_currency` = 1 (mai slab): încasări în altă monedă decât USD, fără etichetă de piață și fără „worldwide”. De obicei sunt doar cele de acasă, dar nu e sigur: 2.796 filme, 35,3% profitabile.

La aceste filme, ROI-ul e **subestimat**, deci unele sunt etichetate greșit ca neprofitabile. Încasările parțiale (o singură piață sau rentals) sunt concentrate în filmele vechi:

| Deceniu | Filme | O singură piață | Rentals | % din deceniu | % profitabile (cu ele / fără ele) |
|---|---|---|---|---|---|
| 1950–1959 | 774 | 223 | 192 | 33,5% | 33,6% / 35,9% |
| 1960–1969 | 592 | 200 | 141 | 40,5% | 42,2% / 54,3% |
| 1970–1979 | 860 | 187 | 109 | 26,6% | 56,9% / 63,7% |
| 1980–1989 | 1.777 | 329 | 12 | 19,1% | 33,4% / 36,0% |
| 1990–1999 | 2.424 | 285 | 0 | 11,8% | 30,0% / 32,8% |
| 2000–2009 | 4.335 | 76 | 0 | 1,8% | 28,0% / 28,3% |
| 2010–2019 | 5.539 | 101 | 2 | 1,8% | 33,1% / 33,2% |
| 2020–2024 | 1.560 | 13 | 0 | 0,8% | 25,5% / 25,3% |

Sumele în moneda locală afectează filmele din afara SUA. Fără o corecție, un model ar învăța că filmele non-americane eșuează, iar asta ar veni din date, nu din realitate. Tabelul arată filmele și % profitabile pentru fiecare categorie (procentul apare de la 30 de filme în sus):

| Moneda încasărilor | o singură piață / rentals | monedă locală fără „worldwide” | restul |
|---|---|---|---|
| USD | 1.156 filme, 24,2% | 0 filme | 13.498 filme, 32,6% |
| INR | 17 filme | 1.568 filme, 43,2% | 7 filme |
| alte monede | 371 filme, 21,6% | 1.224 filme, 24,9% | 20 filme |

Filmele cu încasări în alte monede (fără USD și INR) sunt profitabile mult mai rar decât filmele în USD fără semnale, ceea ce confirmă că suma lor e de obicei doar cea de acasă. Filmele indiene (INR) merg în sens invers. Infobox-urile lor dau de regulă încasările mondiale, adesea anunțate de producători și cu taxe incluse, deci probabil umflate. Același semnal (monedă locală fără „worldwide”) are așadar sensuri diferite pentru cele două grupuri.

`extraction/sensitivity.py` arată ținta și fără aceste filme. Pentru antrenare, varianta cea mai sigură e să le scoatem sau să le dăm o pondere mai mică, iar concluziile despre țară sau limbă trebuie verificate și fără ele.

## Ținta

Un film e **profitabil** dacă încasările sunt de cel puțin 2,5 ori bugetul (`roi = încasări / buget`).

| | Filme | % |
|---|---|---|
| profitabile | 5.764 | 32,3% |
| neprofitabile | 12.097 | 67,7% |

Clasele sunt dezechilibrate: un model care spune mereu „neprofitabil” are dreptate în 67,7% din cazuri.

## De unde vin bugetul și încasările

Când mai multe surse au o valoare, o luăm pe cea din sursa mai sigură:
- buget: tmdb → dbpedia → wikidata → wikipedia, dar prin **majoritate**: din fiecare grup independent (tmdb, wikidata, wiki = dbpedia/wikipedia) intră o singură valoare, a sursei celei mai prioritare din grup. Câștigă valoarea cu care sunt de acord (la ±25%) cele mai multe grupuri; la egalitate rămâne ordinea de mai sus. Așa, o greșeală izolată a TMDB nu mai câștigă doar pentru că TMDB e primul (Dragon Ball Super: Broly: TMDB 1 mil. $, Wikidata și Wikipedia 8,5 mil. $). Limită: Wikidata are exact aceeași valoare ca Wikipedia la 70,5% din bugete, deci și „majoritatea” poate fi uneori aceeași informație copiată.
- încasări: wikipedia → dbpedia → wikidata → tmdb (doar ordinea; TMDB raportează sistematic încasări mai mici, de aceea e ultima)

| Sursă | Buget | Încasări |
|---|---|---|
| tmdb | 11.995 | 1.405 |
| dbpedia | 3.372 | 53 |
| wikidata | 482 | 249 |
| wikipedia | 2.012 | 16.154 |

Box Office Mojo și The Numbers nu sunt folosite: interzic colectarea automată (vezi README).

### Cât de sigure sunt valorile

Câte surse au o valoare pentru fiecare film:

**Buget**

| Surse | Filme | % |
|---|---|---|
| 1 | 4.303 | 24,1% |
| 2 | 4.579 | 25,6% |
| 3 | 5.943 | 33,3% |
| 4 | 3.036 | 17,0% |

**Încasări**

| Surse | Filme | % |
|---|---|---|
| 1 | 3.238 | 18,1% |
| 2 | 4.564 | 25,6% |
| 3 | 7.283 | 40,8% |
| 4 | 2.776 | 15,5% |

Sursele nu se pun mereu de acord: la 1.842 filme (10,3%) bugetele diferă cu peste 25%, iar la 2.322 (13,0%) încasările.

DBpedia copiază infobox-ul Wikipedia (valori identice la 96,5% din bugete și 91,4% din încasări), deci nu e o sursă independentă. De aceea `budget_n_agree` și `gross_n_agree` (câte surse confirmă valoarea aleasă, la ±25%) le numără ca **o singură sursă**: tmdb, wikidata și wiki (dbpedia sau wikipedia) — maximum 3.

## Monede

Bugetul și încasările se iau în moneda în care sunt scrise (`budget_<sursă>_amount`, `budget_<sursă>_currency`, la fel pentru încasări) și se convertesc în USD cu cursul mediu al anului lansării (World Bank, indicatorul `PA.NUS.FCRF`, licență CC BY 4.0; `fx_year`, `budget_fx_rate`, `gross_fx_rate`). Abia apoi se aplică regulile de mai sus: intervalele plauzibile, alegerea prin majoritate, toleranța de ±25% și regula pentru ROI extrem.

- 3.795 filme (21,2%) au bugetul sau încasările dintr-o sumă în altă monedă; doar 2,4% dintre ele sunt (co)producții americane. La 142 dintre ele, suma în USD vine chiar din infobox („£800,000 ($1.4 million)”) și are prioritate față de conversia noastră.
- Când bugetul și încasările alese sunt în aceeași monedă, ROI-ul se calculează direct din sumele native (un raport nu are nevoie de curs): 1.825 filme. `roi_currency` arată moneda.

**Moneda în care s-a calculat ROI-ul**

| Monedă | Filme | % |
|---|---|---|
| USD | 16.036 | 89,8% |
| INR | 1.027 | 5,7% |
| AUD | 182 | 1,0% |
| GBP | 95 | 0,5% |
| CAD | 93 | 0,5% |
| EUR | 70 | 0,4% |
| BRL | 55 | 0,3% |
| HKD | 40 | 0,2% |
| RUB | 40 | 0,2% |
| JPY | 32 | 0,2% |
| altele | 191 | 1,1% |

**Sume care nu s-au putut converti** (filme scoase la pașii „buget” / „încasări” doar din acest motiv)

| Motiv | Buget | Încasări |
|---|---|---|
| fără curs înainte de 1960 | 204 | 51 |
| hiperinflație (curs schimbat cu peste 50% față de anul anterior) | 15 | 5 |
| monedă sau an fără curs World Bank | 115 | 39 |

Seria World Bank începe în 1960: pentru filmele din 1950–1959 nu folosim alt curs (nu inventăm valori), deci filmele din anii '50 cu sume doar în altă monedă se pierd. Monedele redenominate (Brazilia înainte de 1994, Turcia înainte de 2005) apar în serie în unitatea de azi; anii lor au aproape toți hiperinflație, iar restul cad la verificarea intervalelor plauzibile.

- DBpedia marchează orice sumă ca USD, deși uneori a citit „A$2,417,000” sau „MXN$142.3 million”. Când suma DBpedia e chiar prima sumă din infobox-ul Wikipedia, iar aceea e în altă monedă, DBpedia primește moneda din infobox: 1.456 bugete și 855 încasări.
- Simboluri ambigue („¥” = JPY sau CNY, „kr” = SEK / NOK / DKK, „Rs”, „$”) se lămuresc după țara de producție din TMDB. Dacă tot rămân ambigue, suma e ignorată: 37 bugete și 17 încasări din Wikipedia.
- Un „$” simplu înseamnă USD, cu o excepție: un film fără SUA printre țările de producție, dar dintr-o țară cu dolar propriu (Australia, Canada, Hong Kong, Noua Zeelandă, Mexic ...), primește moneda acelei țări. Limită: Wikidata are aceeași sumă marcată USD la multe dintre aceste filme, dar și acolo valorile sunt adesea importate automat din infobox, deci nu e o confirmare independentă.
- Limită: cursul e media anului lansării; bugetul se cheltuie de obicei cu 1–2 ani înainte, iar încasările vin în anul lansării și după.

## Mare studio: producție sau distribuție

| Coloană | Ce înseamnă | Filme cu 1 | % profitabile (1) | % profitabile (0) |
|---|---|---|---|---|
| `major_studio` | producție: o companie TMDB e un mare studio | 4.351 | 41,5% | 29,3% |
| `major_distributor` | distribuitorul principal din infobox e un mare studio | 6.480 | 39,3% | 28,2% |
| `major_any` | oricare din cele două | 6.701 | 39,1% | 28,2% |

Față de `major_studio`, 2.350 filme trec în clasa „mare studio” doar prin distribuitor (de exemplu filme independente cumpărate de Universal sau Sony Pictures Classics), iar 221 au un mare studio printre producători, dar alt distribuitor principal. `major_any` le cuprinde pe toate. Contează doar distribuitorul principal: lista completă din infobox poate avea distribuitori adăugați după lansare (reluări, alte piețe).

`budget_is_range` = 1 la 888 filme (5,0%): bugetul final e mijlocul unui interval din infobox („$10–15 million”). E o coloană despre calitatea datelor, nu o caracteristică pentru model.

## De ce excludem coloanele META

`extraction/columns.py` împarte coloanele în TARGETS, IDS, META, LEAKAGE și FEATURES; în model intră doar cele 79 din FEATURES (`feature_columns(df)`), iar `extraction/test_columns.py` verifică asta pe CSV. META sunt cele 41 coloane despre **date**, nu despre film: sursa valorii, câte surse au o valoare și câte sunt de acord, sumele brute, monedele și cursurile. Ele prezic ținta, pentru că un film cu încasări raportate peste tot e de obicei un film cunoscut, care a mers bine:

**% profitabile după numărul de surse cu încasări**

| Surse cu încasări | Filme | % profitabile |
|---|---|---|
| 1 | 3.238 | 27,3% |
| 2 | 4.564 | 23,8% |
| 3 | 7.283 | 30,1% |
| 4 | 2.776 | 57,6% |

**% profitabile după sursa bugetului ales**

| Sursa bugetului | Filme | % profitabile |
|---|---|---|
| dbpedia | 3.372 | 19,9% |
| tmdb | 11.995 | 36,2% |
| wikidata | 482 | 26,1% |
| wikipedia | 2.012 | 31,2% |

**% profitabile după sursa încasărilor alese**

| Sursa încasărilor | Filme | % profitabile |
|---|---|---|
| dbpedia | 53 | 34,0% |
| tmdb | 1.405 | 19,6% |
| wikidata | 249 | 21,3% |
| wikipedia | 16.154 | 33,5% |

**Corelația coloanelor META numerice cu `profitable`**

| Coloană | Corelație cu `profitable` |
|---|---|
| `gross_wd` | 0,370 |
| `worldwide_gross_dbp` | 0,351 |
| `gross_dbp_amount` | 0,350 |
| `revenue_tmdb` | 0,326 |
| `gross_tmdb_amount` | 0,326 |
| `gross_wp` | 0,318 |
| `gross_n_agree` | 0,222 |
| `budget_n_agree` | 0,189 |
| `gross_n_sources` | 0,176 |
| `budget_n_sources` | 0,160 |
| `gross_wd_amount` | 0,089 |
| `budget_wd` | 0,083 |
| `fx_year` | -0,071 |
| `budget_wp` | 0,065 |
| `gross_is_single_market` | -0,064 |
| `budget_dbp` | 0,061 |
| `budget_is_range` | 0,056 |
| `budget_dbp_amount` | 0,056 |
| `gross_is_domestic` | -0,050 |
| `budget_tmdb_amount` | 0,046 |
| `budget_tmdb` | 0,045 |
| `budget_conflict` | 0,044 |
| `gross_fx_rate` | 0,040 |
| `gross_conflict` | 0,034 |
| `gross_is_local_currency` | 0,028 |
| `gross_wp_amount` | 0,024 |
| `budget_fx_rate` | 0,021 |
| `gross_is_rental` | -0,020 |
| `budget_wp_amount` | 0,005 |
| `budget_wd_amount` | -0,003 |

La predicție, pentru un film care nu a fost încă lansat, nu știm câte surse îi vor raporta încasările. Un model care le folosește ar învăța „filmul e documentat bine”, nu „filmul va fi profitabil”.

## Date lipsă

| Coloană | Lipsesc | % |
|---|---|---|
| `runtime_min` | 33 | 0,2% |
| `certification` | 7.443 | 41,7% |
| `sequel_number` | 13 | 0,1% |
| `director_prev_mean_roi` | 7.407 | 41,5% |
| `cast_prev_mean_roi` | 2.779 | 15,6% |

Istoricul regizorului și al actorilor lipsește când e primul lor film din dataset.

## Istoricul ROI: de ce nu ajunge media

Câteva filme au un ROI uriaș (Paranormal Activity: 12.947), deși la jumătate dintre filme ROI-ul e sub 1,35. Un singur astfel de hit strică media pentru toată cariera unui regizor. De aceea, pe lângă media cerută în issue, am adăugat:

- `*_prev_profitable_rate`: ce parte din filmele anterioare au fost profitabile;
- `*_prev_mean_log_roi`: media pe logaritm, unde un hit contează mult mai puțin.

| Coloană | Corelație cu `profitable` |
|---|---|
| `director_prev_mean_roi` | 0,029 |
| `director_prev_mean_log_roi` | 0,202 |
| `director_prev_profitable_rate` | 0,189 |
| `cast_prev_mean_roi` | 0,033 |
| `cast_prev_mean_log_roi` | 0,186 |
| `cast_prev_profitable_rate` | 0,179 |

Media simplă aproape nu are legătură cu ținta; variantele noi au.

## Fără informații de după lansare (anti-leakage)

- Premiile și istoricul regizorului/actorilor vin doar din filmele lansate **înainte**.
- **Franciza** (#47): colecțiile TMDB se creează după ce apare o continuare, iar continuarea se face de obicei pentru că primul film a avut succes. Cu vechea definiție (`is_franchise` = filmul e într-o colecție), primul film al unei serii ar fi purtat informație din viitor:

  | Vechea definiție | Filme | % profitabile |
  |---|---|---|
  | nu e în nicio colecție | 14.595 | 26,2% |
  | primul film dintr-o colecție | 1.651 | 65,6% |
  | continuare (au apărut deja filme din serie) | 1.602 | 53,3% |

  Acum `is_franchise` = 1 doar pentru continuări, iar `sequel_number` = câte filme din serie au apărut înainte (0 = primul film sau film independent).
- Am scos keywords ca `amused` sau `suspenseful`: arată cum a reacționat publicul, deci nu se știu înainte de lansare.
- **Keywords de după lansare** (#47): `duringcreditsstinger` / `aftercreditsstinger` (scenă după generic) se află abia la vizionare și anunțau succesul (774 filme, 55,0% profitabile). Le-am scos, împreună cu `controversy`, `banned film` și altele asemănătoare.
- **Limită rămasă:** keywords sunt adăugate de utilizatorii TMDB, după lansare. Filmele populare primesc mai multe: numărul de keywords se corelează 0,62 cu log(voturi) și 0,22 cu `profitable`, iar cele 2.584 filme fără niciun keyword au doar 24,0% profitabile. Chiar dacă fiecare keyword descrie conținutul, un vector `kw_*` plin de zerouri spune „film puțin cunoscut”. În experimentul cu leakage, merită comparat modelul cu și fără `kw_*`.
- `vote_average` și `vote_count` rămân în tabel doar pentru experimentul cu leakage.
- La antrenare, e mai sigur să împărțim filmele după an (vechi = antrenare, noi = test). La o împărțire aleatoare, istoricul unui film vechi ar putea folosi ROI-ul unui film din test.
