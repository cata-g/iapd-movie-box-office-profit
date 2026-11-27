# Preprocesare pentru model (#40)

Cod: `modeling/preprocessing.py`. Teste: `python -m pytest modeling`. Indicele de preț: `python extraction/cpi.py`.

Dataset-ul de intrare: `data/processed/movies_dataset.parquet`, **18.056 filme**, 31,9% profitabile (ROI ≥ 2,5). Din cele 50 de caracteristici fără leakage (`extraction/columns.py`) rezultă 54 de coloane de intrare și **67 de coloane după preprocesare** (one-hot inclus).

## Două etape

1. **`add_features`**: transformări rând cu rând (log, sin/cos, indicatori). Nu învață nimic din date, deci se pot aplica înainte de split fără leakage.
2. **`build_preprocessor`**: tot ce învață ceva din date (medianele, media și deviația pentru scalare, categoriile one-hot) e un `ColumnTransformer`, ajustat **doar pe train** (`fit` pe train, `transform` pe validare și test). Testul `test_fit_only_on_train` verifică că medianele și mediile învățate sunt exact cele din train.

## Split-uri

| Split | Set | Filme | % din total | Ani | % profitabile |
|---|---|---|---|---|---|
| **temporal (principal)** | train | 13.624 | 75,5% | 1950–2014 | 32,5% |
| | validare | 2.313 | 12,8% | 2015–2018 | 33,5% |
| | test | 2.119 | 11,7% | 2019–2024 | 26,1% |
| stratificat (comparație) | train | 12.638 | 70,0% | 1950–2024 | 31,9% |
| | validare | 2.709 | 15,0% | 1950–2024 | 31,9% |
| | test | 2.709 | 15,0% | 1950–2024 | 31,9% |

- **De ce temporal:** așa s-ar folosi modelul în realitate, pe filme care încă nu s-au lansat. La un split aleator, un film din 2010 din train ar fi învățat lângă filme din 2015, iar tendințele din viitor ar ajunge în antrenare. Tăierea pe an calendaristic (nu pe procente exacte) face setul de test ușor de descris: „filmele din 2019–2024”.
- **Istoricul regizorului și al actorilor** (`*_prev_*`) folosește doar filmele lansate cu cel puțin 6 luni înainte (#49). Pentru un film din test, istoricul poate include filme din train: e informație cunoscută la data lansării, deci nu e leakage.
- **Limită:** testul conține anii COVID (2020–2021, 549 de filme), de aici rata mai mică de profitabile (26,1%). Rezultatele pe test se raportează și fără 2020–2021, în analiza erorilor (#57).
- Split-ul stratificat (seed 42) păstrează rata de 31,9% în fiecare parte; servește doar pentru comparație cu cel temporal.

## Date lipsă

| Coloană | Lipsă | Tip | Strategie | Motiv |
|---|---|---|---|---|
| `director_prev_mean_log_roi`, `director_prev_profitable_rate` | 42,0% | structurală | indicator `has_director_history` + mediana din train | Lipsește **exact** când regizorul nu are filme anterioare în dataset (`director_prev_films = 0`, verificat pe 100% din rânduri). Nu e o valoare necunoscută, ci una care nu există; indicatorul îi spune modelului asta, iar mediana doar umple căsuța. |
| `cast_prev_mean_log_roi`, `cast_prev_profitable_rate` | 16,3% | structurală | indicator `has_cast_history` + mediana din train | La fel, pentru actori (`cast_prev_films = 0`). |
| `certification` | 41,6% | MAR (depinde de limbă) | indicator `certification_missing` + mediana din train | Clasificarea americană (G/PG/PG-13/R/NC-17) lipsește la 84,1% din filmele care nu sunt în engleză, față de 18,8% la cele în engleză. Lipsa e explicată în mare parte de o coloană pe care o avem (`is_english`), deci e MAR; indicatorul păstrează semnalul „film fără lansare clasificată în SUA”. Testul complet MAR / MNAR: #42. |
| `runtime_min` | 0,2% | MCAR (probabil) | mediana din train | Prea puține filme ca să conteze. |
| `sequel_number` | 0,07% | MCAR (probabil) | mediana din train | La fel. |

`build_preprocessor(imputer="knn" | "iterative")` oferă și `KNNImputer` / `IterativeImputer` (pe date scalate), pentru comparația din #41.

## Transformări

| Coloană nouă | Din | Transformare | Motiv |
|---|---|---|---|
| `log_budget` | `budget` | log | Bugetul e foarte asimetric (skewness 4,31 → −0,48 după log). |
| `log_budget_real` | `budget`, `release_year` | log(buget × CPI(2024) / CPI(an)) | Bugetul în dolari 2024. 5 mil. $ în 1965 erau o producție mare, iar în 2015 una mică. Vezi mai jos. |
| `month_sin`, `month_cos` | `release_month` | poziția pe un cerc cu 12 puncte | Decembrie (12) și ianuarie (1) par departe ca numere, dar sunt vecine și fac parte din același sezon de sărbători; pe cerc ajung alăturate. |
| `weekday_sin`, `weekday_cos` | `release_weekday` | cerc cu 7 puncte | Același motiv: duminică și luni sunt vecine. |
| `log1p_*` | numărul de filme anterioare și de premii | log(1 + x) | Numărători cu multe zerouri și câteva valori mari; `log1p` păstrează zeroul. |
| `has_director_history`, `has_cast_history`, `certification_missing` | vezi „Date lipsă” | 0 / 1 | |
| one-hot | `original_language` (7 valori), `adaptation_type` (10) | o coloană 0/1 per categorie | Categoriile cu sub 50 de filme în train (`video_game`, `franchise`, `true_story`) intră împreună la `infrequent`. O categorie nouă în test merge tot acolo. |
| ținta `log_roi` | `roi` | log | ROI are skewness 112 și kurtosis 13.679; după log, −0,75. |
| ținta `profitable` | `roi` | ROI ≥ prag | `load_dataset(threshold=2 / 2,5 / 3)`: 38,5% / 31,9% / 26,8% profitabile. |

**Coloane scoase în favoarea variantei transformate:** `budget`, `release_month`, `release_weekday`, numărătorile brute, `director_prev_mean_roi` și `cast_prev_mean_roi`. Ultimele două conțin aceeași informație ca `*_prev_mean_log_roi`, dar cu ROI-uri extreme care ar domina un model liniar. Lista completă e `REPLACED` în cod; testul `test_every_feature_used` verifică că fiecare caracteristică din `extraction/columns.py` intră direct sau prin înlocuitorul ei. Ce mai eliminăm ca redundant (de ex. `major_any` față de `major_studio` + `major_distributor`) se decide în EDA (#42).

`certification` rămâne ordinală (G = 0 … NC-17 = 4): ordinea are sens, de la public general la restricționat.

## Bugetul ajustat la inflație

- **Sursa:** FRED, seria `CPIAUCNS` (Federal Reserve Bank of St. Louis), care republică indicele oficial al prețurilor de consum din SUA (Bureau of Labor Statistics, domeniu public). Media celor 12 luni pentru fiecare an, în `data/interim/cpi.parquet`.
- **De ce nu World Bank** (`FP.CPI.TOTL`, cum scria inițial în issue): seria începe în 1960, iar dataset-ul are 774 de filme din 1950–1959. `CPIAUCNS` începe în 1913, deci toate filmele au buget ajustat.
- **De ce inflația SUA:** bugetele sunt deja în USD, convertite cu cursul din anul lansării (#45).
- **Exemple:** 1 USD din 1950 = 13,03 USD în 2024; 1965: 9,96; 1980: 3,81; 2000: 1,82; 2015: 1,32.
- **Ținta nu se schimbă:** ROI = încasări / buget, ambele din același an, deci inflația se simplifică. Ajustarea contează doar pentru bugetul folosit ca intrare. Bugetul nominal rămâne în dataset.

## Scalare

- `build_preprocessor("linear")`: `StandardScaler` pe coloanele numerice (regresie logistică, Ridge, kNN, SVM). Coloanele 0/1 nu se scalează.
- `build_preprocessor("tree")`: fără scalare; arborii aleg praguri și nu depind de scară.

## Ce nu intră în model

- **Ținte:** `worldwide_gross`, `roi`, `profitable`, `log_roi`.
- **Leakage:** `vote_average`, `vote_count`, `kw_*` (#54). Intră doar cu `X_y(df, leakage=True)` și `build_preprocessor(leakage=True)`, pentru experimentul leakage vs. no-leakage (#41).
- **Calitatea datelor** (`gross_is_domestic`, `gross_is_rental`, `gross_is_single_market`, `gross_is_local_currency`, `budget_is_range`, `*_conflict`, `*_n_agree` ...): descriu cum am găsit cifrele, nu filmul, și nu se știu dinainte în forma asta. Un film cu încasări în toate sursele e mai cunoscut, deci mai des profitabil: ar fi leakage. Rămân pentru analiza erorilor (#57) și pentru evaluarea pe filmele cu surse care se pun de acord (#41).
