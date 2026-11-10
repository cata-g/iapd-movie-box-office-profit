# Test de sensibilitate (#45)

Generat cu `python extraction/sensitivity.py`. Fiecare variantă schimbă **o singură** decizie față de varianta implicită (`python extraction/integrate.py`: buget prin majoritate, încasări cu Wikipedia prima, `--roi-rule roi-recalculat`, prag 2,5) și refolosește funcțiile din `integrate.py`. „Etichete schimbate” = filmele prezente în ambele variante la care `profitable` diferă.

Varianta implicită: **18.056 filme, 31,9% profitabile**.

### Buget

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| majoritate (implicit) | 18.056 | 31,9% | 0 (0,0%) | +0 / −0 |
| majoritate fără mediană (vechi) | 18.056 | 31,9% | 3 (0,0%) | +0 / −0 |
| prioritate tmdb → dbpedia → wikidata → wikipedia | 18.054 | 31,9% | 49 (0,3%) | +0 / −2 |
| prioritate wikipedia → tmdb → dbpedia → wikidata | 18.032 | 31,8% | 328 (1,8%) | +18 / −42 |

### Încasări

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| wikipedia → dbpedia → wikidata → tmdb (implicit) | 18.056 | 31,9% | 0 (0,0%) | +0 / −0 |
| tmdb → wikipedia → dbpedia → wikidata | 18.039 | 31,0% | 446 (2,5%) | +15 / −32 |

### ROI extrem

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| vechi | 18.071 | 31,9% | 0 (0,0%) | +20 / −5 |
| surse-independente | 18.044 | 31,8% | 0 (0,0%) | +0 / −12 |
| roi-recalculat (implicit) | 18.056 | 31,9% | 0 (0,0%) | +0 / −0 |

### Prag

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| 2 | 18.056 | 38,5% | 1.199 (6,6%) | +0 / −0 |
| 2,5 (implicit) | 18.056 | 31,9% | 0 (0,0%) | +0 / −0 |
| 3 | 18.056 | 26,8% | 920 (5,1%) | +0 / −0 |

### Streaming

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| fără filtru (implicit) | 18.056 | 31,9% | 0 (0,0%) | +0 / −0 |
| filtrul vechi (semnal de streaming și ROI < 0,05) | 17.863 | 32,2% | 0 (0,0%) | +0 / −193 |

### Calitatea încasărilor

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| toate filmele (implicit) | 18.056 | 31,9% | 0 (0,0%) | +0 / −0 |
| fără încasări parțiale (o singură piață, rentals) | 16.506 | 32,6% | 0 (0,0%) | +0 / −1.550 |
| și fără moneda locală fără „worldwide” | 13.703 | 32,2% | 0 (0,0%) | +0 / −4.353 |

## Concluzie

- Deciziile despre date (ce buget sau ce încasări alegem, ce ROI extrem păstrăm) mută rata de filme profitabile doar între 31,0% și 31,9% și schimbă cel mult 2,5% din etichete (încasări: tmdb → wikipedia → dbpedia → wikidata). Ținta e **robustă** la aceste alegeri: doar câteva sute de filme, cele cu surse în dezacord, își schimbă clasa.
- Fără filmele cu încasări parțiale (o singură piață sau „rentals”) rămân 16.506 filme, cu 32,6% profitabile; fără și cele în moneda locală fără „worldwide”, 13.703 filme, cu 32,2% profitabile (implicit: 31,9%). Etichetele celorlalte filme nu se schimbă: rata crește doar pentru că se scot filme etichetate, cel mai probabil, greșit ca neprofitabile.
- Filtrul vechi „streaming” (#52) ar scoate 193 filme și ar urca rata de profitabile la 32,2%, față de 31,9%: scotea filme alese după rezultat (ROI < 0,05), deci umfla rata. Acum distribuitorul de streaming e o caracteristică (`streaming_distributor`), nu un filtru.
- Pragul contează mult mai mult: varianta cu cel mai mare efect e prag 2, cu 6,6% etichete schimbate. Pragul de 2,5 e o convenție (bugetul nu include marketingul, iar cinematografele păstrează aproximativ jumătate din bilete), deci rezultatele modelului trebuie raportate și pentru pragurile 2 și 3, nu doar pentru 2,5.
- Dintre cele 762 filme care își schimbă eticheta într-una din variantele despre date, 87,3% au surse în dezacord la buget sau la încasări (`budget_conflict` / `gross_conflict`), față de 20,2% în tot dataset-ul. Incertitudinea e concentrată în filmele cu surse care nu se pun de acord, nu în regula aleasă.
