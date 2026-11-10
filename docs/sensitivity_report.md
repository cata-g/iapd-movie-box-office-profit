# Test de sensibilitate (#45)

Generat cu `python extraction/sensitivity.py`. Fiecare variantă schimbă **o singură** decizie față de varianta implicită (`python extraction/integrate.py`: buget prin majoritate, încasări cu Wikipedia prima, `--roi-rule roi-recalculat`, prag 2,5) și refolosește funcțiile din `integrate.py`. „Etichete schimbate” = filmele prezente în ambele variante la care `profitable` diferă.

Varianta implicită: **17.863 filme, 32,2% profitabile**.

### Buget

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| majoritate (implicit) | 17.863 | 32,2% | 0 (0,0%) | +0 / −0 |
| majoritate fără mediană (vechi) | 17.863 | 32,2% | 3 (0,0%) | +0 / −0 |
| prioritate tmdb → dbpedia → wikidata → wikipedia | 17.860 | 32,3% | 49 (0,3%) | +0 / −3 |
| prioritate wikipedia → tmdb → dbpedia → wikidata | 17.837 | 32,1% | 327 (1,8%) | +22 / −48 |

### Încasări

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| wikipedia → dbpedia → wikidata → tmdb (implicit) | 17.863 | 32,2% | 0 (0,0%) | +0 / −0 |
| tmdb → wikipedia → dbpedia → wikidata | 17.826 | 31,4% | 444 (2,5%) | +25 / −62 |

### ROI extrem

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| vechi | 17.875 | 32,2% | 0 (0,0%) | +17 / −5 |
| surse-independente | 17.851 | 32,2% | 0 (0,0%) | +0 / −12 |
| roi-recalculat (implicit) | 17.863 | 32,2% | 0 (0,0%) | +0 / −0 |

### Prag

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| 2 | 17.863 | 38,9% | 1.199 (6,7%) | +0 / −0 |
| 2,5 (implicit) | 17.863 | 32,2% | 0 (0,0%) | +0 / −0 |
| 3 | 17.863 | 27,1% | 920 (5,2%) | +0 / −0 |

### Calitatea încasărilor

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| toate filmele (implicit) | 17.863 | 32,2% | 0 (0,0%) | +0 / −0 |
| fără încasări parțiale (o singură piață, rentals) | 16.319 | 33,0% | 0 (0,0%) | +0 / −1.544 |
| și fără moneda locală fără „worldwide” | 13.527 | 32,6% | 0 (0,0%) | +0 / −4.336 |

## Concluzie

- Deciziile despre date (ce buget sau ce încasări alegem, ce ROI extrem păstrăm) mută rata de filme profitabile doar între 31,4% și 32,3% și schimbă cel mult 2,5% din etichete (încasări: tmdb → wikipedia → dbpedia → wikidata). Ținta e **robustă** la aceste alegeri: doar câteva sute de filme, cele cu surse în dezacord, își schimbă clasa.
- Fără filmele cu încasări parțiale (o singură piață sau „rentals”) rămân 16.319 filme, cu 33,0% profitabile; fără și cele în moneda locală fără „worldwide”, 13.527 filme, cu 32,6% profitabile (implicit: 32,2%). Etichetele celorlalte filme nu se schimbă: rata crește doar pentru că se scot filme etichetate, cel mai probabil, greșit ca neprofitabile.
- Pragul contează mult mai mult: varianta cu cel mai mare efect e prag 2, cu 6,7% etichete schimbate. Pragul de 2,5 e o convenție (bugetul nu include marketingul, iar cinematografele păstrează aproximativ jumătate din bilete), deci rezultatele modelului trebuie raportate și pentru pragurile 2 și 3, nu doar pentru 2,5.
- Dintre cele 759 filme care își schimbă eticheta într-una din variantele despre date, 87,2% au surse în dezacord la buget sau la încasări (`budget_conflict` / `gross_conflict`), față de 20,2% în tot dataset-ul. Incertitudinea e concentrată în filmele cu surse care nu se pun de acord, nu în regula aleasă.
