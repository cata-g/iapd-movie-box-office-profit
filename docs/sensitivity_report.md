# Test de sensibilitate (#45)

Generat cu `python extraction/sensitivity.py`. Fiecare variantă schimbă **o singură** decizie față de varianta implicită (`python extraction/integrate.py`: buget prin majoritate, încasări cu Wikipedia prima, `--roi-rule roi-recalculat`, prag 2,5) și refolosește funcțiile din `integrate.py`. „Etichete schimbate” = filmele prezente în ambele variante la care `profitable` diferă.

Varianta implicită: **17.911 filme, 32,2% profitabile**.

### Buget

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| majoritate (implicit) | 17.911 | 32,2% | 0 (0,0%) | +0 / −0 |
| prioritate tmdb → dbpedia → wikidata → wikipedia | 17.908 | 32,2% | 46 (0,3%) | +0 / −3 |
| prioritate wikipedia → tmdb → dbpedia → wikidata | 17.916 | 32,0% | 335 (1,9%) | +20 / −15 |

### Încasări

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| wikipedia → dbpedia → wikidata → tmdb (implicit) | 17.911 | 32,2% | 0 (0,0%) | +0 / −0 |
| tmdb → wikipedia → dbpedia → wikidata | 17.889 | 31,3% | 453 (2,5%) | +26 / −48 |

### ROI extrem

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| vechi | 17.923 | 32,2% | 0 (0,0%) | +17 / −5 |
| surse-independente | 17.899 | 32,1% | 0 (0,0%) | +0 / −12 |
| roi-recalculat (implicit) | 17.911 | 32,2% | 0 (0,0%) | +0 / −0 |

### Prag

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| 2 | 17.911 | 38,9% | 1.198 (6,7%) | +0 / −0 |
| 2,5 (implicit) | 17.911 | 32,2% | 0 (0,0%) | +0 / −0 |
| 3 | 17.911 | 27,0% | 920 (5,1%) | +0 / −0 |

### Calitatea încasărilor

| Varianta | Filme | % profitabile | Etichete schimbate | Filme intrate / ieșite |
|---|---|---|---|---|
| toate filmele (implicit) | 17.911 | 32,2% | 0 (0,0%) | +0 / −0 |
| fără încasări doar SUA/Canada sau rentals | 16.739 | 32,7% | 0 (0,0%) | +0 / −1.172 |

## Concluzie

- Deciziile despre date (ce buget sau ce încasări alegem, ce ROI extrem păstrăm) mută rata de filme profitabile doar între 31,3% și 32,2% și schimbă cel mult 2,5% din etichete (încasări: tmdb → wikipedia → dbpedia → wikidata). Ținta e **robustă** la aceste alegeri: doar câteva sute de filme, cele cu surse în dezacord, își schimbă clasa.
- Fără filmele cu încasări doar SUA/Canada sau „rentals” (#47) rămân 16.739 filme, cu 32,7% profitabile, față de 32,2%. Etichetele celorlalte filme nu se schimbă; rata se mișcă doar pentru că se scot filme vechi, etichetate cel mai probabil greșit ca neprofitabile.
- Pragul contează mult mai mult: varianta cu cel mai mare efect e prag 2, cu 6,7% etichete schimbate. Pragul de 2,5 e o convenție (bugetul nu include marketingul, iar cinematografele păstrează aproximativ jumătate din bilete), deci rezultatele modelului trebuie raportate și pentru pragurile 2 și 3, nu doar pentru 2,5.
- Dintre cele 771 filme care își schimbă eticheta într-una din variantele despre date, 87,4% au surse în dezacord la buget sau la încasări (`budget_conflict` / `gross_conflict`), față de 20,2% în tot dataset-ul. Incertitudinea e concentrată în filmele cu surse care nu se pun de acord, nu în regula aleasă.
