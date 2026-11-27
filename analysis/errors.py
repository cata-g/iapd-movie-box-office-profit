"""
Analiza erorilor (#57): unde și de ce greșesc modelele, pe testul temporal (2019–2024).

Run:
  python analysis/errors.py     # -> docs/error_analysis.md, docs/figures/errors_*.png, results/errors/*.csv

Pornește de la probabilitățile pe test și pragurile salvate de `modeling/compare.py` (#56). Modelele
analizate: gradient boosting (principal) și regresia logistică + buget² (model simplu de comparație).
Pentru regresie, modelele se reantrenează cu hiperparametrii din #41.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.eda import ACCENT, INK_2, MUTED, PAIR, md_table, num, pct, save
from extraction.utils import PROJECT_ROOT
from modeling.baselines import REGRESSORS
from modeling.compare import predict_reg
from modeling.preprocessing import load_dataset, temporal_split

CMP_DIR = PROJECT_ROOT / "results" / "compare"
OUT_DIR = PROJECT_ROOT / "results" / "errors"
REPORT = PROJECT_ROOT / "docs" / "error_analysis.md"
PARAMS = json.loads((PROJECT_ROOT / "results" / "baselines" / "best_params.json").read_text(encoding="utf-8"))
THRESHOLDS = json.loads((CMP_DIR / "thresholds_chosen.json").read_text(encoding="utf-8"))
MAIN, SIMPLE = "Gradient boosting", "Regresie logistică + buget²"
LANG = {"en": "engleză", "fr": "franceză", "hi": "hindi", "ru": "rusă", "es": "spaniolă", "ko": "coreeană",
        "other": "altă limbă"}


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    """Testul, cu caracteristicile, ținta și predicțiile celor două modele."""
    df = load_dataset()
    _, _, test = temporal_split(df)
    preds = pd.read_csv(CMP_DIR / "test_predictions.csv")
    test = test.merge(preds[["imdb_id", f"proba_{MAIN}", f"proba_{SIMPLE}"]], on="imdb_id", how="left")
    assert test[f"proba_{MAIN}"].notna().all(), "predicțiile din #56 nu corespund testului"
    for name in (MAIN, SIMPLE):
        test[f"pred_{name}"] = (test[f"proba_{name}"] >= THRESHOLDS[name]).astype(int)
    # segmente
    edges = df[df["release_year"] <= 2014]["log_budget_real"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    labels = [f"{k + 1}" for k in range(5)]
    test["grupa de buget (train)"] = pd.Categorical(np.array(labels)[np.digitize(test["log_budget_real"], edges)],
                                                    categories=labels, ordered=True)
    test["gen principal"] = test["genres"].fillna("").str.split("|").str[0].replace("", "necunoscut")
    test["limbă"] = test["original_language"].map(LANG)
    test["franciză"] = test["is_franchise"].map({1: "da", 0: "nu"})
    test["studio major"] = test["major_studio"].map({1: "da", 0: "nu"})
    test["regizor cu istoric"] = test["has_director_history"].map({1: "da", 0: "nu"})
    test["an"] = test["release_year"].astype(str)
    test["surse în dezacord"] = np.where(test["budget_conflict"].astype(bool) | test["gross_conflict"].astype(bool),
                                         "da (buget sau încasări)", "nu")
    test["încasări în moneda locală"] = test["gross_is_local_currency"].map({1: "da", 0: "nu"})
    dist = np.abs(np.log(test["roi"] / 2.5))
    test["distanța ROI față de 2,5"] = pd.cut(dist, [-np.inf, np.log(1.2), np.log(1.5), np.log(2), np.inf],
                                              labels=["sub 1,2×", "1,2–1,5×", "1,5–2×", "peste 2×"])
    return test, edges


def segment_table(test: pd.DataFrame, col: str, model: str, min_n: int = 30) -> pd.DataFrame:
    y, p, pred = test["profitable"], test[f"proba_{model}"], test[f"pred_{model}"]
    rows = []
    for value, idx in test.groupby(col, observed=True).groups.items():
        if len(idx) < min_n:
            continue
        yy, pp, pr = y[idx], p[idx], pred[idx]
        rows.append({col: value, "filme": len(idx), "% profitabile": yy.mean(), "% prezise profitabile": pr.mean(),
                     "precision": precision_score(yy, pr, zero_division=0), "recall": recall_score(yy, pr, zero_division=0),
                     "F1": f1_score(yy, pr, zero_division=0),
                     "ROC-AUC": roc_auc_score(yy, pp) if yy.nunique() == 2 else np.nan,
                     "% greșite": (yy != pr).mean()})
    return pd.DataFrame(rows)


def top_errors(test: pd.DataFrame, model: str, kind: str, n: int = 10) -> pd.DataFrame:
    p = test[f"proba_{model}"]
    if kind == "FP":
        sel = test[(test["profitable"] == 0) & (test[f"pred_{model}"] == 1)].assign(_s=p).nlargest(n, "_s")
    else:
        sel = test[(test["profitable"] == 1) & (test[f"pred_{model}"] == 0)].assign(_s=-p).nlargest(n, "_s")
    return pd.DataFrame({
        "film": sel["title"] + " (" + sel["release_year"].astype(str) + ")",
        "P(profitabil)": sel[f"proba_{model}"], "ROI real": sel["roi"],
        "buget (mil. $)": sel["budget"].astype(float) / 1e6, "încasări (mil. $)": sel["worldwide_gross"].astype(float) / 1e6,
        "gen": sel["gen principal"], "limbă": sel["limbă"], "franciză": sel["franciză"], "studio major": sel["studio major"],
        "ROI mediu regizor": np.exp(sel["director_prev_mean_log_roi"]),
        "surse în dezacord": sel["surse în dezacord"], "sursă încasări": sel["gross_source"],
    })


def calibration(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in (MAIN, SIMPLE):
        y, p = test["profitable"], test[f"proba_{model}"]
        rows.append({"model": model, "Brier": brier_score_loss(y, p),
                     "Brier, baseline (rata de bază)": brier_score_loss(y, np.full(len(y), y.mean())),
                     "probabilitate medie prezisă": p.mean(), "% profitabile real": y.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------
def fig_reliability(test: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], color="#cbd5e1", linewidth=1)
    ax.annotate("calibrare perfectă", (0.62, 0.66), color=MUTED, fontsize=8, rotation=42)
    for model, color in zip((MAIN, SIMPLE), PAIR):
        frac, mean = calibration_curve(test["profitable"], test[f"proba_{model}"], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, color=color, marker="o", markersize=5, linewidth=2, label=model)
    ax.set(xlabel="probabilitatea prezisă (medie pe decilă)", ylabel="% filme profitabile în realitate",
           xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Calibrare pe test (10 grupe egale)")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, "errors_reliability")


def fig_segments(tables: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(1, len(tables), figsize=(4 * len(tables), 3.6), sharey=True)
    for ax, (col, t) in zip(axes, tables.items()):
        x = np.arange(len(t))
        w = 0.38
        ax.bar(x - w / 2 - 0.01, 100 * t["% profitabile"], width=w, color="#94a3b8", label="profitabile în realitate")
        ax.bar(x + w / 2 + 0.01, 100 * t["% prezise profitabile"], width=w, color=ACCENT, label="prezise profitabile")
        ax.set_xticks(x, [str(v) for v in t[col]], rotation=30 if len(t) > 5 else 0, ha="right" if len(t) > 5 else "center")
        ax.set_title(col, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% din filmele grupei")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return save(fig, "errors_segments")


def fig_year(t: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = t["an"].astype(int)
    ax.plot(x, 100 * t["% profitabile"], color="#94a3b8", marker="o", linewidth=2, label="profitabile în realitate")
    ax.plot(x, 100 * t["% prezise profitabile"], color=ACCENT, marker="o", linewidth=2, label="prezise profitabile")
    for xi, f in zip(x, t["F1"]):
        ax.annotate(f"F1 {num(f, 2)}", (xi, 3), ha="center", fontsize=8, color=INK_2)
    ax.set_ylim(0, 75)
    ax.set_ylabel("% filme")
    ax.set_title(f"Pe ani de test ({MAIN})")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    return save(fig, "errors_year")


def fig_residuals(test: pd.DataFrame, resid: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, col, label in ((axes[0], "log_budget_real", "log(buget în dolari 2024)"), (axes[1], "release_year", "anul lansării")):
        x = test[col].to_numpy(dtype=float)
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(x)) if col == "release_year" else 0
        ax.scatter(x + jitter, resid, s=4, alpha=0.25, color=ACCENT, linewidths=0)
        ax.axhline(0, color=INK_2, linewidth=1)
        bins = pd.qcut(x, 10, duplicates="drop") if col != "release_year" else x
        med = pd.Series(resid).groupby(bins, observed=True).median()
        centers = [b.mid for b in med.index] if col != "release_year" else med.index
        ax.plot(centers, med.to_numpy(), color=PAIR[1], linewidth=2, marker="o", markersize=4, label="mediana reziduurilor")
        ax.set_xlabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("log_roi real − prezis")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Reziduurile regresiei (Ridge + buget²) pe test", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, "errors_residuals")


# ---------------------------------------------------------------------------
def fmt(df: pd.DataFrame, digits=3, pct_cols=()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(pct)
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, digits))
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, 0))
    return out


SEG_PCT = ["% profitabile", "% prezise profitabile", "% greșite"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test, edges = load()

    seg_cols = ["grupa de buget (train)", "gen principal", "limbă", "franciză", "studio major", "regizor cu istoric",
                "surse în dezacord", "încasări în moneda locală", "distanța ROI față de 2,5", "an"]
    segs = {c: segment_table(test, c, MAIN) for c in seg_cols}
    segs_simple = {c: segment_table(test, c, SIMPLE) for c in ("grupa de buget (train)", "surse în dezacord", "an")}
    covid = pd.DataFrame([{"test": name, "filme": len(t), "% profitabile": t["profitable"].mean(),
                           "% prezise profitabile": t[f"pred_{MAIN}"].mean(),
                           "F1": f1_score(t["profitable"], t[f"pred_{MAIN}"]),
                           "ROC-AUC": roc_auc_score(t["profitable"], t[f"proba_{MAIN}"])}
                          for name, t in (("tot testul", test), ("fără 2020–2021", test[~test.release_year.isin([2020, 2021])]),
                                          ("doar 2020–2021", test[test.release_year.isin([2020, 2021])]))])
    fp, fn = top_errors(test, MAIN, "FP"), top_errors(test, MAIN, "FN")
    cal = calibration(test)

    # regresie: reziduuri
    df = load_dataset()
    train, _, _ = temporal_split(df)
    spec = REGRESSORS[2]
    pred_log = predict_reg(spec, PARAMS["regresie"][spec.name], train, test)
    resid = test["log_roi"].to_numpy() - pred_log
    res_tab = test.assign(prezis_roi=np.exp(pred_log), rezidual=resid)
    big = res_tab.reindex(res_tab["rezidual"].abs().sort_values(ascending=False).index).head(10)
    big = pd.DataFrame({"film": big["title"] + " (" + big["release_year"].astype(str) + ")", "ROI real": big["roi"],
                        "ROI prezis": big["prezis_roi"], "rezidual (log)": big["rezidual"],
                        "buget (mil. $)": big["budget"].astype(float) / 1e6, "sursă încasări": big["gross_source"],
                        "surse în dezacord": big["surse în dezacord"]})
    resid_seg = pd.DataFrame([{"grupă": g, "filme": len(r), "rezidual median": r.median(), "|rezidual| median": r.abs().median()}
                              for g, r in pd.Series(resid).groupby(test["grupa de buget (train)"].to_numpy(), observed=True)])

    for name, t in {**{f"segment_{k}": v for k, v in segs.items()}, "covid": covid, "top_fp": fp, "top_fn": fn,
                    "calibration": cal, "residuals_top": big, "residuals_by_budget": resid_seg}.items():
        t.to_csv(OUT_DIR / f"{name.replace(' ', '_').replace(',', '')}.csv", index=False, float_format="%.6f")

    figs = {"seg": fig_segments({c: segs[c] for c in ("grupa de buget (train)", "limbă", "surse în dezacord", "distanța ROI față de 2,5")}),
            "year": fig_year(segs["an"]), "cal": fig_reliability(test), "res": fig_residuals(test, resid)}
    write_report(test, edges, segs, segs_simple, covid, fp, fn, cal, big, resid_seg, figs)
    print(f"Scris: {REPORT}")


def write_report(test, edges, segs, segs_simple, covid, fp, fn, cal, big, resid_seg, figs) -> None:
    bounds = [0, *np.exp(edges) / 1e6]
    budget_note = ", ".join(f"{k + 1}: {num(bounds[k], 1)}–{num(bounds[k + 1], 1) if k < 4 else '∞'}" for k in range(5))
    seg_md = "\n\n".join(f"### {c}\n\n{md_table(fmt(segs[c], pct_cols=SEG_PCT))}" for c in segs if c != "an")
    text = f"""# Analiza erorilor (#57)

Generat cu `python analysis/errors.py`. Tabelele complete: `results/errors/*.csv`.

Testul temporal: **{num(len(test), 0)} filme din 2019–2024**, {pct(test['profitable'].mean())} profitabile. Modelul principal: **{MAIN}** (prag de decizie {num(THRESHOLDS[MAIN], 2)}, ales pe validare); model de comparație: **{SIMPLE}** (prag {num(THRESHOLDS[SIMPLE], 2)}). Probabilitățile și pragurile vin din #56 (`results/compare/`).

## Pe scurt

{CONCLUSIONS}

## 1. Erori pe segmente ({MAIN})

Pentru fiecare grupă: câte filme sunt profitabile în realitate, câte prezice modelul ca profitabile, precizia, recall-ul, F1, ROC-AUC (cât de bine ordonează filmele în grupă) și ce procent din filme e clasificat greșit. Grupele cu sub 30 de filme sunt omise. „Distanța ROI față de 2,5” e raportul dintre ROI și prag, în orice sens: sub 1,2× = ROI între 2,1 și 3; 1,2–1,5× = ROI 1,7–2,1 sau 3–3,75; 1,5–2× = ROI 1,25–1,7 sau 3,75–5; peste 2× = ROI sub 1,25 sau peste 5. Grupele de buget sunt cele 5 grupe egale din train, în mil. $ 2024: {budget_note}.

{figs['seg']}

{seg_md}

**Același lucru pentru {SIMPLE}** (grupe de buget și surse):

{md_table(fmt(segs_simple['grupa de buget (train)'], pct_cols=SEG_PCT))}

{md_table(fmt(segs_simple['surse în dezacord'], pct_cols=SEG_PCT))}

## 2. În timp: COVID și schimbarea ratei de bază

{md_table(fmt(segs['an'], pct_cols=SEG_PCT))}

{figs['year']}

{md_table(fmt(covid, pct_cols=['% profitabile', '% prezise profitabile']))}

## 3. Cele mai sigure greșeli ({MAIN})

**Fals pozitive** (modelul era cel mai sigur că sunt profitabile, dar nu au fost):

{md_table(fmt(fp, digits=2))}

**Fals negative** (modelul era cel mai sigur că nu sunt profitabile, dar au fost):

{md_table(fmt(fn, digits=2))}

{ERROR_TYPES}

## 4. Calibrare

Un model calibrat spune „30%” pentru filmele din care chiar 30% sunt profitabile. Brier = media pătratului diferenței dintre probabilitate și realitate (0 = perfect; baseline-ul prezice rata de bază pentru toate filmele).

{md_table(fmt(cal, pct_cols=['probabilitate medie prezisă', '% profitabile real']))}

{figs['cal']}

## 5. Regresie: reziduuri (Ridge + buget²)

Rezidual = `log_roi` real − prezis; pozitiv = filmul a mers mai bine decât a prezis modelul.

{figs['res']}

{md_table(fmt(resid_seg))}

**Cele mai mari reziduuri:**

{md_table(fmt(big, digits=2))}
"""
    REPORT.write_text(text, encoding="utf-8")


CONCLUSIONS = """1. **Modelul e prea optimist pe test.** Gradient boosting prezice 48,0% din filme ca profitabile, iar în
   realitate sunt 26,1%; probabilitatea medie prezisă e 44,8%. Modelul a învățat pe filme cu 32,5%
   profitabile (train), iar pragul a fost ales pe 33,5% (validare); testul are o rată mult mai mică
   (COVID, streaming). Consecința: recall mare (0,79), precizie mică (0,43). Gradient boosting
   e chiar **mai prost calibrat decât baseline-ul** (Brier 0,195 vs 0,193); regresia logistică + buget²
   e mai bine calibrată (Brier 0,178).
2. **Cele mai mari greșeli sunt la filmele mari de studio și francize:** la studiourile majore modelul
   prezice 89% profitabile, real 39% (52,6% greșite); la francize 81% vs 49%. Fals pozitivele cele mai
   sigure sunt blockbustere (Star Wars: The Rise of Skywalker, Fast X, Twisters, Mission: Impossible)
   care **au avut succes, dar sub pragul de 2,5** (ROI 1,96–2,40), plus filmele Warner din 2020–2021
   lansate simultan în cinema și pe HBO Max (Wonder Woman 1984, The Suicide Squad, Space Jam: A New
   Legacy), pe care coloana `streaming_distributor` nu le prinde (distribuitorul e Warner).
3. **Fals negativele sunt hituri-surpriză mici, mai ales din afara SUA:** comedii franceze, filme
   rusești pentru copii, un film chinezesc cu 674 mil. $ încasări (Full River Red), un film danez premiat
   (Another Round). Modelul nu are informații despre piața locală, iar regizorii lor nu au istoric în
   dataset.
4. **Aproape de prag, modelul ghicește:** pentru filmele cu ROI între 2,1 și 3 (cel mult 1,2× distanță de
   2,5), ROC-AUC e 0,45, adică sub nivelul unei alegeri aleatoare, cu 52% greșite. Pentru filmele cu ROI
   foarte departe de prag (peste 2× sau sub jumătate), ROC-AUC e 0,84. O mare parte din eroare vine din
   pragul fix: un film cu ROI 2,4 și unul cu ROI 2,6 sunt aproape identici, dar au etichete diferite.
5. **COVID:** în 2020–2021 doar 17,9% din filme au fost profitabile, dar modelul ordonează filmele la fel
   de bine ca în ceilalți ani (ROC-AUC 0,772 vs 0,775). F1 scade (0,43 vs 0,59) doar pentru că pragul
   nu se adaptează la rata de bază mai mică. Problema e de prag / calibrare, nu de ordonare.
6. **Sursele în dezacord nu explică erorile:** filmele cu buget sau încasări în dezacord între surse au
   34,2% greșite, față de 32,7% la celelalte (la regresia logistică: 30,4% vs 28,9%). Incertitudinea
   etichetei (#45: 87% din filmele care își schimbă eticheta au surse în dezacord) există, dar e mică
   față de restul erorii.
7. **Pe limbi, comportament foarte diferit:** filmele franțuzești din test sunt doar 4,3% profitabile, iar
   modelul le prezice corect ca neprofitabile (6,8% greșite); la cele coreene modelul nu ordonează deloc
   (ROC-AUC 0,47, doar 57 de filme), iar la hindi greșește 45%.
8. **Regresia:** reziduurile sunt pozitive în toate grupele de buget (mediană 0,14–0,40), deci modelul
   subestimează ușor ROI-ul tipic. Cele mai mari erori sunt filme cu ROI aproape 0: lansări de streaming
   cu o încasare simbolică în cinema (The Gray Man și White Noise, Netflix; Cinderella, Amazon), unde
   modelul prezice ROI 0,4–1,1. În test sunt 209 filme cu ROI sub 0,05 (9,9%, față de 6,3% în train).
   **Avengement (2019)** are încasările doar din TMDB (89,9 mil. $ pentru un film de acțiune britanic
   cu buget de 2,4 mil. $ și lansare foarte limitată în cinema, ROI 37), fără confirmare din altă
   sursă: foarte probabil o eroare de date, nu o eroare de model.

**Ce s-ar putea face:** recalibrarea probabilităților (de ex. Platt / izotonic pe validare) și alegerea
pragului după rata de bază așteptată; o variabilă pentru lansările simultane în cinema și streaming;
verificarea încasărilor venite dintr-o singură sursă (mai ales TMDB)."""
ERROR_TYPES = """**Tipuri de greșeli:**

- **Fals pozitive = blockbustere de franciză care au avut succes, dar nu destul.** Toate primele 9 sunt
  francize de la studiouri majore, cu buget de 20–490 mil. $. Patru au ROI între 1,96 și 2,40, adică au
  dublat bugetul (Star Wars, Fast X, Twisters, Mission: Impossible): pentru presă au fost succese, dar sub
  pragul nostru de 2,5. Alte trei sunt filme Warner din 2020–2021, lansate în același timp în cinema și
  pe HBO Max, în plin COVID. Modelul a învățat corect că francizele mari sunt de obicei profitabile, dar
  nu are cum să știe de pandemie sau de lansarea simultană pe streaming.
- **Fals negative = filme mici, fără istoric, care au surprins.** Majoritatea nu sunt în engleză, nu au
  studio major, iar regizorul nu are filme anterioare în dataset (ROI mediu „–”) sau are un istoric slab.
  Succesul lor a venit din piața locală (Franța, Rusia, China) sau din premii (Another Round, Belfast),
  informații pe care modelul nu le are înainte de lansare.
- **Avengement (2019)** e cel mai „sigur” fals negativ (P = 0,10, ROI 37): încasările vin dintr-o singură
  sursă (TMDB) și sunt foarte probabil greșite. Aici, probabil, modelul a avut dreptate, iar eticheta e greșită."""

if __name__ == "__main__":
    main()
