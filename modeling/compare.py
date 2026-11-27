"""
Comparația finală a modelelor (#56): toate modelele pe același test temporal, cu intervale de
încredere, teste statistice, curbe ROC / PR și rezultatele la pragurile ROI 2 / 2,5 / 3.

Run:
  python modeling/compare.py     # -> docs/results.md, results/final_metrics.csv, results/compare/*.csv, figuri

Modelele și hiperparametrii vin din #41 (`results/baselines/best_params.json`) și #43 (genuri ca
componente PCA). Fiecare model: antrenat pe train (1950–2014), pragul de decizie ales pe validare
(2015–2018), evaluat o singură dată pe test (2019–2024). Varianta principală e fără leakage;
modelele cu voturi + keywords apar doar ca referință.

Incertitudinea: bootstrap pe test (1.000 de eșantioane, aceleași pentru toate modelele, deci
diferențele sunt pereche). Testul McNemar compară erorile a două clasificatoare pe aceleași filme.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (f1_score, precision_recall_curve, r2_score, roc_auc_score, roc_curve,
                             root_mean_squared_error, average_precision_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.eda import INK, INK_2, MUTED, md_table, num, pct, save
from analysis.reduction import fit_eval, genre_columns, pca_pipe
from extraction.utils import PROJECT_ROOT
from modeling.baselines import (CLASSIFIERS, REGRESSORS, best_threshold, clf_metrics, make_pipe,
                                reg_metrics, xy)
from modeling.preprocessing import SEED, load_dataset, temporal_split

OUT_DIR = PROJECT_ROOT / "results" / "compare"
FINAL = PROJECT_ROOT / "results" / "final_metrics.csv"
REPORT = PROJECT_ROOT / "docs" / "results.md"
PARAMS = json.loads((PROJECT_ROOT / "results" / "baselines" / "best_params.json").read_text(encoding="utf-8"))
N_BOOT = 1000
K_PCA = 12          # componentele pentru 90% din varianța genurilor (#43)

# Categoriale validate (dataviz): albastru, portocaliu, aqua, galben, magenta în ordine fixă; gri = referință
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# ---------------------------------------------------------------------------
# Predicții
# ---------------------------------------------------------------------------
def predict_clf(spec, params, train, val, test, leakage=False) -> dict:
    """Probabilitățile pe test și pragul ales pe validare."""
    pipe = make_pipe(spec, params, leakage).fit(*xy(train, "profitable", spec, leakage))
    p_va = pipe.predict_proba(xy(val, "profitable", spec, leakage)[0])[:, 1]
    threshold = 0.5 if spec.name.startswith("Dummy") else best_threshold(val["profitable"], p_va)
    return {"proba": pipe.predict_proba(xy(test, "profitable", spec, leakage)[0])[:, 1], "prag": threshold}


def predict_clf_pca(spec, params, df, train, val, test) -> dict:
    genres = genre_columns(df)
    pipe, cols = pca_pipe(spec, K_PCA, genres, list(df.columns))
    pipe.fit(train[cols], train["profitable"])
    threshold = best_threshold(val["profitable"], pipe.predict_proba(val[cols])[:, 1])
    return {"proba": pipe.predict_proba(test[cols])[:, 1], "prag": threshold}


def predict_reg(spec, params, train, test, leakage=False) -> np.ndarray:
    pipe = make_pipe(spec, params, leakage).fit(*xy(train, "log_roi", spec, leakage))
    return pipe.predict(xy(test, "log_roi", spec, leakage)[0])


# ---------------------------------------------------------------------------
# Incertitudine și teste
# ---------------------------------------------------------------------------
def boot_indices(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))


def clf_boot(y: np.ndarray, proba: np.ndarray, threshold: float, idx: np.ndarray) -> dict[str, np.ndarray]:
    pred = (proba >= threshold).astype(int)
    f1 = np.array([f1_score(y[i], pred[i], zero_division=0) for i in idx])
    auc = np.array([roc_auc_score(y[i], proba[i]) if len(np.unique(proba[i])) > 1 else 0.5 for i in idx])
    pr = np.array([average_precision_score(y[i], proba[i]) for i in idx])
    return {"F1": f1, "ROC-AUC": auc, "PR-AUC": pr}


def reg_boot(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    return {"RMSE log_roi": np.array([root_mean_squared_error(y[i], pred[i]) for i in idx]),
            "R² log_roi": np.array([r2_score(y[i], pred[i]) for i in idx])}


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar(y, pred_a, pred_b) -> dict:
    """Pe filmele unde exact un model greșește: dacă erorile sunt la fel de probabile, b ≈ c."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())       # A corect, B greșit
    c = int((~a_ok & b_ok).sum())       # A greșit, B corect
    chi = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"A corect, B greșit": b, "A greșit, B corect": c, "χ² (corecție de continuitate)": chi,
            "p (χ²)": float(stats.chi2.sf(chi, 1)), "p (exact, binomial)": float(stats.binomtest(b, b + c).pvalue)}


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------
def fig_curves(y: np.ndarray, preds: dict[str, dict], leak: dict[str, dict]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for (name, p), color in zip(preds.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, p["proba"])
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} ({num(roc_auc_score(y, p['proba']), 3)})")
        prec, rec, _ = precision_recall_curve(y, p["proba"])
        ax2.plot(rec, prec, color=color, linewidth=2,
                 label=f"{name} ({num(average_precision_score(y, p['proba']), 3)})")
    for (name, p), style in zip(leak.items(), ("--", ":")):
        fpr, tpr, _ = roc_curve(y, p["proba"])
        ax1.plot(fpr, tpr, color=MUTED, linewidth=1.5, linestyle=style,
                 label=f"{name} ({num(roc_auc_score(y, p['proba']), 3)})")
        prec, rec, _ = precision_recall_curve(y, p["proba"])
        ax2.plot(rec, prec, color=MUTED, linewidth=1.5, linestyle=style,
                 label=f"{name} ({num(average_precision_score(y, p['proba']), 3)})")
    ax1.plot([0, 1], [0, 1], color="#cbd5e1", linewidth=1)
    ax1.annotate("ghicit la întâmplare", (0.55, 0.5), color=MUTED, fontsize=8, rotation=33)
    ax2.axhline(y.mean(), color="#cbd5e1", linewidth=1)
    ax2.annotate(f"rata de bază {pct(y.mean())}", (0.02, y.mean() + 0.01), color=MUTED, fontsize=8)
    ax1.set(xlabel="rata de fals pozitive", ylabel="rata de adevărat pozitive (recall)", xlim=(0, 1), ylim=(0, 1))
    ax2.set(xlabel="recall", ylabel="precizie", xlim=(0, 1), ylim=(0, 1))
    ax1.set_title("Curba ROC (în paranteză: ROC-AUC)")
    ax2.set_title("Curba precizie–recall (în paranteză: PR-AUC)")
    ax1.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    return save(fig, "results_roc_pr")


def fig_ci(table: pd.DataFrame) -> str:
    t = table[~table["model"].str.startswith("Dummy")].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(t) + 1.2), sharey=True)
    for ax, metric in zip(axes, ("F1", "ROC-AUC")):
        y = np.arange(len(t))
        lo, hi = t[f"{metric} IC 2,5%"], t[f"{metric} IC 97,5%"]
        colors = [MUTED if "leakage" in m else "#0f766e" for m in t["model"]]
        ax.hlines(y, lo, hi, color=colors, linewidth=2)
        ax.scatter(t[metric], y, color=colors, s=30, zorder=3)
        ax.set_yticks(y, t["model"])
        ax.set_title(f"{metric} pe test, cu interval de încredere 95%")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "results_ci")


# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train, val, test = temporal_split(df)
    y = test["profitable"].to_numpy()
    idx = boot_indices(len(test))
    pc, pr = PARAMS["clasificare"], PARAMS["regresie"]

    # 1. Clasificare
    preds = {spec.name: predict_clf(spec, pc[spec.name], train, val, test) for spec in CLASSIFIERS}
    lin, gb = CLASSIFIERS[1], CLASSIFIERS[4]
    preds[f"Regresie logistică, genuri → {K_PCA} PCA"] = predict_clf_pca(lin, pc[lin.name], df, train, val, test)
    leak = {f"{s.name} + leakage": predict_clf(s, pc[s.name], train, val, test, leakage=True) for s in (lin, gb)}

    rows, boots = [], {}
    for name, p in {**preds, **leak}.items():
        m = clf_metrics(y, p["proba"], p["prag"])
        b = boots[name] = clf_boot(y, p["proba"], p["prag"], idx)
        rows.append({"model": name, **{k: m[k] for k in ("accuracy", "precision", "recall")},
                     "F1": m["F1"], "F1 IC 2,5%": ci(b["F1"])[0], "F1 IC 97,5%": ci(b["F1"])[1],
                     "ROC-AUC": m["ROC-AUC"], "ROC-AUC IC 2,5%": ci(b["ROC-AUC"])[0],
                     "ROC-AUC IC 97,5%": ci(b["ROC-AUC"])[1], "PR-AUC": m["PR-AUC"], "prag": m["prag"]})
    clf_table = pd.DataFrame(rows)

    # 2. Testele între modele (doar fără leakage)
    real = clf_table[~clf_table["model"].str.contains("leakage|Dummy")].sort_values("ROC-AUC", ascending=False)
    first, second = real["model"].iloc[0], real["model"].iloc[1]
    best_f1 = real.sort_values("F1", ascending=False)["model"].iloc[0]
    pred_of = {n: (p["proba"] >= p["prag"]).astype(int) for n, p in preds.items()}
    tests = []
    pairs = [(first, second), (best_f1, first) if best_f1 != first else (first, real["model"].iloc[2]),
             (first, "Dummy (clasa majoritară)")]
    for a, b in pairs:
        mc = mcnemar(y, pred_of[a], pred_of[b])
        d_auc = boots[a]["ROC-AUC"] - boots[b]["ROC-AUC"]
        d_f1 = boots[a]["F1"] - boots[b]["F1"]
        tests.append({"A": a, "B": b, **mc,
                      "Δ ROC-AUC (A − B)": float(np.mean(d_auc)), "Δ ROC-AUC IC": f"[{num(ci(d_auc)[0])}; {num(ci(d_auc)[1])}]",
                      "Δ F1 (A − B)": float(np.mean(d_f1)), "Δ F1 IC": f"[{num(ci(d_f1)[0])}; {num(ci(d_f1)[1])}]"})
    tests = pd.DataFrame(tests)

    # 3. Regresie
    y_log = test["log_roi"].to_numpy()
    reg_rows = []
    reg_specs = [(s, False) for s in REGRESSORS] + [(REGRESSORS[1], True), (REGRESSORS[4], True)]
    for spec, leakage in reg_specs:
        pred = predict_reg(spec, pr[spec.name], train, test, leakage)
        m, b = reg_metrics(y_log, pred), reg_boot(y_log, pred, idx)
        reg_rows.append({"model": spec.name + (" + leakage" if leakage else ""), "RMSE log_roi": m["RMSE log_roi"],
                         "RMSE IC": f"[{num(ci(b['RMSE log_roi'])[0])}; {num(ci(b['RMSE log_roi'])[1])}]",
                         "MAE log_roi": m["MAE log_roi"], "R² log_roi": m["R² log_roi"],
                         "R² IC": f"[{num(ci(b['R² log_roi'])[0])}; {num(ci(b['R² log_roi'])[1])}]",
                         "MedAE ROI": m["MedAE ROI"]})
    reg_table = pd.DataFrame(reg_rows)

    # 4. Pragurile ROI
    thr_rows = []
    for t in (2.0, 2.5, 3.0):
        d = load_dataset(threshold=t)
        tr, va, te = temporal_split(d)
        yt = te["profitable"].to_numpy()
        it = boot_indices(len(te))
        for spec in CLASSIFIERS[1:]:
            p = predict_clf(spec, pc[spec.name], tr, va, te)
            m, b = clf_metrics(yt, p["proba"], p["prag"]), clf_boot(yt, p["proba"], p["prag"], it)
            thr_rows.append({"prag ROI": num(t, 1), "% profitabile (test)": yt.mean(), "model": spec.name,
                             "F1": m["F1"], "F1 IC": f"[{num(ci(b['F1'])[0])}; {num(ci(b['F1'])[1])}]",
                             "ROC-AUC": m["ROC-AUC"], "ROC-AUC IC": f"[{num(ci(b['ROC-AUC'])[0])}; {num(ci(b['ROC-AUC'])[1])}]",
                             "PR-AUC": m["PR-AUC"]})
    thr_table = pd.DataFrame(thr_rows)

    # 5. Salvare
    clf_table.to_csv(FINAL, index=False, float_format="%.6f")
    for name, t in (("classification", clf_table), ("tests", tests), ("regression", reg_table), ("thresholds", thr_table)):
        t.to_csv(OUT_DIR / f"{name}.csv", index=False, float_format="%.6f")
    pd.DataFrame({"imdb_id": test["imdb_id"], "title": test["title"], "release_year": test["release_year"],
                  "profitable": y, "roi": test["roi"],
                  **{f"proba_{n}": p["proba"] for n, p in preds.items() if not n.startswith("Dummy")}}
                 ).to_csv(OUT_DIR / "test_predictions.csv", index=False, float_format="%.6f")
    (OUT_DIR / "thresholds_chosen.json").write_text(json.dumps({n: p["prag"] for n, p in {**preds, **leak}.items()},
                                                               indent=2, ensure_ascii=False), encoding="utf-8")

    figs = {"curves": fig_curves(y, {n: p for n, p in preds.items() if not n.startswith("Dummy")}, leak),
            "ci": fig_ci(clf_table)}
    write_report(train, val, test, clf_table, tests, reg_table, thr_table, figs)
    print(f"Scris: {REPORT}")


def fmt(df: pd.DataFrame, digits=3, pct_cols=(), p_cols=()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(pct)
        elif c in p_cols:
            out[c] = out[c].map(lambda v: f"{v:.1e}" if v < 0.001 else num(v, 3))
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, digits))
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, 0))
    return out


def write_report(train, val, test, clf_table, tests, reg_table, thr_table, figs) -> None:
    show = clf_table.copy()
    for metric in ("F1", "ROC-AUC"):
        show[f"{metric} (IC 95%)"] = [f"{num(v)} [{num(lo)}; {num(hi)}]" for v, lo, hi in
                                      zip(show[metric], show[f"{metric} IC 2,5%"], show[f"{metric} IC 97,5%"])]
    show = show[["model", "F1 (IC 95%)", "ROC-AUC (IC 95%)", "PR-AUC", "precision", "recall", "accuracy", "prag"]]
    text = f"""# Rezultate: comparația finală a modelelor (#56)

Generat cu `python modeling/compare.py`. Tabele: `results/final_metrics.csv`, `results/compare/*.csv`; probabilitățile pe test ale fiecărui model: `results/compare/test_predictions.csv` (pentru analiza erorilor, #57).

## Protocol

- **Același test pentru toate modelele:** filmele din 2019–2024 ({num(len(test), 0)} filme, {pct(test['profitable'].mean())} profitabile), folosite o singură dată. Antrenare pe 1950–2014 ({num(len(train), 0)} filme); hiperparametrii (#41) și pragul de decizie aleși pe 2015–2018 ({num(len(val), 0)} filme).
- **Intervale de încredere 95%** prin bootstrap pe test: {N_BOOT} de eșantioane de filme cu înlocuire, aceleași pentru toate modelele, deci diferențele dintre modele sunt calculate pe aceleași filme.
- **Testul McNemar** compară două clasificatoare pe filmele unde exact unul dintre ele greșește. Dacă modelele sunt la fel de bune, cele două tipuri de dezacord ar trebui să fie cam la fel de frecvente.
- Varianta principală e **fără leakage**; modelele cu voturi + keywords TMDB apar doar ca referință (gri / „+ leakage”).

## Clasificare: `profitable` (ROI ≥ 2,5)

{md_table(fmt(show))}

{figs['ci']}

{figs['curves']}

## Sunt diferențele reale?

{md_table(fmt(tests, p_cols=['p (χ²)', 'p (exact, binomial)'], digits=3))}

Δ = diferența medie pe cele {N_BOOT} de eșantioane bootstrap; IC = intervalul de 95% al diferenței. Dacă intervalul conține 0, diferența nu e semnificativă.

## Regresie: `log_roi`

{md_table(fmt(reg_table))}

## Pragurile ROI 2 / 2,5 / 3

Ținta recalculată la fiecare prag; aceiași hiperparametri, pragul de decizie ales din nou pe validare.

{md_table(fmt(thr_table, pct_cols=['% profitabile (test)']))}

## Concluzii

{CONCLUSIONS}
"""
    REPORT.write_text(text, encoding="utf-8")


CONCLUSIONS = """1. **Nu există un câștigător clar între modelele reale.** Toate au ROC-AUC 0,77–0,78 și F1 0,54–0,56,
   iar intervalele de încredere se suprapun aproape complet. Diferențele perechi pe bootstrap conțin 0:
   Random Forest vs gradient boosting (ROC-AUC +0,002, IC [−0,01; 0,01]), regresia logistică + buget²
   vs Random Forest (F1 +0,022, IC [−0,00; 0,05]). Pe datele noastre, un model liniar simplu, cu bugetul
   la pătrat, e la fel de bun ca un ansamblu de arbori, și e mai ușor de explicat.
2. **Toate bat clar baseline-ul:** ROC-AUC +0,28 (IC [0,26; 0,30]) și F1 +0,54 față de Dummy. Acuratețea
   spune povestea inversă: Dummy are 73,9%, iar Random Forest 62,2%, pentru că Dummy nu spune
   niciodată „profitabil”. De aceea acuratețea nu e metrica după care alegem.
3. **McNemar iese semnificativ chiar când F1 și ROC-AUC nu diferă.** Testul compară greșelile la
   pragul ales al fiecărui model, adică acuratețea. Modelele au praguri diferite (0,39 vs 0,45):
   unul spune „profitabil” mai des și greșește pe alte filme. McNemar arată că modelele **se comportă
   diferit**, nu că unul e mai bun: diferența ține de alegerea pragului (precizie vs recall), nu de
   cât de bine ordonează modelele filmele.
4. **Leakage:** cu voturile și keywords TMDB, ROC-AUC urcă la 0,83 (regresie logistică) și 0,87
   (gradient boosting), cu intervale care nu se suprapun cu modelele fără leakage. Câștigul de
   aproximativ 0,1 e exact informația care nu există înainte de lansare.
5. **Regresie:** R² ≈ 0,25 (IC [0,22; 0,28] la Ridge + buget²). Modelele liniare au eroarea mediană pe
   ROI mai mică (0,55 vs 0,71–0,76 la arbori): arborii greșesc mai mult la filmul obișnuit.
6. **Pragul ROI:** ROC-AUC rămâne 0,77–0,79 la 2, 2,5 și 3, cu intervale care se suprapun; F1 scade odată
   cu pragul (≈ 0,60 → 0,55 → 0,51), pentru că filmele „profitabile” devin mai rare (31,5% → 22,1% în
   test). Clasamentul modelelor nu se schimbă semnificativ la niciun prag.
7. **Modelul de referință pentru restul proiectului** (analiza erorilor #57, interpretabilitatea #58):
   **gradient boosting** pentru interpretabilitate (prinde interacțiuni și forma de U fără coloane
   construite manual) și **regresia logistică + buget²** ca model simplu de comparație."""

if __name__ == "__main__":
    main()
