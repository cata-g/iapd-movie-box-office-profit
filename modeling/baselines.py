"""
Modele de bază și experimente (#41).

Run:
  python modeling/baselines.py    # -> docs/baseline_results.md, results/baselines/*.csv|json, figuri

Protocol (split temporal din #40):
  - fiecare model se antrenează pe train (1950–2014);
  - hiperparametrii se aleg pe validare (2015–2018): ROC-AUC la clasificare, RMSE pe log_roi la
    regresie; fără CV aleator, ca să nu intre filme din viitor în antrenare;
  - pragul de decizie al clasificatorilor se alege tot pe validare (F1 maxim);
  - testul (2019–2024) se folosește o singură dată, la final, cu modelul antrenat pe train.

Experimente (fiecare schimbă o singură decizie față de varianta principală):
  class_weight balanced vs fără, prag ales pe validare vs 0,5, leakage (voturi + keywords) vs fără,
  cu / fără outliers în train, fără premii, cu buget² (forma de U), imputare median / kNN /
  iterativă, evaluare doar pe filmele cu surse care se pun de acord, test fără 2020–2021 (COVID)
  și split stratificat în loc de temporal.
"""

import json
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, r2_score, recall_score,
                             roc_auc_score, root_mean_squared_error)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import PROJECT_ROOT
from modeling.preprocessing import (SEED, X_y, build_preprocessor, load_dataset, stratified_split,
                                    temporal_split)

OUT_DIR = PROJECT_ROOT / "results" / "baselines"
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
REPORT = PROJECT_ROOT / "docs" / "baseline_results.md"

AWARDS = ("log1p_director_award_noms_before", "log1p_director_award_wins_before",
          "log1p_cast_award_noms_before", "log1p_cast_award_wins_before")
BUDGET_SQ = ("log_budget_real_sq",)

INK, INK_2, MUTED = "#0f172a", "#334155", "#64748b"


@dataclass
class Spec:
    """Un model: tipul de preprocesare, estimatorul și grila de hiperparametri (pe validare)."""
    name: str
    kind: str                       # "linear" (scalat) sau "tree"
    estimator: object
    grid: dict = field(default_factory=dict)
    extra: tuple[str, ...] = ()     # coloane numerice în plus (buget²)


CLASSIFIERS = [
    Spec("Dummy (clasa majoritară)", "tree", DummyClassifier(strategy="prior")),
    Spec("Regresie logistică", "linear",
         LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
         {"C": [0.01, 0.1, 1, 10]}),
    Spec("Regresie logistică + buget²", "linear",
         LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
         {"C": [0.01, 0.1, 1, 10]}, extra=BUDGET_SQ),
    Spec("Random Forest", "tree",
         RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
         {"max_depth": [None, 12], "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.3]}),
    Spec("Gradient boosting", "tree",
         HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, early_stopping=False,
                                        random_state=SEED),
         {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50],
          "l2_regularization": [0.0, 1.0]}),
]

REGRESSORS = [
    Spec("Dummy (mediana)", "tree", DummyRegressor(strategy="median")),
    Spec("Ridge", "linear", Ridge(), {"alpha": [0.1, 1, 10, 100]}),
    Spec("Ridge + buget²", "linear", Ridge(), {"alpha": [0.1, 1, 10, 100]}, extra=BUDGET_SQ),
    Spec("Random Forest", "tree",
         RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
         {"max_depth": [None, 12], "min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.3]}),
    Spec("Gradient boosting", "tree",
         HistGradientBoostingRegressor(max_iter=300, early_stopping=False, random_state=SEED),
         {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50],
          "l2_regularization": [0.0, 1.0]}),
]


# ---------------------------------------------------------------------------
# Antrenare și metrici
# ---------------------------------------------------------------------------
def make_pipe(spec: Spec, params: dict, leakage=False, imputer="median", exclude=(), **est_params) -> Pipeline:
    pre = build_preprocessor(spec.kind, leakage=leakage, imputer=imputer, extra=spec.extra, exclude=exclude)
    return Pipeline([("pre", pre), ("model", clone(spec.estimator).set_params(**params, **est_params))])


def xy(df, target, spec: Spec, leakage=False, exclude=()):
    return X_y(df, target=target, leakage=leakage, extra=spec.extra, exclude=exclude)


def grid_points(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*grid.values())] or [{}]


def best_threshold(y, proba) -> float:
    """Pragul cu F1 maxim (pe validare)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def clf_metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    single_class = len(np.unique(proba)) == 1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": 0.5 if single_class else roc_auc_score(y, proba),
            "PR-AUC": average_precision_score(y, proba), "prag": threshold,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "n": int(len(y))}


def reg_metrics(log_true, log_pred) -> dict:
    roi_true, roi_pred = np.exp(log_true), np.exp(log_pred)
    return {"MAE log_roi": mean_absolute_error(log_true, log_pred),
            "RMSE log_roi": root_mean_squared_error(log_true, log_pred),
            "R² log_roi": r2_score(log_true, log_pred),
            "MAE ROI": mean_absolute_error(roi_true, roi_pred),
            "MedAE ROI": float(np.median(np.abs(roi_true - roi_pred))),
            # ce ar da regresia folosită drept clasificator (ROI prezis ≥ 2,5)
            "F1 din regresie": f1_score(roi_true >= 2.5, roi_pred >= 2.5, zero_division=0),
            "n": int(len(log_true))}


def tune(spec: Spec, train, val, task: str, log: list) -> dict:
    """Alege hiperparametrii pe validare; păstrează toată grila în `log`."""
    target = "profitable" if task == "clf" else "log_roi"
    X_tr, y_tr = xy(train, target, spec)
    X_va, y_va = xy(val, target, spec)
    best, best_score = {}, -np.inf
    for params in grid_points(spec.grid):
        pipe = make_pipe(spec, params).fit(X_tr, y_tr)
        if task == "clf":
            score = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1]) if spec.grid else 0.5
        else:
            score = -root_mean_squared_error(y_va, pipe.predict(X_va))
        log.append({"task": task, "model": spec.name, **{k: str(v) for k, v in params.items()},
                    "scor validare": score if task == "clf" else -score})
        if score > best_score:
            best, best_score = params, score
    return best


def run_clf(spec: Spec, params, train, val, tests: dict, leakage=False, imputer="median", exclude=(),
            fixed_threshold=None, **est_params) -> dict:
    """Antrenează pe train, alege pragul pe validare, evaluează pe fiecare set din `tests`."""
    pipe = make_pipe(spec, params, leakage, imputer, exclude, **est_params)
    pipe.fit(*xy(train, "profitable", spec, leakage, exclude))
    X_va, y_va = xy(val, "profitable", spec, leakage, exclude)
    p_va = pipe.predict_proba(X_va)[:, 1]
    threshold = fixed_threshold if fixed_threshold is not None else best_threshold(y_va, p_va)
    out = {"validare": clf_metrics(y_va, p_va, threshold)}
    for name, df in tests.items():
        X, y = xy(df, "profitable", spec, leakage, exclude)
        out[name] = clf_metrics(y, pipe.predict_proba(X)[:, 1], threshold)
    out["_pipe"] = pipe
    return out


def run_reg(spec: Spec, params, train, val, tests: dict, leakage=False, exclude=()) -> dict:
    pipe = make_pipe(spec, params, leakage, exclude=exclude)
    pipe.fit(*xy(train, "log_roi", spec, leakage, exclude))
    out = {}
    for name, df in {"validare": val, **tests}.items():
        X, y = xy(df, "log_roi", spec, leakage, exclude)
        out[name] = reg_metrics(y.to_numpy(), pipe.predict(X))
    return out


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------
def num(x, digits=3) -> str:
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", ".")
    if pd.isna(x):
        return "–"
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def md_table(df: pd.DataFrame, digits=3) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(v if isinstance(v, str) else num(v, digits) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def fig_confusion(results: dict, names: list[str]) -> str:
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.3))
    for ax, name in zip(np.atleast_1d(axes), names):
        m = results[name]["test"]
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=cm.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["prezis: nu", "prezis: da"])
        ax.set_yticks([0, 1], ["real: nu", "real: da"])
        ax.set_title(f"{name}\nF1 {num(m['F1'], 2)}, prag {num(m['prag'], 2)}", fontsize=10, loc="left", color=INK)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "baseline_confusion.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "![baseline_confusion](figures/baseline_confusion.png)"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train, val, test = temporal_split(df)
    no_covid = test[~test["release_year"].isin([2020, 2021])]
    agree = test[(test["budget_n_agree"] >= 2) & (test["gross_n_agree"] >= 2)]
    tests = {"test": test, "test fără 2020–2021": no_covid, "test, surse de acord": agree}
    grid_log: list[dict] = []

    # 1. Clasificare: tuning pe validare, evaluare pe test
    best_clf, clf = {}, {}
    for spec in CLASSIFIERS:
        best_clf[spec.name] = tune(spec, train, val, "clf", grid_log)
        clf[spec.name] = run_clf(spec, best_clf[spec.name], train, val, tests,
                                 fixed_threshold=0.5 if spec.name.startswith("Dummy") else None)
        print(f"clf  {spec.name}: {best_clf[spec.name]}  test F1 {clf[spec.name]['test']['F1']:.3f} "
              f"AUC {clf[spec.name]['test']['ROC-AUC']:.3f}")

    # 2. Regresie pe log_roi
    best_reg, reg = {}, {}
    for spec in REGRESSORS:
        best_reg[spec.name] = tune(spec, train, val, "reg", grid_log)
        reg[spec.name] = run_reg(spec, best_reg[spec.name], train, val, tests)
        print(f"reg  {spec.name}: {best_reg[spec.name]}  test RMSE {reg[spec.name]['test']['RMSE log_roi']:.3f}")

    # 3. Experimente: o singură schimbare față de varianta principală, pe 2 modele
    lin, gb = CLASSIFIERS[1], CLASSIFIERS[4]
    q1, q3 = train["log_roi"].quantile([0.25, 0.75])
    iqr = q3 - q1
    train_no_out = train[train["log_roi"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    strat_tr, strat_va, strat_te = stratified_split(df)
    experiments = []

    def add(name, spec, res, note=""):
        experiments.append({"experiment": name, "model": spec.name, **{k: res["test"][k] for k in
                            ("F1", "ROC-AUC", "PR-AUC", "precision", "recall", "accuracy")},
                            "prag": res["test"]["prag"], "observații": note})

    for spec in (lin, gb):
        p = best_clf[spec.name]
        add("principal", spec, clf[spec.name])
        add("fără class_weight", spec, run_clf(spec, p, train, val, {"test": test}, class_weight=None))
        add("prag fix 0,5", spec, run_clf(spec, p, train, val, {"test": test}, fixed_threshold=0.5))
        add("cu leakage (voturi + keywords)", spec, run_clf(spec, p, train, val, {"test": test}, leakage=True))
        add("fără outliers în train", spec, run_clf(spec, p, train_no_out, val, {"test": test}),
            f"{len(train) - len(train_no_out)} filme scoase (log_roi în afara 1,5×IQR)")
        add("fără premii", spec, run_clf(spec, p, train, val, {"test": test}, exclude=AWARDS))
        add("test fără 2020–2021", spec, {"test": clf[spec.name]["test fără 2020–2021"]},
            f"{len(no_covid)} filme")
        add("test, surse de acord", spec, {"test": clf[spec.name]["test, surse de acord"]},
            f"{len(agree)} filme cu buget și încasări confirmate de ≥ 2 surse")
        add("split stratificat", spec, run_clf(spec, p, strat_tr, strat_va, {"test": strat_te}),
            "aleator 70/15/15, doar pentru comparație")
    for imputer in ("knn", "iterative"):
        add(f"imputare {imputer}", lin, run_clf(lin, best_clf[lin.name], train, val, {"test": test}, imputer=imputer))

    reg_exp = []
    for spec in (REGRESSORS[1], REGRESSORS[4]):
        p = best_reg[spec.name]
        for name, res in (("principal", reg[spec.name]),
                          ("cu leakage", run_reg(spec, p, train, val, {"test": test}, leakage=True)),
                          ("fără outliers în train", run_reg(spec, p, train_no_out, val, {"test": test}))):
            reg_exp.append({"experiment": name, "model": spec.name, **res["test"]})

    # 4. Salvare
    clf_rows = [{"model": n, "set": s, **m} for n, r in clf.items() for s, m in r.items() if s != "_pipe"]
    reg_rows = [{"model": n, "set": s, **m} for n, r in reg.items() for s, m in r.items()]
    pd.DataFrame(clf_rows).to_csv(OUT_DIR / "classification.csv", index=False, float_format="%.6f")
    pd.DataFrame(reg_rows).to_csv(OUT_DIR / "regression.csv", index=False, float_format="%.6f")
    pd.DataFrame(experiments).to_csv(OUT_DIR / "experiments_classification.csv", index=False, float_format="%.6f")
    pd.DataFrame(reg_exp).to_csv(OUT_DIR / "experiments_regression.csv", index=False, float_format="%.6f")
    pd.DataFrame(grid_log).to_csv(OUT_DIR / "grid_search.csv", index=False, float_format="%.6f")
    (OUT_DIR / "best_params.json").write_text(json.dumps(
        {"clasificare": best_clf, "regresie": best_reg}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    write_report(df, train, val, test, no_covid, agree, clf, reg, best_clf, best_reg,
                 pd.DataFrame(experiments), pd.DataFrame(reg_exp))
    print(f"Scris: {REPORT}")


def write_report(df, train, val, test, no_covid, agree, clf, reg, best_clf, best_reg, exp, reg_exp) -> None:
    cols_c = ["F1", "ROC-AUC", "PR-AUC", "precision", "recall", "accuracy", "prag"]
    main_c = pd.DataFrame([{"model": n, **{c: r["test"][c] for c in cols_c},
                            "F1 validare": r["validare"]["F1"], "AUC validare": r["validare"]["ROC-AUC"]}
                           for n, r in clf.items()])
    cols_r = ["RMSE log_roi", "MAE log_roi", "R² log_roi", "MAE ROI", "MedAE ROI", "F1 din regresie"]
    main_r = pd.DataFrame([{"model": n, **{c: r["test"][c] for c in cols_r},
                            "RMSE validare": r["validare"]["RMSE log_roi"]} for n, r in reg.items()])
    params = pd.DataFrame([{"sarcină": t, "model": n, "hiperparametri aleși": ", ".join(f"{k} = {v}" for k, v in p.items()) or "–"}
                           for t, d in (("clasificare", best_clf), ("regresie", best_reg)) for n, p in d.items()])
    fig = fig_confusion(clf, ["Regresie logistică", "Regresie logistică + buget²", "Gradient boosting"])
    best = main_c.iloc[1:].sort_values("ROC-AUC", ascending=False).iloc[0]
    best_r = main_r.iloc[1:].sort_values("RMSE log_roi").iloc[0]
    dummy = clf["Dummy (clasa majoritară)"]["test"]

    text = f"""# Modele de bază și experimente (#41)

Generat cu `python modeling/baselines.py` (seed {SEED}; rulat de două ori, dă aceleași cifre). Tabelele complete: `results/baselines/*.csv`; hiperparametrii: `results/baselines/best_params.json`.

## Protocol

- **Split temporal** (#40): train 1950–2014 ({num(len(train))} filme), validare 2015–2018 ({num(len(val))}), test 2019–2024 ({num(len(test))}, {num(100 * test['profitable'].mean(), 1)}% profitabile).
- Fiecare model se antrenează pe train. **Hiperparametrii** se aleg pe validare (ROC-AUC la clasificare, RMSE pe `log_roi` la regresie), dintr-o grilă mică; fără validare încrucișată aleatoare, ca să nu intre filme din viitor în antrenare.
- **Pragul de decizie** al clasificatorilor (probabilitatea peste care spunem „profitabil”) se alege pe validare, ca F1 să fie maxim. Pragul implicit 0,5 nu e potrivit când doar o treime din filme sunt profitabile.
- **Testul** se folosește o singură dată, la final. Clasele sunt dezechilibrate, deci metrica principală e **F1** (și ROC-AUC, care nu depinde de prag); acuratețea singură înșală (vezi Dummy).
- Coloanele de intrare: cele 63 din #40 / #42, fără leakage. Regresia logistică și Ridge primesc date scalate; arborii, nescalate.

## Clasificare: `profitable` (ROI ≥ 2,5), pe test

{md_table(main_c)}

- **Baseline-ul trivial** (mereu „neprofitabil”) are acuratețe {num(100 * dummy['accuracy'], 1)}%, dar F1 = 0: nu găsește niciun film profitabil. Orice model trebuie judecat față de el pe F1 și ROC-AUC, nu pe acuratețe.
- **Cel mai bun model** după ROC-AUC pe test: **{best['model']}** (ROC-AUC {num(best['ROC-AUC'])}, F1 {num(best['F1'])}).

{fig}

### Hiperparametri aleși pe validare

{md_table(params)}

## Regresie: `log_roi`, pe test

{md_table(main_r)}

- RMSE și MAE sunt pe `log_roi` (o eroare de 1 înseamnă un ROI greșit de aproximativ e ≈ 2,7 ori). MAE pe ROI e dominat de câteva filme cu ROI uriaș; mediana erorii absolute (MedAE) descrie mai bine un film obișnuit.
- „F1 din regresie”: regresia folosită ca un clasificator (ROI prezis ≥ 2,5), comparabilă cu tabelul de clasificare.
- **Cel mai bun model** după RMSE: **{best_r['model']}** (RMSE {num(best_r['RMSE log_roi'])}, R² {num(best_r['R² log_roi'])}).

## Experimente (clasificare, pe test)

Fiecare rând schimbă **o singură** decizie față de varianta principală, pe regresia logistică și pe gradient boosting, cu hiperparametrii aleși mai sus.

{md_table(exp)}

## Experimente (regresie, pe test)

{md_table(reg_exp[['experiment', 'model', 'RMSE log_roi', 'MAE log_roi', 'R² log_roi', 'MedAE ROI']])}

## Concluzii

{CONCLUSIONS}
"""
    REPORT.write_text(text, encoding="utf-8")


CONCLUSIONS = """1. **Toate modelele reale bat clar baseline-ul**, dar ajung la același plafon: ROC-AUC 0,77–0,78 pe
   test, F1 0,54–0,56. Diferența dintre regresia logistică și gradient boosting e mică (sub 0,01 la
   ROC-AUC). Limita vine din informația disponibilă înainte de lansare, nu din model; dacă un model
   e semnificativ mai bun decât altul se testează în #56 (bootstrap, McNemar).
2. **Bugetul la pătrat ajută modelul liniar** să prindă forma de U din EDA: F1 0,544 → 0,563 la
   regresia logistică (la ROC-AUC diferența e în marja de zgomot), iar Ridge + buget² are cel mai mic
   RMSE la regresie. Arborii o prind singuri.
3. **Leakage-ul se vede imediat:** cu voturile și keywords TMDB (cunoscute abia după lansare),
   ROC-AUC urcă de la 0,775 la 0,867 la gradient boosting, iar R² la regresie de la 0,22 la 0,42.
   Un model care le folosește ar părea mult mai bun decât e în realitate; de aceea sunt excluse (#54).
4. **Dezechilibrul de clase se rezolvă prin prag, nu prin ponderi:** fără `class_weight`, regresia
   logistică dă exact aceleași predicții după ce pragul e ales pe validare (0,24 în loc de 0,39);
   ponderile doar mută probabilitățile. Contează ca pragul să fie ales, nu lăsat la 0,5.
5. **Pragul ales pe validare nu se transferă perfect pe test:** validarea are 33,5% profitabile, testul
   26,1%. La regresia logistică, pragul fix 0,5 dă pe test un F1 ușor mai mare (0,556 vs 0,544). Asta e
   o schimbare a ratei de bază în timp (COVID, §EDA), de urmărit în analiza erorilor (#57).
6. **COVID:** fără 2020–2021, F1 crește la 0,57–0,59, iar ROC-AUC rămâne la fel: modelul ordonează
   filmele la fel de bine, dar în acei ani mai puține filme au trecut pragul.
7. **Calitatea țintei:** pe cele 817 filme de test cu buget și încasări confirmate de cel puțin 2 surse,
   F1 crește (0,60–0,61), dar ROC-AUC scade (0,69–0,71). Subsetul e diferit (38% profitabile, filme
   mai cunoscute), deci nu e o comparație directă; o reluăm în #57, pe segmente.
8. **Outliers, premii, imputare:** fără outliers în train, clasificarea rămâne practic la fel, dar
   regresia pierde (R² 0,25 → 0,21 la Ridge): filmele cu ROI extrem sunt reale și utile. Fără premii,
   rezultatele nu se schimbă (premiile aduc foarte puțin, cum arăta EDA). Imputarea kNN sau
   iterativă dă același rezultat ca mediana, deci păstrăm mediana (mai simplă).
9. **Split stratificat vs temporal:** gradient boosting are F1 0,604 pe un test aleator față de 0,555
   pe testul temporal. Un split aleator ar fi supraestimat performanța reală, pe filme viitoare.
10. **Regresia e mai grea decât clasificarea:** R² ≈ 0,25 pe `log_roi`, iar folosită ca clasificator
   (ROI prezis ≥ 2,5) dă F1 de 0,18–0,45, mult sub clasificatorii direcți. Pentru decizia „profitabil
   sau nu” folosim clasificarea; regresia rămâne pentru a estima cât de mare e ROI-ul."""

if __name__ == "__main__":
    main()
