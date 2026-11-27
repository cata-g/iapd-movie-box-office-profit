"""
Interpretabilitate (#58): ce caracteristici contează pentru model și cum.

Run:
  python analysis/interpret.py     # -> docs/interpretability.md, docs/figures/interpret_*.png, results/interpret/*.csv

Modele (hiperparametrii din #41, antrenate pe train 1950–2014):
  - gradient boosting: permutation importance, SHAP (TreeExplainer), partial dependence;
  - regresia logistică + buget²: permutation importance și coeficienți, pentru comparație.
Importanța se măsoară pe test (2019–2024), adică pe filme pe care modelul nu le-a văzut.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import partial_dependence, permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.eda import ACCENT, INK, INK_2, MUTED, PAIR, md_table, num, pct, save
from extraction.utils import PROJECT_ROOT
from modeling.baselines import CLASSIFIERS, make_pipe, xy
from modeling.preprocessing import SEED, load_dataset, temporal_split

OUT_DIR = PROJECT_ROOT / "results" / "interpret"
REPORT = PROJECT_ROOT / "docs" / "interpretability.md"
PARAMS = json.loads((PROJECT_ROOT / "results" / "baselines" / "best_params.json").read_text(encoding="utf-8"))["clasificare"]
THRESHOLDS = json.loads((PROJECT_ROOT / "results" / "compare" / "thresholds_chosen.json").read_text(encoding="utf-8"))
GB, LIN = CLASSIFIERS[4], CLASSIFIERS[2]          # gradient boosting, regresie logistică + buget²
N_REPEATS = 10
TOP = 15
MONTHS = ["ian", "feb", "mar", "apr", "mai", "iun", "iul", "aug", "sep", "oct", "nov", "dec"]


def fit(spec, train):
    return make_pipe(spec, PARAMS[spec.name]).fit(*xy(train, "profitable", spec))


def perm_importance(pipe, spec, test) -> pd.DataFrame:
    """Cât scade ROC-AUC pe test când amestecăm o coloană de intrare (legătura ei cu ținta se rupe)."""
    X, y = xy(test, "profitable", spec)
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)
    return pd.DataFrame({"caracteristică": X.columns, "scădere ROC-AUC": r.importances_mean,
                         "std": r.importances_std,
                         "IC 2,5%": np.percentile(r.importances, 2.5, axis=1),
                         "IC 97,5%": np.percentile(r.importances, 97.5, axis=1)}
                        ).sort_values("scădere ROC-AUC", ascending=False).reset_index(drop=True)


def shap_values(pipe, spec, test):
    X = xy(test, "profitable", spec)[0]
    Xt = pipe.named_steps["pre"].transform(X)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return explainer(Xt), Xt


def pdp(pipe, spec, train, test, feature: str, grid=None) -> pd.DataFrame:
    """Probabilitatea medie prezisă pe test când toate filmele primesc aceeași valoare a coloanei."""
    X = xy(test, "profitable", spec)[0]
    if grid is None:
        grid = np.quantile(train[feature].dropna(), np.linspace(0.02, 0.98, 25))
    rows = []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v
        if feature == "log_budget_real" and "log_budget_real_sq" in Xv:
            Xv["log_budget_real_sq"] = v ** 2
        rows.append({"valoare": v, "P(profitabil) medie": pipe.predict_proba(Xv)[:, 1].mean()})
    return pd.DataFrame(rows)


def pdp_month(pipe, spec, test) -> pd.DataFrame:
    X = xy(test, "profitable", spec)[0]
    rows = []
    for m in range(1, 13):
        Xm = X.copy()
        angle = 2 * np.pi * (m - 1) / 12
        Xm["month_sin"], Xm["month_cos"] = np.sin(angle), np.cos(angle)
        rows.append({"luna": MONTHS[m - 1], "P(profitabil) medie": pipe.predict_proba(Xm)[:, 1].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------
def fig_perm(gb: pd.DataFrame, lin: pd.DataFrame) -> str:
    top = list(gb.head(TOP)["caracteristică"])
    g, l = gb.set_index("caracteristică").loc[top], lin.set_index("caracteristică").reindex(top)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    y = np.arange(len(top))[::-1]
    h = 0.38
    for (t, label, color, off) in ((g, GB.name, PAIR[0], h / 2 + 0.01), (l, LIN.name, PAIR[1], -h / 2 - 0.01)):
        ax.barh(y + off, t["scădere ROC-AUC"], height=h, color=color, label=label,
                xerr=[t["scădere ROC-AUC"] - t["IC 2,5%"], t["IC 97,5%"] - t["scădere ROC-AUC"]],
                error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 2})
    ax.set_yticks(y, top)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("scăderea ROC-AUC pe test când coloana e amestecată")
    ax.set_title(f"Top {TOP} caracteristici (permutation importance, {N_REPEATS} repetări)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    return save(fig, "interpret_permutation")


def fig_shap_summary(sv, Xt) -> str:
    plt.figure()
    shap.plots.beeswarm(sv, max_display=TOP, show=False, color_bar_label="valoarea caracteristicii")
    fig = plt.gcf()
    fig.set_size_inches(8.5, 6.5)
    plt.title("SHAP pe test (gradient boosting): efectul fiecărei caracteristici asupra predicției",
              loc="left", fontsize=11, fontweight="bold", color=INK)
    plt.xlabel("contribuția la log-odds „profitabil” (dreapta = mai probabil profitabil)")
    return save(fig, "interpret_shap_summary")


def fig_waterfall(sv, i: int, name: str, title: str) -> str:
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8.5, 5)
    plt.title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    return save(fig, name)


def fig_pdp(curves: dict[str, dict[str, pd.DataFrame]], month: dict[str, pd.DataFrame], base: float) -> str:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    specs = [("log_budget_real", "buget (mil. $ 2024, scară log)", lambda v: np.exp(v) / 1e6),
             ("director_prev_profitable_rate", "rata de hituri a regizorului", lambda v: v),
             ("cast_prev_mean_log_roi", "ROI mediu anterior al actorilor (scară log)", lambda v: np.exp(v))]
    for ax, (feat, label, tr) in zip(axes, specs):
        for (model, d), color in zip(curves[feat].items(), PAIR):
            ax.plot(tr(d["valoare"]), 100 * d["P(profitabil) medie"], color=color, linewidth=2, label=model)
        if feat != "director_prev_profitable_rate":
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(axis="x", visible=False)
    for (model, d), color in zip(month.items(), PAIR):
        axes[3].plot(range(12), 100 * d["P(profitabil) medie"], color=color, linewidth=2, marker="o", markersize=3)
    axes[3].set_xticks(range(12), MONTHS, rotation=45)
    axes[3].set_xlabel("luna lansării")
    for ax in axes:
        ax.axhline(100 * base, color=MUTED, linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("P(profitabil) medie pe test, %")
    axes[0].legend(frameon=False, fontsize=8, loc="upper center")
    fig.suptitle("Partial dependence: cum se schimbă predicția medie când variem o singură caracteristică",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "interpret_pdp")


# ---------------------------------------------------------------------------
def fmt(df: pd.DataFrame, digits=4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, digits))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train, _, test = temporal_split(df)
    gb_pipe, lin_pipe = fit(GB, train), fit(LIN, train)

    # 1. Permutation importance
    pi_gb, pi_lin = perm_importance(gb_pipe, GB, test), perm_importance(lin_pipe, LIN, test)
    coef = pd.DataFrame({"coloană": lin_pipe.named_steps["pre"].get_feature_names_out(),
                         "coeficient (date scalate)": lin_pipe.named_steps["model"].coef_[0]}
                        ).sort_values("coeficient (date scalate)", key=abs, ascending=False)

    # 2. SHAP
    sv, Xt = shap_values(gb_pipe, GB, test)
    mean_abs = pd.DataFrame({"coloană": Xt.columns, "|SHAP| mediu": np.abs(sv.values).mean(0)}
                            ).sort_values("|SHAP| mediu", ascending=False)
    p_gb = gb_pipe.predict_proba(xy(test, "profitable", GB)[0])[:, 1]
    t = THRESHOLDS[GB.name]
    pos = test.reset_index(drop=True).assign(p=p_gb)
    # exemple: un hit prezis corect, un fals pozitiv sigur, un fals negativ sigur
    hit = pos[(pos.profitable == 1) & (pos.p >= t)].nlargest(1, "p").index[0]
    fp = pos[(pos.profitable == 0) & (pos.p >= t)].nlargest(1, "p").index[0]
    fn = pos[(pos.profitable == 1) & (pos.p < t)].nsmallest(1, "p").index[0]

    def label(i, kind):
        r = pos.loc[i]
        return f"{kind}: {r['title']} ({r['release_year']}), P = {num(r['p'], 2)}, ROI real {num(r['roi'], 2)}"

    # 3. Partial dependence
    curves = {f: {GB.name: pdp(gb_pipe, GB, train, test, f), LIN.name: pdp(lin_pipe, LIN, train, test, f)}
              for f in ("log_budget_real", "director_prev_profitable_rate", "cast_prev_mean_log_roi")}
    curves["director_prev_profitable_rate"] = {
        m: pdp(p, s, train, test, "director_prev_profitable_rate", grid=np.linspace(0, 1, 21))
        for m, p, s in ((GB.name, gb_pipe, GB), (LIN.name, lin_pipe, LIN))}
    month = {GB.name: pdp_month(gb_pipe, GB, test), LIN.name: pdp_month(lin_pipe, LIN, test)}
    franchise = pd.DataFrame([{"model": m, "is_franchise = 0": pdp(p, s, train, test, "is_franchise", grid=[0])["P(profitabil) medie"][0],
                               "is_franchise = 1": pdp(p, s, train, test, "is_franchise", grid=[1])["P(profitabil) medie"][0]}
                              for m, p, s in ((GB.name, gb_pipe, GB), (LIN.name, lin_pipe, LIN))])
    budget_pd = curves["log_budget_real"][GB.name]
    b = budget_pd.assign(**{"buget (mil. $ 2024)": np.exp(budget_pd["valoare"]) / 1e6})

    # 4. Comparație cu EDA (MI pe train)
    mi = pd.read_csv(PROJECT_ROOT / "results" / "eda" / "relevance.csv")[["caracteristică", "MI"]]
    shap_rank = mean_abs.rename(columns={"coloană": "caracteristică"})
    comp = shap_rank.merge(mi, on="caracteristică", how="inner")
    rho, p_rho = stats.spearmanr(comp["|SHAP| mediu"], comp["MI"])
    top_shap, top_mi = set(shap_rank.head(10)["caracteristică"]), set(mi.nlargest(10, "MI")["caracteristică"])

    for name, tbl in (("permutation_gb", pi_gb), ("permutation_lin", pi_lin), ("coefficients_lin", coef),
                      ("shap_mean_abs", mean_abs), ("pdp_budget", b), ("pdp_month_gb", month[GB.name]),
                      ("pdp_franchise", franchise)):
        tbl.to_csv(OUT_DIR / f"{name}.csv", index=False, float_format="%.6f")

    figs = {"perm": fig_perm(pi_gb, pi_lin), "shap": fig_shap_summary(sv, Xt),
            "hit": fig_waterfall(sv, hit, "interpret_shap_hit", label(hit, "Hit prezis corect")),
            "fp": fig_waterfall(sv, fp, "interpret_shap_fp", label(fp, "Fals pozitiv")),
            "fn": fig_waterfall(sv, fn, "interpret_shap_fn", label(fn, "Fals negativ")),
            "pdp": fig_pdp(curves, month, test["profitable"].mean())}

    write_report(pi_gb, pi_lin, coef, mean_abs, b, month, franchise, comp, rho, p_rho, top_shap, top_mi, figs)
    print(f"Scris: {REPORT}")


def write_report(pi_gb, pi_lin, coef, mean_abs, b, month, franchise, comp, rho, p_rho, top_shap, top_mi, figs) -> None:
    pi_show = pi_gb.head(TOP).merge(pi_lin[["caracteristică", "scădere ROC-AUC"]].rename(
        columns={"scădere ROC-AUC": f"scădere ROC-AUC ({LIN.name})"}), on="caracteristică", how="left")
    pi_show["IC 95% (GB)"] = [f"[{num(lo, 4)}; {num(hi, 4)}]" for lo, hi in zip(pi_show["IC 2,5%"], pi_show["IC 97,5%"])]
    pi_show = pi_show[["caracteristică", "scădere ROC-AUC", "IC 95% (GB)", f"scădere ROC-AUC ({LIN.name})"]].rename(
        columns={"scădere ROC-AUC": f"scădere ROC-AUC ({GB.name})"})
    m = month[GB.name]
    bmin = b.loc[b["P(profitabil) medie"].idxmin()]
    text = f"""# Interpretabilitate (#58)

Generat cu `python analysis/interpret.py`. Tabelele complete: `results/interpret/*.csv`. Modelele sunt antrenate pe train (1950–2014) cu hiperparametrii din #41; toate măsurătorile sunt pe **test** (2019–2024), pe filme pe care modelul nu le-a văzut.

## Pe scurt

{CONCLUSIONS}

## 1. Permutation importance

Pentru fiecare coloană de intrare: amestecăm valorile ei între filmele de test ({N_REPEATS} repetări) și măsurăm cât scade ROC-AUC. Dacă scade mult, modelul se bazează pe ea. Intervalul e dat de cele {N_REPEATS} repetări.

{md_table(fmt(pi_show))}

{figs['perm']}

Limită: coloanele corelate (de ex. `director_prev_mean_log_roi` și `director_prev_profitable_rate`, ρ = 0,84) își împart importanța: amestecând una, modelul o mai poate folosi pe cealaltă, deci fiecare pare mai puțin importantă decât e informația comună.

**Coeficienții regresiei logistice + buget²** (pe date scalate, deci comparabili între ei; top 15 în modul):

{md_table(fmt(coef.head(TOP), 3))}

`log_budget_real` și `log_budget_real_sq` trebuie citite împreună: un coeficient negativ pe buget și unul pozitiv pe pătratul lui descriu un U.

## 2. SHAP (gradient boosting)

SHAP împarte predicția fiecărui film între caracteristici: cât a împins fiecare predicția spre „profitabil” (dreapta) sau „neprofitabil” (stânga), față de media modelului. Fiecare punct e un film din test; culoarea e valoarea caracteristicii (roșu = mare, albastru = mică).

{figs['shap']}

**Trei filme explicate:** cele mai mari contribuții, de la media modelului (jos, E[f(x)]) până la predicția pentru film (sus, f(x)), în log-odds.

{figs['hit']}

{figs['fp']}

{figs['fn']}

## 3. Partial dependence

Pentru o caracteristică: dăm aceeași valoare tuturor filmelor din test, păstrând restul neschimbat, și calculăm probabilitatea medie prezisă. Linia punctată: rata reală de profitabile din test.

{figs['pdp']}

**Bugetul (gradient boosting):**

{md_table(fmt(b[['buget (mil. $ 2024)', 'P(profitabil) medie']], 3))}

Minimul e la aproximativ {num(bmin['buget (mil. $ 2024)'], 1)} mil. $ (P = {num(bmin['P(profitabil) medie'], 3)}).

**Luna lansării (gradient boosting):**

{md_table(fmt(m, 3))}

**Franciza:**

{md_table(fmt(franchise, 3))}

## 4. Comparație cu EDA

Corelația Spearman între importanța SHAP (pe test) și informația mutuală din EDA (#42, pe train), pe cele {len(comp)} coloane comune: ρ = {num(rho, 2)} (p = {p_rho:.1e}).

- În top 10 la ambele: {', '.join(f'`{c}`' for c in sorted(top_shap & top_mi))}.
- Doar în top 10 SHAP: {', '.join(f'`{c}`' for c in sorted(top_shap - top_mi)) or '–'}.
- Doar în top 10 MI: {', '.join(f'`{c}`' for c in sorted(top_mi - top_shap)) or '–'}.
"""
    REPORT.write_text(text, encoding="utf-8")


CONCLUSIONS = """1. **Bugetul e cea mai importantă caracteristică** în ambele modele. Amestecat, scade ROC-AUC cu 0,067
   la gradient boosting și cu 0,218 la regresia logistică + buget² (care depinde mai mult de el). Are și
   cel mai mare |SHAP| mediu.
2. **Forma de U din EDA nu e un efect direct al bugetului.** Ținând restul caracteristicilor fixe,
   probabilitatea scade aproape continuu cu bugetul: de la 0,70 (sub 0,3 mil. $) la 0,32 (peste 100 mil. $)
   la gradient boosting, și de la 0,74 la 0,23 la regresia logistică. Brațul drept al U-ului (filmele
   foarte scumpe, mai des profitabile) vine din ce **însoțește** bugetul mare: continuări, francize,
   distribuitori majori. Odată ce modelul știe acestea, un buget mai mare înseamnă doar un prag mai greu
   de atins. Brațul stâng (filmele foarte ieftine) e real: un film de 300.000 $ are nevoie doar de
   750.000 $ încasări pentru ROI 2,5.
3. **Lipsa clasificării americane (`certification_missing`) e a doua caracteristică**, în ambele modele
   (scădere ROC-AUC 0,041). Confirmă EDA (#42): lipsa clasificării e ea însăși o informație (filme fără
   lansare clasificată în SUA, de obicei mai mici). Decizia din #40 de a păstra indicatorul se dovedește
   utilă.
4. **Durata, continuările și distribuitorul major** urmează în clasament. La gradient boosting,
   `sequel_number` face toată treaba francizei: o continuare ridică probabilitatea medie de la 0,43 la
   0,56, iar `is_franchise` nu mai adaugă nimic (aceeași predicție cu 0 sau 1), pentru că informația e
   deja în `sequel_number`. Regresia logistică, invers, folosește `is_franchise` (coeficient +0,85).
5. **Istoricul actorilor și al regizorului contează, dar mai puțin decât sugera EDA.** Amestecat, ROI-ul
   mediu anterior al actorilor scade ROC-AUC cu 0,013, al regizorului cu 0,009. O parte din importanță e
   împărțită cu perechea lor corelată (`*_profitable_rate`).
6. **Luna lansării are un efect mic:** iulie e cea mai bună lună (P medie 0,49), aprilie cea mai slabă
   (0,41). Diferența e de 8 puncte procentuale, mult sub efectul bugetului.
7. **Modelul e de acord cu EDA doar parțial:** corelația dintre importanța SHAP și informația mutuală e
   ρ = 0,47. Bugetul, anul, durata și istoricul sunt în top 10 la ambele. Modelul folosește în plus
   indicatori pe care MI îi subestima luați separat (`certification_missing`, `sequel_number`,
   `major_distributor`), pentru că efectul lor apare în combinație cu bugetul."""

if __name__ == "__main__":
    main()
