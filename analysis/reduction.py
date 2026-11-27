"""
Reducerea dimensiunii și discretizare (#43): PCA, analiză factorială, benzi de buget, praguri ROI.

Run:
  python analysis/reduction.py     # -> docs/reduction.md, docs/figures/reduction_*.png, results/reduction/*.csv

Totul se ajustează doar pe train-ul temporal (1950–2014): PCA, factorii, limitele benzilor de buget.
Modelele (regresia logistică și gradient boosting din #41) folosesc hiperparametrii aleși acolo
(`results/baselines/best_params.json`); pragul de decizie se alege tot pe validare.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.eda import ACCENT, DIVERGING, INK, INK_2, MUTED, PAIR, md_table, num, pct, save
from extraction.utils import PROJECT_ROOT
from modeling.baselines import CLASSIFIERS, best_threshold, clf_metrics, run_clf
from modeling.preprocessing import (NUMERIC, SEED, X_y, build_preprocessor, load_dataset,
                                    temporal_split)

OUT_DIR = PROJECT_ROOT / "results" / "reduction"
REPORT = PROJECT_ROOT / "docs" / "reduction.md"
PARAMS = json.loads((PROJECT_ROOT / "results" / "baselines" / "best_params.json").read_text(encoding="utf-8"))["clasificare"]
LIN, GB = CLASSIFIERS[1], CLASSIFIERS[4]
N_BOOT = 100
N_BINS = 5
# numericele fără sin/cos (perechile ciclice au sens doar împreună, nu ca variabile separate)
PCA_NUMERIC = [c for c in NUMERIC if not c.endswith(("_sin", "_cos"))]


def genre_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("genre_")]


# ---------------------------------------------------------------------------
# 1. PCA
# ---------------------------------------------------------------------------
def pca_block(X: pd.DataFrame, standardize: bool) -> tuple[PCA, np.ndarray]:
    """Genurile (0/1, aceeași scară) doar centrate; numericele standardizate (scări diferite)."""
    Z = StandardScaler(with_std=standardize).fit_transform(X)
    return PCA(random_state=SEED).fit(Z), Z


def pca_summary(pca: PCA, name: str) -> dict:
    cum = np.cumsum(pca.explained_variance_ratio_)
    # Kaiser: componente cu valoare proprie > media valorilor proprii (= 1 pe date standardizate)
    kaiser = int((pca.explained_variance_ > pca.explained_variance_.mean()).sum())
    return {"bloc": name, "variabile": pca.n_components_, "PC1 %": pca.explained_variance_ratio_[0],
            "componente 90%": int(np.searchsorted(cum, 0.90) + 1),
            "componente 95%": int(np.searchsorted(cum, 0.95) + 1), "Kaiser": kaiser}


def loadings(pca: PCA, cols: list[str], k: int) -> pd.DataFrame:
    return pd.DataFrame(pca.components_[:k].T, index=cols, columns=[f"PC{i + 1}" for i in range(k)])


def bootstrap_stability(Z: np.ndarray, ref: PCA, k: int) -> pd.DataFrame:
    """|cos| între componentele din eșantioane bootstrap și cele de referință (semnul nu contează)."""
    rng = np.random.default_rng(SEED)
    sims = np.zeros((N_BOOT, k))
    for b in range(N_BOOT):
        sample = Z[rng.integers(0, len(Z), len(Z))]
        comp = PCA(n_components=k, random_state=SEED).fit(sample).components_
        sims[b] = np.abs(np.sum(comp * ref.components_[:k], axis=1))
    return pd.DataFrame({"componentă": [f"PC{i + 1}" for i in range(k)], "cos mediu (modul)": sims.mean(0),
                         "cos minim (modul)": sims.min(0), "% bootstrap cu cos ≥ 0,9": (sims >= 0.9).mean(0)})


# ---------------------------------------------------------------------------
# 2. Analiză factorială
# ---------------------------------------------------------------------------
def kmo_bartlett(X: pd.DataFrame) -> tuple[float, pd.Series, float, float]:
    """KMO (adecvarea eșantionului) și testul de sfericitate Bartlett, din matricea de corelații."""
    R = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    inv = np.linalg.inv(R)
    partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    off = ~np.eye(len(R), dtype=bool)
    r2, p2 = (R ** 2) * off, (partial ** 2) * off
    kmo_total = r2.sum() / (r2.sum() + p2.sum())
    kmo_var = pd.Series(r2.sum(0) / (r2.sum(0) + p2.sum(0)), index=X.columns)
    n, p = X.shape
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    dof = p * (p - 1) / 2
    return float(kmo_total), kmo_var, float(chi), float(stats.chi2.sf(chi, dof))


def promax(L: np.ndarray, power: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Rotație oblică promax pornind de la încărcările varimax; întoarce încărcările și corelațiile factorilor."""
    target = np.sign(L) * np.abs(L) ** power
    T, *_ = np.linalg.lstsq(L, target, rcond=None)
    T = T @ np.diag(np.sqrt(np.diag(np.linalg.inv(T.T @ T))))
    phi = np.linalg.inv(T.T @ T)
    return L @ T, phi


def factor_analysis(X: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    Z = StandardScaler().fit_transform(X)
    fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=SEED).fit(Z)
    L_var = fa.components_.T
    L_pro, phi = promax(L_var)
    cols = [f"F{i + 1}" for i in range(k)]
    return (pd.DataFrame(L_var, index=X.columns, columns=cols),
            pd.DataFrame(L_pro, index=X.columns, columns=cols), phi)


# ---------------------------------------------------------------------------
# 3. Discretizarea bugetului
# ---------------------------------------------------------------------------
def budget_edges(train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Limitele interioare ale celor 5 benzi, în log(dolari 2024), învățate pe train."""
    x = train[["log_budget_real"]].to_numpy()
    edges = {}
    for strategy, label in (("uniform", "lățime egală"), ("quantile", "frecvență egală")):
        kb = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy=strategy,
                              quantile_method="averaged_inverted_cdf")
        edges[label] = kb.fit(x).bin_edges_[0][1:-1]
    # entropie: pragurile unui arbore de decizie pe `profitable`, cu 5 frunze
    tree = DecisionTreeClassifier(max_leaf_nodes=N_BINS, min_samples_leaf=200, random_state=SEED)
    tree.fit(x, train["profitable"])
    edges["entropie (arbore)"] = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
    return edges


def add_band(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    return df.assign(budget_band=np.digitize(df["log_budget_real"], edges).astype(str))


def band_table(train: pd.DataFrame, edges: dict) -> pd.DataFrame:
    rows = []
    for name, e in edges.items():
        band = np.digitize(train["log_budget_real"], e)
        bounds = [0, *np.exp(e) / 1e6, np.inf]
        for k in range(len(e) + 1):
            m = band == k
            rows.append({"discretizare": name, "banda": k + 1,
                         "buget (mil. $ 2024)": f"{num(bounds[k], 1)} – {num(bounds[k + 1], 1) if np.isfinite(bounds[k + 1]) else '∞'}",
                         "filme (train)": int(m.sum()), "% profitabile": train.loc[m, "profitable"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modele cu reprezentări alternative
# ---------------------------------------------------------------------------
def fit_eval(pipe: Pipeline, cols: list[str], train, val, test) -> dict:
    pipe.fit(train[cols], train["profitable"])
    p_va = pipe.predict_proba(val[cols])[:, 1]
    t = best_threshold(val["profitable"], p_va)
    return clf_metrics(test["profitable"], pipe.predict_proba(test[cols])[:, 1], t)


def pca_pipe(spec, k: int, genres: list[str], columns: list[str]) -> tuple[Pipeline, list[str]]:
    """Genurile înlocuite de primele k componente PCA (ajustate pe train), restul ca în #40."""
    base = build_preprocessor(spec.kind, exclude=tuple(genres))
    X_cols = list(X_y(pd.DataFrame(columns=columns), exclude=tuple(genres))[0].columns)
    pca = Pipeline([("center", StandardScaler(with_std=False)), ("pca", PCA(n_components=k, random_state=SEED))])
    if spec.kind == "linear":
        pca.steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([("rest", base, X_cols), ("genres", pca, genres)], verbose_feature_names_out=False)
    model = clone(spec.estimator).set_params(**PARAMS[spec.name])
    return Pipeline([("pre", pre), ("model", model)]), X_cols + genres


def band_pipe(spec, edges: np.ndarray, columns: list[str]) -> tuple[Pipeline, list[str]]:
    """`log_budget_real` înlocuit de banda de buget (one-hot); restul ca în #40."""
    excl = ("log_budget_real",)
    base = build_preprocessor(spec.kind, exclude=excl)
    X_cols = list(X_y(pd.DataFrame(columns=columns), exclude=excl)[0].columns)
    pre = ColumnTransformer([("rest", base, X_cols),
                             ("band", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["budget_band"])],
                            verbose_feature_names_out=False)
    model = clone(spec.estimator).set_params(**PARAMS[spec.name])
    return Pipeline([("pre", pre), ("model", model)]), X_cols + ["budget_band"]


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------
def fig_scree(pcas: dict[str, PCA]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, pca) in zip(axes, pcas.items()):
        k = np.arange(1, pca.n_components_ + 1)
        ax.bar(k, 100 * pca.explained_variance_ratio_, color=ACCENT, width=0.7, label="per componentă")
        ax.plot(k, 100 * np.cumsum(pca.explained_variance_ratio_), color=INK_2, marker="o", markersize=3,
                linewidth=1.5, label="cumulat")
        for level in (90, 95):
            ax.axhline(level, color=MUTED, linestyle="--", linewidth=0.8)
            ax.annotate(f"{level}%", (k[0], level), xytext=(0, 2), textcoords="offset points", color=MUTED,
                        fontsize=8, ha="left", va="bottom")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_title(name)
        ax.set_xlabel("componenta")
        ax.set_ylabel("% din varianță")
        ax.set_ylim(0, 102)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.25))
    fig.tight_layout()
    return save(fig, "reduction_scree")


def fig_heat(tables: dict[str, pd.DataFrame], name: str, title: str) -> str:
    fig, axes = plt.subplots(1, len(tables), figsize=(4.2 * len(tables) + 1, 6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, t) in zip(axes, tables.items()):
        im = ax.imshow(t.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(t.shape[1]), t.columns)
        ax.set_yticks(range(t.shape[0]), [c.removeprefix("genre_").replace("_", " ") for c in t.index])
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.iloc[i, j]
                if abs(v) >= 0.3:
                    ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 else INK)
        ax.set_title(label, fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return save(fig, name)


def fig_bands(bands: pd.DataFrame) -> str:
    names = list(bands["discretizare"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3.4), sharey=True)
    for ax, name in zip(axes, names):
        b = bands[bands["discretizare"] == name]
        ax.bar(b["banda"], 100 * b["% profitabile"], color=ACCENT, width=0.65)
        for x, v, n in zip(b["banda"], b["% profitabile"], b["filme (train)"]):
            ax.annotate(f"{pct(v, 0)}\nn={num(n, 0)}", (x, 100 * v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_2)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(b["banda"])
        ax.set_xlabel("banda de buget (1 = cel mai mic)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% profitabile (train)")
    axes[0].set_ylim(0, 80)
    fig.tight_layout()
    return save(fig, "reduction_bands")


# ---------------------------------------------------------------------------
def fmt(df: pd.DataFrame, pct_cols=(), digits=3) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(pct)
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, digits))
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, 0))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train, val, test = temporal_split(df)
    genres = genre_columns(df)

    # 1. PCA pe genuri (centrat) și pe numericele imputate + standardizate
    num_tr = pd.DataFrame(build_preprocessor("tree").fit(X_y(train)[0]).transform(X_y(train)[0])[PCA_NUMERIC])
    pca_g, Zg = pca_block(train[genres], standardize=False)
    pca_n, Zn = pca_block(num_tr, standardize=True)
    summary = pd.DataFrame([pca_summary(pca_g, f"genuri ({len(genres)}, 0/1)"), pca_summary(pca_n, f"numerice ({len(PCA_NUMERIC)})")])
    load_g, load_n = loadings(pca_g, genres, 4), loadings(pca_n, PCA_NUMERIC, 4)
    boot_g, boot_n = bootstrap_stability(Zg, pca_g, 5), bootstrap_stability(Zn, pca_n, 5)

    # 2. Analiză factorială pe genuri
    kmo, kmo_var, bart_chi, bart_p = kmo_bartlett(train[genres])
    k_fa = int(summary.loc[0, "Kaiser"])
    fa_var, fa_pro, phi = factor_analysis(train[genres], k_fa)

    # 3. Discretizare
    edges = budget_edges(train)
    bands = band_table(train, edges)
    rows = []
    for spec in (LIN, GB):
        rows.append({"reprezentare buget": "continuu (log_budget_real)", "model": spec.name,
                     **_short(run_clf(spec, PARAMS[spec.name], train, val, {"test": test})["test"])})
        if spec is LIN:
            sq = CLASSIFIERS[2]
            rows.append({"reprezentare buget": "continuu + pătrat", "model": spec.name,
                         **_short(run_clf(sq, PARAMS[sq.name], train, val, {"test": test})["test"])})
        for name, e in edges.items():
            pipe, cols = band_pipe(spec, e, list(df.columns))
            res = fit_eval(pipe, cols, add_band(train, e), add_band(val, e), add_band(test, e))
            rows.append({"reprezentare buget": f"5 benzi, {name}", "model": spec.name, **_short(res)})
    disc = pd.DataFrame(rows)

    # 4. Modele cu PCA în loc de genuri
    k90 = int(summary.loc[0, "componente 90%"])
    pca_rows = []
    for spec in (LIN, GB):
        pca_rows.append({"genuri": f"{len(genres)} coloane 0/1 (#41)", "model": spec.name,
                         **_short(run_clf(spec, PARAMS[spec.name], train, val, {"test": test})["test"])})
        for k in sorted({3, 5, k90}):
            pipe, cols = pca_pipe(spec, k, genres, list(df.columns))
            pca_rows.append({"genuri": f"{k} componente PCA" + (" (90%)" if k == k90 else ""), "model": spec.name,
                             **_short(fit_eval(pipe, cols, train, val, test))})
    pca_models = pd.DataFrame(pca_rows)

    # 5. Pragul ROI
    thr_rows = []
    for t in (2.0, 2.5, 3.0):
        d = load_dataset(threshold=t)
        tr, va, te = temporal_split(d)
        for spec in (LIN, GB):
            m = run_clf(spec, PARAMS[spec.name], tr, va, {"test": te})["test"]
            thr_rows.append({"prag ROI": num(t, 1), "% profitabile (tot)": d["profitable"].mean(),
                             "% profitabile (test)": te["profitable"].mean(), "model": spec.name, **_short(m)})
    thresholds = pd.DataFrame(thr_rows)

    for name, t in (("pca_summary", summary), ("pca_loadings_genres", load_g), ("pca_loadings_numeric", load_n),
                    ("pca_bootstrap_genres", boot_g), ("pca_bootstrap_numeric", boot_n), ("fa_varimax", fa_var),
                    ("fa_promax", fa_pro), ("budget_bands", bands), ("discretization_models", disc),
                    ("pca_models", pca_models), ("thresholds", thresholds)):
        t.to_csv(OUT_DIR / f"{name}.csv", index=name.startswith(("pca_loadings", "fa_")), float_format="%.6f")

    figs = {"scree": fig_scree({"Genuri (centrate)": pca_g, "Numerice (standardizate)": pca_n}),
            "pca": fig_heat({"genuri": load_g}, "reduction_pca_genres", "Încărcările primelor 4 componente PCA (genuri, train)"),
            "pca_n": fig_heat({"numerice": load_n}, "reduction_pca_numeric", "Încărcările primelor 4 componente PCA (numerice, train)"),
            "fa": fig_heat({"varimax (ortogonală)": fa_var, "promax (oblică)": fa_pro}, "reduction_fa",
                           f"Analiză factorială pe genuri, {k_fa} factori (train)"),
            "bands": fig_bands(bands)}

    write_report(train, summary, load_g, load_n, boot_g, boot_n, kmo, kmo_var, bart_chi, bart_p, k_fa,
                 fa_var, fa_pro, phi, bands, disc, pca_models, thresholds, figs, k90)
    print(f"Scris: {REPORT}")


def _short(m: dict) -> dict:
    return {k: m[k] for k in ("F1", "ROC-AUC", "PR-AUC", "precision", "recall", "prag")}


def top_loadings(t: pd.DataFrame, col: str, n: int = 4) -> str:
    s = t[col].sort_values(key=abs, ascending=False).head(n)
    return ", ".join(f"{i.removeprefix('genre_').replace('_', ' ')} ({num(v, 2)})" for i, v in s.items())


def write_report(train, summary, load_g, load_n, boot_g, boot_n, kmo, kmo_var, bart_chi, bart_p, k_fa,
                 fa_var, fa_pro, phi, bands, disc, pca_models, thresholds, figs, k90) -> None:
    phi_df = pd.DataFrame(phi, index=fa_pro.columns, columns=fa_pro.columns).reset_index(names="factor")
    pc_desc = "\n".join(f"- **{c}**: {top_loadings(load_g, c)}" for c in load_g.columns)
    pcn_desc = "\n".join(f"- **{c}**: {top_loadings(load_n, c)}" for c in load_n.columns)
    fa_desc = "\n".join(f"- **{c}** (promax): {top_loadings(fa_pro, c)}" for c in fa_pro.columns)
    text = f"""# Reducerea dimensiunii și discretizare (#43)

Generat cu `python analysis/reduction.py`. Tabelele complete: `results/reduction/*.csv`. Totul (PCA, factori, limitele benzilor) e ajustat **doar pe train** (1950–2014, {num(len(train), 0)} filme). Modelele folosesc hiperparametrii aleși pe validare în #41; pragul de decizie se alege tot pe validare, iar scorurile sunt pe test (2019–2024).

## Pe scurt

{CONCLUSIONS}

## 1. PCA

Două blocuri, separat: **genurile** ({summary.loc[0, 'variabile']} coloane 0/1, aceeași scară, deci doar centrate) și **numericele** ({len(PCA_NUMERIC)} coloane după imputare, standardizate pentru că au scări diferite: buget în log, ani, minute, proporții). Sin/cos pentru lună și zi nu intră: au sens doar ca pereche.

{md_table(fmt(summary, pct_cols=['PC1 %']))}

Criteriul Kaiser: componentele cu valoare proprie peste medie (pe date standardizate, peste 1).

{figs['scree']}

**Ce înseamnă primele componente (genuri):** încărcările cele mai mari, în modul.

{pc_desc}

{figs['pca']}

**Numerice:**

{pcn_desc}

{figs['pca_n']}

**Stabilitate prin bootstrap** ({N_BOOT} eșantioane din train, cu înlocuire): cosinusul (în modul, semnul unei componente e arbitrar) dintre componenta din eșantion și cea de referință; 1 = aceeași direcție.

Genuri:

{md_table(fmt(pd.DataFrame(boot_g), pct_cols=['% bootstrap cu cos ≥ 0,9']))}

Numerice:

{md_table(fmt(pd.DataFrame(boot_n), pct_cols=['% bootstrap cu cos ≥ 0,9']))}

## 2. Analiză factorială (genuri)

- **Adecvarea eșantionului:** KMO = {num(kmo, 3)} (sub 0,5 = nepotrivit, 0,5–0,6 slab, peste 0,6 acceptabil). Testul Bartlett (corelațiile nu sunt toate zero): χ² = {num(bart_chi, 0)}, p {'< 1e-300' if bart_p == 0 else f'= {bart_p:.1e}'}. Eșantionul ({num(len(train), 0)} filme pentru {len(kmo_var)} variabile, peste 700 de filme per variabilă) e mai mult decât suficient ca mărime.
- KMO pe variabile, cele mai mici: {', '.join(f"{c.removeprefix('genre_')} ({num(v, 2)})" for c, v in kmo_var.sort_values().head(4).items())}.
- **{k_fa} factori** (numărul dat de Kaiser la PCA pe genuri), estimați cu `FactorAnalysis` (sklearn). Limită: genurile sunt 0/1, iar analiza factorială clasică presupune variabile continue; corelațiile Pearson între variabile binare subestimează legăturile (corelația tetracorică ar fi mai potrivită). Rezultatul e descriptiv, nu intră în model.
- **Varimax** (ortogonală: factorii rămân necorelați) vs **promax** (oblică: factorii se pot corela, mai realist pentru genuri care apar împreună).

{figs['fa']}

{fa_desc}

**Corelațiile dintre factori după promax** (la varimax sunt zero prin construcție):

{md_table(fmt(phi_df, digits=2))}

## 3. Discretizarea bugetului

Bugetul ajustat (`log_budget_real`) în **5 benzi**, cu limitele învățate pe train, în trei feluri: **lățime egală** (pe scara log), **frecvență egală** (cuantile, câte 20% din filme) și **pe bază de entropie** (pragurile unui arbore de decizie cu 5 frunze care separă filmele profitabile de cele neprofitabile).

{md_table(fmt(bands, pct_cols=['% profitabile']))}

{figs['bands']}

**În model** (bugetul continuu înlocuit de banda lui, one-hot; restul caracteristicilor neschimbate; pe test):

{md_table(fmt(disc))}

## 4. Modele cu componente PCA în loc de genuri

Cele {summary.loc[0, 'variabile']} coloane de gen înlocuite de primele k componente PCA (ajustate pe train în pipeline), restul caracteristicilor neschimbate; pe test.

{md_table(fmt(pca_models))}

## 5. Pragul ROI: 2 / 2,5 / 3

Ținta `profitable` recalculată la fiecare prag; aceleași modele și hiperparametri (aleși la 2,5 în #41), pragul de decizie ales din nou pe validare.

{md_table(fmt(thresholds, pct_cols=['% profitabile (tot)', '% profitabile (test)']))}
"""
    REPORT.write_text(text, encoding="utf-8")


CONCLUSIONS = """1. **PCA nu reduce mult și nu ajută modelele.** La genuri, prima componentă explică doar 18,8%
   din varianță și sunt nevoie de 12 din 18 componente pentru 90%: genurile sunt aproape independente
   între ele (un film are în medie 2–3 genuri, în combinații foarte variate). Înlocuirea genurilor cu
   3, 5 sau 12 componente dă același ROC-AUC (0,772–0,775) ca one-hot-ul, la ambele modele.
   **Decizie: păstrăm cele 18 coloane de gen**, mai ușor de interpretat; 18 coloane nu sunt o problemă
   pentru 13.623 de filme.
2. **Componentele au sens și sunt stabile:** la genuri, primele 5 componente se regăsesc în toate
   cele 100 de eșantioane bootstrap (cos ≥ 0,95). PC1 opune comedia dramei și thriller-ului. La
   numerice, PC1 = „notorietate” (premii + buget), PC2 = „istoric de succes” (ROI-ul anterior al
   regizorului și al actorilor), ambele foarte stabile; de la PC3 încolo componentele se rotesc între
   eșantioane (cos minim 0,03–0,32), pentru că au valori proprii apropiate: nu le interpretăm.
3. **Analiza factorială găsește „teme” de gen recognoscibile:** filme pentru copii (family +
   animation), acțiune / aventură / SF, crimă / thriller / mister, istoric / război. Promax arată că
   factorii nu sunt independenți (corelație până la 0,51). KMO = 0,58 e însă „slab”, iar genurile 0/1
   nu respectă ipotezele modelului: rezultatul e descriptiv, nu intră în model.
4. **Discretizarea bugetului face vizibilă forma de U**, cel mai clar la benzile pe bază de entropie:
   53,6% profitabile sub 1,6 mil. $, 27,2% între 3,9 și 51,2 mil. $, 50,8% peste 149,9 mil. $ (dolari
   2024). În model însă **benzile nu bat bugetul continuu** (ROC-AUC 0,763–0,769 vs 0,774 la regresia
   logistică): se pierde informație în interiorul benzii. Pentru modelul liniar, forma de U se prinde
   mai bine cu bugetul la pătrat (F1 0,563). **Decizie: bugetul rămâne continuu.**
5. **Pragul ROI:** ROC-AUC rămâne 0,77–0,79 la toate cele trei praguri, deci modelul ordonează filmele
   la fel de bine. F1 scade când pragul crește (0,60 → 0,55 → 0,51–0,53), pentru că filmele
   „profitabile” devin mai rare (31,5% → 26,1% → 22,1% în test). Rezultatele finale (#56) se raportează
   la toate cele trei praguri."""

if __name__ == "__main__":
    main()
