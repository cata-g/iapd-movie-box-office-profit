"""
Analiza exploratorie (#42): statistici descriptive, outliers, date lipsă, redundanță și
relevanța caracteristicilor față de țintă.

Run:
  python analysis/eda.py      # -> docs/eda.md, docs/figures/eda_*.png, results/eda/*.csv

Statisticile descriptive, outliers și datele lipsă se calculează pe tot dataset-ul (descriu
datele). Redundanța și relevanța față de țintă se calculează DOAR pe train-ul temporal
(1950–2014): decizia de a scoate o caracteristică e tot o „învățare” din date și nu are voie să
vadă filmele din validare / test.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.columns import feature_columns
from extraction.utils import PROJECT_ROOT
from modeling.preprocessing import (DATASET, DUPLICATES, NUMERIC, SEED, X_y, build_preprocessor,
                                    load_dataset, temporal_split)

FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR = PROJECT_ROOT / "results" / "eda"
REPORT = PROJECT_ROOT / "docs" / "eda.md"

# Culori: un accent pentru o singură serie (ca în prezentare), perechea albastru / portocaliu
# (paleta validată, docs dataviz) când se compară două serii, gri pentru referințe
INK, INK_2, MUTED, GRID = "#0f172a", "#334155", "#64748b", "#e2e8f0"
ACCENT = "#0f766e"
PAIR = ("#2a78d6", "#eb6834")
DIVERGING = "RdBu_r"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%".replace(".", ",")


def num(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "–"
    s = f"{x:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".")


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
    return f"![{name}](figures/{name}.png)"


# ---------------------------------------------------------------------------
# 1. Tipuri de date
# ---------------------------------------------------------------------------
TYPES = [
    ("budget", "raport (continuă)", "USD nominal; zero absolut"),
    ("release_year", "interval (discretă)", "anul nu are zero natural"),
    ("release_month", "ordinală ciclică", "1–12, decembrie vecină cu ianuarie"),
    ("release_weekday", "ordinală ciclică", "0–6"),
    ("runtime_min", "raport (discretă)", "minute"),
    ("certification", "ordinală", "G < PG < PG-13 < R < NC-17"),
    ("sequel_number", "raport (discretă)", "0 = nu e continuare"),
    ("director_prev_films, cast_prev_films", "raport (numărătoare)", ""),
    ("*_award_noms_before, *_award_wins_before", "raport (numărătoare)", "4 coloane"),
    ("director_prev_mean_roi / _mean_log_roi", "raport / interval", "log-ul poate fi negativ"),
    ("*_prev_profitable_rate", "raport (proporție)", "0–1"),
    ("original_language", "nominală", "7 categorii"),
    ("adaptation_type", "nominală", "10 categorii"),
    ("genre_* (18)", "binară asimetrică", "1 = filmul are genul; 0 nu spune nimic despre film"),
    ("country_* (4)", "binară asimetrică", "o țară de producție poate fi în mai multe grupe"),
    ("major_studio, major_distributor, major_any, streaming_distributor", "binară asimetrică", ""),
    ("is_franchise, is_adaptation", "binară asimetrică", ""),
    ("is_english", "binară simetrică", "ambele valori sunt la fel de informative"),
    ("profitable (țintă)", "binară simetrică", "ROI ≥ 2,5"),
    ("roi (țintă)", "raport (continuă)", "încasări / buget"),
]


# ---------------------------------------------------------------------------
# 2. Statistici descriptive
# ---------------------------------------------------------------------------
DESCRIBE = {
    "budget": "budget", "log_budget": "log_budget", "log_budget_real": "log_budget_real",
    "worldwide_gross": "worldwide_gross", "log_gross": "log_gross",
    "roi": "roi", "log_roi": "log_roi",
    "runtime_min": "runtime_min", "release_year": "release_year",
    "director_prev_films": "director_prev_films", "cast_prev_films": "cast_prev_films",
    "director_prev_mean_roi": "director_prev_mean_roi",
    "director_prev_mean_log_roi": "director_prev_mean_log_roi",
    "director_prev_profitable_rate": "director_prev_profitable_rate",
    "cast_prev_mean_log_roi": "cast_prev_mean_log_roi",
    "cast_award_noms_before": "cast_award_noms_before",
}


def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in DESCRIBE.items():
        x = df[col].astype(float).dropna()
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        rows.append({
            "coloană": name, "n": len(x), "medie": x.mean(), "mediană": med,
            "medie trunchiată 10%": stats.trim_mean(x, 0.1), "mod": x.mode().iloc[0],
            "min": x.min(), "Q1": q1, "Q3": q3, "max": x.max(), "IQR": q3 - q1, "std": x.std(),
            "skewness": x.skew(), "kurtosis": x.kurt(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Outliers
# ---------------------------------------------------------------------------
def outliers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["budget", "log_budget", "worldwide_gross", "log_gross", "roi", "log_roi", "runtime_min"]:
        x = df[col].astype(float).dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = np.abs((x - x.mean()) / x.std())
        rows.append({"coloană": col, "sub 1,5×IQR": int((x < low).sum()), "peste 1,5×IQR": int((x > high).sum()),
                     "% IQR": (((x < low) | (x > high)).mean()), "|z| > 3": int((z > 3).sum()),
                     "limite IQR": f"[{num(low)}; {num(high)}]"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Date lipsă
# ---------------------------------------------------------------------------
def missing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cols = [c for c in feature_columns(pd.read_parquet(DATASET).head(1)) if df[c].isna().any()]
    table = pd.DataFrame({"coloană": cols, "lipsă": [int(df[c].isna().sum()) for c in cols],
                          "%": [df[c].isna().mean() for c in cols]}).sort_values("%", ascending=False)
    m = df["certification"].isna()
    by = []
    for col, label in [("is_english", "limba engleză"), ("country_us", "producție SUA"),
                       ("major_any", "studio / distribuitor major"), ("profitable", "profitabil")]:
        ct = pd.crosstab(df[col], m)
        chi, p, _, _ = stats.chi2_contingency(ct)
        by.append({"grupă": label, "lipsă dacă 1": m[df[col] == 1].mean(), "lipsă dacă 0": m[df[col] == 0].mean(),
                   "χ²": chi, "p": p})
    by = pd.DataFrame(by)
    # MAR vs MNAR: lipsa mai depinde de țintă după ce ținem cont de limbă și țară?
    strata = []
    for (eng, us), g in df.groupby(["is_english", "country_us"]):
        strata.append({"is_english": eng, "country_us": us, "filme": len(g),
                       "profitabile cu clasificare": g.loc[g.certification.notna(), "profitable"].mean(),
                       "profitabile fără clasificare": g.loc[g.certification.isna(), "profitable"].mean()})
    return table, by, {"strata": pd.DataFrame(strata)}


# ---------------------------------------------------------------------------
# 5. Redundanță (pe train)
# ---------------------------------------------------------------------------
LASSO_C = (0.01, 0.05, 0.1, 1.0)
LASSO_MAIN = 0.1
REDUNDANCY_NUMERIC = ["log_budget"] + [c for c in NUMERIC if c not in ("month_sin", "month_cos", "weekday_sin", "weekday_cos")]


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    chi = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi / (n * (min(ct.shape) - 1))))


def redundancy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_df = train[REDUNDANCY_NUMERIC].astype(float)
    spearman = num_df.corr("spearman")
    pearson = num_df.corr("pearson")
    pairs = []
    for i, a in enumerate(REDUNDANCY_NUMERIC):
        for b in REDUNDANCY_NUMERIC[i + 1:]:
            pairs.append({"a": a, "b": b, "Pearson": pearson.loc[a, b], "Spearman": spearman.loc[a, b]})
    pairs = pd.DataFrame(pairs)
    pairs = pairs[pairs["Spearman"].abs() >= 0.6].sort_values("Spearman", key=abs, ascending=False)
    cat = ["is_english", "original_language", "is_adaptation", "adaptation_type", "major_studio",
           "major_distributor", "major_any", "streaming_distributor", "is_franchise", "country_us",
           "country_europe", "country_asia", "country_other", "has_director_history", "has_cast_history",
           "certification_missing"]
    vpairs = []
    for i, a in enumerate(cat):
        for b in cat[i + 1:]:
            vpairs.append({"a": a, "b": b, "V Cramér": cramers_v(train[a], train[b])})
    vpairs = pd.DataFrame(vpairs)
    vpairs = vpairs[vpairs["V Cramér"] >= 0.5].sort_values("V Cramér", ascending=False)
    return spearman, pairs, vpairs


# ---------------------------------------------------------------------------
# 6. Relevanță față de țintă (pe train)
# ---------------------------------------------------------------------------
REDUNDANT = ["log_budget", "major_any", "is_english", "is_adaptation"]


def relevance(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pe coloanele modelului PLUS cele scoase ca redundante (§9), ca decizia să se vadă în tabel."""
    X, y = X_y(train)
    pre_tree = build_preprocessor("tree").fit(X)
    Xt = pd.concat([pre_tree.transform(X), train[REDUNDANT].astype(float)], axis=1)
    cols = list(Xt.columns)
    log_roi = train["log_roi"].to_numpy()
    rows = []
    for c in cols:
        x = Xt[c].to_numpy(dtype=float)
        r_pb, p_pb = stats.pointbiserialr(y, x) if x.std() > 0 else (np.nan, np.nan)
        rho, _ = stats.spearmanr(x, log_roi)
        rows.append({"caracteristică": c, "r (profitable)": r_pb, "p": p_pb, "ρ (log_roi)": rho})
    rel = pd.DataFrame(rows)
    rel["MI"] = mutual_info_classif(Xt, y, random_state=SEED,
                                    discrete_features=[Xt[c].nunique() <= 2 for c in cols])
    binary = [c for c in cols if set(np.unique(Xt[c])) <= {0, 1}]
    chi_vals, chi_p = chi2(Xt[binary], y)
    rel = rel.merge(pd.DataFrame({"caracteristică": binary, "χ²": chi_vals, "p χ²": chi_p}),
                    on="caracteristică", how="left")
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED,
                                class_weight="balanced").fit(Xt, y)
    rel["importanță RF"] = rf.feature_importances_
    pre_lin = build_preprocessor("linear").fit(X)
    Xl = pre_lin.transform(X)
    path = {}
    for C in LASSO_C:
        lasso = LogisticRegression(l1_ratio=1, solver="liblinear", C=C, class_weight="balanced",
                                   random_state=SEED).fit(Xl, y)
        path[C] = pd.Series(lasso.coef_[0], index=Xl.columns)
    rel = rel.merge(pd.DataFrame({"caracteristică": Xl.columns, "coef. L1": path[LASSO_MAIN].to_numpy()}),
                    on="caracteristică", how="left")
    n_tests = rel["p"].notna().sum()
    alpha = 0.05
    info = {"n_tests": int(n_tests), "sig_raw": int((rel["p"] < alpha).sum()),
            "sig_bonf": int((rel["p"] < alpha / n_tests).sum()), "alpha_bonf": alpha / n_tests,
            "lasso_zero": sorted(rel.loc[rel["coef. L1"] == 0, "caracteristică"]),
            "lasso_path": {C: int((coef == 0).sum()) for C, coef in path.items()}, "n_model": Xl.shape[1],
            "n_features": len(cols)}
    return rel.sort_values("MI", ascending=False), info


# ---------------------------------------------------------------------------
# 7. Duplicate semantice
# ---------------------------------------------------------------------------
def duplicates(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw.duplicated(["title", "release_year"], keep=False)].sort_values(["title", "release_year", "imdb_id"])
    out = d[["title", "release_year", "imdb_id", "original_language", "runtime_min", "budget", "worldwide_gross"]].copy()
    out["decizie"] = out["imdb_id"].map(lambda i: f"scos: {DUPLICATES[i]}" if i in DUPLICATES else "păstrat")
    return out


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------
def fig_hist(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5))
    for row, (col, logc, label) in enumerate([("budget", "log_budget", "Buget (USD)"),
                                              ("worldwide_gross", "log_gross", "Încasări mondiale (USD)"),
                                              ("roi", "log_roi", "ROI (încasări / buget)")]):
        for ax, c, title in ((axes[row, 0], col, f"{label}: brut"), (axes[row, 1], logc, f"{label}: log")):
            x = df[c].astype(float)
            ax.hist(x, bins=60, color=ACCENT, edgecolor="white", linewidth=0.4)
            ax.set_title(f"{title}   skew {num(x.skew())}", fontsize=10)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("filme" if c == col else "")
        if col == "roi":
            axes[row, 1].axvline(np.log(2.5), color=INK_2, linestyle="--", linewidth=1)
            axes[row, 1].annotate("prag ROI 2,5", (np.log(2.5), axes[row, 1].get_ylim()[1] * 0.9),
                                  xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=9)
    fig.suptitle("Distribuțiile brute sunt foarte asimetrice; după log devin aproape simetrice",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return save(fig, "eda_hist_log")


def fig_box_decade(df: pd.DataFrame) -> str:
    dec = (df["release_year"] // 10 * 10)
    labels = sorted(dec.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, col, title in ((axes[0], "log_budget_real", "log(buget în dolari 2024)"), (axes[1], "log_roi", "log(ROI)")):
        data = [df.loc[dec == d, col] for d in labels]
        ax.boxplot(data, tick_labels=[f"{d}s" for d in labels], widths=0.55, patch_artist=True, showfliers=True,
                   boxprops={"facecolor": "#ccfbf1", "edgecolor": ACCENT}, medianprops={"color": ACCENT, "linewidth": 2},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                   flierprops={"marker": ".", "markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": MUTED, "alpha": 0.4})
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[1].axhline(np.log(2.5), color=INK_2, linestyle="--", linewidth=1)
    fig.tight_layout()
    return save(fig, "eda_box_decade")


def fig_rate_year(df: pd.DataFrame) -> str:
    g = df.groupby("release_year").agg(rate=("profitable", "mean"), n=("profitable", "size"))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, height_ratios=[3, 1.2])
    ax.plot(g.index, 100 * g["rate"], color=ACCENT, linewidth=2)
    ax.axhline(100 * df["profitable"].mean(), color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"media\n{pct(df['profitable'].mean())}", (2025, 100 * df["profitable"].mean()),
                xytext=(2, 3), textcoords="offset points", va="bottom", color=MUTED, fontsize=9)
    ax.set_xlim(1949, 2029)
    for x0, x1, label in ((1950, 2014, "train"), (2015, 2018, "validare"), (2019, 2024, "test")):
        ax.axvspan(x0 - 0.5, x1 + 0.5, color="#f1f5f9" if label != "train" else "white", zorder=0)
        ax.annotate(label, ((x0 + x1) / 2, 72), ha="center", color=MUTED, fontsize=9)
    ax.set_ylim(0, 78)
    ax.set_ylabel("% profitabile")
    ax.set_title("Rata de filme profitabile pe an")
    ax2.bar(g.index, g["n"], color="#94a3b8", width=0.8)
    ax2.set_ylabel("filme")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "eda_rate_year")


def _rate_bars(ax, labels, rates, ns, overall, title, horizontal=False):
    pos = np.arange(len(labels))
    if horizontal:
        ax.barh(pos, 100 * rates, color=ACCENT, height=0.65)
        ax.set_yticks(pos, labels)
        ax.axvline(100 * overall, color=MUTED, linestyle="--", linewidth=1)
        for p, r, n in zip(pos, rates, ns):
            ax.annotate(f"{pct(r, 0)}  (n={n:,})".replace(",", "."), (100 * r, p), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% profitabile")
        ax.invert_yaxis()
    else:
        ax.bar(pos, 100 * rates, color=ACCENT, width=0.65)
        ax.set_xticks(pos, labels)
        ax.axhline(100 * overall, color=MUTED, linestyle="--", linewidth=1)
        for p, r in zip(pos, rates):
            ax.annotate(pct(r, 0), (p, 100 * r), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_2)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("% profitabile")
    ax.set_title(title)


def fig_rate_month(df: pd.DataFrame) -> str:
    g = df.groupby("release_month")["profitable"].agg(["mean", "size"])
    names = ["ian", "feb", "mar", "apr", "mai", "iun", "iul", "aug", "sep", "oct", "nov", "dec"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _rate_bars(ax, names, g["mean"].to_numpy(), g["size"].to_numpy(), df["profitable"].mean(),
               "Rata de filme profitabile pe luna lansării (linia: media)")
    return save(fig, "eda_rate_month")


def fig_rate_genre(df: pd.DataFrame) -> str:
    genres = [c for c in df.columns if c.startswith("genre_")]
    rows = [(c.removeprefix("genre_").replace("_", " "), df.loc[df[c] == 1, "profitable"].mean(), int(df[c].sum()))
            for c in genres]
    rows.sort(key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(8, 5.6))
    _rate_bars(ax, [r[0] for r in rows], np.array([r[1] for r in rows]), [r[2] for r in rows],
               df["profitable"].mean(), "Rata de filme profitabile pe gen (un film poate avea mai multe genuri)",
               horizontal=True)
    return save(fig, "eda_rate_genre")


def fig_rate_flags(df: pd.DataFrame) -> str:
    flags = [("major_studio", "studio major"), ("is_franchise", "franciză"), ("is_adaptation", "adaptare"),
             ("is_english", "limba engleză"), ("has_director_history", "regizor cu istoric"),
             ("certification_missing", "fără clasificare SUA")]
    fig, axes = plt.subplots(1, len(flags), figsize=(12, 3), sharey=True)
    for ax, (col, label) in zip(axes, flags):
        r = df.groupby(col)["profitable"].mean()
        n = df[col].value_counts()
        ax.bar([0, 1], 100 * r.reindex([0, 1]).to_numpy(), color=["#94a3b8", ACCENT], width=0.6)
        ax.set_xticks([0, 1], ["nu", "da"])
        for x, v in zip([0, 1], r.reindex([0, 1])):
            ax.annotate(pct(v, 0), (x, 100 * v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9, color=INK_2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(f"n: {n.get(0, 0):,} / {n.get(1, 0):,}".replace(",", "."), fontsize=8, color=MUTED)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% profitabile")
    fig.tight_layout()
    return save(fig, "eda_rate_flags")


def budget_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in (("log_budget", "nominal"), ("log_budget_real", "dolari 2024")):
        q = pd.qcut(df[col], 5, labels=False)
        edges = np.exp(df.groupby(q)[col].max().to_numpy()[:-1]) / 1e6
        for k, (rate, n) in df.groupby(q)["profitable"].agg(["mean", "size"]).iterrows():
            rows.append({"buget": label, "grupa": k + 1, "% profitabile": rate, "filme": n,
                         "limita superioară (mil. $)": edges[k] if k < 4 else np.nan})
    return pd.DataFrame(rows)


def fig_budget_u(q: pd.DataFrame, n: int) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.38
    for i, (label, color) in enumerate(zip(("nominal", "dolari 2024"), PAIR)):
        s = q[q["buget"] == label]
        x = s["grupa"].to_numpy() + (i - 0.5) * (w + 0.02)
        ax.bar(x, 100 * s["% profitabile"], width=w, color=color, label=f"buget {label}")
        for xx, v in zip(x, s["% profitabile"]):
            ax.annotate(pct(v, 0), (xx, 100 * v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK_2)
    ax.set_xticks([1, 2, 3, 4, 5], ["1\ncele mai ieftine", "2", "3", "4", "5\ncele mai scumpe"])
    ax.set_ylabel("% profitabile")
    ax.set_title(f"Forma de U pe 5 grupe egale de buget, toate cele {num(n, 0)} de filme")
    ax.legend(frameon=False, loc="upper center", ncols=2)
    ax.set_ylim(0, 58)
    ax.grid(axis="x", visible=False)
    return save(fig, "eda_budget_u")


def fig_missing(df: pd.DataFrame) -> str:
    cols = ["certification", "director_prev_mean_log_roi", "cast_prev_mean_log_roi", "runtime_min", "sequel_number"]
    d = df.sort_values("release_date")
    m = d[cols].isna().to_numpy().T.astype(float)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.imshow(m, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#f1f5f9", ACCENT]), interpolation="nearest")
    ax.set_yticks(range(len(cols)), [f"{c}  ({pct(df[c].isna().mean())})" for c in cols])
    years = d["release_year"].to_numpy()
    # puține filme înainte de 1980: etichetele de deceniu s-ar suprapune
    tick_years = [1950, 1980, 1990, 2000, 2010, 2020]
    ax.set_xticks([np.searchsorted(years, y) for y in tick_years], tick_years)
    ax.grid(False)
    ax.set_title("Valori lipsă (colorat = lipsă), filmele ordonate după data lansării")
    return save(fig, "eda_missing")


def fig_corr(spearman: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(spearman, cmap=DIVERGING, vmin=-1, vmax=1)
    n = len(spearman)
    ax.set_xticks(range(n), spearman.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), spearman.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = spearman.iloc[i, j]
            if i != j and abs(v) >= 0.5:
                ax.text(j, i, num(v, 2), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.75 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="ρ Spearman")
    ax.set_title("Corelații Spearman între caracteristicile numerice (train 1950–2014)")
    return save(fig, "eda_corr")


def fig_relevance(rel: pd.DataFrame) -> str:
    top = rel.nlargest(20, "MI").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["caracteristică"], top["MI"], color=ACCENT, height=0.65)
    ax.set_xlabel("informație mutuală cu profitable (nați)")
    ax.set_title("Top 20 caracteristici după informația mutuală (train)")
    ax.grid(axis="y", visible=False)
    return save(fig, "eda_relevance")


# ---------------------------------------------------------------------------
# Raport
# ---------------------------------------------------------------------------
def fmt(df: pd.DataFrame, pct_cols=(), int_cols=(), digits=2, p_cols=()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(pct)
        elif c in int_cols:
            out[c] = out[c].map(lambda v: num(v, 0) if pd.notna(v) else "–")
        elif c in p_cols:
            out[c] = out[c].map(lambda v: "–" if pd.isna(v) else ("< 1e-300" if v == 0 else f"{v:.1e}" if v < 0.001 else num(v, 3)))
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: num(v, digits))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(DATASET)
    df = load_dataset()
    df["log_gross"] = np.log(df["worldwide_gross"].astype(float))
    train, _, _ = temporal_split(df)

    desc = descriptive(df)
    out = outliers(df)
    miss, miss_by, miss_extra = missing(df)
    spearman, pairs, vpairs = redundancy(train)
    rel, info = relevance(train)
    dups = duplicates(raw)
    quint = budget_quintiles(df)
    top_roi = df.nlargest(10, "roi")[["title", "release_year", "budget", "worldwide_gross", "roi"]]
    low_roi = df.nsmallest(5, "roi")[["title", "release_year", "budget", "worldwide_gross", "roi"]]

    for name, table in (("descriptive", desc), ("outliers", out), ("missing", miss), ("missing_certification", miss_by),
                        ("redundancy_numeric", pairs), ("redundancy_categorical", vpairs), ("relevance", rel),
                        ("budget_quintiles", quint), ("duplicates", dups)):
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)

    figs = {"hist": fig_hist(df), "box": fig_box_decade(df), "year": fig_rate_year(df), "month": fig_rate_month(df),
            "genre": fig_rate_genre(df), "flags": fig_rate_flags(df), "u": fig_budget_u(quint, len(df)),
            "missing": fig_missing(df), "corr": fig_corr(spearman), "rel": fig_relevance(rel)}

    q_nom = quint[quint["buget"] == "nominal"]["% profitabile"].to_numpy()
    q_real = quint[quint["buget"] == "dolari 2024"]["% profitabile"].to_numpy()
    rel_show = rel.head(20)[["caracteristică", "MI", "r (profitable)", "p", "ρ (log_roi)", "χ²", "importanță RF", "coef. L1"]]
    strata = miss_extra["strata"]

    text = f"""# Analiza exploratorie (#42)

Generat cu `python analysis/eda.py`. Tabelele complete: `results/eda/*.csv`; figurile: `docs/figures/eda_*.png`.

Dataset: **{len(df):,} filme** (după scoaterea a {len(DUPLICATES)} duplicate, vezi §8), {pct(df['profitable'].mean())} profitabile (ROI ≥ 2,5). Statisticile descriptive, outliers și datele lipsă sunt pe tot dataset-ul; redundanța și relevanța față de țintă sunt calculate **doar pe train-ul temporal** (1950–2014, {len(train):,} filme), ca decizia de a scoate o caracteristică să nu vadă filmele de test.

## Pe scurt

1. **Banii sunt foarte asimetrici.** ROI are skewness {num(desc.set_index('coloană').loc['roi', 'skewness'])} și kurtosis {num(desc.set_index('coloană').loc['roi', 'kurtosis'], 0)}: media ({num(desc.set_index('coloană').loc['roi', 'medie'])}) e de {num(desc.set_index('coloană').loc['roi', 'medie'] / desc.set_index('coloană').loc['roi', 'mediană'], 1)} ori mediana ({num(desc.set_index('coloană').loc['roi', 'mediană'])}), din cauza câtorva filme cu buget mic și încasări uriașe (Paranormal Activity: ROI {num(df['roi'].max(), 0)}). După log, skewness scade la {num(desc.set_index('coloană').loc['log_roi', 'skewness'])}. Modelele lucrează pe `log_budget_real` și `log_roi` (#40).
2. **Forma de U pe buget** se păstrează pe toate filmele, nominal ({' / '.join(pct(v, 0) for v in q_nom)}) și în dolari 2024 ({' / '.join(pct(v, 0) for v in q_real)}): filmele foarte ieftine și cele foarte scumpe sunt mai des profitabile decât cele de mijloc. Un model liniar pe `log_budget_real` nu poate prinde un U (ar avea nevoie de un termen pătratic, de testat în #41); arborii pot.
3. **Rata de profitabile se schimbă în timp**: anii '70 au cea mai mare rată ({pct(df[df.release_year.between(1970, 1979)]['profitable'].mean())}), iar setul de test (2019–2024) are {pct(df[df.release_year >= 2019]['profitable'].mean())}, cu minimul în 2020 și 2021 (COVID: {pct(df[df.release_year == 2020]['profitable'].mean())} și {pct(df[df.release_year == 2021]['profitable'].mean())}).
4. **Cele mai informative caracteristici** (informație mutuală, pe train): {', '.join(f'`{c}`' for c in rel.head(5)['caracteristică'])}. Bugetul, anul lansării și istoricul actorilor și al regizorului poartă cel mai mult semnal, dar fiecare efect e mic luat separat (cea mai mare corelație cu ținta: {num(rel['r (profitable)'].abs().max())}), deci ținta nu se prezice dintr-o singură coloană.
5. **Redundanță**: 4 coloane se calculează exact din altele și ies din model (§9): `log_budget`, `major_any`, `is_english`, `is_adaptation`. După ele, modelul primește {info['n_model']} de coloane.

## 1. Tipuri de date

{md_table(pd.DataFrame(TYPES, columns=['caracteristică', 'tip', 'observații']))}

Binarele **asimetrice** (genuri, țări, studio) au un „1” informativ și un „0” care înseamnă doar „nu știm / nu e cazul”: pentru similaritate se folosește Jaccard, nu potrivirea simplă. `is_english` e **simetrică**.

## 2. Statistici descriptive

{md_table(fmt(desc, int_cols=['n']))}

- `budget`: media ({num(desc.set_index('coloană').loc['budget', 'medie'] / 1e6, 1)} mil. $) e mult peste mediană ({num(desc.set_index('coloană').loc['budget', 'mediană'] / 1e6, 1)} mil. $); skewness {num(desc.set_index('coloană').loc['budget', 'skewness'])} → {num(desc.set_index('coloană').loc['log_budget', 'skewness'])} după log.
- Media trunchiată (10% din fiecare capăt) e aproape de mediană la toate coloanele de bani: diferența dintre medie și mediană vine doar din cozi.
- `director_prev_mean_roi` are aceeași problemă ca ROI (kurtosis {num(desc.set_index('coloană').loc['director_prev_mean_roi', 'kurtosis'], 0)}); de aceea modelul folosește `director_prev_mean_log_roi` (#40).

{figs['hist']}

{figs['box']}

## 3. Outliers

{md_table(fmt(out, pct_cols=['% IQR']))}

- Pe datele brute, regula 1,5×IQR marchează mii de filme ca outliers, pentru că distribuția are o coadă lungă, nu pentru că valorile sunt greșite. Pe log, numărul scade mult: regula se aplică pe `log_roi`, nu pe ROI.
- Experimentul cu / fără outliers din #41 folosește limitele IQR pe `log_roi` (calculate pe train).

**Cele mai mari ROI-uri** (reale, nu erori: filme cu buget foarte mic devenite fenomene):

{md_table(fmt(top_roi, int_cols=['budget', 'worldwide_gross', 'release_year'], digits=0))}

**Cele mai mici ROI-uri** (încasări de câteva mii de dolari: lansări aproape doar simbolice):

{md_table(fmt(low_roi, int_cols=['budget', 'worldwide_gross', 'release_year'], digits=4))}

## 4. Date lipsă

{md_table(fmt(miss, pct_cols=['%'], int_cols=['lipsă']))}

{figs['missing']}

**Lipsa `certification`**: depinde puternic de limbă și de țara de producție (test χ²):

{md_table(fmt(miss_by, pct_cols=['lipsă dacă 1', 'lipsă dacă 0'], p_cols=['p']))}

Lipsa e explicată în mare parte de coloane pe care le avem (limbă, țară), deci e cel puțin **MAR**. Rămâne o legătură cu ținta și în interiorul fiecărei grupe limbă × țară (lipsa nu e doar o chestiune de limbă):

{md_table(fmt(strata, pct_cols=['profitabile cu clasificare', 'profitabile fără clasificare'], int_cols=['filme']))}

Diferența din fiecare grupă sugerează o componentă **MNAR**: filmele fără lansare clasificată în SUA sunt, de obicei, filme mai mici. De aceea #40 păstrează indicatorul `certification_missing` în loc să completeze doar cu mediana: indicatorul e el însuși o informație.

Istoricul regizorului / actorilor lipsește **structural** (primul film, verificat în #40), iar `runtime_min` și `sequel_number` au sub 0,3% lipsă.

## 5. Ce se vede față de țintă

{figs['u']}

{md_table(fmt(quint, pct_cols=['% profitabile'], int_cols=['filme']))}

{figs['year']}

{figs['month']}

{figs['genre']}

{figs['flags']}

## 6. Redundanță (train)

**Perechi numerice cu |ρ Spearman| ≥ 0,6:**

{md_table(fmt(pairs))}

{figs['corr']}

**Perechi categorice / binare cu V Cramér ≥ 0,5:**

{md_table(fmt(vpairs))}

## 7. Relevanță față de țintă (train)

Pentru fiecare dintre cele {info['n_features']} coloane după preprocesare: informația mutuală (MI) cu `profitable`, corelația punct-biserială (r) cu `profitable` și p-valoarea ei, ρ Spearman cu `log_roi`, χ² (doar pentru coloanele 0/1), importanța într-un Random Forest (300 de arbori) și coeficientul unei regresii logistice cu penalizare L1 (Lasso, C = 0,1, date scalate). Tabelul include și cele 4 coloane scoase ca redundante (§9); ele nu au coeficient L1, pentru că nu intră în model. Top 20 după MI:

{md_table(fmt(rel_show, p_cols=['p'], digits=3))}

{figs['rel']}

- **Corecția Bonferroni:** cu {info['n_tests']} teste, pragul 0,05 devine {info['alpha_bonf']:.1e}. Corelații semnificative: {info['sig_raw']} fără corecție, **{info['sig_bonf']} cu corecție**. Cu {num(len(train), 0)} de filme, chiar și corelații foarte mici ies semnificative; contează mărimea efectului, nu doar p.
- **Lasso:** câte dintre cele {info['n_model']} de coloane ale modelului ajung la coeficient zero, în funcție de penalizare (C mic = penalizare mare): {', '.join(f"C = {num(C, 2)}: {n}" for C, n in info['lasso_path'].items())}. La C = 0,1 rămân la zero: {', '.join(f'`{c}`' for c in info['lasso_zero'])}. Lasso vede doar efecte liniare și independente; o coloană pusă la zero aici poate conta într-un arbore (prin interacțiuni), deci nu scoatem nimic doar pe baza lui.

## 8. Duplicate semantice

Perechile cu același titlu și același an în `data/processed/movies_dataset.parquet`, verificate manual:

{md_table(fmt(dups, int_cols=['budget', 'worldwide_gross', 'release_year', 'runtime_min'], digits=0))}

- Majoritatea sunt **omonime**: filme diferite (alt `imdb_id`, altă limbă, alt buget). Rămân.
- **Alone (2020)**: două filme diferite, dar cu buget și încasări identice, ambele din același infobox Wikipedia. Pagina e a filmului `tt7711170` (confirmată și de DBpedia); `tt10192566` a primit cifrele altui film → scos.
- **The Message (1976)**: versiunea în engleză și cea în arabă, filmate simultan, aceeași producție și același buget. Wikipedia are un singur articol pentru ambele, deci bugetul ar fi numărat de două ori → păstrăm doar versiunea în engleză (`tt0074896`).
- Cele două filme sunt scoase în `modeling/preprocessing.py` (`DUPLICATES`), nu în dataset-ul final, ca decizia să fie vizibilă și reversibilă.

## 9. Propunere: ce scoatem din model

{DECISIONS}
"""
    REPORT.write_text(text, encoding="utf-8")
    print(f"Scris: {REPORT}")
    print(f"Figuri: {len(figs)} în {FIG_DIR}")


DECISIONS = """Regula: scoatem o coloană doar dacă e **redundantă exact** (se calculează din alte coloane ale
modelului) sau aproape (ρ sau V Cramér ≥ 0,9). Pentru relevanță mică nu scoatem nimic aici:
arborii pot folosi o coloană slabă în combinație cu altele, iar comparația cu / fără se face pe
validare, în #41.

| Coloană | Decizie | Motiv (verificat pe date) |
|---|---|---|
| `log_budget` (nominal) | **scos** | = `log_budget_real` − log(factorul CPI al anului), diferență maximă 4·10⁻¹⁵: cu `release_year` în model nu aduce nimic nou (ρ = 0,90 cu varianta reală). Păstrăm varianta reală: e comparabilă între ani, iar informația mutuală e practic aceeași (0,030 la ambele). |
| `major_any` | **scos** | = `major_studio` SAU `major_distributor` pe 100% din filme (V = 0,98 cu `major_distributor`). |
| `is_english` | **scos** | = `original_language == "en"` pe 100% din filme (V = 1,00); one-hot-ul limbii o conține deja. |
| `is_adaptation` | **scos** | = `adaptation_type != "none"` pe 100% din filme (V = 1,00). |
| `director_prev_mean_log_roi` / `_profitable_rate` | păstrate amândouă | ρ = 0,84 (sub 0,9). Media ROI-ului și proporția de hituri măsoară lucruri diferite (un hit uriaș vs hituri constante), iar Lasso le păstrează pe amândouă. |
| `cast_prev_mean_log_roi` / `_profitable_rate` | păstrate amândouă | ρ = 0,79; același motiv. |
| `*_award_noms_before` / `*_award_wins_before` | păstrate | ρ = 0,84 (actori) și 0,69 (regizori). Relevanța e foarte mică (corelație cu ținta sub 0,02, nesemnificativă după Bonferroni), dar au importanță nenulă în Random Forest. Varianta fără premii se testează în #41. |
| `country_*` vs `original_language` | păstrate | V = 0,77: legate, dar nu identice (filme în engleză produse în Europa, coproducții). |
| `certification_missing` vs limbă / țară | păstrat | V ≈ 0,58; e și o informație proprie (§4). |
| genurile | păstrate | Lasso pune la zero o parte dintre ele, dar rata de profitabile diferă mult între genuri (§5). |

Rezultat: **4 coloane scoase**, aplicate în `modeling/preprocessing.py` (`REPLACED`). Modelul
primește 63 de coloane după preprocesare (înainte: 67)."""

if __name__ == "__main__":
    main()
