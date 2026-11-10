"""
Test de sensibilitate (#45): cât de mult depinde ținta `profitable` de deciziile de curățare.

Refolosește funcțiile din integrate.py (aceleași date din data/interim/, fără cereri HTTP) și
schimbă câte o singură decizie față de varianta implicită:
  - alegerea bugetului: majoritate (implicit) / prioritate TMDB → DBpedia → Wikidata → Wikipedia /
    prioritate cu Wikipedia prima
  - alegerea încasărilor: prioritatea curentă (Wikipedia prima) / TMDB prima
  - --roi-rule: vechi / surse-independente / roi-recalculat (implicit)
  - pragul de profitabilitate: 2 / 2,5 (implicit) / 3
  - calitatea încasărilor: toate filmele (implicit) / fără încasările parțiale (o singură piață sau
    „rentals”, #47, #49) / și fără încasările în moneda locală fără „worldwide” (#49)

Pentru fiecare variantă: câte filme rămân, % profitabile și câte etichete `profitable` se
schimbă față de varianta implicită (la filmele prezente în ambele), plus filmele intrate / ieșite.

Usage:
    python extraction/sensitivity.py      # -> docs/sensitivity_report.md
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.integrate import (BUDGET_SOURCES, GROSS_SOURCES, INTERIM, _md_table, _n, _pct,
                                  clean, load_fx, load_sources)
from extraction.utils import PROJECT_ROOT

REPORT = PROJECT_ROOT / "docs" / "sensitivity_report.md"
DEFAULT = {"budget_choice": "majoritate", "budget_sources": BUDGET_SOURCES, "gross_sources": GROSS_SOURCES,
           "roi_rule": "roi-recalculat", "threshold": 2.5, "drop_partial_gross": 0}


def _order(sources: dict, first: str) -> dict:
    return {first: sources[first], **{k: v for k, v in sources.items() if k != first}}


VARIANTS = [
    # (grup, nume, schimbarea față de DEFAULT)
    ("Buget", "majoritate (implicit)", {}),
    ("Buget", "prioritate tmdb → dbpedia → wikidata → wikipedia", {"budget_choice": "prioritate"}),
    ("Buget", "prioritate wikipedia → tmdb → dbpedia → wikidata",
     {"budget_choice": "prioritate", "budget_sources": _order(BUDGET_SOURCES, "wikipedia")}),
    ("Încasări", "wikipedia → dbpedia → wikidata → tmdb (implicit)", {}),
    ("Încasări", "tmdb → wikipedia → dbpedia → wikidata", {"gross_sources": _order(GROSS_SOURCES, "tmdb")}),
    ("ROI extrem", "vechi", {"roi_rule": "vechi"}),
    ("ROI extrem", "surse-independente", {"roi_rule": "surse-independente"}),
    ("ROI extrem", "roi-recalculat (implicit)", {}),
    ("Prag", "2", {"threshold": 2.0}),
    ("Prag", "2,5 (implicit)", {}),
    ("Prag", "3", {"threshold": 3.0}),
    ("Calitatea încasărilor", "toate filmele (implicit)", {}),
    ("Calitatea încasărilor", "fără încasări parțiale (o singură piață, rentals)", {"drop_partial_gross": 1}),
    ("Calitatea încasărilor", "și fără moneda locală fără „worldwide”", {"drop_partial_gross": 2}),
]


def run(base: pd.DataFrame, fx: dict, cfg: dict) -> pd.DataFrame:
    """`profitable` (și dacă sursele sunt în dezacord) pentru filmele care trec curățarea."""
    df = clean(base.copy(), [], roi_rule=cfg["roi_rule"], budget_choice=cfg["budget_choice"],
               budget_sources=cfg["budget_sources"], gross_sources=cfg["gross_sources"], fx=fx, stats={})
    if cfg["drop_partial_gross"] >= 1:
        df = df[~(df["gross_is_single_market"] | df["gross_is_rental"])]
    if cfg["drop_partial_gross"] >= 2:
        df = df[~df["gross_is_local_currency"]]
    return pd.DataFrame({"profitable": (df["roi"] >= cfg["threshold"]).astype(int),
                         "conflict": df["budget_conflict"] | df["gross_conflict"]})


def compare(default_df: pd.DataFrame, other_df: pd.DataFrame) -> dict:
    default, other = default_df["profitable"], other_df["profitable"]
    common = default.index.intersection(other.index)
    flips = common[default[common] != other[common]]
    changed = len(flips)
    return {"Filme": int(len(other)), "% profitabile": _pct(other.mean()),
            "Etichete schimbate": f"{_n(changed)} ({_pct(changed / len(common))})",
            "Filme intrate / ieșite": f"+{_n(len(other.index.difference(default.index)))} / "
                                      f"−{_n(len(default.index.difference(other.index)))}",
            "_changed": changed / len(common), "_pos": other.mean(),
            "_flips": flips, "_conflict": default_df.loc[flips, "conflict"].mean() if changed else 0.0}


def write(results: list[dict]) -> None:
    default = next(r for r in results if r["_default"])
    others = [r for r in results if not r["_default"]]
    worst = max(others, key=lambda r: r["_changed"])
    film_rules = [r for r in others if r["Grup"] in ("Buget", "Încasări", "ROI extrem")]
    partial, local = [r for r in others if r["Grup"] == "Calitatea încasărilor"]
    worst_rule = max(film_rules, key=lambda r: r["_changed"])
    pos = [r["_pos"] for r in film_rules] + [default["_pos"]]
    flips = pd.Index([]).append([r["_flips"] for r in film_rules]).unique()
    conflict_share = default["_conflict_all"].reindex(flips).mean()
    base_conflict = default["_conflict_all"].mean()

    sections = []
    for group in dict.fromkeys(r["Grup"] for r in results):
        rows = [{k: v for k, v in r.items() if not k.startswith("_") and k != "Grup"}
                for r in results if r["Grup"] == group]
        sections.append(f"### {group}\n\n{_md_table(rows)}")

    text = f"""# Test de sensibilitate (#45)

Generat cu `python extraction/sensitivity.py`. Fiecare variantă schimbă **o singură** decizie față de varianta implicită (`python extraction/integrate.py`: buget prin majoritate, încasări cu Wikipedia prima, `--roi-rule roi-recalculat`, prag 2,5) și refolosește funcțiile din `integrate.py`. „Etichete schimbate” = filmele prezente în ambele variante la care `profitable` diferă.

Varianta implicită: **{_n(default["Filme"])} filme, {default["% profitabile"]} profitabile**.

{(chr(10) * 2).join(sections)}

## Concluzie

- Deciziile despre date (ce buget sau ce încasări alegem, ce ROI extrem păstrăm) mută rata de filme profitabile doar între {_pct(min(pos))} și {_pct(max(pos))} și schimbă cel mult {_pct(worst_rule["_changed"])} din etichete ({worst_rule["Grup"].lower()}: {worst_rule["Varianta"]}). Ținta e **robustă** la aceste alegeri: doar câteva sute de filme, cele cu surse în dezacord, își schimbă clasa.
- Fără filmele cu încasări parțiale (o singură piață sau „rentals”) rămân {_n(partial["Filme"])} filme, cu {partial["% profitabile"]} profitabile; fără și cele în moneda locală fără „worldwide”, {_n(local["Filme"])} filme, cu {local["% profitabile"]} profitabile (implicit: {default["% profitabile"]}). Etichetele celorlalte filme nu se schimbă: rata crește doar pentru că se scot filme etichetate, cel mai probabil, greșit ca neprofitabile.
- Pragul contează mult mai mult: varianta cu cel mai mare efect e {worst["Grup"].lower()} {worst["Varianta"]}, cu {_pct(worst["_changed"])} etichete schimbate. Pragul de 2,5 e o convenție (bugetul nu include marketingul, iar cinematografele păstrează aproximativ jumătate din bilete), deci rezultatele modelului trebuie raportate și pentru pragurile 2 și 3, nu doar pentru 2,5.
- Dintre cele {_n(len(flips))} filme care își schimbă eticheta într-una din variantele despre date, {_pct(conflict_share)} au surse în dezacord la buget sau la încasări (`budget_conflict` / `gross_conflict`), față de {_pct(base_conflict)} în tot dataset-ul. Incertitudinea e concentrată în filmele cu surse care nu se pun de acord, nu în regula aleasă.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    manifest = json.loads((INTERIM / "movies_manifest.json").read_text(encoding="utf-8"))
    base = load_sources(manifest)
    fx = load_fx()
    default = run(base, fx, DEFAULT)
    results, cache = [], {}
    for group, name, change in VARIANTS:
        cfg = {**DEFAULT, **change}
        key = json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in cfg.items()}, sort_keys=True)
        if key not in cache:
            cache[key] = default if not change else run(base, fx, cfg)
        r = {"Grup": group, "Varianta": name, **compare(default, cache[key]), "_default": not change,
             "_conflict_all": default["conflict"]}
        results.append(r)
        print(f"  {group:10s} {name:50s} {r['Filme']:>7,} {r['% profitabile']:>7s}  {r['Etichete schimbate']}")
    write(results)
    print(f"Scris {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
