"""
Indicele prețurilor de consum SUA (#40): FRED, seria CPIAUCNS ("Consumer Price Index for All
Urban Consumers: All Items in U.S. City Average", lunar, nedesezonalizat). FRED republică
seria oficială a Bureau of Labor Statistics (domeniu public).
https://fred.stlouisfed.org/series/CPIAUCNS

Run:
  python extraction/cpi.py     # -> data/interim/cpi.parquet (o cerere, apoi din cache)

De ce FRED și nu World Bank (FP.CPI.TOTL): seria World Bank începe în 1960, iar dataset-ul are
774 de filme din 1950–1959. CPIAUCNS începe în 1913.

Ieșire: un rând per an cu `cpi` = media celor 12 luni și `factor_2024` = CPI(2024) / CPI(an).
Bugetele sunt deja în USD (convertite cu cursul din anul lansării), deci se ajustează cu
inflația SUA: budget_real = budget * factor_2024.
"""

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import PROJECT_ROOT, YEAR_MAX, YEAR_MIN, fetch

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = "CPIAUCNS"
BASE_YEAR = YEAR_MAX
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "cpi.parquet"


def download() -> pd.DataFrame:
    """Seria lunară, o singură cerere (cache în data/raw/fred/)."""
    text = fetch(URL, source="fred", params={"id": SERIES}, as_json=False)
    if not text:
        raise RuntimeError("FRED: răspuns gol, vezi logs/fred.log")
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "cpi"]
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    return df.dropna(subset=["cpi"])


def build(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.assign(year=pd.to_datetime(monthly["date"]).dt.year)
    yearly = monthly.groupby("year").agg(cpi=("cpi", "mean"), months=("cpi", "size")).reset_index()
    # doar anii compleți: media unui an parțial nu e comparabilă
    yearly = yearly[(yearly["months"] == 12) & yearly["year"].between(YEAR_MIN, YEAR_MAX)]
    missing = sorted(set(range(YEAR_MIN, YEAR_MAX + 1)) - set(yearly["year"]))
    if missing:
        raise RuntimeError(f"FRED: lipsesc anii {missing}")
    base = yearly.loc[yearly["year"] == BASE_YEAR, "cpi"].item()
    yearly["factor_2024"] = base / yearly["cpi"]
    return yearly[["year", "cpi", "factor_2024"]].reset_index(drop=True)


if __name__ == "__main__":
    df = build(download())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Scris: {OUT_PATH} ({len(df)} ani)")
    for y in (1950, 1965, 1980, 2000, 2015, 2024):
        print(f"  {y}: CPI {df.loc[df.year == y, 'cpi'].item():.1f}, "
              f"1 USD = {df.loc[df.year == y, 'factor_2024'].item():.2f} USD în {BASE_YEAR}")
