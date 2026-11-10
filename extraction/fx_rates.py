"""
Cursuri de schimb (#45): World Bank, indicatorul PA.NUS.FCRF
("Official exchange rate, LCU per US$, period average"), licență CC BY 4.0.
https://data.worldbank.org/indicator/PA.NUS.FCRF

Run:
  python extraction/fx_rates.py     # -> data/interim/fx_rates.parquet (o cerere, apoi din cache)

Ieșire: un rând per (monedă, an) cu `rate` = unități de monedă locală pentru 1 USD (media
anului) și `hyperinflation` = cursul s-a schimbat cu peste 50% față de anul anterior (din
aceeași serie). Conversia: suma / rate.

Moneda -> țara a cărei serie o folosim (FX_COUNTRY). Zona euro: moneda națională (DEM, FRF,
ITL ...) până la intrarea în euro (seria țării e în moneda națională pentru acei ani), EUR
(seria agregată „Euro area”, EMU) de la 1999. Seria începe în 1960; anii fără curs rămân fără
curs: nu inventăm valori.

Monedele redenominate (Brazilia înainte de 1994, Turcia înainte de 2005, Rusia înainte de
1998) apar în serie în unitatea de azi, deci o sumă veche în unitatea veche ar ieși greșit.
Acei ani au aproape toți hiperinflație (exclusă aici); restul cad la verificarea intervalelor
plauzibile din integrate.py. Un curs 0 (rotunjit de World Bank) e tratat ca lipsă.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.utils import PROJECT_ROOT, YEAR_MAX, YEAR_MIN, _get_logger, fetch

API = "https://api.worldbank.org/v2/country/all/indicator/PA.NUS.FCRF"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "fx_rates.parquet"
HYPERINFLATION = 0.5        # |curs / curs anul trecut - 1| > 50% -> an exclus

log = _get_logger("worldbank")

# Moneda -> codul World Bank (ISO3) al țării sau zonei a cărei serie o folosim
FX_COUNTRY = {
    "EUR": "EMU", "GBP": "GBR", "JPY": "JPN", "CNY": "CHN", "INR": "IND", "KRW": "KOR", "RUB": "RUS",
    "CAD": "CAN", "AUD": "AUS", "HKD": "HKG", "NZD": "NZL", "SGD": "SGP", "BRL": "BRA", "MXN": "MEX",
    "ARS": "ARG", "CLP": "CHL", "COP": "COL", "PEN": "PER", "VES": "VEN", "DOP": "DOM", "JMD": "JAM",
    "CHF": "CHE", "SEK": "SWE", "NOK": "NOR", "DKK": "DNK", "ISK": "ISL", "PLN": "POL", "CZK": "CZE",
    "HUF": "HUN", "RON": "ROU", "BGN": "BGR", "HRK": "HRV", "RSD": "SRB", "TRY": "TUR", "UAH": "UKR",
    "KZT": "KAZ", "GEL": "GEO", "ILS": "ISR", "EGP": "EGY", "NGN": "NGA", "ZAR": "ZAF", "MAD": "MAR",
    "IRR": "IRN", "PKR": "PAK", "BDT": "BGD", "LKR": "LKA", "NPR": "NPL", "MVR": "MDV", "MYR": "MYS",
    "IDR": "IDN", "THB": "THA", "PHP": "PHL", "VND": "VNM", "KES": "KEN", "GHS": "GHA", "ETB": "ETH",
    "TND": "TUN", "DZD": "DZA", "LBP": "LBN", "JOD": "JOR", "AED": "ARE", "SAR": "SAU", "QAR": "QAT",
    "KWD": "KWT",
    # monede naționale înlocuite de euro: seria țării (în euro) * cursul fix
    "DEM": "DEU", "FRF": "FRA", "ITL": "ITA", "ESP": "ESP", "NLG": "NLD", "BEF": "BEL", "ATS": "AUT",
    "PTE": "PRT", "IEP": "IRL", "FIM": "FIN", "GRD": "GRC", "LUF": "LUX", "SIT": "SVN", "SKK": "SVK",
    "EEK": "EST", "LVL": "LVA", "LTL": "LTU",
}
# Primul an în euro: de atunci seria țării e în euro, iar moneda națională nu mai are curs
EURO_LEGACY = {
    "DEM": 1999, "FRF": 1999, "ITL": 1999, "ESP": 1999, "NLG": 1999, "BEF": 1999, "ATS": 1999,
    "PTE": 1999, "IEP": 1999, "FIM": 1999, "LUF": 1999, "GRD": 2001, "SIT": 2007, "SKK": 2009,
    "EEK": 2011, "LVL": 2014, "LTL": 2015,
}
# Monedele fără serie (sovietică, cehoslovacă, taiwaneză) rămân fără curs: SUR, CSK, TWD


def download() -> pd.DataFrame:
    """Seria pentru toate țările, o singură cerere (cache în data/raw/worldbank/)."""
    data = fetch(API, source="worldbank",
                 params={"format": "json", "per_page": 20000, "date": f"{YEAR_MIN}:{YEAR_MAX}"})
    if not data or len(data) < 2:
        raise RuntimeError("World Bank: răspuns gol, vezi logs/worldbank.log")
    meta, rows = data
    if meta.get("pages", 1) > 1:
        raise RuntimeError(f"World Bank: {meta['pages']} pagini, mărește per_page")
    df = pd.DataFrame([{"country": r["countryiso3code"] or r["country"]["id"], "year": int(r["date"]),
                        "lcu_per_usd": r["value"]} for r in rows])
    return df.dropna(subset=["lcu_per_usd"])


def build(raw: pd.DataFrame) -> pd.DataFrame:
    series = {c: g.set_index("year")["lcu_per_usd"] for c, g in raw.groupby("country")}
    rows = []
    for currency, country in FX_COUNTRY.items():
        s = series.get(country)
        if s is None:
            log.warning("%s: nicio serie pentru %s", currency, country)
            continue
        s = s[s > 0].sort_index()
        if currency in EURO_LEGACY:
            s = s[s.index < EURO_LEGACY[currency]]
        change = (s / s.shift(1) - 1).abs()
        # schimbarea se compară doar cu anul imediat anterior
        change = change.where(pd.Series(s.index, index=s.index).diff() == 1)
        for year, rate in s.items():
            rows.append({"currency": currency, "year": int(year), "rate": float(rate),
                         "country": country, "change": change.get(year),
                         "hyperinflation": bool(change.get(year, 0) > HYPERINFLATION)})
    df = pd.DataFrame(rows)
    # EUR înainte de 1999 nu există: anii aceia din seria EMU nu se folosesc
    return df[~((df["currency"] == "EUR") & (df["year"] < 1999))].reset_index(drop=True)


def report(df: pd.DataFrame) -> str:
    first = df.groupby("currency")["year"].min()
    hyper = df[df["hyperinflation"]]
    return "\n".join([
        f"Monede cu curs: {df['currency'].nunique()}, rânduri (monedă, an): {len(df)}",
        f"Primul an cu curs: {int(first.min())} (monede care încep după 1960: "
        f"{', '.join(f'{c}={y}' for c, y in first[first > 1960].sort_values().items())})",
        f"Ani cu hiperinflație (> 50% față de anul anterior), excluși: {len(hyper)} "
        f"({', '.join(sorted(hyper['currency'].unique()))})",
        f"Monede fără curs: {', '.join(sorted(set(FX_COUNTRY) - set(df['currency'])))}",
    ])


if __name__ == "__main__":
    df = build(download())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Scris: {OUT_PATH}\n")
    print(report(df))
