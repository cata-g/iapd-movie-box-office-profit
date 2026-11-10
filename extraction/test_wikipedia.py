"""
Manual test for the infobox money parser in extraction/wikipedia.py.

Run: python extraction/test_wikipedia.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.wikipedia import MULT, _num, gross_scope, infobox_field, parse_money, parse_names

# raw infobox value -> (amount, currency, is_range, usd)   (usd = suma în USD scrisă în infobox)
CASES = {
    "$15 million": (15e6, "USD", False, 15e6),
    "$10–15 million": (12.5e6, "USD", True, 12.5e6),
    "US$2.5 million": (2.5e6, "USD", False, 2.5e6),
    "$1,200,000": (1.2e6, "USD", False, 1.2e6),
    "$200,000–$750,000<ref name=\"THR\">{{cite web |url=x}}</ref>": (475_000, "USD", True, 475_000),
    "$15,000-$450,000": (232_500, "USD", True, 232_500),
    "$3.5&nbsp;million<ref name=\"Mojo\" />": (3.5e6, "USD", False, 3.5e6),
    "$356–400{{nbsp}}million": (378e6, "USD", True, 378e6),
    "$2.94{{nbsp}}billion": (2.94e9, "USD", False, 2.94e9),
    "{{US$|160 million}}": (160e6, "USD", False, 160e6),
    "$35.0 Million": (35e6, "USD", False, 35e6),
    "Production: $7,225<br />Post-production: $200,000": (7_225, "USD", False, 7_225),
    "USD 4 million": (4e6, "USD", False, 4e6),
    "{{USD}}5 million": (5e6, "USD", False, 5e6),
    "{{usd|65,000}}": (65_000, "USD", False, 65_000),
    # alte monede: suma și codul ISO
    "A$2,417,000": (2_417_000, "AUD", False, None),
    "MXN$142.3 million": (142.3e6, "MXN", False, None),
    "{{INR|55 crore}}": (5.5e8, "INR", False, None),
    "₹40 crore": (4e8, "INR", False, None),
    "₹35 lakhs": (3.5e6, "INR", False, None),
    "{{INR}}30 million": (30e6, "INR", False, None),
    "{{INRConvert|3.6|c|year=1978}}": (3.6e7, "INR", False, None),
    "{{Currency|1.2 million|CAD}}": (1.2e6, "CAD", False, None),
    "CAD $1.5 million": (1.5e6, "CAD", False, None),
    "£3 million": (3e6, "GBP", False, None),
    "€10 million": (10e6, "EUR", False, None),
    "MYR 2.80 million": (2.8e6, "MYR", False, None),
    "8 169 771 ESP": (8_169_771, "ESP", False, None),
    "{{Philippine peso|5 million}}": (5e6, "PHP", False, None),
    # suma în USD scrisă explicit în paranteză are prioritate la conversie
    "£800,000 ($1.4 million)": (800_000, "GBP", False, 1.4e6),
    "MYR 35.88 million (USD 7.68 million)": (35.88e6, "MYR", False, 7.68e6),
    "£3 million (US$4.5 million)": (3e6, "GBP", False, 4.5e6),
    "415,000,000 ITL ($665,000)": (415e6, "ITL", False, 665_000),
    "{{JPY|200 million|link=yes}} ({{US$|{{To USD|200000|JPN|year=1977|round=yes}},000|long=no}})":
        (200e6, "JPY", False, None),     # USD calculat de un șablon, nu scris: se ignoră
    # într-o listă, elementul „worldwide” e cel bun
    "{{Plainlist| * {{JPY|1.5 billion}} (Japan) * {{US$|14.7 million|long=no}} (worldwide)}}":
        (14.7e6, "USD", False, 14.7e6),
    # virgula zecimală europeană (#52): 1–2 cifre după virgulă și apoi o unitate = zecimale
    "€13,7 million": (13.7e6, "EUR", False, None),
    "$4,5 million": (4.5e6, "USD", False, 4.5e6),
    "0,3 million $": (0.3e6, "USD", False, 0.3e6),
    "11,6 million guilders": (11.6e6, "NLG", False, None),
    "[[United States dollar|US$]]8,7 million": (8.7e6, "USD", False, 8.7e6),
    "€2,35 Mio.": (2.35e6, "EUR", False, None),
    "US$1,250 million": (1.25e9, "USD", False, 1.25e9),   # 3 cifre = separator de mii
    "$4,500,000": (4.5e6, "USD", False, 4.5e6),
    "": (None, None, False, None),
}

# simboluri ambigue: (raw, țări de producție TMDB) -> (amount, currency, note)
AMBIGUOUS_CASES = [
    ("¥1 billion", ["JP"], (1e9, "JPY", "")),
    ("¥1 billion", ["CN"], (1e9, "CNY", "")),
    ("¥1 billion", [], (1e9, None, "ambiguous")),          # fără țară clară: ignorat și numărat
    ("¥1 billion", ["JP", "CN"], (1e9, None, "ambiguous")),
    ("4 million [[Swedish krona|kr]]", ["SE"], (4e6, "SEK", "")),
    ("kr 4 million", ["DK", "SE"], (4e6, None, "ambiguous")),
    ("Rs 12 [[lakh]]", ["IN"], (1.2e6, "INR", "")),
    ("$5 million", ["AU"], (5e6, "AUD", "")),
    ("$5 million", ["AU", "US"], (5e6, "USD", "")),
    ("$5 million", ["GB"], (5e6, "USD", "")),
    ("$5 million", ["AU", "CA"], (5e6, None, "ambiguous")),
    ("ZW$5 million", [], (None, None, "unknown")),
]


def test_parse_money():
    for raw, expected in CASES.items():
        m = parse_money(raw)
        got = (m.amount, m.currency, m.is_range, m.usd)
        assert got == expected, f"{raw!r}: expected {expected}, got {got}"
    print(f"Test 1 passed: {len(CASES)} formate de sume parsate corect.")


def test_decimal_comma():
    assert parse_money("€13,7 million").misread == 137e6        # cum citea parserul vechi
    assert _num("1,250", "million") * MULT["million"] == 1.25e9  # „1,250 million”: separator de mii
    assert _num("13,7", "million") * MULT["million"] == 13.7e6   # „13,7 million”: zecimală
    assert _num("4,500,000") == 4.5e6
    assert parse_money("$1,250 million").amount == 1.25e9
    assert parse_money("$4,500,000").misread is None            # citire neschimbată
    print("Test 6 passed: virgula zecimală vs. separatorul de mii.")


def test_ambiguous():
    for raw, countries, expected in AMBIGUOUS_CASES:
        m = parse_money(raw, countries)
        got = (m.amount, m.currency, m.note)
        assert got == expected, f"{raw!r} {countries}: expected {expected}, got {got}"
    print(f"Test 2 passed: {len(AMBIGUOUS_CASES)} simboluri ambigue rezolvate după țară (sau ignorate).")


# raw -> (doar SUA/Canada, rentals, o singură piață, worldwide)
SCOPE_CASES = {
    "$1.8 million (US/Canada rentals)": (True, True, True, False),
    "$12 million (U.S.)": (True, False, True, False),
    "$4.25 million (US rentals)<br />$9.25 million (worldwide rentals)": (False, True, False, True),
    "{{Plainlist| * $12 million (U.S.) * $33 million (Worldwide)}}": (False, False, False, True),
    "£49,981 (UK)": (False, False, True, False),
    "A$3.2 million (Australia)": (False, False, True, False),
    "US$92.2 million": (False, False, False, False),          # „US$” e moneda, nu teritoriul
    "$839 million": (False, False, False, False),
}


def test_gross_scope():
    for raw, expected in SCOPE_CASES.items():
        assert tuple(gross_scope(raw)) == expected, f"{raw!r}: expected {expected}, got {gross_scope(raw)}"
    print(f"Test 5 passed: {len(SCOPE_CASES)} etichete de încasări (piață, rentals, worldwide) recunoscute.")


def test_infobox_field():
    text = "{{Infobox film\n| name = X\n| budget = $5 million<br />\n (est.)\n| gross = $9 million\n}}"
    assert infobox_field(text, "budget") == "$5 million<br />  (est.)"
    assert infobox_field(text, "gross") == "$9 million"
    assert infobox_field(text, "runtime") is None
    print("Test 3 passed: câmpurile din infobox sunt extrase corect.")


def test_parse_names():
    cases = {
        "[[Warner Bros. Pictures]]<ref name=\"AFI\"/>": ["Warner Bros. Pictures"],
        "20th Century Fox<ref name=\"BFI\"/>{{efn|name=Disney|Distributed by [[Walt Disney Studios Motion Pictures]] in the 2022 re-release}}": ["20th Century Fox"],
        "{{plainlist| * Paramount Pictures (United States and Canada) * 20th Century Fox (International)}}": ["Paramount Pictures", "20th Century Fox"],
        "{{Plainlist| * [[Roadshow Entertainment]] (Australia) * [[Warner Bros. Pictures]] (International)}}": ["Warner Bros. Pictures", "Roadshow Entertainment"],
        "[[Columbia Pictures|Columbia]]<br />[[Sony Pictures Releasing]]": ["Columbia", "Sony Pictures Releasing"],
        "{{ill|Neue Visionen|de|Neue Visionen Filmverleih}}": ["Neue Visionen"],
        "": [],
    }
    for raw, expected in cases.items():
        got = parse_names(raw)
        assert got == expected, f"{raw!r}: expected {expected}, got {got}"
    print(f"Test 4 passed: {len(cases)} formate de distribuitori parsate corect.")


if __name__ == "__main__":
    test_parse_money()
    test_ambiguous()
    test_infobox_field()
    test_parse_names()
    test_gross_scope()
    test_decimal_comma()
    print("\nToate testele au trecut.")
