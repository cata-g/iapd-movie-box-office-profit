"""
Manual test for the infobox money parser in extraction/wikipedia.py.

Run: python extraction/test_wikipedia.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.wikipedia import infobox_field, parse_money

# raw infobox value -> (value, is_range, has_other_currency)
CASES = {
    "$15 million": (15e6, False, False),
    "$10–15 million": (12.5e6, True, False),
    "US$2.5 million": (2.5e6, False, False),
    "$1,200,000": (1.2e6, False, False),
    "$200,000–$750,000<ref name=\"THR\">{{cite web |url=x}}</ref>": (475_000, True, False),
    "$15,000-$450,000": (232_500, True, False),
    "$3.5&nbsp;million<ref name=\"Mojo\" />": (3.5e6, False, False),
    "$356–400{{nbsp}}million": (378e6, True, False),
    "$2.94{{nbsp}}billion": (2.94e9, False, False),
    "{{US$|160 million}}": (160e6, False, False),
    "Production: $7,225<br />Post-production: $200,000": (7_225, False, False),
    "£3 million": (None, False, True),
    "€10 million": (None, False, True),
    "{{INR|55 crore}}": (None, False, True),
    "₹40 crore": (None, False, True),
    "A$8 million": (None, False, True),
    "£3 million (US$4.5 million)": (None, False, True),
    "MYR 2.80 million": (None, False, True),
    "CAD 10.5 million": (None, False, True),
    "{{INR}}30 million": (None, False, True),
    "USD 4 million": (4e6, False, False),
    "{{USD}}5 million": (5e6, False, False),
    "{{usd|65,000}}": (65_000, False, False),
    "{{JPY|289 million}}<br>({{USD|800,000|long=no}})": (None, False, True),
    "": (None, False, False),
}


def test_parse_money():
    for raw, expected in CASES.items():
        got = parse_money(raw)
        assert got == expected, f"{raw!r}: expected {expected}, got {got}"
    print(f"Test 1 passed: {len(CASES)} formate de sume parsate corect.")


def test_infobox_field():
    text = "{{Infobox film\n| name = X\n| budget = $5 million<br />\n (est.)\n| gross = $9 million\n}}"
    assert infobox_field(text, "budget") == "$5 million<br />  (est.)"
    assert infobox_field(text, "gross") == "$9 million"
    assert infobox_field(text, "runtime") is None
    print("Test 2 passed: câmpurile din infobox sunt extrase corect.")


if __name__ == "__main__":
    test_parse_money()
    test_infobox_field()
    print("\nToate testele au trecut.")
