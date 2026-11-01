"""
Manual smoke test for extraction/utils.py.

Run: python extraction/test_utils.py

Tests:
  1. Second fetch of same URL comes from disk cache (no HTTP request made).
  2. A 404 response returns None without raising an exception.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Make sure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extraction.utils as utils


def _mock_response(status_code: int, body: dict | str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    if isinstance(body, dict):
        resp.json.return_value = body
        resp.text = json.dumps(body)
    else:
        resp.text = body
        resp.json.side_effect = ValueError("not json")
    return resp


def test_cache():
    # Use a temporary raw dir so we don't pollute real cache
    with tempfile.TemporaryDirectory() as tmp:
        original_raw = utils.RAW_DIR
        utils.RAW_DIR = Path(tmp)

        call_count = 0

        def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(200, {"id": 27205, "title": "Inception"})

        with patch("requests.get", side_effect=fake_get):
            result1 = utils.fetch("https://api.tmdb.org/3/movie/27205", source="tmdb")
            result2 = utils.fetch("https://api.tmdb.org/3/movie/27205", source="tmdb")

        utils.RAW_DIR = original_raw

    assert result1 == {"id": 27205, "title": "Inception"}, "First fetch failed"
    assert result2 == {"id": 27205, "title": "Inception"}, "Second fetch returned wrong data"
    assert call_count == 1, f"Expected 1 HTTP call, got {call_count} (cache not working)"
    print("Test 1 passed: al doilea apel a venit din cache (un singur request HTTP).")


def test_404():
    with tempfile.TemporaryDirectory() as tmp:
        original_raw = utils.RAW_DIR
        utils.RAW_DIR = Path(tmp)

        def fake_get(url, **kwargs):
            return _mock_response(404, "Not Found")

        with patch("requests.get", side_effect=fake_get):
            result = utils.fetch("https://api.tmdb.org/3/movie/9999999999", source="tmdb")

        utils.RAW_DIR = original_raw

    assert result is None, f"Expected None for 404, got {result}"
    print("Test 2 passed: 404 a returnat None fara exceptie.")


if __name__ == "__main__":
    test_cache()
    test_404()
    print("\nToate testele au trecut.")
