"""
HTTP utility with disk cache, per-source rate limiting, retry with exponential backoff,
and per-source file logging.
"""

import hashlib
import json
import logging
import os
import time
import threading
from pathlib import Path
from typing import Optional, Union

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Rate limiting: minimum seconds between consecutive requests per source
# ---------------------------------------------------------------------------
RATE_LIMITS: dict[str, float] = {
    "tmdb": 1 / 20,       # 20 req/s
    "the_numbers": 1.0,   # 1 req/s
    "bom": 1.0,            # 1 req/s (Box Office Mojo)
    "wikipedia": 1 / 5,   # 5 req/s
    "wikidata": 1 / 5,
    "dbpedia": 1.0,        # public SPARQL endpoint, be gentle
}
DEFAULT_RATE = 1.0

_rate_locks: dict[str, threading.Lock] = {}
_last_request: dict[str, float] = {}
_rate_meta_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Loggers cache
# ---------------------------------------------------------------------------
_loggers: dict[str, logging.Logger] = {}

USER_AGENT = "iapd-student-project (contact: barabas.catalin@gmail.com)"


def _get_logger(source: str) -> logging.Logger:
    if source in _loggers:
        return _loggers[source]
    logger = logging.getLogger(f"iapd.{source}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOGS_DIR / f"{source}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    _loggers[source] = logger
    return logger


def _get_lock(source: str) -> threading.Lock:
    with _rate_meta_lock:
        if source not in _rate_locks:
            _rate_locks[source] = threading.Lock()
        return _rate_locks[source]


def _rate_wait(source: str) -> None:
    interval = RATE_LIMITS.get(source, DEFAULT_RATE)
    lock = _get_lock(source)
    with lock:
        last = _last_request.get(source, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_request[source] = time.monotonic()


def _cache_path(source: str, url: str, params: Optional[dict], as_json: bool) -> Path:
    key = url + json.dumps(params or {}, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()
    ext = "json" if as_json else "html"
    folder = RAW_DIR / source
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.{ext}"


def fetch(
    url: str,
    source: str,
    params: Optional[dict] = None,
    as_json: bool = True,
) -> Optional[Union[dict, str]]:
    """
    Fetch a URL with caching, rate limiting, and retry.

    Returns parsed JSON (dict/list) or raw text depending on as_json.
    Returns None on 404 or after exhausting retries.
    """
    logger = _get_logger(source)
    cache_file = _cache_path(source, url, params, as_json)

    if cache_file.exists():
        logger.info("CACHE HIT %s", url)
        content = cache_file.read_text(encoding="utf-8")
        return json.loads(content) if as_json else content

    headers = {"User-Agent": USER_AGENT}
    max_retries = 3
    backoff = 2.0

    for attempt in range(1, max_retries + 1):
        _rate_wait(source)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d/%d): %s — %s", attempt, max_retries, url, exc)
            if attempt == max_retries:
                return None
            time.sleep(backoff ** attempt)
            continue

        if resp.status_code == 404:
            logger.info("404 %s", url)
            return None

        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning("HTTP %d (attempt %d/%d): %s", resp.status_code, attempt, max_retries, url)
            if attempt == max_retries:
                return None
            time.sleep(backoff ** attempt)
            continue

        if not resp.ok:
            logger.warning("HTTP %d %s", resp.status_code, url)
            return None

        # Success — write to cache
        cache_file.write_text(resp.text, encoding="utf-8")
        logger.info("FETCHED %s → %s", url, cache_file.name)

        return resp.json() if as_json else resp.text

    return None
