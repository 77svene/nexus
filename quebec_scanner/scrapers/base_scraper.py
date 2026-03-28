"""
Base scraper with common functionality: rate limiting, retries, logging.
"""

import logging
import random
import time
import requests
from bs4 import BeautifulSoup

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    USER_AGENT, REQUEST_TIMEOUT, REQUEST_DELAY_MIN, REQUEST_DELAY_MAX, MAX_RETRIES
)

logger = logging.getLogger(__name__)


class BaseScraper:
    """Base class for all scrapers with respectful rate limiting."""

    def __init__(self, name: str, knowledge_base=None):
        self.name = name
        self.kb = knowledge_base
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "fr-CA,fr;q=0.9,en-CA;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        self._last_request_time = 0

    def _rate_limit(self):
        """Ensure minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def fetch(self, url: str, params=None) -> BeautifulSoup | None:
        """Fetch a URL with retries and rate limiting. Returns BeautifulSoup or None."""
        self._rate_limit()
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    return BeautifulSoup(resp.text, "lxml")
                elif resp.status_code == 403:
                    logger.warning(f"[{self.name}] 403 Forbidden: {url}")
                    return None
                elif resp.status_code == 429:
                    wait = (attempt + 1) * 10
                    logger.warning(f"[{self.name}] Rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"[{self.name}] HTTP {resp.status_code}: {url}")
            except requests.exceptions.Timeout:
                logger.warning(f"[{self.name}] Timeout (attempt {attempt + 1}): {url}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[{self.name}] Connection error (attempt {attempt + 1}): {url}")
            except Exception as e:
                logger.error(f"[{self.name}] Error fetching {url}: {e}")
                return None

            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

        return None

    def fetch_json(self, url: str, params=None) -> dict | None:
        """Fetch JSON endpoint with retries."""
        self._rate_limit()
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    time.sleep((attempt + 1) * 10)
                else:
                    logger.warning(f"[{self.name}] HTTP {resp.status_code}: {url}")
            except Exception as e:
                logger.warning(f"[{self.name}] JSON error: {e}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

        return None

    def log_result(self, mrc, status, records_found=0, error_message="", duration=0.0):
        """Log scrape result to database."""
        if self.kb:
            self.kb.log_scrape(self.name, mrc, status, records_found, error_message, duration)
        level = logging.INFO if status == "success" else logging.WARNING
        logger.log(level, f"[{self.name}] {mrc}: {status} ({records_found} records, {duration:.1f}s)")
