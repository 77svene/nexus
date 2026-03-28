"""
Base scraper with common functionality: rate limiting, retries, logging.
Includes DuckDuckGo web search as primary search engine (Google blocked in many envs).
"""

import logging
import random
import re
import time
import requests
from urllib.parse import unquote, urlparse, parse_qs
from bs4 import BeautifulSoup

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    USER_AGENT, REQUEST_TIMEOUT, REQUEST_DELAY_MIN, REQUEST_DELAY_MAX, MAX_RETRIES
)

logger = logging.getLogger(__name__)

DDG_URL = "https://html.duckduckgo.com/html/"


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

    def web_search(self, query: str, max_results: int = 20) -> list:
        """
        Search the web using DuckDuckGo HTML API (works without JS, no API key).
        Returns list of dicts: [{"title": str, "url": str, "snippet": str}, ...]
        """
        results = []
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                resp = self.session.post(DDG_URL, data={"q": query, "b": ""},
                                         timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    break
                elif resp.status_code in (202, 429):
                    # Rate limited - back off and retry
                    wait = (attempt + 1) * 5
                    logger.debug(f"[{self.name}] DDG rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"[{self.name}] DDG search HTTP {resp.status_code}")
                    return results
            except requests.exceptions.Timeout:
                logger.warning(f"[{self.name}] DDG timeout (attempt {attempt + 1})")
                time.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error(f"[{self.name}] DDG error: {e}")
                return results
        else:
            logger.warning(f"[{self.name}] DDG search failed after {MAX_RETRIES} retries")
            return results

        try:
            soup = BeautifulSoup(resp.text, "lxml")

            for result_div in soup.select("div.result"):
                title_el = result_div.select_one("a.result__a")
                snippet_el = result_div.select_one("a.result__snippet")

                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                raw_url = title_el.get("href", "")

                # DDG wraps URLs in a redirect - extract the real URL
                url = self._extract_ddg_url(raw_url)

                snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                if title:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    })

                if len(results) >= max_results:
                    break

        except requests.exceptions.Timeout:
            logger.warning(f"[{self.name}] DDG search timeout")
        except Exception as e:
            logger.error(f"[{self.name}] DDG search error: {e}")

        return results

    def _extract_ddg_url(self, raw_url: str) -> str:
        """Extract the real URL from a DuckDuckGo redirect link."""
        if "uddg=" in raw_url:
            try:
                parsed = urlparse(raw_url)
                qs = parse_qs(parsed.query)
                if "uddg" in qs:
                    return unquote(qs["uddg"][0])
            except Exception:
                pass
        if raw_url.startswith("//"):
            return "https:" + raw_url
        return raw_url

    def web_search_site(self, site: str, query: str, max_results: int = 15) -> list:
        """Search within a specific site using DDG."""
        return self.web_search(f"site:{site} {query}", max_results)

    def web_search_recent(self, query: str, max_results: int = 15) -> list:
        """Search with date bias towards recent results."""
        # DDG doesn't have a direct date filter in HTML API, but we add year
        from datetime import datetime
        year = datetime.now().year
        return self.web_search(f"{query} {year}", max_results)

    def log_result(self, mrc, status, records_found=0, error_message="", duration=0.0):
        """Log scrape result to database."""
        if self.kb:
            self.kb.log_scrape(self.name, mrc, status, records_found, error_message, duration)
        level = logging.INFO if status == "success" else logging.WARNING
        logger.log(level, f"[{self.name}] {mrc}: {status} ({records_found} records, {duration:.1f}s)")
