"""
REQ (Registraire des entreprises du Québec) Scraper.
Tracks business registrations by sector and region.
Also attempts to count active businesses per category/MRC.
"""

import logging
import re
import time
from urllib.parse import quote_plus

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class REQScraper(BaseScraper):
    """Scrapes Quebec business registry for supply-side data."""

    def __init__(self, knowledge_base=None):
        super().__init__("req", knowledge_base)
        self.search_url = "https://www.registreentreprises.gouv.qc.ca"

    def search_businesses(self, category: str, mrc: str, max_results: int = 50) -> int:
        """
        Search for registered businesses in a category within an MRC.
        Since REQ's direct search is complex, we use web search to search the registry.
        Returns count of found businesses.
        """
        cat_data = BUSINESS_CATEGORIES.get(category, {})
        keywords = cat_data.get("keywords_fr", [category])
        search_term = keywords[0] if keywords else category

        # Search for REQ entries via web search
        query = f"site:registreentreprises.gouv.qc.ca {search_term} {mrc}"

        start_time = time.time()
        search_results = self.web_search(query, max_results=max_results)
        duration = time.time() - start_time

        if not search_results:
            self.log_result(mrc, "error", 0, "Failed to fetch", duration)
            return 0

        businesses = []

        for result in search_results:
            title = result.get("title", "")
            url = result.get("url", "")

            # Extract business name (REQ results typically show business name)
            # Filter out non-business results
            if "registreentreprises" in url:
                biz_name = title.split(" - ")[0].strip() if " - " in title else title
                businesses.append(biz_name)

                if self.kb:
                    self.kb.add_supply_provider(
                        source="req",
                        mrc=mrc,
                        category=category,
                        business_name=biz_name,
                    )

        count = len(businesses)
        self.log_result(mrc, "success", count, duration=duration)
        logger.info(f"[req] {mrc}/{category}: {count} registered businesses found")
        return count

    def search_new_registrations(self, category: str, mrc: str) -> int:
        """Search for recently registered businesses (new entrants)."""
        cat_data = BUSINESS_CATEGORIES.get(category, {})
        keywords = cat_data.get("keywords_fr", [category])
        search_term = keywords[0] if keywords else category

        # Search for recent registrations via web search
        query = f"site:registreentreprises.gouv.qc.ca {search_term} {mrc}"
        search_results = self.web_search_recent(query, max_results=20)
        if not search_results:
            return 0

        return len(search_results)

    def scan_mrc(self, mrc: str, categories: list = None) -> dict:
        """Scan an MRC for all business categories."""
        if not categories:
            categories = list(BUSINESS_CATEGORIES.keys())

        results = {}
        for cat in categories:
            count = self.search_businesses(cat, mrc)
            results[cat] = count

        return results
