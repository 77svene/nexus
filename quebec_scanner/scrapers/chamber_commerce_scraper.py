"""
Chamber of Commerce Scraper - Tracks active businesses via chamber directories.
New members = business launches. Lost members = closures.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ChamberCommerceScraper(BaseScraper):
    """Scrapes chamber of commerce directories for supply data."""

    def __init__(self, knowledge_base=None):
        super().__init__("chamber_commerce", knowledge_base)

    def search_chamber_members(self, mrc: str, category: str) -> int:
        """Count chamber of commerce members in a category/MRC."""
        cat_data = BUSINESS_CATEGORIES.get(category, {})
        keywords = cat_data.get("keywords_fr", [category])
        search_term = keywords[0]

        query = f"chambre commerce {mrc} {search_term} répertoire"
        search_results = self.web_search(query, max_results=15)

        if not search_results:
            return 0

        count = 0

        for result in search_results:
            title = result.get("title", "")

            # Filter for actual business listings (not generic pages)
            if any(kw.lower() in title.lower() for kw in keywords[:3]):
                count += 1
                if self.kb:
                    self.kb.add_supply_provider(
                        source="chamber_commerce",
                        mrc=mrc,
                        category=category,
                        business_name=title[:200],
                    )

        return count

    def search_new_businesses(self, mrc: str) -> list:
        """Search for recently announced new businesses in an MRC."""
        new_businesses = []

        query = f"nouvelle entreprise ouverture {mrc} Québec 2026"
        search_results = self.web_search_recent(query, max_results=15)

        if not search_results:
            return new_businesses

        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")

            full_text = f"{title} {snippet}".lower()

            # Classify
            for cat, data in BUSINESS_CATEGORIES.items():
                for kw in data["keywords_fr"]:
                    if kw.lower() in full_text:
                        new_businesses.append({
                            "name": title,
                            "category": cat,
                            "mrc": mrc,
                            "signal": "new_entrant",
                        })
                        break

        return new_businesses

    def scan_mrc(self, mrc: str, categories: list = None) -> dict:
        """Scan chamber data for an MRC."""
        if not categories:
            categories = list(BUSINESS_CATEGORIES.keys())[:10]

        results = {"member_counts": {}, "new_businesses": []}
        for cat in categories:
            results["member_counts"][cat] = self.search_chamber_members(mrc, cat)

        results["new_businesses"] = self.search_new_businesses(mrc)
        self.log_result(mrc, "success", sum(results["member_counts"].values()))
        return results
