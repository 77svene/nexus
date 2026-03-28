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
        soup = self.fetch("https://www.google.com/search", params={
            "q": query, "num": 15, "hl": "fr",
        })

        if not soup:
            return 0

        results = soup.select("div.g")
        count = 0

        for result in results:
            title_el = result.select_one("h3")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)

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
        soup = self.fetch("https://www.google.com/search", params={
            "q": query, "num": 15, "hl": "fr", "tbs": "qdr:m3",
        })

        if not soup:
            return new_businesses

        for result in soup.select("div.g"):
            title_el = result.select_one("h3")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            snippet_el = result.select_one("div.VwiC3b")
            snippet = snippet_el.get_text(strip=True) if snippet_el else ""

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
