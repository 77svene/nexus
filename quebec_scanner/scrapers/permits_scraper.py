"""
Building Permits Scraper - Monitors construction permit activity via Google searches.
Tracks new residential/commercial permits as leading indicators.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class PermitsScraper(BaseScraper):
    """Scrapes building permit data as a leading indicator for construction activity."""

    def __init__(self, knowledge_base=None):
        super().__init__("permits", knowledge_base)

    def search_permits(self, mrc: str) -> dict:
        """Search for building permit data for an MRC."""
        results = {"residential": 0, "commercial": 0, "total_mentions": 0}
        start_time = time.time()

        queries = [
            f"permis construction {mrc} Québec 2026",
            f"permis bâtir {mrc} résidentiel",
            f"building permit {mrc} Quebec",
        ]

        for query in queries:
            try:
                search_results = self.web_search(query, max_results=10)
                if not search_results:
                    continue

                page_text = " ".join(
                    f"{r.get('title', '')} {r.get('snippet', '')}" for r in search_results
                ).lower()

                # Count mentions of permits
                permit_keywords = ["permis", "permit", "construction", "bâtir"]
                for kw in permit_keywords:
                    results["total_mentions"] += page_text.count(kw)

                # Try to extract numbers
                numbers = re.findall(r'(\d+)\s*(?:permis|permits)', page_text)
                for n in numbers:
                    val = int(n)
                    if val < 10000:  # Sanity check
                        results["residential"] = max(results["residential"], val)

            except Exception as e:
                logger.error(f"[permits] Error searching {mrc}: {e}")

        duration = time.time() - start_time
        self.log_result(mrc, "success", results["total_mentions"], duration=duration)
        return results

    def scrape_all(self, mrcs: list) -> dict:
        """Scrape permit data for a list of MRCs."""
        all_results = {}
        for mrc in mrcs:
            all_results[mrc] = self.search_permits(mrc)
        return all_results
