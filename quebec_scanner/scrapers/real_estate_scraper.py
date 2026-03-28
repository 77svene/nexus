"""
Real Estate Scraper - Monitors housing market as leading indicator.
More sales = more renovation/moving/service demand.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class RealEstateScraper(BaseScraper):
    """Scrapes real estate data as a leading indicator for service demand."""

    def __init__(self, knowledge_base=None):
        super().__init__("real_estate", knowledge_base)

    def get_market_activity(self, mrc: str) -> dict:
        """Get real estate market activity for an MRC."""
        data = {
            "listings_count": 0,
            "market_trend": "unknown",
            "new_construction": 0,
            "avg_price_signal": "unknown",
        }
        start_time = time.time()

        # Search Centris/DuProprio listings
        queries = [
            f"site:centris.ca maison à vendre {mrc}",
            f"site:duproprio.com à vendre {mrc}",
            f"immobilier {mrc} Québec marché 2026",
        ]

        total_listings = 0
        for query in queries:
            search_results = self.web_search(query, max_results=20)
            if search_results:
                total_listings += len(search_results)

                # Extract price/trend signals from titles and snippets
                text = " ".join(
                    f"{r.get('title', '')} {r.get('snippet', '')}" for r in search_results
                ).lower()

                # Market trend detection
                hot_keywords = ["marché vendeur", "surenchère", "hausse prix", "forte demande"]
                cold_keywords = ["marché acheteur", "baisse prix", "ralentissement"]

                hot_count = sum(text.count(k) for k in hot_keywords)
                cold_count = sum(text.count(k) for k in cold_keywords)

                if hot_count > cold_count:
                    data["market_trend"] = "hot"
                elif cold_count > hot_count:
                    data["market_trend"] = "cold"
                else:
                    data["market_trend"] = "balanced"

                # New construction mentions
                new_const = re.findall(r'(?:neuf|nouvelle construction|nouveau)', text)
                data["new_construction"] += len(new_const)

        data["listings_count"] = total_listings
        duration = time.time() - start_time
        self.log_result(mrc, "success", total_listings, duration=duration)

        # Hot real estate = demand for renovation, moving, cleaning services
        if data["market_trend"] == "hot" and self.kb:
            for cat in ["construction", "demenagement", "menage", "peinture", "paysagement"]:
                self.kb.add_demand_signal(
                    source="real_estate_indicator",
                    mrc=mrc,
                    category=cat,
                    title=f"Marché immobilier actif à {mrc}",
                    description=f"Indicateur avancé: {total_listings} annonces, marché {data['market_trend']}",
                )

        return data

    def scrape_all(self, mrcs: list) -> dict:
        """Get real estate data for multiple MRCs."""
        results = {}
        for mrc in mrcs:
            results[mrc] = self.get_market_activity(mrc)
        return results
