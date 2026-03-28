"""
Facebook Marketplace Scraper - Monitors Facebook groups and marketplace for Quebec service demand.
Uses Google search to find public Facebook posts (no FB API needed).
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

# Quebec Facebook groups commonly used for services (public ones only)
FB_GROUP_SEARCHES = {
    "Sherbrooke": [
        "recherche service Sherbrooke",
        "besoin service Sherbrooke Estrie",
        "recommendation Sherbrooke entrepreneur",
    ],
    "Granby": [
        "recherche service Granby",
        "besoin service Granby Haute-Yamaska",
    ],
    "Brome-Missisquoi": [
        "recherche service Brome-Missisquoi",
        "besoin service Cowansville Sutton Bedford",
    ],
    "Montreal": [
        "recherche service Montréal",
    ],
    "Quebec": [
        "recherche service Québec",
    ],
}

LOCATION_MRC_MAP = {
    "Sherbrooke": "Sherbrooke",
    "Granby": "La Haute-Yamaska",
    "Brome-Missisquoi": "Brome-Missisquoi",
    "Montreal": "Montréal",
    "Quebec": "Québec",
}

LOCATION_REGION_MAP = {
    "Sherbrooke": "Estrie",
    "Granby": "Montérégie",
    "Brome-Missisquoi": "Estrie",
    "Montreal": "Montréal",
    "Quebec": "Capitale-Nationale",
}


class FacebookScraper(BaseScraper):
    """
    Scrapes Facebook demand signals via Google search.
    We search Google for public Facebook posts about service requests in Quebec regions.
    """

    def __init__(self, knowledge_base=None, llm_extractor=None):
        super().__init__("facebook", knowledge_base)
        self.llm = llm_extractor

    def scrape_location(self, location: str) -> int:
        """Search for Facebook service requests in a location."""
        searches = FB_GROUP_SEARCHES.get(location, [])
        mrc = LOCATION_MRC_MAP.get(location, location)
        region = LOCATION_REGION_MAP.get(location, "")
        start_time = time.time()
        total_found = 0

        for search_term in searches:
            query = f"site:facebook.com {search_term}"

            try:
                search_results = self.web_search(query, max_results=20)
                if not search_results:
                    continue

                for result in search_results:
                    title = result.get("title", "")
                    url = result.get("url", "")
                    snippet = result.get("snippet", "")

                    full_text = f"{title} {snippet}"

                    from llm.minimal_extraction import classify_by_keywords
                    category = classify_by_keywords(full_text)
                    if not category:
                        continue

                    added = self.kb.add_demand_signal(
                        source="facebook",
                        mrc=mrc,
                        category=category,
                        title=title[:500],
                        description=snippet[:1000],
                        url=url,
                        raw_text=full_text[:1000],
                        region=region,
                    )
                    if added:
                        total_found += 1

            except Exception as e:
                logger.error(f"[facebook] Error searching {search_term}: {e}")

        duration = time.time() - start_time
        self.log_result(mrc, "success", total_found, duration=duration)
        return total_found

    def scrape_all(self, locations: list = None) -> int:
        """Scrape all locations."""
        if not locations:
            locations = list(FB_GROUP_SEARCHES.keys())

        total = 0
        for loc in locations:
            count = self.scrape_location(loc)
            total += count
        logger.info(f"[facebook] Total: {total} new demand signals")
        return total
