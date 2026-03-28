"""
LesPAC Quebec Classifieds Scraper - Secondary demand signal source.
Uses DuckDuckGo to find LesPAC service listings across Quebec.
"""

import logging
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

LOCATION_MRC_MAP = {
    "Sherbrooke": "Sherbrooke", "Granby": "La Haute-Yamaska",
    "Montreal": "Montréal", "Quebec": "Québec",
    "Trois-Rivieres": "Trois-Rivières", "Gatineau": "Gatineau",
    "Drummondville": "Drummond",
}

LOCATION_REGION_MAP = {
    "Sherbrooke": "Estrie", "Granby": "Montérégie",
    "Montreal": "Montréal", "Quebec": "Capitale-Nationale",
    "Trois-Rivieres": "Mauricie", "Gatineau": "Outaouais",
    "Drummondville": "Centre-du-Québec",
}


class LesPACScraper(BaseScraper):
    """Scrapes LesPAC for service demand signals via DuckDuckGo search."""

    def __init__(self, knowledge_base=None, llm_extractor=None):
        super().__init__("lespac", knowledge_base)
        self.llm = llm_extractor

    def scrape_location(self, location: str, max_pages: int = 2) -> int:
        """Search DDG for LesPAC service listings in a location."""
        mrc = LOCATION_MRC_MAP.get(location, location)
        region = LOCATION_REGION_MAP.get(location, "")
        start_time = time.time()
        total_found = 0

        try:
            # Search for LesPAC listings
            queries = [
                f"site:lespac.com services {location}",
                f"lespac.com {location} service recherché",
            ]

            for query in queries:
                results = self.web_search(query, max_results=15)

                for r in results:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    url = r.get("url", "")

                    if "lespac.com" not in url:
                        continue

                    full_text = f"{title} {snippet}"
                    from llm.minimal_extraction import classify_by_keywords
                    category = classify_by_keywords(full_text) or "unknown"

                    if category == "unknown":
                        continue

                    added = self.kb.add_demand_signal(
                        source="lespac",
                        mrc=mrc,
                        category=category,
                        title=title[:500],
                        description=snippet[:500],
                        url=url,
                        raw_text=full_text[:1000],
                        region=region,
                    )
                    if added:
                        total_found += 1

            duration = time.time() - start_time
            self.log_result(mrc, "success", total_found, duration=duration)

        except Exception as e:
            duration = time.time() - start_time
            self.log_result(mrc, "error", total_found, str(e), duration)
            logger.error(f"[lespac] Error scraping {location}: {e}")

        return total_found

    def scrape_all(self, locations: list = None) -> int:
        if not locations:
            locations = list(LOCATION_MRC_MAP.keys())
        total = 0
        for loc in locations:
            count = self.scrape_location(loc)
            total += count
        logger.info(f"[lespac] Total: {total} new demand signals")
        return total
