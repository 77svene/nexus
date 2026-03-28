"""
LesPAC Quebec Classifieds Scraper - Secondary demand signal source.
Scrapes service requests from LesPAC.com.
"""

import logging
import re
import time
from urllib.parse import urljoin

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# LesPAC category URLs for services
LESPAC_BASE = "https://www.lespac.com"
LESPAC_SERVICE_PATHS = {
    "Sherbrooke": "/sherbrooke/services/",
    "Granby": "/granby/services/",
    "Montreal": "/montreal/services/",
    "Quebec": "/quebec/services/",
    "Trois-Rivieres": "/trois-rivieres/services/",
    "Gatineau": "/gatineau/services/",
    "Drummondville": "/drummondville/services/",
}

LOCATION_MRC_MAP = {
    "Sherbrooke": "Sherbrooke",
    "Granby": "La Haute-Yamaska",
    "Montreal": "Montréal",
    "Quebec": "Québec",
    "Trois-Rivieres": "Trois-Rivières",
    "Gatineau": "Gatineau",
    "Drummondville": "Drummond",
}

LOCATION_REGION_MAP = {
    "Sherbrooke": "Estrie",
    "Granby": "Montérégie",
    "Montreal": "Montréal",
    "Quebec": "Capitale-Nationale",
    "Trois-Rivieres": "Mauricie",
    "Gatineau": "Outaouais",
    "Drummondville": "Centre-du-Québec",
}


class LesPACScraper(BaseScraper):
    """Scrapes LesPAC for service demand signals."""

    def __init__(self, knowledge_base=None, llm_extractor=None):
        super().__init__("lespac", knowledge_base)
        self.llm = llm_extractor

    def scrape_location(self, location: str, max_pages: int = 2) -> int:
        """Scrape a LesPAC location for service listings."""
        path = LESPAC_SERVICE_PATHS.get(location)
        if not path:
            return 0

        mrc = LOCATION_MRC_MAP.get(location, location)
        region = LOCATION_REGION_MAP.get(location, "")
        start_time = time.time()
        total_found = 0

        try:
            for page in range(1, max_pages + 1):
                url = f"{LESPAC_BASE}{path}"
                if page > 1:
                    url += f"?page={page}"

                soup = self.fetch(url)
                if not soup:
                    break

                listings = self._extract_listings(soup)
                if not listings:
                    break

                for listing in listings:
                    full_text = f"{listing['title']} {listing.get('description', '')}"

                    from llm.minimal_extraction import classify_by_keywords
                    category = classify_by_keywords(full_text) or "unknown"

                    if category == "unknown":
                        continue

                    added = self.kb.add_demand_signal(
                        source="lespac",
                        mrc=mrc,
                        category=category,
                        title=listing["title"],
                        description=listing.get("description", ""),
                        url=listing.get("url", ""),
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

    def _extract_listings(self, soup) -> list:
        """Extract listings from LesPAC page."""
        listings = []

        # Try multiple selectors (LesPAC changes structure)
        items = (
            soup.select("div.listing-item") or
            soup.select("article.ad-card") or
            soup.select("div[class*='listing']") or
            soup.select("div.result-item")
        )

        if not items:
            # Fallback: look for links with service-like text
            for link in soup.select("a[href*='/services/']"):
                title = link.get_text(strip=True)
                if title and len(title) > 10:
                    url = link.get("href", "")
                    if not url.startswith("http"):
                        url = urljoin(LESPAC_BASE, url)
                    listings.append({"title": title, "url": url, "description": ""})
            return listings[:50]

        for item in items:
            try:
                title_el = item.select_one("h2, h3, a.title, [class*='title']")
                title = title_el.get_text(strip=True) if title_el else ""

                link_el = item.select_one("a[href]")
                url = ""
                if link_el:
                    url = link_el.get("href", "")
                    if not url.startswith("http"):
                        url = urljoin(LESPAC_BASE, url)

                desc_el = item.select_one("p, .description, [class*='desc']")
                description = desc_el.get_text(strip=True) if desc_el else ""

                if title:
                    listings.append({"title": title, "url": url, "description": description})
            except Exception:
                continue

        return listings

    def scrape_all(self, locations: list = None) -> int:
        """Scrape all configured locations."""
        if not locations:
            locations = list(LESPAC_SERVICE_PATHS.keys())

        total = 0
        for loc in locations:
            count = self.scrape_location(loc)
            total += count
        logger.info(f"[lespac] Total: {total} new demand signals")
        return total
