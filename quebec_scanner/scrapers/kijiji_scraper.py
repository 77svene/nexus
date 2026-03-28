"""
Kijiji Quebec Scraper - Primary demand signal source.
Scrapes service requests and "recherché" postings across Quebec.
"""

import logging
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KIJIJI_BASE_URL, KIJIJI_LOCATIONS, BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class KijijiScraper(BaseScraper):
    """Scrapes Kijiji for service demand signals in Quebec."""

    def __init__(self, knowledge_base=None, llm_extractor=None):
        super().__init__("kijiji", knowledge_base)
        self.llm = llm_extractor
        # Build French keyword search terms
        self.search_terms = self._build_search_terms()

    def _build_search_terms(self):
        """Build search terms from all business categories."""
        terms = set()
        for cat, data in BUSINESS_CATEGORIES.items():
            for kw in data["keywords_fr"]:
                terms.add(kw)
        # Add generic "recherché" terms
        terms.update(["recherche", "cherche", "besoin", "service recherché"])
        return list(terms)

    def _mrc_for_location(self, location_name):
        """Map Kijiji location name to MRC."""
        mapping = {
            "Sherbrooke": "Sherbrooke",
            "Granby": "La Haute-Yamaska",
            "Drummondville": "Drummond",
            "Montreal": "Montréal",
            "Quebec": "Québec",
            "Trois-Rivieres": "Trois-Rivières",
            "Gatineau": "Gatineau",
            "Saguenay": "Le Fjord-du-Saguenay",
            "Rimouski": "Rimouski-Neigette",
            "Laval": "Laval",
        }
        return mapping.get(location_name, location_name)

    def _region_for_location(self, location_name):
        mapping = {
            "Sherbrooke": "Estrie",
            "Granby": "Montérégie",
            "Drummondville": "Centre-du-Québec",
            "Montreal": "Montréal",
            "Quebec": "Capitale-Nationale",
            "Trois-Rivieres": "Mauricie",
            "Gatineau": "Outaouais",
            "Saguenay": "Saguenay–Lac-Saint-Jean",
            "Rimouski": "Bas-Saint-Laurent",
            "Laval": "Laval",
        }
        return mapping.get(location_name, "")

    def scrape_location(self, location_name: str, max_pages: int = 3) -> int:
        """Scrape a single Kijiji location for service requests. Returns count of new records."""
        path = KIJIJI_LOCATIONS.get(location_name)
        if not path:
            logger.warning(f"Unknown Kijiji location: {location_name}")
            return 0

        mrc = self._mrc_for_location(location_name)
        region = self._region_for_location(location_name)
        start_time = time.time()
        total_found = 0

        try:
            for page in range(1, max_pages + 1):
                page_url = f"{KIJIJI_BASE_URL}{path}"
                if page > 1:
                    page_url += f"/page-{page}"

                soup = self.fetch(page_url)
                if not soup:
                    break

                listings = self._extract_listings(soup)
                if not listings:
                    break

                for listing in listings:
                    # Classify the listing
                    full_text = f"{listing['title']} {listing.get('description', '')}"

                    if self.llm:
                        info = self.llm.extract_service_type(full_text)
                        category = info["category"]
                        urgency = info["urgency"]
                    else:
                        from llm.minimal_extraction import classify_by_keywords
                        category = classify_by_keywords(full_text) or "unknown"
                        urgency = "normal"

                    if category == "unknown":
                        continue

                    added = self.kb.add_demand_signal(
                        source="kijiji",
                        mrc=mrc,
                        category=category,
                        title=listing["title"],
                        description=listing.get("description", ""),
                        url=listing.get("url", ""),
                        raw_text=full_text[:1000],
                        urgency=urgency,
                        posted_date=listing.get("date", ""),
                        region=region,
                    )
                    if added:
                        total_found += 1

                logger.info(f"[kijiji] {location_name} page {page}: {len(listings)} listings parsed")

            duration = time.time() - start_time
            self.log_result(mrc, "success", total_found, duration=duration)

        except Exception as e:
            duration = time.time() - start_time
            self.log_result(mrc, "error", total_found, str(e), duration)
            logger.error(f"[kijiji] Error scraping {location_name}: {e}")

        return total_found

    def _extract_listings(self, soup) -> list:
        """Extract listing data from Kijiji search results page."""
        listings = []

        # Kijiji uses various container classes - try multiple selectors
        selectors = [
            "div[data-listing-id]",
            "li[data-listing-id]",
            "div.search-item",
            "div.regular-ad",
            "section[data-testid='listing-card']",
            "li[data-testid='listing-card']",
        ]

        items = []
        for sel in selectors:
            items = soup.select(sel)
            if items:
                break

        if not items:
            # Fallback: look for any links that look like listings
            items = soup.select("a[href*='/v-']")
            for link in items[:50]:  # Limit
                title = link.get_text(strip=True)
                url = link.get("href", "")
                if url and not url.startswith("http"):
                    url = urljoin(KIJIJI_BASE_URL, url)
                if title and len(title) > 10:
                    listings.append({
                        "title": title,
                        "url": url,
                        "description": "",
                        "date": "",
                    })
            return listings

        for item in items:
            try:
                # Extract title
                title_el = (
                    item.select_one("a.title") or
                    item.select_one("h3") or
                    item.select_one("[data-testid='listing-title']") or
                    item.select_one("a[class*='title']") or
                    item.select_one(".info-container a")
                )
                title = title_el.get_text(strip=True) if title_el else ""

                # Extract URL
                link_el = item.select_one("a[href]") or title_el
                url = ""
                if link_el and link_el.get("href"):
                    url = link_el["href"]
                    if not url.startswith("http"):
                        url = urljoin(KIJIJI_BASE_URL, url)

                # Extract description snippet
                desc_el = (
                    item.select_one(".description") or
                    item.select_one("[data-testid='listing-description']") or
                    item.select_one("p")
                )
                description = desc_el.get_text(strip=True) if desc_el else ""

                # Extract date
                date_el = (
                    item.select_one(".date-posted") or
                    item.select_one("time") or
                    item.select_one("[data-testid='listing-date']") or
                    item.select_one(".location span")
                )
                date_text = date_el.get_text(strip=True) if date_el else ""

                if title:
                    listings.append({
                        "title": title,
                        "url": url,
                        "description": description,
                        "date": date_text,
                    })
            except Exception as e:
                logger.debug(f"Error parsing listing: {e}")
                continue

        return listings

    def scrape_with_search(self, location_name: str, keywords: list = None, max_pages: int = 2) -> int:
        """Search Kijiji with specific keywords to find service requests."""
        if not keywords:
            # Use most common demand keywords
            keywords = [
                "recherche service",
                "besoin entrepreneur",
                "cherche plombier",
                "cherche électricien",
                "besoin rénovation",
                "recherche thermopompe",
                "cherche couvreur",
                "besoin déneigement",
                "cherche paysagiste",
                "recherche ménage",
                "besoin mécanicien",
                "cherche informatique",
            ]

        mrc = self._mrc_for_location(location_name)
        region = self._region_for_location(location_name)
        total = 0

        for kw in keywords:
            search_url = f"{KIJIJI_BASE_URL}/b-services/{location_name.lower()}/k0c72"
            params = {"q": kw}

            soup = self.fetch(search_url, params=params)
            if not soup:
                continue

            listings = self._extract_listings(soup)
            for listing in listings:
                full_text = f"{listing['title']} {listing.get('description', '')}"
                from llm.minimal_extraction import classify_by_keywords
                category = classify_by_keywords(full_text) or "unknown"

                if category != "unknown":
                    added = self.kb.add_demand_signal(
                        source="kijiji_search",
                        mrc=mrc,
                        category=category,
                        title=listing["title"],
                        description=listing.get("description", ""),
                        url=listing.get("url", ""),
                        raw_text=full_text[:1000],
                        region=region,
                    )
                    if added:
                        total += 1

        logger.info(f"[kijiji] Search scrape {location_name}: {total} new demand signals")
        return total

    def scrape_all(self, locations: list = None, max_pages: int = 3) -> int:
        """Scrape all configured Kijiji locations."""
        if not locations:
            locations = list(KIJIJI_LOCATIONS.keys())

        total = 0
        for loc in locations:
            count = self.scrape_location(loc, max_pages)
            total += count
            # Also do keyword searches
            count2 = self.scrape_with_search(loc)
            total += count2

        logger.info(f"[kijiji] Total scrape complete: {total} new demand signals")
        return total
