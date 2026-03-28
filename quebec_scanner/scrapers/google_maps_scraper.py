"""
Google Maps Supply Scraper - Counts business providers per category per MRC.
Uses web search (no API key needed) to estimate business density.
"""

import logging
import re
import time
from urllib.parse import quote_plus

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES, PRIORITY_MRCS
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class GoogleMapsScraper(BaseScraper):
    """Estimates business supply by scraping Google search results for Maps listings."""

    def __init__(self, knowledge_base=None):
        super().__init__("google_maps", knowledge_base)
        # Use Google search to find Maps results (avoids Maps API)
        self.search_url = "https://www.google.com/search"

    def count_providers(self, mrc: str, category: str) -> int:
        """
        Count approximate number of service providers for a category in an MRC.
        Uses Google search results to estimate count.
        """
        cat_data = BUSINESS_CATEGORIES.get(category, {})
        keywords_fr = cat_data.get("keywords_fr", [category])

        # Use the first 2 keywords for search
        search_terms = keywords_fr[:2]
        total_providers = 0
        businesses_found = set()

        for term in search_terms:
            query = f"{term} {mrc} Québec"
            params = {
                "q": query,
                "num": 20,
                "hl": "fr",
                "gl": "ca",
            }

            start_time = time.time()
            soup = self.fetch(self.search_url, params=params)
            duration = time.time() - start_time

            if not soup:
                continue

            # Extract business names from search results
            providers = self._extract_providers_from_search(soup, mrc, category)
            for p in providers:
                biz_key = p["business_name"].lower().strip()
                if biz_key not in businesses_found:
                    businesses_found.add(biz_key)
                    if self.kb:
                        self.kb.add_supply_provider(
                            source="google_search",
                            mrc=mrc,
                            category=category,
                            business_name=p["business_name"],
                            address=p.get("address", ""),
                            rating=p.get("rating", 0.0),
                            review_count=p.get("review_count", 0),
                        )

            total_providers = len(businesses_found)

        if self.kb:
            self.log_result(mrc, "success", total_providers, duration=0)

        logger.info(f"[google] {mrc}/{category}: {total_providers} providers found")
        return total_providers

    def _extract_providers_from_search(self, soup, mrc, category):
        """Extract business info from Google search results."""
        providers = []

        # Look for local pack / maps results
        # Google wraps these in various divs
        local_results = soup.select("div.VkpGBb") or soup.select("div[data-attrid='kc:/local:one box']")

        if local_results:
            for result in local_results:
                name_el = result.select_one("span.OSrXXb") or result.select_one("div.dbg0pd")
                if name_el:
                    name = name_el.get_text(strip=True)
                    rating = 0.0
                    review_count = 0

                    rating_el = result.select_one("span.yi40Hd") or result.select_one("span[aria-label*='étoile']")
                    if rating_el:
                        try:
                            rating = float(re.search(r'[\d.]+', rating_el.get_text()).group())
                        except (ValueError, AttributeError):
                            pass

                    reviews_el = result.select_one("span.RDApEe") or result.select_one("span[aria-label*='avis']")
                    if reviews_el:
                        try:
                            review_count = int(re.search(r'\d+', reviews_el.get_text().replace(',', '')).group())
                        except (ValueError, AttributeError):
                            pass

                    providers.append({
                        "business_name": name,
                        "rating": rating,
                        "review_count": review_count,
                    })

        # Also check regular search results for business listings
        for result in soup.select("div.g"):
            title_el = result.select_one("h3")
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            # Check if this looks like a local business (not a directory)
            skip_patterns = ["yelp", "pages jaunes", "yellow pages", "411", "canpages",
                             "facebook.com", "linkedin.com", "wikipedia", "indeed"]
            if any(pat in title.lower() for pat in skip_patterns):
                continue

            # Check if title contains category-related keywords
            cat_data = BUSINESS_CATEGORIES.get(category, {})
            all_keywords = cat_data.get("keywords_fr", []) + cat_data.get("keywords_en", [])
            if any(kw.lower() in title.lower() for kw in all_keywords):
                # Extract snippet for more info
                snippet_el = result.select_one("div.VwiC3b") or result.select_one("span.aCOpRe")
                address = ""
                if snippet_el:
                    snippet_text = snippet_el.get_text(strip=True)
                    # Try to find address pattern
                    addr_match = re.search(r'\d+\s+\w+.*(?:rue|av|boul|ch|rte)', snippet_text, re.I)
                    if addr_match:
                        address = addr_match.group()

                providers.append({
                    "business_name": title,
                    "address": address,
                    "rating": 0.0,
                    "review_count": 0,
                })

        return providers

    def scrape_mrc(self, mrc: str, categories: list = None) -> dict:
        """Count all providers for all categories in an MRC."""
        if not categories:
            categories = list(BUSINESS_CATEGORIES.keys())

        results = {}
        for cat in categories:
            count = self.count_providers(mrc, cat)
            results[cat] = count

        return results

    def scrape_all_priority(self) -> dict:
        """Scrape all priority MRCs for provider counts."""
        all_results = {}
        for mrc in PRIORITY_MRCS:
            logger.info(f"[google] Scanning supply in {mrc}...")
            results = self.scrape_mrc(mrc)
            all_results[mrc] = results
        return all_results
