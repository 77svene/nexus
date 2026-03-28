"""
Supply Provider Scraper - Counts business providers per category per MRC.
Uses DuckDuckGo web search to find and count local businesses.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES, PRIORITY_MRCS
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class GoogleMapsScraper(BaseScraper):
    """Estimates business supply by searching for providers via DuckDuckGo."""

    def __init__(self, knowledge_base=None):
        super().__init__("supply_search", knowledge_base)

    def count_providers(self, mrc: str, category: str) -> int:
        """Count approximate number of service providers for a category in an MRC."""
        cat_data = BUSINESS_CATEGORIES.get(category, {})
        keywords_fr = cat_data.get("keywords_fr", [category])

        search_term = keywords_fr[0]
        query = f"{search_term} {mrc} Québec"
        businesses_found = set()

        start_time = time.time()
        results = self.web_search(query, max_results=20)

        # Directories and aggregators to skip (not actual businesses)
        skip_domains = {"yelp.ca", "yelp.com", "pagesjaunes.ca", "yellowpages.ca",
                        "411.ca", "canpages.ca", "facebook.com", "linkedin.com",
                        "wikipedia.org", "indeed.com", "jobillico.com", "kijiji.ca",
                        "google.com", "google.ca", "duckduckgo.com"}

        for r in results:
            url = r.get("url", "")
            title = r.get("title", "")
            snippet = r.get("snippet", "")

            # Skip directory/aggregator sites
            try:
                domain = url.split("//")[1].split("/")[0].lower() if "//" in url else ""
                domain = domain.replace("www.", "")
            except (IndexError, AttributeError):
                domain = ""

            if any(skip in domain for skip in skip_domains):
                continue

            # Skip results that are clearly not local businesses
            if not title or len(title) < 5:
                continue

            # Check if title contains category-related keywords (indicates a real business)
            all_kw = keywords_fr + cat_data.get("keywords_en", [])
            full_text = f"{title} {snippet}".lower()
            if not any(kw.lower() in full_text for kw in all_kw):
                continue

            biz_key = title.lower().strip()[:50]
            if biz_key not in businesses_found:
                businesses_found.add(biz_key)

                # Extract rating from snippet if present
                rating = 0.0
                rating_match = re.search(r'(\d[.,]\d)\s*/?\s*5|(\d[.,]\d)\s*étoiles?|rating:?\s*(\d[.,]\d)', snippet.lower())
                if rating_match:
                    val = rating_match.group(1) or rating_match.group(2) or rating_match.group(3)
                    try:
                        rating = float(val.replace(",", "."))
                    except ValueError:
                        pass

                review_count = 0
                review_match = re.search(r'(\d+)\s*(?:avis|reviews?|évaluations?)', snippet.lower())
                if review_match:
                    try:
                        review_count = int(review_match.group(1))
                    except ValueError:
                        pass

                if self.kb:
                    self.kb.add_supply_provider(
                        source="web_search",
                        mrc=mrc,
                        category=category,
                        business_name=title[:200],
                        address=snippet[:200] if mrc.lower() in snippet.lower() else "",
                        rating=rating,
                        review_count=review_count,
                    )

        total = len(businesses_found)
        duration = time.time() - start_time
        if self.kb:
            self.log_result(mrc, "success", total, duration=duration)

        logger.info(f"[supply] {mrc}/{category}: {total} providers found")
        return total

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
            logger.info(f"[supply] Scanning supply in {mrc}...")
            results = self.scrape_mrc(mrc)
            all_results[mrc] = results
        return all_results
