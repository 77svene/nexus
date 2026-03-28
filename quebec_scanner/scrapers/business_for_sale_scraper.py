"""
Business For Sale Scraper - Detects businesses being sold/closing.
A business for sale = potential supply reduction = opportunity for new entrant.
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


class BusinessForSaleScraper(BaseScraper):
    """Detects businesses being sold as supply-reduction signals."""

    def __init__(self, knowledge_base=None):
        super().__init__("business_for_sale", knowledge_base)

    def search_businesses_for_sale(self, mrc: str) -> list:
        """Search for businesses being sold in an MRC."""
        results = []
        start_time = time.time()

        queries = [
            f"entreprise à vendre {mrc} Québec",
            f"commerce à vendre {mrc}",
            f"business for sale {mrc} Quebec",
            f"site:bizbuysell.com {mrc}",
            f"site:businessforsale.com {mrc} Quebec",
        ]

        for query in queries[:3]:  # Limit requests
            soup = self.fetch("https://www.google.com/search", params={
                "q": query, "num": 15, "hl": "fr",
            })
            if not soup:
                continue

            for result in soup.select("div.g"):
                title_el = result.select_one("h3")
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                snippet_el = result.select_one("div.VwiC3b, span.aCOpRe")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                link_el = result.select_one("a[href]")
                url = link_el.get("href", "") if link_el else ""

                full_text = f"{title} {snippet}".lower()

                # Classify the business category
                matched_category = None
                for cat, data in BUSINESS_CATEGORIES.items():
                    for kw in data["keywords_fr"] + data["keywords_en"]:
                        if kw.lower() in full_text:
                            matched_category = cat
                            break
                    if matched_category:
                        break

                # Extract asking price
                price = ""
                price_match = re.search(r'(\d[\d\s,]*\d)\s*\$', f"{title} {snippet}")
                if price_match:
                    price = price_match.group(0)

                results.append({
                    "title": title,
                    "snippet": snippet[:500],
                    "url": url,
                    "category": matched_category or "unknown",
                    "asking_price": price,
                    "mrc": mrc,
                    "signal_type": "supply_reduction",
                })

        duration = time.time() - start_time
        self.log_result(mrc, "success", len(results), duration=duration)
        logger.info(f"[biz_sale] {mrc}: {len(results)} businesses for sale found")
        return results

    def scrape_all(self, mrcs: list) -> dict:
        """Search for businesses for sale across MRCs."""
        all_results = {}
        for mrc in mrcs:
            all_results[mrc] = self.search_businesses_for_sale(mrc)
        return all_results
