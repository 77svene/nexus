"""
Professional Orders Scraper - Tracks licensed professionals across Quebec.
Monitors member counts, retirements, and new registrations as supply indicators.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# Quebec professional orders relevant to our categories
PROFESSIONAL_ORDERS = {
    "comptabilite": {
        "order": "Ordre des comptables professionnels agréés du Québec",
        "url": "cpaquebec.ca",
        "search_terms": ["CPA Québec", "comptable professionnel agréé"],
    },
    "sante_bienetre": {
        "order": "Ordre des physiothérapeutes / massothérapeutes",
        "url": "oppq.qc.ca",
        "search_terms": ["physiothérapeute Québec", "massothérapeute ordre"],
    },
    "electricite": {
        "order": "Corporation des maîtres électriciens du Québec (CMEQ)",
        "url": "cmeq.org",
        "search_terms": ["maître électricien CMEQ", "électricien Québec membre"],
    },
    "plomberie": {
        "order": "Corporation des maîtres mécaniciens en tuyauterie du Québec (CMMTQ)",
        "url": "cmmtq.org",
        "search_terms": ["maître mécanicien tuyauterie", "plombier CMMTQ"],
    },
}


class ProfessionalOrdersScraper(BaseScraper):
    """Scrapes professional order data for supply-side intelligence."""

    def __init__(self, knowledge_base=None):
        super().__init__("professional_orders", knowledge_base)

    def count_members(self, category: str, mrc: str) -> dict:
        """Count professional order members in a category/MRC."""
        order_info = PROFESSIONAL_ORDERS.get(category)
        if not order_info:
            return {"count": 0, "source": "none"}

        total = 0
        for term in order_info["search_terms"][:1]:
            query = f"site:{order_info['url']} {term} {mrc}"
            soup = self.fetch("https://www.google.com/search", params={
                "q": query, "num": 20, "hl": "fr",
            })
            if soup:
                results = soup.select("div.g")
                total += len(results)

                # Also try to extract actual member counts from text
                text = soup.get_text(separator=" ", strip=True)
                count_match = re.search(r'(\d+)\s*(?:membres?|member)', text, re.I)
                if count_match:
                    parsed = int(count_match.group(1))
                    if parsed < 50000:  # Sanity check
                        total = max(total, parsed)

        # Store as supply data
        if total > 0 and self.kb:
            self.kb.add_supply_provider(
                source="professional_order",
                mrc=mrc,
                category=category,
                business_name=f"{order_info['order']} - {mrc}",
            )

        return {
            "count": total,
            "order": order_info["order"],
            "source": order_info["url"],
        }

    def scan_mrc(self, mrc: str) -> dict:
        """Count all professional order members in an MRC."""
        results = {}
        for category in PROFESSIONAL_ORDERS:
            results[category] = self.count_members(category, mrc)
        return results
