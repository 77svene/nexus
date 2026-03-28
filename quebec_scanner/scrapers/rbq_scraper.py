"""
RBQ (Régie du bâtiment du Québec) Scraper.
Tracks licensed contractors and construction businesses.
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

RBQ_CATEGORIES = {
    "construction": ["entrepreneur général", "1.1"],
    "plomberie": ["plomberie", "4.1"],
    "electricite": ["électricité", "16"],
    "thermopompe_climatisation": ["climatisation ventilation", "15.2", "réfrigération"],
    "toiture": ["couverture", "7.2"],
    "excavation": ["excavation", "2.1", "2.6"],
}


class RBQScraper(BaseScraper):
    def __init__(self, knowledge_base=None):
        super().__init__("rbq", knowledge_base)

    def search_licensed_contractors(self, category: str, mrc: str) -> int:
        rbq_terms = RBQ_CATEGORIES.get(category)
        if not rbq_terms:
            return 0

        total = 0
        for term in rbq_terms[:1]:
            query = f"site:rbq.gouv.qc.ca {term} {mrc}"
            results = self.web_search(query, max_results=20)
            if not results:
                continue

            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")

                license_match = re.search(r'\b(\d{4}-\d{4}-\d{2})\b', title)
                license_num = license_match.group(1) if license_match else ""
                biz_name = title.split(" - ")[0].strip() if " - " in title else title

                if self.kb:
                    self.kb.add_supply_provider(
                        source="rbq", mrc=mrc, category=category,
                        business_name=biz_name, license_number=license_num,
                        license_type=term,
                    )
                    total += 1

        self.log_result(mrc, "success", total)
        return total

    def scan_mrc(self, mrc: str) -> dict:
        results = {}
        for category in RBQ_CATEGORIES:
            count = self.search_licensed_contractors(category, mrc)
            results[category] = count
        return results
