"""
Government Programs Scraper - Monitors Quebec subsidies, grants, and regulatory changes.
Tracks programs that create business opportunities.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOVERNMENT_PROGRAMS, BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class GovernmentScraper(BaseScraper):
    """Scrapes Quebec government websites for subsidy programs and regulations."""

    def __init__(self, knowledge_base=None):
        super().__init__("government", knowledge_base)

    def check_known_programs(self) -> list:
        """Check all known government programs and update their status."""
        active_programs = []
        start_time = time.time()

        for program in GOVERNMENT_PROGRAMS:
            try:
                soup = self.fetch(program["url"])
                if not soup:
                    continue

                page_text = soup.get_text(separator=" ", strip=True).lower()

                # Check if program appears active
                is_active = True
                inactive_signals = ["terminé", "expiré", "fermé", "suspendu", "plus disponible", "ended", "expired"]
                for signal in inactive_signals:
                    if signal in page_text:
                        is_active = False
                        break

                # Try to extract amounts/limits
                amount = ""
                amount_patterns = [
                    r"jusqu[\'']?à\s*([\d\s,]+\s*\$)",
                    r"([\d\s,]+\s*\$)\s*(?:par|maximum|jusqu)",
                    r"aide\s+(?:financière|maximale)?\s*(?:de|:)?\s*([\d\s,]+\s*\$)",
                    r"subvention\s+(?:de|:)?\s*([\d\s,]+\s*\$)",
                    r"crédit\s+(?:d[\'']impôt)?\s*(?:de|:)?\s*([\d\s,]+\s*\$)",
                ]
                for pat in amount_patterns:
                    m = re.search(pat, page_text)
                    if m:
                        amount = m.group(1).strip()
                        break

                if self.kb:
                    self.kb.add_government_program(
                        name=program["name"],
                        url=program["url"],
                        sector=program["sector"],
                        description=program["description"],
                        is_active=is_active,
                        amount=amount,
                    )

                if is_active:
                    active_programs.append({
                        "name": program["name"],
                        "sector": program["sector"],
                        "amount": amount,
                        "url": program["url"],
                    })

                logger.info(f"[gov] {program['name']}: {'active' if is_active else 'inactive'}")

            except Exception as e:
                logger.error(f"[gov] Error checking {program['name']}: {e}")

        duration = time.time() - start_time
        self.log_result("all", "success", len(active_programs), duration=duration)
        return active_programs

    def search_new_programs(self) -> list:
        """Search for newly announced Quebec programs and subsidies."""
        new_programs = []
        search_queries = [
            "site:quebec.ca nouveau programme subvention",
            "site:transitionenergetique.gouv.qc.ca programme",
            "québec nouveau programme aide financière 2026",
            "québec subvention rénovation énergie 2026",
            "québec programme construction habitation 2026",
        ]

        for query in search_queries:
            search_results = self.web_search_recent(query, max_results=10)
            if not search_results:
                continue

            for result in search_results:
                title = result.get("title", "")
                url = result.get("url", "")

                # Classify which sector this program might affect
                title_lower = title.lower()
                for cat, data in BUSINESS_CATEGORIES.items():
                    for kw in data["keywords_fr"]:
                        if kw.lower() in title_lower:
                            new_programs.append({
                                "name": title,
                                "url": url,
                                "sector": cat,
                            })
                            break

        return new_programs

    def scrape_all(self) -> dict:
        """Run all government program checks."""
        result = {
            "known_programs": self.check_known_programs(),
            "new_programs": self.search_new_programs(),
        }
        logger.info(f"[gov] Found {len(result['known_programs'])} active known programs, "
                     f"{len(result['new_programs'])} new program signals")
        return result
