"""
Emploi Quebec Scraper - Job vacancies, labor shortages, and wage data.
Monitors hiring difficulty as a supply constraint indicator.
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

# Map business categories to job search terms
JOB_SEARCH_TERMS = {
    "construction": ["charpentier", "menuisier", "manoeuvre construction"],
    "plomberie": ["plombier", "tuyauteur"],
    "electricite": ["électricien", "technicien électrique"],
    "thermopompe_climatisation": ["frigoriste", "technicien HVAC", "climatisation"],
    "toiture": ["couvreur", "poseur bardeaux"],
    "deneigement": ["opérateur souffleuse", "déneigement"],
    "paysagement": ["paysagiste", "horticulteur"],
    "menage": ["préposé entretien", "ménage", "nettoyage"],
    "peinture": ["peintre bâtiment"],
    "comptabilite": ["comptable", "technicien comptabilité"],
    "mecanique_auto": ["mécanicien automobile", "technicien auto"],
    "soudure_metal": ["soudeur", "assembleur soudeur"],
    "informatique": ["technicien informatique", "support technique"],
    "excavation": ["opérateur pelle", "excavation"],
}


class EmploiQuebecScraper(BaseScraper):
    """Scrapes job data as indicator of labor shortages and service demand."""

    def __init__(self, knowledge_base=None):
        super().__init__("emploi_quebec", knowledge_base)

    def count_job_postings(self, category: str, mrc: str) -> dict:
        """Count job postings for a category in an MRC. More postings = labor shortage."""
        terms = JOB_SEARCH_TERMS.get(category, [category])
        total_postings = 0

        for term in terms[:2]:
            # Search Indeed Quebec (free, no API)
            query = f"site:indeed.com {term} {mrc} Québec"
            soup = self.fetch("https://www.google.com/search", params={
                "q": query, "num": 20, "hl": "fr",
            })

            if soup:
                results = soup.select("div.g")
                total_postings += len(results)

            # Also search Jobillico
            query2 = f"site:jobillico.com {term} {mrc}"
            soup2 = self.fetch("https://www.google.com/search", params={
                "q": query2, "num": 10, "hl": "fr",
            })

            if soup2:
                results2 = soup2.select("div.g")
                total_postings += len(results2)

        # Many job postings = labor shortage = opportunity for independent providers
        shortage_level = "none"
        if total_postings >= 10:
            shortage_level = "critical"
        elif total_postings >= 5:
            shortage_level = "high"
        elif total_postings >= 2:
            shortage_level = "moderate"

        result = {
            "job_postings": total_postings,
            "shortage_level": shortage_level,
            "category": category,
            "mrc": mrc,
        }

        self.log_result(mrc, "success", total_postings)
        return result

    def scan_labor_market(self, mrc: str, categories: list = None) -> dict:
        """Scan the labor market for all categories in an MRC."""
        if not categories:
            categories = list(JOB_SEARCH_TERMS.keys())

        results = {}
        for cat in categories:
            results[cat] = self.count_job_postings(cat, mrc)

        # Sort by shortage level
        critical = [c for c, d in results.items() if d["shortage_level"] == "critical"]
        high = [c for c, d in results.items() if d["shortage_level"] == "high"]

        if critical or high:
            logger.info(f"[emploi] {mrc}: Critical shortages in {critical}, high in {high}")

        return results
