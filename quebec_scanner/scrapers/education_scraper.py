"""
Education Institutions Scraper - Tracks trade school enrollments and graduations.
More graduations = future supply. Low enrollment = future shortage.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# DEP (Diplôme d'études professionnelles) programs mapped to our categories
DEP_PROGRAMS = {
    "construction": ["charpenterie-menuiserie", "DEP 5319"],
    "plomberie": ["plomberie-chauffage", "DEP 5333"],
    "electricite": ["électricité", "DEP 5295"],
    "thermopompe_climatisation": ["réfrigération", "DEP 5315", "frigoriste"],
    "toiture": ["couverture", "ferblanterie"],
    "peinture": ["peinture en bâtiment", "DEP 5336"],
    "mecanique_auto": ["mécanique automobile", "DEP 5298"],
    "soudure_metal": ["soudage-montage", "DEP 5195"],
    "excavation": ["conduite engins chantier", "DEP 5220"],
    "comptabilite": ["comptabilité", "DEP 5231"],
    "informatique": ["soutien informatique", "DEP 5229"],
}


class EducationScraper(BaseScraper):
    """Monitors trade school programs as supply pipeline indicators."""

    def __init__(self, knowledge_base=None):
        super().__init__("education", knowledge_base)

    def check_program_availability(self, category: str, region: str) -> dict:
        """Check if a DEP program is available and active in a region."""
        programs = DEP_PROGRAMS.get(category)
        if not programs:
            return {"available": False, "program": "none"}

        search_term = programs[0]
        query = f"DEP {search_term} {region} Québec inscription"
        search_results = self.web_search(query, max_results=10)

        result = {
            "available": False,
            "program": search_term,
            "schools_found": 0,
            "wait_list_signal": False,
        }

        if not search_results:
            return result

        result["schools_found"] = len(search_results)
        result["available"] = len(search_results) > 0

        # Check for waitlist/capacity signals
        text = " ".join(
            f"{r.get('title', '')} {r.get('snippet', '')}" for r in search_results
        ).lower()
        if any(w in text for w in ["liste d'attente", "complet", "places limitées"]):
            result["wait_list_signal"] = True

        return result

    def check_supply_pipeline(self, category: str, region: str) -> dict:
        """
        Estimate the future supply pipeline for a category.
        Low enrollment / no program = future shortage = sustained opportunity.
        """
        availability = self.check_program_availability(category, region)

        # No training program in region = structural shortage
        if not availability["available"]:
            return {
                "pipeline_status": "critical_shortage",
                "description": (
                    f"Aucun programme DEP en {availability['program']} trouvé dans la région {region}. "
                    f"Pénurie structurelle probable."
                ),
                "supply_forecast": "declining",
                **availability,
            }

        # Wait lists = high demand for training = many want to enter
        if availability["wait_list_signal"]:
            return {
                "pipeline_status": "growing",
                "description": (
                    f"Programme DEP {availability['program']} populaire avec liste d'attente. "
                    f"Nouvelle offre dans 12-24 mois."
                ),
                "supply_forecast": "increasing_18mo",
                **availability,
            }

        return {
            "pipeline_status": "stable",
            "description": f"Programme DEP {availability['program']} disponible.",
            "supply_forecast": "stable",
            **availability,
        }

    def scan_region(self, region: str) -> dict:
        """Check supply pipeline for all categories in a region."""
        results = {}
        for category in DEP_PROGRAMS:
            results[category] = self.check_supply_pipeline(category, region)
        return results
