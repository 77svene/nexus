"""
Subsidy Tracker - Aggregates ALL Quebec subsidy/grant programs across government levels.
Tracks budget remaining and uptake rates as urgency signals.
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

# Comprehensive list of Quebec subsidy/grant programs
SUBSIDY_PROGRAMS = [
    {"name": "Chauffez vert", "sector": "thermopompe_climatisation",
     "search": "chauffez vert programme aide", "level": "provincial"},
    {"name": "Rénovert", "sector": "construction",
     "search": "rénovert crédit impôt rénovation", "level": "provincial"},
    {"name": "Novoclimat", "sector": "construction",
     "search": "novoclimat maison neuve écoénergétique", "level": "provincial"},
    {"name": "Logis vert", "sector": "construction",
     "search": "logis vert efficacité énergétique", "level": "provincial"},
    {"name": "Greener Homes (fédéral)", "sector": "thermopompe_climatisation",
     "search": "maisons plus vertes canada subvention", "level": "federal"},
    {"name": "Programme AccèsLogis", "sector": "construction",
     "search": "accèslogis québec logement social", "level": "provincial"},
    {"name": "Programme habitation durable", "sector": "construction",
     "search": "habitation durable québec aide", "level": "provincial"},
    {"name": "Aide aux entreprises (MRC)", "sector": "construction",
     "search": "aide entreprise démarrage MRC Québec", "level": "municipal"},
    {"name": "Programme Emploi Québec", "sector": "education_tutorat",
     "search": "subvention salariale emploi québec formation", "level": "provincial"},
    {"name": "Aide technique auto électrique", "sector": "mecanique_auto",
     "search": "aide véhicule électrique borne Québec", "level": "provincial"},
    {"name": "Programme d'aide aux aînés", "sector": "sante_bienetre",
     "search": "aide maintien domicile aînés Québec", "level": "provincial"},
    {"name": "Subvention décontamination", "sector": "excavation",
     "search": "décontamination terrain subvention Québec", "level": "provincial"},
]


class SubsidyTrackerScraper(BaseScraper):
    """Aggregates and tracks all Quebec subsidy programs."""

    def __init__(self, knowledge_base=None):
        super().__init__("subsidy_tracker", knowledge_base)

    def check_program(self, program: dict) -> dict:
        """Check status and details of a single subsidy program."""
        soup = self.fetch("https://www.google.com/search", params={
            "q": f"{program['search']} 2026", "num": 5, "hl": "fr",
        })

        result = {
            "name": program["name"],
            "sector": program["sector"],
            "level": program["level"],
            "is_active": True,
            "amount": "",
            "urgency": "normal",
            "budget_signal": "unknown",
        }

        if not soup:
            return result

        text = soup.get_text(separator=" ", strip=True).lower()

        # Check if program is still active
        inactive_words = ["terminé", "expiré", "fermé", "suspendu", "annulé"]
        for word in inactive_words:
            if word in text:
                result["is_active"] = False
                break

        # Extract amounts
        amount_match = re.search(r'(?:jusqu.à|maximum|aide de)\s*([\d\s,]+\s*\$)', text)
        if amount_match:
            result["amount"] = amount_match.group(1).strip()

        # Budget urgency signals
        if any(w in text for w in ["budget limité", "fonds restants", "date limite", "avant le"]):
            result["urgency"] = "high"
            result["budget_signal"] = "running_out"
        elif any(w in text for w in ["bonification", "augmentation", "prolongé", "élargi"]):
            result["budget_signal"] = "expanding"

        # Store in database
        if self.kb and result["is_active"]:
            self.kb.add_government_program(
                name=result["name"],
                url="",
                sector=result["sector"],
                description=f"Level: {result['level']}, Amount: {result['amount']}",
                is_active=result["is_active"],
                amount=result["amount"],
            )

        return result

    def check_all_programs(self) -> list:
        """Check all known subsidy programs."""
        results = []
        for program in SUBSIDY_PROGRAMS:
            result = self.check_program(program)
            results.append(result)
            logger.info(f"[subsidy] {result['name']}: {'active' if result['is_active'] else 'inactive'}"
                         f" {result['budget_signal']}")

        active = [r for r in results if r["is_active"]]
        urgent = [r for r in results if r["urgency"] == "high"]

        logger.info(f"[subsidy] {len(active)}/{len(results)} programs active, "
                     f"{len(urgent)} with budget urgency")
        return results

    def get_subsidies_by_sector(self) -> dict:
        """Group active subsidies by business sector."""
        results = self.check_all_programs()
        by_sector = {}
        for r in results:
            if r["is_active"]:
                sector = r["sector"]
                if sector not in by_sector:
                    by_sector[sector] = []
                by_sector[sector].append(r)
        return by_sector
