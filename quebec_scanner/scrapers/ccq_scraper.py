"""
CCQ (Commission de la construction du Québec) Scraper.
Tracks construction labor data: shortages, wages, activity levels.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

CCQ_TRADES = {
    "construction": ["charpentier-menuisier", "manoeuvre"],
    "plomberie": ["plombier", "tuyauteur"],
    "electricite": ["électricien"],
    "toiture": ["couvreur"],
    "peinture": ["peintre"],
    "soudure_metal": ["soudeur", "ferblantier"],
    "excavation": ["opérateur équipement lourd", "grutier"],
}


class CCQScraper(BaseScraper):
    def __init__(self, knowledge_base=None):
        super().__init__("ccq", knowledge_base)

    def get_trade_data(self, trade_category: str, region: str = "Estrie") -> dict:
        trades = CCQ_TRADES.get(trade_category, [trade_category])
        trade_name = trades[0]

        data = {
            "trade": trade_name, "region": region,
            "shortage_signal": False, "wage_trend": "unknown",
            "activity_level": "unknown", "hours_trend": "unknown",
        }

        query = f"CCQ {trade_name} {region} pénurie main-d'oeuvre"
        results = self.web_search(query, max_results=10)
        if not results:
            return data

        text = " ".join(f"{r['title']} {r['snippet']}" for r in results).lower()

        shortage_words = ["pénurie", "manque", "shortage", "rareté", "difficulté recrutement"]
        data["shortage_signal"] = any(w in text for w in shortage_words)

        if any(w in text for w in ["hausse salaire", "augmentation taux", "wage increase"]):
            data["wage_trend"] = "rising"
        elif any(w in text for w in ["baisse", "diminution", "gel"]):
            data["wage_trend"] = "stable_or_declining"
        else:
            data["wage_trend"] = "stable"

        if any(w in text for w in ["croissance", "hausse activité", "record", "forte"]):
            data["activity_level"] = "high"
        elif any(w in text for w in ["ralentissement", "baisse", "recul"]):
            data["activity_level"] = "low"
        else:
            data["activity_level"] = "moderate"

        return data

    def get_construction_outlook(self, region: str = "Estrie") -> dict:
        query = f"CCQ perspectives construction {region} Québec"
        results = self.web_search(query, max_results=10)

        outlook = {
            "region": region, "overall_trend": "unknown",
            "investment_signal": "unknown", "workforce_gap": False,
        }
        if not results:
            return outlook

        text = " ".join(f"{r['title']} {r['snippet']}" for r in results).lower()

        if any(w in text for w in ["croissance", "hausse", "positif", "record"]):
            outlook["overall_trend"] = "growth"
        elif any(w in text for w in ["stable", "maintien"]):
            outlook["overall_trend"] = "stable"
        elif any(w in text for w in ["recul", "baisse", "ralentissement"]):
            outlook["overall_trend"] = "declining"

        amounts = re.findall(r'(\d+[\d,.\s]*)\s*(?:milliards?|millions?)\s*\$', text)
        if amounts:
            outlook["investment_signal"] = "significant"
        if any(w in text for w in ["pénurie", "manque de main", "besoin de travailleurs"]):
            outlook["workforce_gap"] = True

        logger.info(f"[ccq] {region}: trend={outlook['overall_trend']}, gap={outlook['workforce_gap']}")
        return outlook

    def scan_region(self, region: str) -> dict:
        results = {"trades": {}, "outlook": {}}
        results["outlook"] = self.get_construction_outlook(region)
        for cat in CCQ_TRADES:
            results["trades"][cat] = self.get_trade_data(cat, region)
        return results
