"""
Statistics Quebec Scraper - Demographics, economic indicators, labor data.
Scrapes publicly available datasets from statistique.quebec.ca.
"""

import logging
import re
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class StatQuebecScraper(BaseScraper):
    """Scrapes Statistics Quebec for demographic and economic data."""

    def __init__(self, knowledge_base=None):
        super().__init__("stat_quebec", knowledge_base)
        self.base_url = "https://statistique.quebec.ca"

    def get_demographics(self, mrc: str) -> dict:
        """Get demographic data for an MRC."""
        query = f"site:statistique.quebec.ca population {mrc}"
        soup = self.fetch("https://www.google.com/search", params={
            "q": query, "num": 10, "hl": "fr",
        })

        data = {"population": 0, "median_age": 0, "seniors_pct": 0, "growth_rate": 0}

        if not soup:
            return data

        text = soup.get_text(separator=" ", strip=True).lower()

        # Extract population numbers
        pop_match = re.search(r'population.*?(\d[\d\s]{2,}\d)', text)
        if pop_match:
            try:
                data["population"] = int(pop_match.group(1).replace(" ", "").replace("\u00a0", ""))
            except ValueError:
                pass

        # Extract median age
        age_match = re.search(r'âge médian.*?(\d{2}[.,]\d)', text)
        if age_match:
            try:
                data["median_age"] = float(age_match.group(1).replace(",", "."))
            except ValueError:
                pass

        # Extract seniors percentage
        senior_match = re.search(r'65 ans et plus.*?(\d{1,2}[.,]\d)\s*%', text)
        if senior_match:
            try:
                data["seniors_pct"] = float(senior_match.group(1).replace(",", "."))
            except ValueError:
                pass

        self.log_result(mrc, "success", 1)
        return data

    def get_economic_indicators(self, mrc: str) -> dict:
        """Get economic indicators for an MRC."""
        query = f"site:statistique.quebec.ca économie emploi {mrc}"
        soup = self.fetch("https://www.google.com/search", params={
            "q": query, "num": 10, "hl": "fr",
        })

        data = {"unemployment_rate": 0, "business_count": 0, "avg_income": 0}

        if not soup:
            return data

        text = soup.get_text(separator=" ", strip=True).lower()

        # Extract unemployment rate
        unemp_match = re.search(r'(?:chômage|taux d.emploi).*?(\d{1,2}[.,]\d)\s*%', text)
        if unemp_match:
            try:
                data["unemployment_rate"] = float(unemp_match.group(1).replace(",", "."))
            except ValueError:
                pass

        # Extract income data
        income_match = re.search(r'revenu.*?(\d[\d\s]{2,}\d)\s*\$', text)
        if income_match:
            try:
                data["avg_income"] = int(income_match.group(1).replace(" ", "").replace("\u00a0", ""))
            except ValueError:
                pass

        return data

    def get_building_permit_trends(self, mrc: str) -> dict:
        """Get building permit value trends."""
        query = f"statistique quebec permis bâtir {mrc} 2025 2026"
        soup = self.fetch("https://www.google.com/search", params={
            "q": query, "num": 10, "hl": "fr",
        })

        data = {"permit_value_trend": "unknown", "permit_mentions": 0}

        if not soup:
            return data

        text = soup.get_text(separator=" ", strip=True).lower()

        # Count permit-related mentions
        data["permit_mentions"] = text.count("permis")

        # Look for trend words
        positive = ["hausse", "augmentation", "croissance", "record"]
        negative = ["baisse", "diminution", "recul", "déclin"]

        pos_count = sum(text.count(w) for w in positive)
        neg_count = sum(text.count(w) for w in negative)

        if pos_count > neg_count:
            data["permit_value_trend"] = "up"
        elif neg_count > pos_count:
            data["permit_value_trend"] = "down"
        else:
            data["permit_value_trend"] = "stable"

        return data

    def scrape_mrc(self, mrc: str) -> dict:
        """Get all statistics for an MRC."""
        start_time = time.time()
        result = {
            "demographics": self.get_demographics(mrc),
            "economic": self.get_economic_indicators(mrc),
            "permits": self.get_building_permit_trends(mrc),
        }
        duration = time.time() - start_time
        self.log_result(mrc, "success", 3, duration=duration)
        return result
