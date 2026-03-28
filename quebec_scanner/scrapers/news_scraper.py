"""
Quebec News Scraper - Monitors local news for business signals.
Detects: closures, expansions, new investments, layoffs, infrastructure projects.
"""

import logging
import re
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES
from scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# Quebec news sources searchable via Google
NEWS_SOURCES = [
    "lapresse.ca",
    "lesoleil.com",
    "latribune.ca",
    "ledroit.com",
    "journaldequebec.com",
    "journaldemontreal.com",
    "ici.radio-canada.ca",
    "tvanouvelles.ca",
]

# Signal keywords for opportunity detection
SIGNAL_KEYWORDS = {
    "closure": {
        "fr": ["fermeture", "fermé", "cesse ses activités", "fait faillite", "liquidation"],
        "impact": "supply_reduction",
    },
    "expansion": {
        "fr": ["expansion", "agrandissement", "nouveau projet", "investissement", "embauche"],
        "impact": "demand_increase",
    },
    "layoff": {
        "fr": ["licenciement", "mise à pied", "perte d'emplois", "compressions"],
        "impact": "cascade_demand",
    },
    "infrastructure": {
        "fr": ["construction", "chantier", "réfection", "projet infrastructure", "route", "pont"],
        "impact": "demand_increase",
    },
    "housing": {
        "fr": ["développement résidentiel", "nouveau quartier", "condos", "habitation"],
        "impact": "cascade_demand",
    },
    "subsidy": {
        "fr": ["subvention", "aide financière", "programme gouvernemental", "crédit d'impôt"],
        "impact": "demand_catalyst",
    },
}


class NewsScraper(BaseScraper):
    """Scrapes Quebec news for business opportunity signals."""

    def __init__(self, knowledge_base=None):
        super().__init__("news", knowledge_base)

    def scan_news(self, mrc: str, days_back: int = 30) -> list:
        """Scan news for relevant signals about an MRC."""
        signals = []
        start_time = time.time()

        for signal_type, signal_data in SIGNAL_KEYWORDS.items():
            for keyword in signal_data["fr"][:2]:  # Limit to 2 keywords per signal type
                query = f"{keyword} {mrc} Québec"
                params = {
                    "q": query,
                    "num": 10,
                    "hl": "fr",
                    "tbs": f"qdr:m{min(days_back // 30, 3)}",  # Last N months
                    "tbm": "nws",  # News search
                }

                soup = self.fetch("https://www.google.com/search", params=params)
                if not soup:
                    continue

                for result in soup.select("div.SoaBEf, div.g"):
                    title_el = result.select_one("div.mCBkyc, h3")
                    if not title_el:
                        continue

                    title = title_el.get_text(strip=True)
                    link_el = result.select_one("a[href]")
                    url = link_el.get("href", "") if link_el else ""

                    snippet_el = result.select_one("div.GI74Re, div.VwiC3b")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""

                    date_el = result.select_one("span.WG9SHc, div.OSrXXb span")
                    date_text = date_el.get_text(strip=True) if date_el else ""

                    # Determine affected categories
                    full_text = f"{title} {snippet}".lower()
                    affected_categories = []
                    for cat, data in BUSINESS_CATEGORIES.items():
                        for kw in data["keywords_fr"]:
                            if kw.lower() in full_text:
                                affected_categories.append(cat)
                                break

                    signals.append({
                        "type": signal_type,
                        "impact": signal_data["impact"],
                        "title": title,
                        "snippet": snippet[:500],
                        "url": url,
                        "date": date_text,
                        "mrc": mrc,
                        "affected_categories": affected_categories,
                        "keyword_matched": keyword,
                    })

        duration = time.time() - start_time
        self.log_result(mrc, "success", len(signals), duration=duration)

        # Also store as demand signals if they indicate demand
        for signal in signals:
            if signal["impact"] in ("demand_increase", "demand_catalyst"):
                for cat in signal["affected_categories"]:
                    if self.kb:
                        self.kb.add_demand_signal(
                            source="news",
                            mrc=mrc,
                            category=cat,
                            title=f"[NEWS] {signal['title'][:200]}",
                            description=signal["snippet"],
                            url=signal["url"],
                            raw_text=f"{signal['title']} {signal['snippet']}"[:1000],
                        )

        logger.info(f"[news] {mrc}: {len(signals)} news signals found")
        return signals

    def scan_all(self, mrcs: list) -> dict:
        """Scan news for multiple MRCs."""
        all_signals = {}
        for mrc in mrcs:
            all_signals[mrc] = self.scan_news(mrc)
        return all_signals
