"""
Reddit Quebec Scraper - Monitors Quebec subreddits for service demand signals.
Uses DuckDuckGo to search Reddit posts (JSON API is blocked).
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

QUEBEC_SUBREDDITS = [
    "quebec", "montreal", "sherbrooke", "QuebecCity",
    "gatineau", "Estrie", "troisrivieres",
]

DEMAND_KEYWORDS = [
    "cherche", "recherche", "besoin", "recommandation", "recommandez",
    "connaissez", "suggestion", "quelqu'un connaît", "bon", "meilleur",
    "avez-vous un", "qui fait", "qui offre", "où trouver",
    "looking for", "need", "recommend", "anyone know", "suggestion",
]

SUBREDDIT_MRC_MAP = {
    "quebec": "Québec", "montreal": "Montréal", "sherbrooke": "Sherbrooke",
    "QuebecCity": "Québec", "gatineau": "Gatineau", "Estrie": "Sherbrooke",
    "troisrivieres": "Trois-Rivières",
}

SUBREDDIT_REGION_MAP = {
    "quebec": "Capitale-Nationale", "montreal": "Montréal",
    "sherbrooke": "Estrie", "QuebecCity": "Capitale-Nationale",
    "gatineau": "Outaouais", "Estrie": "Estrie", "troisrivieres": "Mauricie",
}


class RedditScraper(BaseScraper):
    """Scrapes Reddit Quebec subreddits for service demand signals via DDG search."""

    def __init__(self, knowledge_base=None, llm_extractor=None):
        super().__init__("reddit", knowledge_base)
        self.llm = llm_extractor

    def _is_demand_post(self, title: str, snippet: str = "") -> bool:
        text = f"{title} {snippet}".lower()
        return any(kw in text for kw in DEMAND_KEYWORDS)

    def scrape_subreddit(self, subreddit: str, max_posts: int = 50) -> int:
        """Search DDG for Reddit posts about services in a Quebec subreddit."""
        mrc = SUBREDDIT_MRC_MAP.get(subreddit, subreddit)
        region = SUBREDDIT_REGION_MAP.get(subreddit, "")
        start_time = time.time()
        total_found = 0

        # Search for service-related posts in this subreddit
        search_queries = [
            f"site:reddit.com/r/{subreddit} cherche service",
            f"site:reddit.com/r/{subreddit} recommandation",
            f"site:reddit.com/r/{subreddit} besoin",
        ]

        try:
            for query in search_queries:
                results = self.web_search(query, max_results=15)

                for r in results:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    url = r.get("url", "")

                    if "reddit.com" not in url:
                        continue

                    if not self._is_demand_post(title, snippet):
                        continue

                    full_text = f"{title} {snippet}"

                    if self.llm:
                        info = self.llm.extract_service_type(full_text)
                        category = info["category"]
                        urgency = info["urgency"]
                    else:
                        from llm.minimal_extraction import classify_by_keywords
                        category = classify_by_keywords(full_text) or "unknown"
                        urgency = "normal"

                    if category == "unknown":
                        continue

                    added = self.kb.add_demand_signal(
                        source="reddit",
                        mrc=mrc,
                        category=category,
                        title=title[:500],
                        description=snippet[:1000],
                        url=url,
                        raw_text=full_text[:1000],
                        urgency=urgency,
                        region=region,
                    )
                    if added:
                        total_found += 1

            duration = time.time() - start_time
            self.log_result(mrc, "success", total_found, duration=duration)

        except Exception as e:
            duration = time.time() - start_time
            self.log_result(mrc, "error", total_found, str(e), duration)
            logger.error(f"[reddit] Error scraping r/{subreddit}: {e}")

        return total_found

    def scrape_all(self) -> int:
        total = 0
        for sub in QUEBEC_SUBREDDITS:
            count = self.scrape_subreddit(sub)
            total += count
            logger.info(f"[reddit] r/{sub}: {count} demand signals found")
        return total
