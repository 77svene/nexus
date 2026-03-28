"""
Reddit Quebec Scraper - Monitors Quebec subreddits for service demand signals.
Uses Reddit's public JSON feeds (no API key needed).
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

# Quebec-related subreddits (public, no auth needed)
QUEBEC_SUBREDDITS = [
    "quebec",
    "montreal",
    "sherbrooke",
    "QuebecCity",
    "gatineau",
    "Estrie",
    "troisrivieres",
]

# Keywords that indicate someone is looking for a service
DEMAND_KEYWORDS_FR = [
    "cherche", "recherche", "besoin", "recommandation", "recommandez",
    "connaissez", "suggestion", "quelqu'un connaît", "bon", "meilleur",
    "avez-vous un", "qui fait", "qui offre", "où trouver",
]

DEMAND_KEYWORDS_EN = [
    "looking for", "need", "recommend", "recommendation", "anyone know",
    "suggestion", "good", "best", "where to find", "who does",
]

# Map subreddits to MRCs
SUBREDDIT_MRC_MAP = {
    "quebec": "Québec",
    "montreal": "Montréal",
    "sherbrooke": "Sherbrooke",
    "QuebecCity": "Québec",
    "gatineau": "Gatineau",
    "Estrie": "Sherbrooke",
    "troisrivieres": "Trois-Rivières",
}

SUBREDDIT_REGION_MAP = {
    "quebec": "Capitale-Nationale",
    "montreal": "Montréal",
    "sherbrooke": "Estrie",
    "QuebecCity": "Capitale-Nationale",
    "gatineau": "Outaouais",
    "Estrie": "Estrie",
    "troisrivieres": "Mauricie",
}


class RedditScraper(BaseScraper):
    """Scrapes Reddit Quebec subreddits for service demand signals."""

    def __init__(self, knowledge_base=None, llm_extractor=None):
        super().__init__("reddit", knowledge_base)
        self.llm = llm_extractor
        # Reddit requires a specific user agent format
        self.session.headers.update({
            "User-Agent": "QuebecScanner/1.0 (educational project)",
        })

    def _is_demand_post(self, title: str, selftext: str = "") -> bool:
        """Check if a post is someone looking for a service."""
        text = f"{title} {selftext}".lower()
        all_keywords = DEMAND_KEYWORDS_FR + DEMAND_KEYWORDS_EN
        return any(kw in text for kw in all_keywords)

    def scrape_subreddit(self, subreddit: str, max_posts: int = 50) -> int:
        """Scrape a subreddit for demand signals. Returns count of new records."""
        mrc = SUBREDDIT_MRC_MAP.get(subreddit, subreddit)
        region = SUBREDDIT_REGION_MAP.get(subreddit, "")
        start_time = time.time()
        total_found = 0

        # Use Reddit's public JSON feed (no auth needed)
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        params = {"limit": min(max_posts, 100)}

        try:
            data = self.fetch_json(url, params=params)
            if not data or "data" not in data:
                self.log_result(mrc, "error", 0, "No data returned", time.time() - start_time)
                return 0

            posts = data["data"].get("children", [])

            for post_wrapper in posts:
                post = post_wrapper.get("data", {})
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                permalink = post.get("permalink", "")
                created_utc = post.get("created_utc", 0)

                if not self._is_demand_post(title, selftext):
                    continue

                full_text = f"{title} {selftext}"

                # Classify
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

                url_full = f"https://www.reddit.com{permalink}" if permalink else ""
                posted_date = datetime.fromtimestamp(created_utc).isoformat() if created_utc else ""

                added = self.kb.add_demand_signal(
                    source="reddit",
                    mrc=mrc,
                    category=category,
                    title=title[:500],
                    description=selftext[:1000],
                    url=url_full,
                    raw_text=full_text[:1000],
                    urgency=urgency,
                    posted_date=posted_date,
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
        """Scrape all Quebec subreddits."""
        total = 0
        for sub in QUEBEC_SUBREDDITS:
            count = self.scrape_subreddit(sub)
            total += count
            logger.info(f"[reddit] r/{sub}: {count} demand signals found")
        return total
