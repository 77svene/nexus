"""
Supply Analyzer - Pure mathematical analysis of supply/provider data.
NO LLM usage - all deterministic scoring.
"""

import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SUPPLY_LOW_THRESHOLD, BUSINESS_CATEGORIES

logger = logging.getLogger(__name__)


class SupplyAnalyzer:
    """Analyzes supply-side data using pure math - no LLM."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def analyze_mrc(self, mrc: str) -> dict:
        """
        Analyze supply for all categories in an MRC.
        Returns: {category: {provider_count, avg_rating, supply_score}}
        """
        results = {}

        for category in BUSINESS_CATEGORIES:
            provider_count = self.kb.get_supply_count(mrc, category)

            # Get average rating and details
            avg_rating, total_reviews = self._get_rating_stats(mrc, category)

            # Supply score (0-10 scale, INVERTED - less supply = higher score)
            # 0 providers = 10 (huge opportunity)
            # 10+ providers = 0 (saturated market)
            supply_score = 10.0 - min(provider_count / (SUPPLY_LOW_THRESHOLD / 5), 10.0)
            supply_score = max(supply_score, 0.0)

            # Quality gap: if existing providers have low ratings, opportunity exists
            quality_gap = 0.0
            if provider_count > 0 and avg_rating > 0:
                # Low average rating = quality gap = opportunity
                quality_gap = max(0, (4.0 - avg_rating) * 2.5)  # 0-10 scale

            results[category] = {
                "provider_count": provider_count,
                "avg_rating": round(avg_rating, 2),
                "total_reviews": total_reviews,
                "supply_score": round(supply_score, 2),
                "quality_gap": round(quality_gap, 2),
            }

        return results

    def _get_rating_stats(self, mrc: str, category: str) -> tuple:
        """Get average rating and total reviews for providers in a category/MRC."""
        with self.kb._get_conn() as conn:
            row = conn.execute("""
                SELECT AVG(rating) as avg_r, SUM(review_count) as total_r
                FROM supply_providers
                WHERE mrc = ? AND category = ? AND is_active = 1 AND rating > 0
            """, (mrc, category)).fetchone()
            avg_rating = row["avg_r"] if row and row["avg_r"] else 0.0
            total_reviews = row["total_r"] if row and row["total_r"] else 0
            return avg_rating, total_reviews

    def get_underserved_categories(self, mrc: str, max_providers: int = 3) -> list:
        """Get categories with very few providers (supply gaps)."""
        analysis = self.analyze_mrc(mrc)
        gaps = [
            (cat, data)
            for cat, data in analysis.items()
            if data["provider_count"] <= max_providers
        ]
        return sorted(gaps, key=lambda x: x[1]["provider_count"])

    def get_quality_gaps(self, mrc: str, min_gap: float = 3.0) -> list:
        """Get categories where existing providers have poor ratings."""
        analysis = self.analyze_mrc(mrc)
        gaps = [
            (cat, data)
            for cat, data in analysis.items()
            if data["quality_gap"] >= min_gap and data["provider_count"] > 0
        ]
        return sorted(gaps, key=lambda x: x[1]["quality_gap"], reverse=True)
