"""
Demand Analyzer - Pure mathematical analysis of demand signals.
NO LLM usage - all deterministic scoring.
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEMAND_HIGH_THRESHOLD, BUSINESS_CATEGORIES

logger = logging.getLogger(__name__)


class DemandAnalyzer:
    """Analyzes demand signals using pure math - no LLM."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def analyze_mrc(self, mrc: str) -> dict:
        """
        Analyze all demand signals for an MRC.
        Returns: {category: {count_30d, count_90d, growth_rate, urgency_ratio, score}}
        """
        results = {}

        for category in BUSINESS_CATEGORIES:
            # Current demand (last 30 days)
            count_30d = self.kb.get_demand_count(mrc, category, days=30)

            # Historical demand (30-90 days ago)
            count_prev = self.kb.get_demand_count_period(mrc, category, 90, 30)

            # Growth rate
            if count_prev > 0:
                growth_rate = (count_30d - count_prev) / count_prev
            elif count_30d > 0:
                growth_rate = 1.0  # 100% growth (from 0)
            else:
                growth_rate = 0.0

            # Demand score (0-10 scale)
            # 0 requests = 0, 10+ requests = 10
            demand_score = min(count_30d / (DEMAND_HIGH_THRESHOLD / 10), 10.0)

            # Check urgency ratio
            urgency_count = self._count_urgent(mrc, category)
            urgency_ratio = urgency_count / max(count_30d, 1)

            results[category] = {
                "count_30d": count_30d,
                "count_90d": count_30d + count_prev,
                "count_prev_period": count_prev,
                "growth_rate": growth_rate,
                "urgency_ratio": urgency_ratio,
                "urgency_count": urgency_count,
                "demand_score": round(demand_score, 2),
            }

        return results

    def _count_urgent(self, mrc: str, category: str) -> int:
        """Count urgent demand signals in last 30 days."""
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        with self.kb._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM demand_signals
                WHERE mrc = ? AND category = ? AND urgency = 'urgent' AND scraped_at >= ?
            """, (mrc, category, cutoff)).fetchone()
            return row["cnt"] if row else 0

    def get_top_demand_categories(self, mrc: str, top_n: int = 5) -> list:
        """Get the top N categories by demand in an MRC."""
        analysis = self.analyze_mrc(mrc)
        sorted_cats = sorted(
            analysis.items(),
            key=lambda x: x[1]["demand_score"],
            reverse=True,
        )
        return [(cat, data) for cat, data in sorted_cats[:top_n] if data["count_30d"] > 0]

    def get_growing_categories(self, mrc: str, min_growth: float = 0.20) -> list:
        """Get categories with significant demand growth."""
        analysis = self.analyze_mrc(mrc)
        growing = [
            (cat, data)
            for cat, data in analysis.items()
            if data["growth_rate"] >= min_growth and data["count_30d"] >= 2
        ]
        return sorted(growing, key=lambda x: x[1]["growth_rate"], reverse=True)
