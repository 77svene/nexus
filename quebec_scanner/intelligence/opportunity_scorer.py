"""
Opportunity Scorer - Pure mathematical scoring of business opportunities.
NO LLM usage whatsoever. 100% deterministic.
"""

import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCORING_WEIGHTS, BUSINESS_CATEGORIES, PRIORITY_MRCS, REGIONS
from intelligence.demand_analyzer import DemandAnalyzer
from intelligence.supply_analyzer import SupplyAnalyzer
from intelligence.convergence_detector import ConvergenceDetector

logger = logging.getLogger(__name__)


def score_opportunity(demand_score, supply_score, regulatory_score, temporal_score):
    """
    Pure mathematical scoring - the core algorithm.
    All inputs on 0-10 scale.
    Returns weighted total on 0-10 scale.
    """
    total = (
        demand_score * SCORING_WEIGHTS["demand"] +
        supply_score * SCORING_WEIGHTS["supply"] +
        regulatory_score * SCORING_WEIGHTS["regulatory"] +
        temporal_score * SCORING_WEIGHTS["temporal"]
    )
    return round(min(total, 10.0), 2)


class OpportunityScorer:
    """Scores and ranks business opportunities across MRCs. Pure math."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.demand_analyzer = DemandAnalyzer(knowledge_base)
        self.supply_analyzer = SupplyAnalyzer(knowledge_base)
        self.convergence = ConvergenceDetector(knowledge_base)

    def _get_region_for_mrc(self, mrc: str) -> str:
        """Look up region name for an MRC."""
        for region_id, region_data in REGIONS.items():
            if mrc in region_data["mrcs"]:
                return region_data["name"]
        return ""

    def score_mrc(self, mrc: str) -> list:
        """
        Score all opportunity categories for a single MRC.
        Returns list of scored opportunities sorted by total score.
        """
        demand_analysis = self.demand_analyzer.analyze_mrc(mrc)
        supply_analysis = self.supply_analyzer.analyze_mrc(mrc)
        region = self._get_region_for_mrc(mrc)

        opportunities = []

        for category in BUSINESS_CATEGORIES:
            demand_data = demand_analysis.get(category, {})
            supply_data = supply_analysis.get(category, {})

            # Skip if absolutely no data
            if demand_data.get("count_30d", 0) == 0 and supply_data.get("provider_count", 0) == 0:
                continue

            # Calculate individual scores (0-10)
            demand_score = demand_data.get("demand_score", 0)
            supply_score = supply_data.get("supply_score", 0)

            # Regulatory score
            subsidies = self.kb.get_active_subsidies(category)
            regulatory_score = 10.0 if subsidies else 5.0

            # Temporal score (growth-based)
            growth_rate = demand_data.get("growth_rate", 0)
            temporal_score = min(max(growth_rate * 20, 0), 10.0)

            # Total score
            total = score_opportunity(demand_score, supply_score, regulatory_score, temporal_score)

            # Convergence analysis
            conv = self.convergence.detect_convergence(mrc, category, demand_data, supply_data)
            conv["region"] = region

            # Boost score if high convergence
            if conv["convergence_count"] >= 4:
                total = min(total * 1.15, 10.0)
            elif conv["convergence_count"] >= 3:
                total = min(total * 1.10, 10.0)

            total = round(total, 2)

            scores = {
                "demand": round(demand_score, 2),
                "supply": round(supply_score, 2),
                "regulatory": round(regulatory_score, 2),
                "temporal": round(temporal_score, 2),
                "total": total,
            }

            # Save to database
            self.kb.upsert_opportunity(mrc, category, scores, conv)

            if total > 0:
                opportunities.append({
                    "mrc": mrc,
                    "region": region,
                    "category": category,
                    "scores": scores,
                    "convergence": conv,
                })

        # Sort by total score descending
        opportunities.sort(key=lambda x: x["scores"]["total"], reverse=True)
        return opportunities

    def score_all_priority(self) -> list:
        """Score all priority MRCs and return top opportunities."""
        all_opportunities = []

        for mrc in PRIORITY_MRCS:
            logger.info(f"[scorer] Scoring {mrc}...")
            opps = self.score_mrc(mrc)
            all_opportunities.extend(opps)

        # Sort all by score
        all_opportunities.sort(key=lambda x: x["scores"]["total"], reverse=True)

        logger.info(f"[scorer] Scored {len(all_opportunities)} opportunities across {len(PRIORITY_MRCS)} MRCs")
        return all_opportunities

    def get_alerts(self, min_score: float = 7.0) -> list:
        """Get opportunities that should trigger alerts."""
        return self.kb.get_unalerted_opportunities(min_score)
