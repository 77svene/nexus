"""
Convergence Detector - Identifies when multiple signals align to create opportunities.
An opportunity is HIGH-CONFIDENCE when 3+ signals converge.
100% deterministic - NO LLM.
"""

import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEMAND_HIGH_THRESHOLD, SUPPLY_LOW_THRESHOLD, GROWTH_THRESHOLD,
    MIN_CONVERGENCE_SIGNALS, BUSINESS_CATEGORIES
)

logger = logging.getLogger(__name__)

# Barriers database (typical barriers per category)
CATEGORY_BARRIERS = {
    "construction": ["Licence RBQ requise", "Capital: ~$15-30K", "Formation: DEP ou expérience"],
    "plomberie": ["Licence RBQ requise", "Certification compagnon", "Capital: ~$10-20K"],
    "electricite": ["Licence RBQ + CMEQ", "Maîtrise requise", "Capital: ~$10-20K"],
    "thermopompe_climatisation": ["Licence RBQ 15.2", "Certification frigoriste", "Capital: ~$15K"],
    "toiture": ["Licence RBQ 7.2", "Assurances élevées", "Capital: ~$10-20K"],
    "deneigement": ["Permis municipal possible", "Capital: ~$5-15K (équipement)"],
    "paysagement": ["Aucune licence requise", "Capital: ~$5-10K"],
    "menage": ["Aucune licence requise", "Capital minimal: ~$500-2K"],
    "demenagement": ["Permis CNESST", "Capital: ~$10-20K (camion)"],
    "peinture": ["Licence RBQ pour commercial", "Capital: ~$2-5K"],
    "comptabilite": ["CPA pour audit/certification", "Capital minimal"],
    "traiteur_cuisine": ["Permis MAPAQ", "Permis municipal", "Capital: ~$5-15K"],
    "sante_bienetre": ["Certification professionnelle", "Ordre professionnel possible"],
    "mecanique_auto": ["Certificat ASE/CPA", "Capital: ~$20-50K"],
    "animaux": ["Aucune licence spécifique", "Assurance RC", "Capital: ~$2-5K"],
    "informatique": ["Aucune licence requise", "Capital minimal: ~$1-3K"],
    "evenementiel": ["Aucune licence requise", "Capital: ~$3-10K (équipement)"],
    "education_tutorat": ["Aucune licence requise", "Capital minimal"],
    "excavation": ["Licence RBQ", "Capital: ~$50-100K (machinerie)"],
    "soudure_metal": ["Certification CWB", "Capital: ~$10-20K"],
}

# Market size estimates (annual revenue per active demand unit)
MARKET_MULTIPLIERS = {
    "construction": 25000,
    "plomberie": 5000,
    "electricite": 5000,
    "thermopompe_climatisation": 15000,
    "toiture": 12000,
    "deneigement": 3000,
    "paysagement": 4000,
    "menage": 2000,
    "demenagement": 3000,
    "peinture": 4000,
    "comptabilite": 3000,
    "traiteur_cuisine": 5000,
    "sante_bienetre": 4000,
    "mecanique_auto": 5000,
    "animaux": 2000,
    "informatique": 3000,
    "evenementiel": 4000,
    "education_tutorat": 2000,
    "excavation": 20000,
    "soudure_metal": 8000,
}


class ConvergenceDetector:
    """Detects opportunity convergence from multiple signal types. Pure math."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def detect_convergence(self, mrc: str, category: str,
                           demand_data: dict, supply_data: dict) -> dict:
        """
        Detect signal convergence for a specific MRC/category combination.
        Returns convergence analysis with signals and score.
        """
        signals = []

        # Signal 1: High demand
        if demand_data.get("count_30d", 0) >= DEMAND_HIGH_THRESHOLD:
            signals.append(f"Haute demande: {demand_data['count_30d']} requêtes/mois")
        elif demand_data.get("count_30d", 0) >= DEMAND_HIGH_THRESHOLD / 2:
            signals.append(f"Demande modérée: {demand_data['count_30d']} requêtes/mois")

        # Signal 2: Low supply
        if supply_data.get("provider_count", 0) < SUPPLY_LOW_THRESHOLD:
            signals.append(f"Faible offre: {supply_data['provider_count']} fournisseurs actifs")

        # Signal 3: Regulatory driver (subsidy)
        subsidies = self.kb.get_active_subsidies(category)
        has_subsidy = len(subsidies) > 0
        if has_subsidy:
            names = [s["name"] for s in subsidies]
            signals.append(f"Programme actif: {', '.join(names)}")

        # Signal 4: Growth trend
        growth_rate = demand_data.get("growth_rate", 0)
        if growth_rate >= GROWTH_THRESHOLD:
            signals.append(f"Croissance: +{growth_rate*100:.0f}% vs période précédente")

        # Signal 5: Quality gap
        quality_gap = supply_data.get("quality_gap", 0)
        if quality_gap >= 3.0:
            signals.append(f"Écart qualité: note moyenne {supply_data.get('avg_rating', 0):.1f}/5")

        # Signal 6: Urgency signals
        urgency_ratio = demand_data.get("urgency_ratio", 0)
        if urgency_ratio >= 0.3:
            signals.append(f"Urgence élevée: {urgency_ratio*100:.0f}% des demandes urgentes")

        convergence_count = len(signals)

        # Market estimation
        demand_count = demand_data.get("count_30d", 0)
        multiplier = MARKET_MULTIPLIERS.get(category, 5000)
        market_estimate = demand_count * multiplier * 12 // 12  # Annual

        # Window estimation (months)
        if convergence_count >= 4:
            window = "12-24 mois"
        elif convergence_count >= 3:
            window = "24-36 mois"
        else:
            window = "36+ mois"

        barriers = CATEGORY_BARRIERS.get(category, ["Information non disponible"])

        return {
            "convergence_count": convergence_count,
            "convergence_signals": signals,
            "is_high_confidence": convergence_count >= MIN_CONVERGENCE_SIGNALS,
            "has_subsidy": has_subsidy,
            "subsidy_programs": [s["name"] for s in subsidies],
            "demand_count": demand_count,
            "supply_count": supply_data.get("provider_count", 0),
            "growth_rate": growth_rate,
            "barriers": barriers,
            "market_estimate": f"${market_estimate:,}/an" if market_estimate > 0 else "À évaluer",
            "window_months": window,
            "region": "",  # Will be set by caller
        }
