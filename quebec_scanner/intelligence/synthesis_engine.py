"""
Cross-Domain Synthesis Engine - Finds non-obvious opportunity patterns.
Detects cascade effects, demographic shifts, infrastructure impacts,
and regulatory arbitrage opportunities.
100% deterministic - NO LLM.
"""

import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES

logger = logging.getLogger(__name__)


# ─── Cascade Effect Maps ───
# When one event happens, it triggers demand for related services
CASCADE_MAPS = {
    "new_housing_development": {
        "trigger": "Nouveau développement résidentiel",
        "immediate": [
            ("construction", 1.0),
            ("excavation", 0.8),
            ("electricite", 0.7),
            ("plomberie", 0.7),
        ],
        "short_term": [  # 1-6 months after
            ("paysagement", 0.6),
            ("peinture", 0.5),
            ("demenagement", 0.4),
            ("menage", 0.3),
        ],
        "long_term": [  # 6-24 months after
            ("deneigement", 0.5),
            ("animaux", 0.2),
            ("education_tutorat", 0.2),
            ("sante_bienetre", 0.2),
        ],
    },
    "population_aging": {
        "trigger": "Vieillissement de la population",
        "immediate": [
            ("sante_bienetre", 0.8),
            ("menage", 0.6),
        ],
        "short_term": [
            ("construction", 0.5),  # Accessibility modifications
            ("deneigement", 0.4),  # Can't do it themselves
            ("traiteur_cuisine", 0.3),  # Meal delivery
        ],
        "long_term": [
            ("comptabilite", 0.3),  # Estate planning
            ("demenagement", 0.4),  # Downsizing
            ("paysagement", 0.3),  # Can't maintain
        ],
    },
    "employer_closure": {
        "trigger": "Fermeture d'employeur majeur",
        "immediate": [
            ("comptabilite", 0.6),
            ("education_tutorat", 0.4),  # Retraining
        ],
        "short_term": [
            ("demenagement", 0.5),
            ("mecanique_auto", -0.2),  # Less spending
        ],
        "long_term": [
            ("construction", -0.3),  # Less building activity
        ],
    },
    "new_employer_arrival": {
        "trigger": "Arrivée nouvel employeur majeur",
        "immediate": [
            ("construction", 0.8),  # Facility build-out
            ("electricite", 0.5),
        ],
        "short_term": [
            ("demenagement", 0.6),  # Workers relocating
            ("construction", 0.5),  # New housing
            ("menage", 0.3),
        ],
        "long_term": [
            ("paysagement", 0.4),
            ("traiteur_cuisine", 0.3),
            ("animaux", 0.2),
            ("education_tutorat", 0.3),
            ("sante_bienetre", 0.3),
        ],
    },
    "environmental_regulation": {
        "trigger": "Nouvelle réglementation environnementale",
        "immediate": [
            ("thermopompe_climatisation", 0.9),
            ("construction", 0.5),
        ],
        "short_term": [
            ("electricite", 0.4),
            ("plomberie", 0.3),
        ],
        "long_term": [
            ("excavation", 0.2),  # Soil remediation
        ],
    },
    "infrastructure_project": {
        "trigger": "Projet d'infrastructure majeur",
        "immediate": [
            ("excavation", 0.9),
            ("construction", 0.8),
        ],
        "short_term": [
            ("electricite", 0.6),
            ("plomberie", 0.5),
            ("soudure_metal", 0.4),
        ],
        "long_term": [
            ("paysagement", 0.5),
            ("menage", 0.3),
            ("deneigement", 0.2),
        ],
    },
    "real_estate_boom": {
        "trigger": "Boom immobilier",
        "immediate": [
            ("construction", 0.9),
            ("toiture", 0.5),
            ("peinture", 0.5),
        ],
        "short_term": [
            ("demenagement", 0.7),
            ("menage", 0.5),
            ("paysagement", 0.5),
            ("electricite", 0.4),
            ("plomberie", 0.4),
        ],
        "long_term": [
            ("deneigement", 0.3),
            ("comptabilite", 0.2),
        ],
    },
}

# ─── Service Complementarity Map ───
# If category A is in demand, category B likely follows
COMPLEMENTARY_SERVICES = {
    "construction": ["electricite", "plomberie", "peinture", "paysagement", "excavation"],
    "toiture": ["peinture", "construction", "soudure_metal"],
    "thermopompe_climatisation": ["electricite", "plomberie"],
    "paysagement": ["excavation", "deneigement", "construction"],
    "demenagement": ["menage", "peinture", "construction"],
    "excavation": ["construction", "plomberie", "paysagement"],
    "soudure_metal": ["construction", "toiture"],
}

# ─── Demographic Impact Categories ───
DEMOGRAPHIC_IMPACTS = {
    "young_families": {
        "services": ["education_tutorat", "animaux", "menage", "construction", "paysagement"],
        "indicators": ["school_enrollment_up", "daycare_waitlists", "family_housing_demand"],
    },
    "seniors": {
        "services": ["sante_bienetre", "menage", "deneigement", "traiteur_cuisine", "construction"],
        "indicators": ["65plus_pct_rising", "healthcare_visits_up", "home_care_requests"],
    },
    "remote_workers": {
        "services": ["informatique", "construction", "menage", "traiteur_cuisine"],
        "indicators": ["home_office_renovations", "internet_speed_complaints", "coworking_demand"],
    },
}


class SynthesisEngine:
    """Discovers non-obvious opportunity patterns through cross-domain analysis."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def detect_cascade_opportunities(self, mrc: str) -> list:
        """
        Find cascade opportunities: when one signal implies future demand for related services.
        """
        cascades = []

        # Get current high-demand categories in this MRC
        high_demand = {}
        for category in BUSINESS_CATEGORIES:
            count = self.kb.get_demand_count(mrc, category, days=30)
            if count >= 3:
                high_demand[category] = count

        if not high_demand:
            return []

        # Check which cascades match current signals
        for cascade_name, cascade in CASCADE_MAPS.items():
            # See if any immediate signals match high demand
            immediate_cats = [cat for cat, _ in cascade["immediate"]]
            matching = [cat for cat in high_demand if cat in immediate_cats]

            if not matching:
                continue

            # This cascade is likely active - predict next phases
            for phase_name in ["short_term", "long_term"]:
                for cat, multiplier in cascade[phase_name]:
                    if multiplier <= 0:
                        continue  # Skip negative impacts for now

                    current_supply = self.kb.get_supply_count(mrc, cat)
                    estimated_additional_demand = sum(high_demand[m] for m in matching) * multiplier

                    if estimated_additional_demand > current_supply * 0.3:
                        timing = "1-6 mois" if phase_name == "short_term" else "6-24 mois"
                        cascades.append({
                            "type": "cascade",
                            "cascade_name": cascade_name,
                            "trigger": cascade["trigger"],
                            "triggered_by": matching,
                            "opportunity_category": cat,
                            "timing": timing,
                            "estimated_demand_increase": round(estimated_additional_demand, 1),
                            "current_supply": current_supply,
                            "confidence": min(len(matching) / 2, 1.0),
                            "description": (
                                f"La forte demande en {', '.join(matching)} indique un effet cascade "
                                f"pour {cat} dans {timing}."
                            ),
                        })

        cascades.sort(key=lambda c: c["confidence"], reverse=True)
        return cascades

    def detect_complementary_gaps(self, mrc: str) -> list:
        """
        Find services that SHOULD be in demand based on complementary categories.
        If construction is booming but nobody's asking for electricians = hidden opportunity.
        """
        gaps = []

        for primary_cat, complement_cats in COMPLEMENTARY_SERVICES.items():
            primary_demand = self.kb.get_demand_count(mrc, primary_cat, days=30)

            if primary_demand < 2:
                continue

            for comp_cat in complement_cats:
                comp_demand = self.kb.get_demand_count(mrc, comp_cat, days=30)
                comp_supply = self.kb.get_supply_count(mrc, comp_cat)

                # Expected demand should be proportional to primary
                expected_demand = primary_demand * 0.5

                if comp_demand < expected_demand and comp_supply < 5:
                    gaps.append({
                        "type": "complementary_gap",
                        "primary_category": primary_cat,
                        "primary_demand": primary_demand,
                        "gap_category": comp_cat,
                        "observed_demand": comp_demand,
                        "expected_demand": round(expected_demand, 1),
                        "supply": comp_supply,
                        "description": (
                            f"Forte demande en {primary_cat} ({primary_demand}/mois) mais "
                            f"faible demande visible en {comp_cat} ({comp_demand}/mois). "
                            f"Demande latente probable."
                        ),
                    })

        gaps.sort(key=lambda g: g["expected_demand"] - g["observed_demand"], reverse=True)
        return gaps

    def detect_regional_arbitrage(self) -> list:
        """
        Find opportunities where a service is oversupplied in one MRC but undersupplied nearby.
        Someone could serve the undersupplied MRC from the oversupplied one.
        """
        arbitrage = []

        # Get all MRCs with data
        with self.kb._get_conn() as conn:
            mrcs = [r["mrc"] for r in
                    conn.execute("SELECT DISTINCT mrc FROM demand_signals UNION SELECT DISTINCT mrc FROM supply_providers").fetchall()]

        for category in BUSINESS_CATEGORIES:
            mrc_data = {}
            for mrc in mrcs:
                demand = self.kb.get_demand_count(mrc, category, days=30)
                supply = self.kb.get_supply_count(mrc, category)
                if demand > 0 or supply > 0:
                    mrc_data[mrc] = {"demand": demand, "supply": supply,
                                      "ratio": demand / max(supply, 1)}

            if len(mrc_data) < 2:
                continue

            # Find pairs with significant imbalance
            sorted_mrcs = sorted(mrc_data.items(), key=lambda x: x[1]["ratio"], reverse=True)

            for i, (high_mrc, high_data) in enumerate(sorted_mrcs[:3]):
                for low_mrc, low_data in sorted_mrcs[-3:]:
                    if high_mrc == low_mrc:
                        continue
                    if high_data["ratio"] > 2 * low_data["ratio"] and high_data["demand"] >= 2:
                        arbitrage.append({
                            "type": "regional_arbitrage",
                            "category": category,
                            "undersupplied_mrc": high_mrc,
                            "undersupplied_demand": high_data["demand"],
                            "undersupplied_supply": high_data["supply"],
                            "oversupplied_mrc": low_mrc,
                            "oversupplied_demand": low_data["demand"],
                            "oversupplied_supply": low_data["supply"],
                            "description": (
                                f"{category}: {high_mrc} a {high_data['demand']} demandes pour "
                                f"{high_data['supply']} fournisseurs, tandis que {low_mrc} a "
                                f"plus de capacité. Opportunité de servir {high_mrc} depuis {low_mrc}."
                            ),
                        })

        return arbitrage

    def detect_underserved_demographics(self, mrc: str, mrc_profile: dict = None) -> list:
        """
        Detect service gaps based on demographic patterns of the MRC.
        """
        opportunities = []

        if not mrc_profile:
            return opportunities

        economic_profile = mrc_profile.get("economic_profile", "").lower()
        population = mrc_profile.get("population", 0)

        # Check if tourism-heavy area needs event services
        if "tourisme" in economic_profile or "villégiature" in economic_profile:
            event_supply = self.kb.get_supply_count(mrc, "evenementiel")
            traiteur_supply = self.kb.get_supply_count(mrc, "traiteur_cuisine")

            if event_supply < 3:
                opportunities.append({
                    "type": "demographic_gap",
                    "category": "evenementiel",
                    "driver": "Zone touristique avec peu de services événementiels",
                    "supply": event_supply,
                    "population": population,
                })
            if traiteur_supply < 3:
                opportunities.append({
                    "type": "demographic_gap",
                    "category": "traiteur_cuisine",
                    "driver": "Zone touristique avec peu de traiteurs",
                    "supply": traiteur_supply,
                    "population": population,
                })

        # Large population + low service supply
        if population > 50000:
            for category in BUSINESS_CATEGORIES:
                supply = self.kb.get_supply_count(mrc, category)
                # Expected supply based on population (rough: 1 per 10K people)
                expected = population / 10000
                if supply < expected * 0.3:
                    opportunities.append({
                        "type": "population_underserved",
                        "category": category,
                        "driver": f"Population de {population:,} avec seulement {supply} fournisseurs",
                        "supply": supply,
                        "expected_supply": round(expected, 1),
                        "population": population,
                    })

        # Agriculture-heavy areas need specialized services
        if "agriculture" in economic_profile or "agroalimentaire" in economic_profile:
            mech_supply = self.kb.get_supply_count(mrc, "mecanique_auto")
            soudure_supply = self.kb.get_supply_count(mrc, "soudure_metal")
            if mech_supply < 3:
                opportunities.append({
                    "type": "economic_profile_gap",
                    "category": "mecanique_auto",
                    "driver": "Zone agricole avec peu de mécaniciens (machinerie agricole)",
                    "supply": mech_supply,
                })
            if soudure_supply < 2:
                opportunities.append({
                    "type": "economic_profile_gap",
                    "category": "soudure_metal",
                    "driver": "Zone agricole avec peu de soudeurs (réparation équipement)",
                    "supply": soudure_supply,
                })

        # Industrial/manufacturing areas
        if "manufacturier" in economic_profile or "industrie" in economic_profile:
            for cat in ["soudure_metal", "electricite", "mecanique_auto"]:
                supply = self.kb.get_supply_count(mrc, cat)
                if supply < 3:
                    opportunities.append({
                        "type": "economic_profile_gap",
                        "category": cat,
                        "driver": f"Zone industrielle/manufacturière avec peu de {cat}",
                        "supply": supply,
                    })

        return opportunities

    def run_full_synthesis(self, mrc: str, mrc_profile: dict = None) -> dict:
        """Run all synthesis analyses for an MRC."""
        return {
            "mrc": mrc,
            "cascades": self.detect_cascade_opportunities(mrc),
            "complementary_gaps": self.detect_complementary_gaps(mrc),
            "demographic_gaps": self.detect_underserved_demographics(mrc, mrc_profile),
            "generated_at": datetime.now().isoformat(),
        }

    def run_global_synthesis(self) -> dict:
        """Run cross-MRC synthesis."""
        return {
            "regional_arbitrage": self.detect_regional_arbitrage(),
            "generated_at": datetime.now().isoformat(),
        }
