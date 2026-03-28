"""
Temporal Predictor - Forecasts demand/supply 60-180 days ahead.
Uses leading indicators, seasonal patterns, and trend analysis.
100% deterministic - NO LLM.
"""

import logging
import math
from datetime import datetime, timedelta
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES

logger = logging.getLogger(__name__)


# ─── Seasonal demand multipliers by category and month ───
# 1.0 = baseline, >1.0 = above average, <1.0 = below average
SEASONAL_PATTERNS = {
    "construction": {
        1: 0.4, 2: 0.5, 3: 0.7, 4: 1.2, 5: 1.5, 6: 1.4,
        7: 1.3, 8: 1.3, 9: 1.2, 10: 0.9, 11: 0.5, 12: 0.3
    },
    "plomberie": {
        1: 1.3, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8, 6: 0.8,
        7: 0.7, 8: 0.7, 9: 0.9, 10: 1.0, 11: 1.2, 12: 1.4
    },
    "electricite": {
        1: 0.7, 2: 0.7, 3: 0.9, 4: 1.1, 5: 1.2, 6: 1.2,
        7: 1.1, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.7
    },
    "thermopompe_climatisation": {
        1: 0.5, 2: 0.6, 3: 0.8, 4: 1.1, 5: 1.5, 6: 1.8,
        7: 1.6, 8: 1.3, 9: 1.2, 10: 1.0, 11: 0.6, 12: 0.4
    },
    "toiture": {
        1: 0.3, 2: 0.3, 3: 0.6, 4: 1.3, 5: 1.6, 6: 1.4,
        7: 1.2, 8: 1.2, 9: 1.3, 10: 1.0, 11: 0.5, 12: 0.2
    },
    "deneigement": {
        1: 1.6, 2: 1.5, 3: 1.3, 4: 0.5, 5: 0.1, 6: 0.0,
        7: 0.0, 8: 0.0, 9: 0.3, 10: 1.0, 11: 1.5, 12: 1.7
    },
    "paysagement": {
        1: 0.1, 2: 0.2, 3: 0.5, 4: 1.4, 5: 1.8, 6: 1.5,
        7: 1.2, 8: 1.0, 9: 1.2, 10: 1.0, 11: 0.3, 12: 0.1
    },
    "menage": {
        1: 0.8, 2: 0.9, 3: 1.1, 4: 1.3, 5: 1.2, 6: 1.0,
        7: 0.8, 8: 0.8, 9: 1.2, 10: 1.0, 11: 0.9, 12: 1.1
    },
    "demenagement": {
        1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8, 6: 1.4,
        7: 2.0, 8: 1.3, 9: 1.0, 10: 0.6, 11: 0.4, 12: 0.3
    },
    "peinture": {
        1: 0.4, 2: 0.5, 3: 0.7, 4: 1.0, 5: 1.3, 6: 1.5,
        7: 1.4, 8: 1.3, 9: 1.1, 10: 0.8, 11: 0.5, 12: 0.3
    },
    "comptabilite": {
        1: 1.0, 2: 1.5, 3: 1.8, 4: 2.0, 5: 1.0, 6: 0.7,
        7: 0.5, 8: 0.5, 9: 0.7, 10: 0.8, 11: 0.9, 12: 1.2
    },
    "traiteur_cuisine": {
        1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.2, 6: 1.5,
        7: 1.3, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.9, 12: 1.5
    },
    "sante_bienetre": {
        1: 1.3, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 0.8,
        7: 0.8, 8: 0.8, 9: 1.1, 10: 1.0, 11: 1.0, 12: 1.0
    },
    "mecanique_auto": {
        1: 0.9, 2: 0.8, 3: 1.0, 4: 1.2, 5: 1.1, 6: 1.0,
        7: 0.9, 8: 0.9, 9: 1.0, 10: 1.3, 11: 1.2, 12: 1.0
    },
    "animaux": {
        1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.3,
        7: 1.4, 8: 1.3, 9: 1.0, 10: 0.9, 11: 0.8, 12: 1.1
    },
    "informatique": {
        1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.9,
        7: 0.8, 8: 0.9, 9: 1.2, 10: 1.1, 11: 1.0, 12: 1.0
    },
    "evenementiel": {
        1: 0.5, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.2, 6: 1.6,
        7: 1.5, 8: 1.4, 9: 1.3, 10: 1.0, 11: 0.7, 12: 1.3
    },
    "education_tutorat": {
        1: 1.2, 2: 1.1, 3: 1.0, 4: 1.1, 5: 1.0, 6: 0.5,
        7: 0.4, 8: 0.8, 9: 1.5, 10: 1.3, 11: 1.2, 12: 0.9
    },
    "excavation": {
        1: 0.2, 2: 0.2, 3: 0.4, 4: 1.0, 5: 1.5, 6: 1.6,
        7: 1.5, 8: 1.5, 9: 1.3, 10: 0.8, 11: 0.3, 12: 0.1
    },
    "soudure_metal": {
        1: 0.7, 2: 0.7, 3: 0.8, 4: 1.0, 5: 1.2, 6: 1.3,
        7: 1.2, 8: 1.2, 9: 1.1, 10: 0.9, 11: 0.7, 12: 0.6
    },
}

# Categories with known prep windows (when to start marketing/preparing)
PREP_WINDOWS = {
    "deneigement": {"marketing_start_month": 8, "contract_signing_peak": 10,
                     "demand_peak": 12, "note": "Commencer marketing en août pour contrats d'hiver"},
    "thermopompe_climatisation": {"marketing_start_month": 3, "contract_signing_peak": 5,
                                   "demand_peak": 6, "note": "Pré-été: marketing mars-avril"},
    "paysagement": {"marketing_start_month": 2, "contract_signing_peak": 4,
                     "demand_peak": 5, "note": "Pré-printemps: marketing février-mars"},
    "toiture": {"marketing_start_month": 2, "contract_signing_peak": 4,
                 "demand_peak": 5, "note": "Post-hiver: inspections en mars"},
    "demenagement": {"marketing_start_month": 4, "contract_signing_peak": 6,
                      "demand_peak": 7, "note": "1er juillet: pic au Québec"},
    "comptabilite": {"marketing_start_month": 1, "contract_signing_peak": 2,
                      "demand_peak": 4, "note": "Saison d'impôts: janvier-avril"},
}


class TemporalPredictor:
    """Forecasts future opportunities using seasonal patterns and trend analysis."""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def get_seasonal_multiplier(self, category: str, month: int) -> float:
        """Get the seasonal demand multiplier for a category in a given month."""
        pattern = SEASONAL_PATTERNS.get(category, {})
        return pattern.get(month, 1.0)

    def predict_demand(self, mrc: str, category: str, horizon_days: int = 90) -> dict:
        """
        Predict future demand for a service in an MRC.
        Uses: current trend + seasonal patterns + leading indicators.
        """
        now = datetime.now()
        target_date = now + timedelta(days=horizon_days)
        target_month = target_date.month

        # Current demand baseline
        current_30d = self.kb.get_demand_count(mrc, category, days=30)
        prev_30d = self.kb.get_demand_count_period(mrc, category, 90, 30)

        # Trend component (linear extrapolation)
        if prev_30d > 0:
            monthly_growth = (current_30d - prev_30d) / 2  # avg monthly change
        else:
            monthly_growth = 0

        months_ahead = horizon_days / 30
        trend_forecast = current_30d + (monthly_growth * months_ahead)

        # Seasonal component
        current_seasonal = self.get_seasonal_multiplier(category, now.month)
        future_seasonal = self.get_seasonal_multiplier(category, target_month)

        if current_seasonal > 0:
            seasonal_adjustment = future_seasonal / current_seasonal
        else:
            seasonal_adjustment = future_seasonal

        # Adjusted forecast
        predicted = max(0, trend_forecast * seasonal_adjustment)

        # Confidence based on data availability
        data_points = current_30d + prev_30d
        confidence = min(data_points / 20, 1.0)  # Need 20+ data points for full confidence

        # Determine if this is an upcoming peak
        is_upcoming_peak = future_seasonal > 1.2 and future_seasonal > current_seasonal

        return {
            "mrc": mrc,
            "category": category,
            "horizon_days": horizon_days,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "current_demand_30d": current_30d,
            "predicted_demand_30d": round(predicted, 1),
            "trend_component": round(trend_forecast, 1),
            "seasonal_multiplier": round(future_seasonal, 2),
            "seasonal_adjustment": round(seasonal_adjustment, 2),
            "confidence": round(confidence, 2),
            "is_upcoming_peak": is_upcoming_peak,
            "growth_rate_monthly": round(monthly_growth, 2),
        }

    def detect_upcoming_peaks(self, mrc: str, look_ahead_months: int = 6) -> list:
        """Find categories with upcoming seasonal demand peaks."""
        peaks = []
        now = datetime.now()

        for category in BUSINESS_CATEGORIES:
            for months_ahead in range(1, look_ahead_months + 1):
                future_month = ((now.month - 1 + months_ahead) % 12) + 1
                multiplier = self.get_seasonal_multiplier(category, future_month)
                current_multiplier = self.get_seasonal_multiplier(category, now.month)

                if multiplier > 1.3 and multiplier > current_multiplier * 1.2:
                    prep_info = PREP_WINDOWS.get(category, {})
                    peaks.append({
                        "category": category,
                        "peak_month": future_month,
                        "months_away": months_ahead,
                        "multiplier": round(multiplier, 2),
                        "current_multiplier": round(current_multiplier, 2),
                        "increase_pct": round((multiplier / max(current_multiplier, 0.1) - 1) * 100),
                        "prep_window": prep_info.get("note", ""),
                        "action": self._recommend_action(category, months_ahead),
                    })

        # Sort by months away (soonest first), then by multiplier
        peaks.sort(key=lambda p: (p["months_away"], -p["multiplier"]))
        return peaks

    def _recommend_action(self, category: str, months_away: int) -> str:
        """Generate actionable recommendation based on timing."""
        if months_away <= 1:
            return "URGENT: Pic imminent. Dernier moment pour préparer."
        elif months_away <= 2:
            return "ACTION: Commencer marketing et préparation maintenant."
        elif months_away <= 3:
            return "PLANIFIER: Bon moment pour se positionner et planifier."
        elif months_away <= 4:
            return "PRÉPARER: Investir dans équipement et formation."
        else:
            return "SURVEILLER: Observer le marché et préparer stratégie."

    def detect_demand_catalysts(self, mrc: str) -> list:
        """
        Detect events that will create future demand spikes.
        Based on subsidies, regulations, seasonal patterns.
        """
        catalysts = []

        # 1. Active subsidies create demand
        for category in BUSINESS_CATEGORIES:
            subsidies = self.kb.get_active_subsidies(category)
            for sub in subsidies:
                catalysts.append({
                    "type": "subsidy_active",
                    "category": category,
                    "event": f"Programme {sub['name']} actif",
                    "impact": "high",
                    "description": f"Le programme {sub['name']} génère de la demande pour {category}",
                    "timing": "ongoing",
                })

        # 2. Seasonal peaks approaching
        peaks = self.detect_upcoming_peaks(mrc, look_ahead_months=3)
        for peak in peaks:
            catalysts.append({
                "type": "seasonal_peak",
                "category": peak["category"],
                "event": f"Pic saisonnier dans {peak['months_away']} mois",
                "impact": "high" if peak["multiplier"] > 1.5 else "medium",
                "description": f"Demande {peak['category']} augmente de {peak['increase_pct']}%",
                "timing": f"{peak['months_away']} mois",
            })

        # 3. Known Quebec-specific catalysts
        now = datetime.now()

        # July 1st moving day (June-July are peak moving)
        if now.month in [4, 5]:
            catalysts.append({
                "type": "cultural_event",
                "category": "demenagement",
                "event": "1er juillet - Jour du déménagement au Québec",
                "impact": "critical",
                "description": "Le plus gros pic de déménagement annuel",
                "timing": f"{7 - now.month} mois",
            })

        # Tax season (Jan-April)
        if now.month in [11, 12]:
            catalysts.append({
                "type": "seasonal_regulatory",
                "category": "comptabilite",
                "event": "Saison d'impôts approche",
                "impact": "high",
                "description": "Forte demande comptabilité/impôts de janvier à avril",
                "timing": f"{(1 - now.month) % 12 + 1} mois",
            })

        # Winter prep (Oct-Nov)
        if now.month in [7, 8, 9]:
            catalysts.append({
                "type": "seasonal_regulatory",
                "category": "deneigement",
                "event": "Signature contrats déneigement",
                "impact": "high",
                "description": "Période de signature des contrats pour l'hiver",
                "timing": f"{10 - now.month} mois",
            })

        catalysts.sort(key=lambda c: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(c["impact"], 4))
        return catalysts

    def calculate_opportunity_window(self, mrc: str, category: str) -> dict:
        """
        Calculate when an opportunity opens and closes.
        Based on supply pipeline, demand trajectory, and competition entry rate.
        """
        current_demand = self.kb.get_demand_count(mrc, category, days=30)
        current_supply = self.kb.get_supply_count(mrc, category)
        prev_demand = self.kb.get_demand_count_period(mrc, category, 90, 30)

        # Demand trajectory
        if prev_demand > 0:
            demand_growth_rate = (current_demand - prev_demand) / prev_demand
        elif current_demand > 0:
            demand_growth_rate = 1.0
        else:
            demand_growth_rate = 0.0

        # Current gap
        gap = current_demand - (current_supply * 3)  # Each provider handles ~3 requests/month

        if gap <= 0:
            # Market is saturated or balanced
            # Estimate when it might open based on demand growth
            if demand_growth_rate > 0.1:
                months_to_open = max(1, int(abs(gap) / max(current_demand * demand_growth_rate, 1)))
                return {
                    "status": "FERMÉE",
                    "opens_in_months": months_to_open,
                    "reason": f"Marché équilibré, mais demande croît de {demand_growth_rate*100:.0f}%/mois",
                    "recommendation": f"Préparer entrée dans ~{months_to_open} mois",
                }
            return {
                "status": "FERMÉE",
                "opens_in_months": None,
                "reason": "Marché saturé, pas de croissance significative",
                "recommendation": "Chercher un autre marché ou niche",
            }

        # Market is undersupplied
        # Estimate how long before competition fills the gap
        # Assume 1 new competitor every 6 months per demand signal
        new_entrant_rate = max(current_demand * 0.05, 0.1)  # 5% of demand converts to new supply
        months_to_saturation = gap / max(new_entrant_rate, 0.01)
        months_to_saturation = min(months_to_saturation, 60)  # Cap at 5 years

        if months_to_saturation < 6:
            urgency = "CRITIQUE"
        elif months_to_saturation < 12:
            urgency = "ÉLEVÉE"
        elif months_to_saturation < 24:
            urgency = "MODÉRÉE"
        else:
            urgency = "FAIBLE"

        return {
            "status": "OUVERTE",
            "current_gap": gap,
            "months_until_saturation": round(months_to_saturation, 1),
            "urgency": urgency,
            "demand_growth": f"+{demand_growth_rate*100:.0f}%/mois",
            "recommendation": self._window_recommendation(urgency, category),
        }

    def _window_recommendation(self, urgency: str, category: str) -> str:
        if urgency == "CRITIQUE":
            return f"Entrer IMMÉDIATEMENT dans {category}. Fenêtre se ferme bientôt."
        elif urgency == "ÉLEVÉE":
            return f"Commencer préparation pour {category} dans les 30 prochains jours."
        elif urgency == "MODÉRÉE":
            return f"Bon moment pour planifier entrée dans {category}. 6-12 mois de préparation."
        else:
            return f"Opportunité stable pour {category}. Temps de bien se former et préparer."

    def get_full_forecast(self, mrc: str) -> dict:
        """Generate a complete temporal forecast for an MRC."""
        forecasts = {}
        peaks = self.detect_upcoming_peaks(mrc)
        catalysts = self.detect_demand_catalysts(mrc)

        for category in BUSINESS_CATEGORIES:
            forecast_90d = self.predict_demand(mrc, category, 90)
            forecast_180d = self.predict_demand(mrc, category, 180)
            window = self.calculate_opportunity_window(mrc, category)

            if forecast_90d["predicted_demand_30d"] > 0 or forecast_180d["predicted_demand_30d"] > 0:
                forecasts[category] = {
                    "forecast_90d": forecast_90d,
                    "forecast_180d": forecast_180d,
                    "window": window,
                }

        return {
            "mrc": mrc,
            "generated_at": datetime.now().isoformat(),
            "category_forecasts": forecasts,
            "upcoming_peaks": peaks,
            "catalysts": catalysts,
        }
