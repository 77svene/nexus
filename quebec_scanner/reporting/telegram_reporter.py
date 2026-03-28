"""
Telegram Reporter - Sends opportunity alerts and daily digests.
Uses python-telegram-bot library.
"""

import json
import logging
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ALERT_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Category display names (French)
CATEGORY_NAMES_FR = {
    "construction": "Construction / Rénovation",
    "plomberie": "Plomberie",
    "electricite": "Électricité",
    "thermopompe_climatisation": "Thermopompe / Climatisation",
    "toiture": "Toiture / Couverture",
    "deneigement": "Déneigement",
    "paysagement": "Paysagement",
    "menage": "Entretien ménager",
    "demenagement": "Déménagement",
    "peinture": "Peinture",
    "comptabilite": "Comptabilité / Impôts",
    "traiteur_cuisine": "Traiteur / Cuisine",
    "sante_bienetre": "Santé / Bien-être",
    "mecanique_auto": "Mécanique auto",
    "animaux": "Services animaliers",
    "informatique": "Informatique / Tech",
    "evenementiel": "Événementiel",
    "education_tutorat": "Éducation / Tutorat",
    "excavation": "Excavation",
    "soudure_metal": "Soudure / Métal",
}


class TelegramReporter:
    """Sends Telegram notifications for business opportunities."""

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning(
                "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.\n"
                "To get these:\n"
                "1. Message @BotFather on Telegram, send /newbot\n"
                "2. Copy the token\n"
                "3. Message your bot, then visit: https://api.telegram.org/bot<TOKEN>/getUpdates\n"
                "4. Find your chat_id in the response"
            )

    async def _send_message(self, text: str, parse_mode: str = "HTML"):
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            logger.info(f"[telegram] (disabled) Would send:\n{text[:200]}...")
            return False

        try:
            import httpx
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                })
                if resp.status_code == 200:
                    logger.info("[telegram] Message sent successfully")
                    return True
                else:
                    logger.error(f"[telegram] Send failed: {resp.status_code} {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"[telegram] Error sending message: {e}")
            return False

    def _send_message_sync(self, text: str):
        """Synchronous version of send message."""
        if not self.enabled:
            logger.info(f"[telegram] (disabled) Would send:\n{text[:300]}...")
            return False

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=30)
            if resp.status_code == 200:
                logger.info("[telegram] Message sent successfully")
                return True
            else:
                logger.error(f"[telegram] Send failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"[telegram] Error sending message: {e}")
            return False

    def format_opportunity_alert(self, opp: dict) -> str:
        """Format a single opportunity as a Telegram message."""
        category_name = CATEGORY_NAMES_FR.get(opp.get("category", ""), opp.get("category", ""))
        mrc = opp.get("mrc", "")
        region = opp.get("region", "")
        total_score = opp.get("total_score", 0)
        convergence_count = opp.get("convergence_count", 0)

        # Parse JSON fields
        try:
            signals = json.loads(opp.get("convergence_signals", "[]"))
        except (json.JSONDecodeError, TypeError):
            signals = []

        try:
            barriers = json.loads(opp.get("barriers", "[]"))
        except (json.JSONDecodeError, TypeError):
            barriers = []

        try:
            subsidy_programs = json.loads(opp.get("subsidy_programs", "[]"))
        except (json.JSONDecodeError, TypeError):
            subsidy_programs = []

        demand_count = opp.get("demand_count", 0)
        supply_count = opp.get("supply_count", 0)
        growth_rate = opp.get("growth_rate", 0)
        market_estimate = opp.get("market_estimate", "À évaluer")
        window = opp.get("window_months", "")

        # Build message
        if total_score >= HIGH_CONFIDENCE_THRESHOLD:
            header = "🍁 QUÉBEC SCANNER - HAUTE CONFIANCE"
        else:
            header = "🍁 QUÉBEC SCANNER ALERT"

        msg = f"""<b>{header}</b>

📍 MRC: <b>{mrc}</b>{f' ({region})' if region else ''}
🎯 Opportunité: <b>{category_name}</b>

⭐ Score: <b>{total_score:.1f}/10</b>
🔗 Convergence: {convergence_count} signaux

<b>DEMANDE:</b>
• {demand_count} requêtes/mois
• Croissance: {'+' if growth_rate >= 0 else ''}{growth_rate*100:.0f}% vs période précédente

<b>OFFRE:</b>
• {supply_count} fournisseurs actifs"""

        if signals:
            msg += "\n\n<b>SIGNAUX:</b>"
            for s in signals:
                msg += f"\n• {s}"

        if subsidy_programs:
            msg += "\n\n<b>CATALYSEURS:</b>"
            for p in subsidy_programs:
                msg += f"\n• Programme: {p}"

        if barriers:
            msg += "\n\n<b>BARRIÈRES:</b>"
            for b in barriers:
                msg += f"\n• {b}"

        if market_estimate or window:
            msg += "\n\n<b>MARCHÉ:</b>"
            if market_estimate:
                msg += f"\n• Estimé: {market_estimate}"
            if window:
                msg += f"\n• Fenêtre: {window}"

        msg += f"\n\n─────────────────────────"
        msg += f"\nDétecté: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return msg

    def send_opportunity_alert(self, opp: dict) -> bool:
        """Send alert for a single opportunity."""
        msg = self.format_opportunity_alert(opp)
        success = self._send_message_sync(msg)
        if success and self.kb and opp.get("id"):
            self.kb.mark_alerted(opp["id"])
        return success

    def send_daily_digest(self, opportunities: list) -> bool:
        """Send a daily summary of top opportunities."""
        if not opportunities:
            msg = ("🍁 <b>QUÉBEC SCANNER - Résumé quotidien</b>\n\n"
                   "Aucune nouvelle opportunité détectée aujourd'hui.\n"
                   f"Prochaine analyse dans 6 heures.\n"
                   f"\n{datetime.now().strftime('%Y-%m-%d %H:%M')}")
            return self._send_message_sync(msg)

        msg = f"🍁 <b>QUÉBEC SCANNER - Résumé quotidien</b>\n\n"
        msg += f"📊 {len(opportunities)} opportunités détectées:\n\n"

        for i, opp in enumerate(opportunities[:10], 1):
            cat_name = CATEGORY_NAMES_FR.get(opp.get("category", ""), opp.get("category", ""))
            score = opp.get("total_score", 0)
            mrc = opp.get("mrc", "")
            conv = opp.get("convergence_count", 0)

            emoji = "🔴" if score >= 8 else "🟡" if score >= 7 else "🟢"
            msg += f"{emoji} #{i} <b>{cat_name}</b> @ {mrc}\n"
            msg += f"   Score: {score:.1f}/10 | {conv} signaux\n\n"

        msg += f"─────────────────────────\n"
        msg += f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"

        return self._send_message_sync(msg)

    def send_startup_message(self) -> bool:
        """Send a message when the scanner starts."""
        msg = ("🍁 <b>QUÉBEC SCANNER - Démarré</b>\n\n"
               "Le scanner autonome est maintenant actif.\n"
               "• Analyse toutes les 6 heures\n"
               "• Alertes pour score ≥ 7.0\n"
               f"\n{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return self._send_message_sync(msg)

    def send_new_alerts(self) -> int:
        """Check for and send any unalerted high-score opportunities."""
        if not self.kb:
            return 0

        alerts = self.kb.get_unalerted_opportunities(ALERT_THRESHOLD)
        sent = 0
        for opp in alerts:
            if self.send_opportunity_alert(dict(opp)):
                sent += 1

        if sent:
            logger.info(f"[telegram] Sent {sent} new opportunity alerts")
        return sent
