#!/usr/bin/env python3
"""
Quebec Business Opportunity Scanner - Main Orchestrator
Autonomous 24/7 scanner that finds supply/demand imbalances across Quebec.

Usage:
    python main.py                  # Run full autonomous loop
    python main.py --scan-once      # Run a single scan cycle
    python main.py --test           # Run quick test (Brome-Missisquoi only)
    python main.py --stats          # Show database stats
    python main.py --top            # Show top opportunities
"""

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DB_PATH, SCRAPE_INTERVAL_HOURS, PRIORITY_MRCS, LOG_DIR, LOG_FILE,
    KIJIJI_LOCATIONS, ALERT_THRESHOLD, BUSINESS_CATEGORIES
)
from storage.knowledge_base import KnowledgeBase
from llm.minimal_extraction import LLMExtractor
from scrapers.kijiji_scraper import KijijiScraper
from scrapers.google_maps_scraper import GoogleMapsScraper
from scrapers.req_scraper import REQScraper
from scrapers.reddit_scraper import RedditScraper
from scrapers.government_scraper import GovernmentScraper
from scrapers.lespac_scraper import LesPACScraper
from scrapers.facebook_scraper import FacebookScraper
from scrapers.rbq_scraper import RBQScraper
from scrapers.permits_scraper import PermitsScraper
from intelligence.opportunity_scorer import OpportunityScorer
from reporting.telegram_reporter import TelegramReporter

# ─── Logging Setup ───
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("quebec_scanner")

# Reduce noise from HTTP libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class QuebecScanner:
    """Main orchestrator for the Quebec business opportunity scanner."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("QUÉBEC BUSINESS OPPORTUNITY SCANNER")
        logger.info("=" * 60)

        # Initialize storage
        self.kb = KnowledgeBase(DB_PATH)
        logger.info(f"Database: {DB_PATH}")

        # Initialize LLM (optional - falls back to keywords)
        self.llm = LLMExtractor(self.kb)
        if self.llm.available:
            logger.info("LLM: Qwen2.5-9B via Ollama (connected)")
        else:
            logger.info("LLM: Not available (using keyword-only mode)")

        # Initialize scrapers
        self.kijiji = KijijiScraper(self.kb, self.llm)
        self.google = GoogleMapsScraper(self.kb)
        self.req = REQScraper(self.kb)
        self.reddit = RedditScraper(self.kb, self.llm)
        self.government = GovernmentScraper(self.kb)
        self.lespac = LesPACScraper(self.kb, self.llm)
        self.facebook = FacebookScraper(self.kb, self.llm)
        self.rbq = RBQScraper(self.kb)
        self.permits = PermitsScraper(self.kb)

        # Initialize intelligence
        self.scorer = OpportunityScorer(self.kb)

        # Initialize reporting
        self.telegram = TelegramReporter(self.kb)

        # Running state
        self._running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("All components initialized")

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received. Finishing current task...")
        self._running = False

    def seed_government_programs(self):
        """Seed the database with known government programs."""
        from config import GOVERNMENT_PROGRAMS
        for prog in GOVERNMENT_PROGRAMS:
            self.kb.add_government_program(
                name=prog["name"],
                url=prog["url"],
                sector=prog["sector"],
                description=prog["description"],
                is_active=True,
            )
        logger.info(f"Seeded {len(GOVERNMENT_PROGRAMS)} government programs")

    def run_demand_scrape(self, locations=None):
        """Run all demand-side scrapers."""
        logger.info("─── DEMAND SCRAPING ───")

        # Kijiji
        try:
            logger.info("[demand] Scraping Kijiji...")
            if locations:
                # Map MRC names to Kijiji locations
                kijiji_locs = [l for l in KIJIJI_LOCATIONS if l in locations or
                               any(l.lower() in mrc.lower() for mrc in locations)]
                if not kijiji_locs:
                    kijiji_locs = list(KIJIJI_LOCATIONS.keys())[:3]
            else:
                kijiji_locs = list(KIJIJI_LOCATIONS.keys())
            kijiji_count = self.kijiji.scrape_all(kijiji_locs, max_pages=2)
            logger.info(f"[demand] Kijiji: {kijiji_count} new signals")
        except Exception as e:
            logger.error(f"[demand] Kijiji error: {e}")
            traceback.print_exc()

        # Reddit
        try:
            logger.info("[demand] Scraping Reddit...")
            reddit_count = self.reddit.scrape_all()
            logger.info(f"[demand] Reddit: {reddit_count} new signals")
        except Exception as e:
            logger.error(f"[demand] Reddit error: {e}")

        # LesPAC
        try:
            logger.info("[demand] Scraping LesPAC...")
            lespac_count = self.lespac.scrape_all()
            logger.info(f"[demand] LesPAC: {lespac_count} new signals")
        except Exception as e:
            logger.error(f"[demand] LesPAC error: {e}")

        # Facebook (via Google)
        try:
            logger.info("[demand] Scraping Facebook signals...")
            fb_count = self.facebook.scrape_all()
            logger.info(f"[demand] Facebook: {fb_count} new signals")
        except Exception as e:
            logger.error(f"[demand] Facebook error: {e}")

    def run_supply_scrape(self, mrcs=None):
        """Run all supply-side scrapers."""
        logger.info("─── SUPPLY SCRAPING ───")
        target_mrcs = mrcs or PRIORITY_MRCS

        # Google search for providers
        try:
            logger.info("[supply] Searching for providers via Google...")
            for mrc in target_mrcs:
                if not self._running:
                    break
                # Only scan top demand categories to save time
                top_cats = list(BUSINESS_CATEGORIES.keys())[:10]
                self.google.scrape_mrc(mrc, categories=top_cats)
        except Exception as e:
            logger.error(f"[supply] Google error: {e}")

        # REQ business registry
        try:
            logger.info("[supply] Checking REQ business registry...")
            for mrc in target_mrcs[:3]:  # Limit to avoid rate limiting
                if not self._running:
                    break
                self.req.scan_mrc(mrc, categories=list(BUSINESS_CATEGORIES.keys())[:5])
        except Exception as e:
            logger.error(f"[supply] REQ error: {e}")

        # RBQ licensed contractors
        try:
            logger.info("[supply] Checking RBQ licenses...")
            for mrc in target_mrcs[:3]:
                if not self._running:
                    break
                self.rbq.scan_mrc(mrc)
        except Exception as e:
            logger.error(f"[supply] RBQ error: {e}")

    def run_regulatory_scrape(self):
        """Run regulatory/government scrapers."""
        logger.info("─── REGULATORY SCRAPING ───")
        try:
            result = self.government.scrape_all()
            logger.info(f"[regulatory] {len(result.get('known_programs', []))} active programs, "
                        f"{len(result.get('new_programs', []))} new signals")
        except Exception as e:
            logger.error(f"[regulatory] Error: {e}")

    def run_scoring(self, mrcs=None):
        """Run opportunity scoring for all MRCs with data."""
        logger.info("─── SCORING & ANALYSIS ───")
        target_mrcs = set(mrcs or PRIORITY_MRCS)

        # Also score any MRC that has demand data
        with self.kb._get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT mrc FROM demand_signals").fetchall()
            for row in rows:
                target_mrcs.add(row["mrc"])
            rows = conn.execute("SELECT DISTINCT mrc FROM supply_providers").fetchall()
            for row in rows:
                target_mrcs.add(row["mrc"])

        all_opps = []
        for mrc in target_mrcs:
            if not self._running:
                break
            try:
                opps = self.scorer.score_mrc(mrc)
                all_opps.extend(opps)
                top = [o for o in opps if o["scores"]["total"] >= ALERT_THRESHOLD]
                if top:
                    logger.info(f"[scoring] {mrc}: {len(top)} opportunities above threshold")
            except Exception as e:
                logger.error(f"[scoring] Error scoring {mrc}: {e}")

        all_opps.sort(key=lambda x: x["scores"]["total"], reverse=True)
        logger.info(f"[scoring] Total: {len(all_opps)} scored opportunities")
        return all_opps

    def run_alerts(self):
        """Send Telegram alerts for new high-score opportunities."""
        logger.info("─── ALERTS ───")
        sent = self.telegram.send_new_alerts()
        logger.info(f"[alerts] Sent {sent} new alerts")
        return sent

    def run_single_cycle(self, mrcs=None, quick=False):
        """Run one complete scan-analyze-alert cycle."""
        cycle_start = time.time()
        logger.info("=" * 60)
        logger.info(f"SCAN CYCLE START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Seed government programs on first run
        self.seed_government_programs()

        # Phase 1: Demand scraping
        if not quick:
            self.run_demand_scrape()
        else:
            # Quick mode: just Kijiji for first few locations
            try:
                locs = list(KIJIJI_LOCATIONS.keys())[:2]
                self.kijiji.scrape_all(locs, max_pages=1)
            except Exception as e:
                logger.error(f"Quick demand scrape error: {e}")

        if not self._running:
            return

        # Phase 2: Supply scraping
        if not quick:
            self.run_supply_scrape(mrcs)
        else:
            # Quick mode: just Google for first MRC
            try:
                target = (mrcs or PRIORITY_MRCS)[:1]
                cats = list(BUSINESS_CATEGORIES.keys())[:5]
                for mrc in target:
                    self.google.scrape_mrc(mrc, categories=cats)
            except Exception as e:
                logger.error(f"Quick supply scrape error: {e}")

        if not self._running:
            return

        # Phase 3: Regulatory check
        if not quick:
            self.run_regulatory_scrape()

        if not self._running:
            return

        # Phase 4: Scoring
        all_opps = self.run_scoring(mrcs)

        # Phase 5: Alerts
        self.run_alerts()

        # Summary
        duration = time.time() - cycle_start
        stats = self.kb.get_stats()
        logger.info("=" * 60)
        logger.info(f"CYCLE COMPLETE in {duration:.0f}s ({duration/60:.1f} min)")
        logger.info(f"Database: {stats}")
        top_opps = self.kb.get_top_opportunities(min_score=ALERT_THRESHOLD, limit=5)
        if top_opps:
            logger.info("Top opportunities:")
            for opp in top_opps:
                cat_name = opp.get("category", "")
                logger.info(f"  {opp['total_score']:.1f}/10 - {cat_name} @ {opp['mrc']}")
        logger.info("=" * 60)

        # Send daily digest
        if top_opps:
            self.telegram.send_daily_digest(top_opps)

    def run_forever(self):
        """Run the scanner in an eternal loop."""
        logger.info("Starting autonomous 24/7 scanning loop")
        logger.info(f"Scan interval: every {SCRAPE_INTERVAL_HOURS} hours")
        logger.info(f"Priority MRCs: {', '.join(PRIORITY_MRCS[:5])}...")

        self.telegram.send_startup_message()

        cycle = 0
        while self._running:
            cycle += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"AUTONOMOUS CYCLE #{cycle}")
            logger.info(f"{'='*60}")

            try:
                self.run_single_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                traceback.print_exc()
                # Continue on error - don't crash the loop
                logger.info("Recovering from error, continuing...")

            if not self._running:
                break

            # Wait for next cycle
            next_scan = SCRAPE_INTERVAL_HOURS * 3600
            logger.info(f"Next scan in {SCRAPE_INTERVAL_HOURS} hours. Sleeping...")
            sleep_start = time.time()
            while self._running and (time.time() - sleep_start) < next_scan:
                time.sleep(30)  # Check shutdown signal every 30s

        logger.info("Scanner stopped gracefully")

    def show_stats(self):
        """Display current database statistics."""
        stats = self.kb.get_stats()
        print("\n" + "=" * 50)
        print("QUÉBEC SCANNER - DATABASE STATS")
        print("=" * 50)
        for table, count in stats.items():
            print(f"  {table}: {count} records")
        print()

        top = self.kb.get_top_opportunities(min_score=0, limit=20)
        if top:
            print("TOP OPPORTUNITIES:")
            print("-" * 50)
            for opp in top:
                cat = opp.get("category", "")
                mrc = opp.get("mrc", "")
                score = opp.get("total_score", 0)
                conv = opp.get("convergence_count", 0)
                demand = opp.get("demand_count", 0)
                supply = opp.get("supply_count", 0)
                print(f"  {score:.1f}/10 | {cat:<30} | {mrc:<20} | D:{demand} S:{supply} C:{conv}")
        else:
            print("No opportunities scored yet. Run a scan first.")
        print()


def main():
    parser = argparse.ArgumentParser(description="Quebec Business Opportunity Scanner")
    parser.add_argument("--scan-once", action="store_true", help="Run a single scan cycle")
    parser.add_argument("--test", action="store_true", help="Quick test scan (Brome-Missisquoi)")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--top", action="store_true", help="Show top opportunities")
    parser.add_argument("--mrcs", nargs="+", help="Specific MRCs to scan")
    args = parser.parse_args()

    scanner = QuebecScanner()

    if args.stats or args.top:
        scanner.show_stats()
    elif args.test:
        logger.info("Running quick test scan...")
        scanner.run_single_cycle(mrcs=["Brome-Missisquoi"], quick=True)
        scanner.show_stats()
    elif args.scan_once:
        mrcs = args.mrcs if args.mrcs else None
        scanner.run_single_cycle(mrcs=mrcs)
        scanner.show_stats()
    else:
        scanner.run_forever()


if __name__ == "__main__":
    main()
