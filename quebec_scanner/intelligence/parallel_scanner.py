"""
Parallel MRC Scanner - Scans all 104 MRCs with concurrent workers.
Implements priority queue, rotation, and geographic clustering.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUSINESS_CATEGORIES

logger = logging.getLogger(__name__)

MRC_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "all_104_mrcs.json")

# Scan frequency by priority level (hours between scans)
PRIORITY_INTERVALS = {
    "critical": 6,
    "high": 12,
    "medium": 24,
    "low": 48,
}

MAX_WORKERS = 5  # Concurrent scraping threads (be respectful)


def load_all_mrcs() -> list:
    """Load the complete 104 MRC database."""
    with open(MRC_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mrcs_by_priority(priority: str) -> list:
    """Get MRCs filtered by priority level."""
    all_mrcs = load_all_mrcs()
    return [m for m in all_mrcs if m["scan_priority"] == priority]


def get_mrcs_by_region(region_id: str) -> list:
    """Get all MRCs in a specific region."""
    all_mrcs = load_all_mrcs()
    return [m for m in all_mrcs if m["region_id"] == region_id]


class ParallelScanner:
    """Manages parallel scanning across all 104 MRCs."""

    def __init__(self, knowledge_base, scrapers: dict, scorer):
        """
        Args:
            knowledge_base: KnowledgeBase instance
            scrapers: dict of scraper instances keyed by name
            scorer: OpportunityScorer instance
        """
        self.kb = knowledge_base
        self.scrapers = scrapers
        self.scorer = scorer
        self.all_mrcs = load_all_mrcs()
        self._last_scan = {}  # mrc_name -> datetime of last scan
        self._running = True

    def _should_scan(self, mrc: dict) -> bool:
        """Check if an MRC is due for scanning based on its priority."""
        name = mrc["name"]
        priority = mrc["scan_priority"]
        interval_hours = PRIORITY_INTERVALS.get(priority, 48)

        last = self._last_scan.get(name)
        if not last:
            return True

        return (datetime.now() - last) >= timedelta(hours=interval_hours)

    def _get_scan_queue(self) -> list:
        """Build prioritized queue of MRCs needing scanning."""
        due = [m for m in self.all_mrcs if self._should_scan(m)]

        # Sort by priority (critical first), then by population (larger first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        due.sort(key=lambda m: (priority_order.get(m["scan_priority"], 4), -m["population"]))
        return due

    def _scan_single_mrc(self, mrc: dict) -> dict:
        """Run all scrapers and scoring for a single MRC. Thread-safe."""
        name = mrc["name"]
        region = mrc["region"]
        start_time = time.time()
        results = {"mrc": name, "signals": 0, "opportunities": 0, "errors": []}

        logger.info(f"[parallel] Scanning {name} ({region})...")

        # Demand scraping
        try:
            kijiji = self.scrapers.get("kijiji")
            if kijiji:
                # Find closest Kijiji location
                from config import KIJIJI_LOCATIONS
                for loc_name in KIJIJI_LOCATIONS:
                    if loc_name.lower() in name.lower() or name.lower() in loc_name.lower():
                        count = kijiji.scrape_location(loc_name, max_pages=1)
                        results["signals"] += count
                        break
        except Exception as e:
            results["errors"].append(f"kijiji: {e}")

        # Supply scraping (Google)
        try:
            google = self.scrapers.get("google")
            if google:
                top_cats = list(BUSINESS_CATEGORIES.keys())[:8]
                google.scrape_mrc(name, categories=top_cats)
        except Exception as e:
            results["errors"].append(f"google: {e}")

        # Scoring
        try:
            opps = self.scorer.score_mrc(name)
            results["opportunities"] = len(opps)
        except Exception as e:
            results["errors"].append(f"scorer: {e}")

        duration = time.time() - start_time
        self._last_scan[name] = datetime.now()

        logger.info(f"[parallel] {name} done: {results['signals']} signals, "
                     f"{results['opportunities']} opportunities ({duration:.1f}s)")
        return results

    def scan_batch(self, batch_size: int = 20) -> list:
        """Scan a batch of MRCs that are due, using parallel workers."""
        queue = self._get_scan_queue()[:batch_size]

        if not queue:
            logger.info("[parallel] No MRCs due for scanning")
            return []

        logger.info(f"[parallel] Scanning batch of {len(queue)} MRCs "
                     f"with {MAX_WORKERS} workers...")

        all_results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._scan_single_mrc, mrc): mrc
                for mrc in queue
            }

            for future in as_completed(futures):
                if not self._running:
                    break
                mrc = futures[future]
                try:
                    result = future.result(timeout=300)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"[parallel] {mrc['name']} failed: {e}")
                    all_results.append({
                        "mrc": mrc["name"],
                        "signals": 0,
                        "opportunities": 0,
                        "errors": [str(e)],
                    })

        # Summary
        total_signals = sum(r["signals"] for r in all_results)
        total_opps = sum(r["opportunities"] for r in all_results)
        total_errors = sum(len(r["errors"]) for r in all_results)

        logger.info(f"[parallel] Batch complete: {len(all_results)} MRCs, "
                     f"{total_signals} signals, {total_opps} opportunities, "
                     f"{total_errors} errors")

        return all_results

    def scan_all_quebec(self) -> list:
        """
        Scan ALL 104 MRCs in order of priority.
        Critical MRCs first, then high, medium, low.
        """
        logger.info("[parallel] Full Quebec scan starting - 104 MRCs")
        all_results = []

        for priority in ["critical", "high", "medium", "low"]:
            if not self._running:
                break

            mrcs = get_mrcs_by_priority(priority)
            logger.info(f"[parallel] Scanning {len(mrcs)} {priority}-priority MRCs...")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(self._scan_single_mrc, mrc): mrc
                    for mrc in mrcs
                }

                for future in as_completed(futures):
                    if not self._running:
                        break
                    try:
                        result = future.result(timeout=300)
                        all_results.append(result)
                    except Exception as e:
                        mrc = futures[future]
                        logger.error(f"[parallel] {mrc['name']}: {e}")

        logger.info(f"[parallel] Full Quebec scan complete: {len(all_results)} MRCs processed")
        return all_results

    def get_scan_status(self) -> dict:
        """Get status of all MRC scans."""
        status = {
            "total_mrcs": len(self.all_mrcs),
            "scanned": len(self._last_scan),
            "pending": len(self._get_scan_queue()),
            "by_priority": {},
        }
        for priority in ["critical", "high", "medium", "low"]:
            mrcs = get_mrcs_by_priority(priority)
            scanned = sum(1 for m in mrcs if m["name"] in self._last_scan)
            status["by_priority"][priority] = {
                "total": len(mrcs),
                "scanned": scanned,
            }
        return status

    def detect_regional_patterns(self) -> list:
        """Detect patterns that span multiple MRCs in the same region."""
        patterns = []
        regions = defaultdict(list)

        # Get top opportunities and group by region
        top_opps = self.kb.get_top_opportunities(min_score=5.0, limit=200)
        for opp in top_opps:
            regions[opp.get("region", "")].append(opp)

        for region, opps in regions.items():
            if not region:
                continue

            # Count category frequency across MRCs in this region
            cat_counts = defaultdict(int)
            cat_mrcs = defaultdict(set)
            for opp in opps:
                cat = opp["category"]
                cat_counts[cat] += 1
                cat_mrcs[cat].add(opp["mrc"])

            # Regional pattern = same category appears in 3+ MRCs
            for cat, count in cat_counts.items():
                if len(cat_mrcs[cat]) >= 3:
                    avg_score = sum(
                        o["total_score"] for o in opps if o["category"] == cat
                    ) / count
                    patterns.append({
                        "type": "regional_pattern",
                        "region": region,
                        "category": cat,
                        "mrc_count": len(cat_mrcs[cat]),
                        "mrcs": list(cat_mrcs[cat]),
                        "avg_score": round(avg_score, 2),
                        "significance": "high" if len(cat_mrcs[cat]) >= 5 else "medium",
                    })

        patterns.sort(key=lambda p: p["avg_score"], reverse=True)
        return patterns

    def stop(self):
        self._running = False
