"""
SQLite Knowledge Base for Quebec Business Opportunity Scanner.
Stores all scraped data, analysis results, and opportunity scores.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager


class KnowledgeBase:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS demand_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    mrc TEXT NOT NULL,
                    region TEXT,
                    category TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    url TEXT,
                    raw_text TEXT,
                    urgency TEXT DEFAULT 'normal',
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    posted_date TEXT,
                    UNIQUE(source, url)
                );

                CREATE TABLE IF NOT EXISTS supply_providers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    mrc TEXT NOT NULL,
                    region TEXT,
                    category TEXT NOT NULL,
                    business_name TEXT,
                    address TEXT,
                    phone TEXT,
                    rating REAL,
                    review_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    license_number TEXT,
                    license_type TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, business_name, mrc, category)
                );

                CREATE TABLE IF NOT EXISTS opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mrc TEXT NOT NULL,
                    region TEXT,
                    category TEXT NOT NULL,
                    demand_score REAL,
                    supply_score REAL,
                    regulatory_score REAL,
                    temporal_score REAL,
                    total_score REAL,
                    convergence_count INTEGER DEFAULT 0,
                    convergence_signals TEXT,
                    demand_count INTEGER DEFAULT 0,
                    supply_count INTEGER DEFAULT 0,
                    growth_rate REAL DEFAULT 0.0,
                    has_subsidy INTEGER DEFAULT 0,
                    subsidy_programs TEXT,
                    barriers TEXT,
                    market_estimate TEXT,
                    window_months TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    alerted INTEGER DEFAULT 0,
                    UNIQUE(mrc, category)
                );

                CREATE TABLE IF NOT EXISTS government_programs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    url TEXT,
                    sector TEXT,
                    description TEXT,
                    is_active INTEGER DEFAULT 1,
                    amount TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, sector)
                );

                CREATE TABLE IF NOT EXISTS scrape_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    mrc TEXT,
                    status TEXT NOT NULL,
                    records_found INTEGER DEFAULT 0,
                    error_message TEXT,
                    duration_seconds REAL,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS llm_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT UNIQUE NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_demand_mrc ON demand_signals(mrc);
                CREATE INDEX IF NOT EXISTS idx_demand_category ON demand_signals(category);
                CREATE INDEX IF NOT EXISTS idx_demand_scraped ON demand_signals(scraped_at);
                CREATE INDEX IF NOT EXISTS idx_supply_mrc ON supply_providers(mrc);
                CREATE INDEX IF NOT EXISTS idx_supply_category ON supply_providers(category);
                CREATE INDEX IF NOT EXISTS idx_opportunities_score ON opportunities(total_score DESC);
                CREATE INDEX IF NOT EXISTS idx_opportunities_mrc ON opportunities(mrc);
                CREATE INDEX IF NOT EXISTS idx_llm_cache_hash ON llm_cache(input_hash);
            """)

    def add_demand_signal(self, source, mrc, category, title="", description="",
                          url="", raw_text="", urgency="normal", posted_date="", region=""):
        with self._get_conn() as conn:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO demand_signals
                    (source, mrc, category, title, description, url, raw_text, urgency, posted_date, region)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (source, mrc, category, title, description, url, raw_text, urgency, posted_date, region))
                return True
            except sqlite3.IntegrityError:
                return False

    def add_supply_provider(self, source, mrc, category, business_name="",
                            address="", phone="", rating=0.0, review_count=0,
                            license_number="", license_type="", region=""):
        with self._get_conn() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO supply_providers
                    (source, mrc, category, business_name, address, phone, rating,
                     review_count, license_number, license_type, region)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (source, mrc, category, business_name, address, phone, rating,
                      review_count, license_number, license_type, region))
                return True
            except sqlite3.IntegrityError:
                return False

    def get_demand_count(self, mrc, category, days=30):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM demand_signals
                WHERE mrc = ? AND category = ? AND scraped_at >= ?
            """, (mrc, category, cutoff)).fetchone()
            return row["cnt"] if row else 0

    def get_demand_count_period(self, mrc, category, days_ago_start, days_ago_end):
        start = (datetime.now() - timedelta(days=days_ago_start)).isoformat()
        end = (datetime.now() - timedelta(days=days_ago_end)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM demand_signals
                WHERE mrc = ? AND category = ? AND scraped_at >= ? AND scraped_at <= ?
            """, (mrc, category, start, end)).fetchone()
            return row["cnt"] if row else 0

    def get_supply_count(self, mrc, category):
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as cnt FROM supply_providers
                WHERE mrc = ? AND category = ? AND is_active = 1
            """, (mrc, category)).fetchone()
            return row["cnt"] if row else 0

    def get_active_subsidies(self, category):
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM government_programs
                WHERE sector = ? AND is_active = 1
            """, (category,)).fetchall()
            return [dict(r) for r in rows]

    def upsert_opportunity(self, mrc, category, scores, details):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO opportunities
                (mrc, category, demand_score, supply_score, regulatory_score,
                 temporal_score, total_score, convergence_count, convergence_signals,
                 demand_count, supply_count, growth_rate, has_subsidy, subsidy_programs,
                 barriers, market_estimate, window_months, region, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(mrc, category) DO UPDATE SET
                    demand_score = excluded.demand_score,
                    supply_score = excluded.supply_score,
                    regulatory_score = excluded.regulatory_score,
                    temporal_score = excluded.temporal_score,
                    total_score = excluded.total_score,
                    convergence_count = excluded.convergence_count,
                    convergence_signals = excluded.convergence_signals,
                    demand_count = excluded.demand_count,
                    supply_count = excluded.supply_count,
                    growth_rate = excluded.growth_rate,
                    has_subsidy = excluded.has_subsidy,
                    subsidy_programs = excluded.subsidy_programs,
                    barriers = excluded.barriers,
                    market_estimate = excluded.market_estimate,
                    window_months = excluded.window_months,
                    last_updated = CURRENT_TIMESTAMP
            """, (
                mrc, category,
                scores.get("demand", 0), scores.get("supply", 0),
                scores.get("regulatory", 0), scores.get("temporal", 0),
                scores.get("total", 0),
                details.get("convergence_count", 0),
                json.dumps(details.get("convergence_signals", [])),
                details.get("demand_count", 0),
                details.get("supply_count", 0),
                details.get("growth_rate", 0.0),
                1 if details.get("has_subsidy") else 0,
                json.dumps(details.get("subsidy_programs", [])),
                json.dumps(details.get("barriers", [])),
                details.get("market_estimate", ""),
                details.get("window_months", ""),
                details.get("region", ""),
            ))

    def get_top_opportunities(self, min_score=7.0, limit=20):
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM opportunities
                WHERE total_score >= ?
                ORDER BY total_score DESC
                LIMIT ?
            """, (min_score, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_unalerted_opportunities(self, min_score=7.0):
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM opportunities
                WHERE total_score >= ? AND alerted = 0
                ORDER BY total_score DESC
            """, (min_score,)).fetchall()
            return [dict(r) for r in rows]

    def mark_alerted(self, opp_id):
        with self._get_conn() as conn:
            conn.execute("UPDATE opportunities SET alerted = 1 WHERE id = ?", (opp_id,))

    def log_scrape(self, source, mrc, status, records_found=0, error_message="", duration=0.0):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO scrape_log (source, mrc, status, records_found, error_message, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source, mrc, status, records_found, error_message, duration))

    def get_llm_cache(self, input_hash):
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT output_text FROM llm_cache WHERE input_hash = ?",
                (input_hash,)
            ).fetchone()
            return row["output_text"] if row else None

    def set_llm_cache(self, input_hash, input_text, output_text, model=""):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO llm_cache (input_hash, input_text, output_text, model)
                VALUES (?, ?, ?, ?)
            """, (input_hash, input_text, output_text, model))

    def add_government_program(self, name, url, sector, description, is_active=True, amount=""):
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO government_programs
                (name, url, sector, description, is_active, amount)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, url, sector, description, 1 if is_active else 0, amount))

    def get_stats(self):
        with self._get_conn() as conn:
            stats = {}
            for table in ["demand_signals", "supply_providers", "opportunities", "scrape_log"]:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                stats[table] = row["cnt"]
            return stats
