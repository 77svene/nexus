"""
Minimal LLM Integration - Qwen2.5-9B via Ollama.
ONLY used for:
1. Extracting service type from unstructured text
2. Classifying business categories
NOTHING ELSE - all scoring/analysis is pure code.
"""

import hashlib
import json
import logging
import re
import requests
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT, LLM_MAX_RETRIES, BUSINESS_CATEGORIES

logger = logging.getLogger(__name__)

# ─── Keyword-based classification (avoid LLM when possible) ───
# Flattened keyword -> category mapping for fast regex matching
_KEYWORD_MAP = {}
for cat, data in BUSINESS_CATEGORIES.items():
    for kw in data["keywords_fr"] + data["keywords_en"]:
        _KEYWORD_MAP[kw.lower()] = cat


def classify_by_keywords(text: str) -> Optional[str]:
    """Try to classify text using simple keyword matching. Returns category or None."""
    text_lower = text.lower()
    # Score each category by keyword matches
    scores = {}
    for kw, cat in _KEYWORD_MAP.items():
        if kw in text_lower:
            scores[cat] = scores.get(cat, 0) + 1
    if scores:
        return max(scores, key=scores.get)
    return None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class LLMExtractor:
    """Minimal LLM wrapper - only for text extraction when keywords fail."""

    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.available = self._check_ollama()
        if not self.available:
            logger.warning("Ollama not available - falling back to keyword-only mode")

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Make a single call to Ollama. Returns response text or None."""
        for attempt in range(LLM_MAX_RETRIES):
            try:
                r = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 100,
                        },
                    },
                    timeout=LLM_TIMEOUT,
                )
                if r.status_code == 200:
                    return r.json().get("response", "").strip()
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout (attempt {attempt + 1}/{LLM_MAX_RETRIES})")
            except Exception as e:
                logger.warning(f"Ollama error (attempt {attempt + 1}): {e}")
        return None

    def classify_text(self, text: str) -> str:
        """
        Classify text into a business category.
        Strategy: keyword match first, LLM only if keywords fail.
        """
        # 1. Try keywords first (free, instant)
        cat = classify_by_keywords(text)
        if cat:
            return cat

        # 2. Check LLM cache
        if self.kb:
            h = _text_hash(text)
            cached = self.kb.get_llm_cache(h)
            if cached:
                return cached

        # 3. Fall back to LLM
        if not self.available:
            return "unknown"

        categories_list = ", ".join(BUSINESS_CATEGORIES.keys())
        prompt = (
            f"Classify this Quebec service request into ONE category.\n"
            f"Categories: {categories_list}\n"
            f"Text: {text[:500]}\n"
            f"Reply with ONLY the category name, nothing else."
        )

        result = self._call_ollama(prompt)
        if result:
            # Normalize - find closest matching category
            result_lower = result.lower().strip()
            for cat in BUSINESS_CATEGORIES:
                if cat in result_lower or result_lower in cat:
                    if self.kb:
                        self.kb.set_llm_cache(_text_hash(text), text[:500], cat, OLLAMA_MODEL)
                    return cat

        return "unknown"

    def extract_service_type(self, text: str) -> dict:
        """
        Extract structured info from unstructured service request text.
        Returns: {"category": str, "urgency": str, "location_hint": str}
        """
        category = self.classify_text(text)

        # Urgency detection by keywords (no LLM needed)
        urgency = "normal"
        urgent_keywords = ["urgent", "urgence", "asap", "immédiat", "vite", "rapidement", "dès que possible", "emergency"]
        text_lower = text.lower()
        for kw in urgent_keywords:
            if kw in text_lower:
                urgency = "urgent"
                break

        # Location extraction by regex (no LLM needed)
        location_hint = ""
        # Match common Quebec location patterns
        loc_patterns = [
            r"(?:à|a|in|near|près de|proche de)\s+([A-ZÀ-Ü][a-zà-ü]+(?:[-\s][A-ZÀ-Ü][a-zà-ü]+)*)",
            r"(?:secteur|quartier|ville de|region de)\s+([A-ZÀ-Ü][a-zà-ü]+(?:[-\s][A-ZÀ-Ü][a-zà-ü]+)*)",
        ]
        for pat in loc_patterns:
            m = re.search(pat, text)
            if m:
                location_hint = m.group(1)
                break

        return {
            "category": category,
            "urgency": urgency,
            "location_hint": location_hint,
        }

    def batch_classify(self, texts: list) -> list:
        """Classify multiple texts efficiently. Uses keywords first, batches LLM calls."""
        results = []
        for text in texts:
            results.append(self.classify_text(text))
        return results
