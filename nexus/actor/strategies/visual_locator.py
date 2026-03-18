"""
Self-Healing Element Detection System
Implements multi-strategy element finding with automatic fallback and retry.
Elements become 'resilient locators' that adapt to DOM changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from playwright.async_api import ElementHandle, Page, Locator

from nexus.actor.element import Element
from nexus.actor.page import BrowserPage
from nexus.actor.utils import retry_async, TimeoutError

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of element detection strategies"""
    CSS_SELECTOR = "css_selector"
    XPATH = "xpath"
    TEXT_CONTENT = "text_content"
    VISUAL_SIMILARITY = "visual_similarity"
    AI_VISION = "ai_vision"
    ACCESSIBILITY = "accessibility"
    COORDINATE_BASED = "coordinate_based"


@dataclass
class ElementSignature:
    """Signature of an element for visual similarity matching"""
    tag_name: str
    attributes: Dict[str, str]
    text_content: str
    bounding_box: Dict[str, float]
    visual_hash: Optional[str] = None
    parent_signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_name": self.tag_name,
            "attributes": self.attributes,
            "text_content": self.text_content,
            "bounding_box": self.bounding_box,
            "visual_hash": self.visual_hash,
            "parent_signature": self.parent_signature
        }


@dataclass
class StrategyResult:
    """Result from a detection strategy"""
    element: Optional[ElementHandle]
    confidence: float
    strategy_type: StrategyType
    selector_used: Optional[str] = None
    bounding_box: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.element is not None and self.confidence > 0.5


@dataclass
class ResilientLocatorConfig:
    """Configuration for resilient locator"""
    max_retries: int = 3
    timeout_ms: int = 10000
    cache_ttl_seconds: int = 300
    enable_ai_vision: bool = True
    enable_visual_similarity: bool = True
    similarity_threshold: float = 0.7
    update_selectors: bool = True
    strategies_order: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.CSS_SELECTOR,
        StrategyType.XPATH,
        StrategyType.TEXT_CONTENT,
        StrategyType.ACCESSIBILITY,
        StrategyType.AI_VISION,
        StrategyType.VISUAL_SIMILARITY,
        StrategyType.COORDINATE_BASED
    ])


class DetectionStrategy(ABC):
    """Base class for element detection strategies"""
    
    def __init__(self, page: BrowserPage, config: ResilientLocatorConfig):
        self.page = page
        self.config = config
        self._cache: Dict[str, StrategyResult] = {}
    
    @abstractmethod
    async def detect(self, 
                     original_selector: str, 
                     context: Dict[str, Any]) -> StrategyResult:
        """Detect element using this strategy"""
        pass
    
    def _get_cache_key(self, selector: str, context: Dict[str, Any]) -> str:
        """Generate cache key for this detection attempt"""
        key_data = f"{selector}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid"""
        if cache_key not in self._cache:
            return False
        
        result = self._cache[cache_key]
        if result.element is None:
            return False
        
        # Check if element is still attached to DOM
        try:
            # This is a simplified check - in reality we'd need async evaluation
            return True
        except:
            return False


class CssSelectorStrategy(DetectionStrategy):
    """CSS selector based detection"""
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        cache_key = self._get_cache_key(original_selector, context)
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            element = await self.page.query_selector(original_selector)
            if element:
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=1.0,
                    strategy_type=StrategyType.CSS_SELECTOR,
                    selector_used=original_selector,
                    bounding_box=bbox
                )
                self._cache[cache_key] = result
                return result
        except Exception as e:
            logger.debug(f"CSS selector failed: {original_selector}, error: {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.CSS_SELECTOR,
            selector_used=original_selector
        )


class XPathStrategy(DetectionStrategy):
    """XPath based detection"""
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        # Convert CSS to XPath if needed, or use provided XPath
        xpath = context.get("xpath", self._css_to_xpath(original_selector))
        
        cache_key = self._get_cache_key(xpath, context)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            element = await self.page.query_selector(f"xpath={xpath}")
            if element:
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=0.9,
                    strategy_type=StrategyType.XPATH,
                    selector_used=xpath,
                    bounding_box=bbox
                )
                self._cache[cache_key] = result
                return result
        except Exception as e:
            logger.debug(f"XPath strategy failed: {xpath}, error: {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.XPATH,
            selector_used=xpath
        )
    
    def _css_to_xpath(self, css_selector: str) -> str:
        """Convert CSS selector to XPath (simplified)"""
        # This is a simplified conversion - production would need full parser
        if css_selector.startswith("#"):
            return f"//*[@id='{css_selector[1:]}']"
        elif css_selector.startswith("."):
            return f"//*[contains(@class, '{css_selector[1:]}')]"
        else:
            return f"//{css_selector}"


class TextContentStrategy(DetectionStrategy):
    """Text content based detection"""
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        text = context.get("text_content", "")
        if not text:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.TEXT_CONTENT
            )
        
        cache_key = self._get_cache_key(f"text:{text}", context)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Try exact text match first
            element = await self.page.get_by_text(text, exact=True).first.element_handle()
            if element:
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=0.85,
                    strategy_type=StrategyType.TEXT_CONTENT,
                    selector_used=f"text={text}",
                    bounding_box=bbox,
                    metadata={"match_type": "exact"}
                )
                self._cache[cache_key] = result
                return result
            
            # Try partial text match
            element = await self.page.get_by_text(text, exact=False).first.element_handle()
            if element:
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=0.7,
                    strategy_type=StrategyType.TEXT_CONTENT,
                    selector_used=f"text*={text}",
                    bounding_box=bbox,
                    metadata={"match_type": "partial"}
                )
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.debug(f"Text content strategy failed for '{text}': {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.TEXT_CONTENT
        )


class AccessibilityStrategy(DetectionStrategy):
    """Accessibility tree based detection"""
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        accessible_name = context.get("accessible_name", "")
        role = context.get("role", "")
        
        if not accessible_name and not role:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.ACCESSIBILITY
            )
        
        cache_key = self._get_cache_key(f"a11y:{accessible_name}:{role}", context)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Use Playwright's accessibility selectors
            selector_parts = []
            if role:
                selector_parts.append(f'role={role}')
            if accessible_name:
                selector_parts.append(f'name="{accessible_name}"')
            
            selector = " >> ".join(selector_parts)
            element = await self.page.query_selector(selector)
            
            if element:
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=0.8,
                    strategy_type=StrategyType.ACCESSIBILITY,
                    selector_used=selector,
                    bounding_box=bbox
                )
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.debug(f"Accessibility strategy failed: {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.ACCESSIBILITY
        )


class AIVisionStrategy(DetectionStrategy):
    """AI vision-based element detection"""
    
    def __init__(self, page: BrowserPage, config: ResilientLocatorConfig):
        super().__init__(page, config)
        self._vision_model = None  # Would be initialized with actual vision model
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        if not self.config.enable_ai_vision:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.AI_VISION
            )
        
        description = context.get("element_description", "")
        if not description:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.AI_VISION
            )
        
        cache_key = self._get_cache_key(f"ai:{description}", context)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Take screenshot for vision analysis
            screenshot = await self.page.screenshot()
            
            # In production, this would call an actual vision model
            # For now, we'll simulate finding elements by description
            elements = await self._find_elements_by_description(description)
            
            if elements:
                # Return the best match
                best_match = max(elements, key=lambda x: x[1])
                element, confidence = best_match
                
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=confidence,
                    strategy_type=StrategyType.AI_VISION,
                    bounding_box=bbox,
                    metadata={"description": description}
                )
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.debug(f"AI vision strategy failed: {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.AI_VISION
        )
    
    async def _find_elements_by_description(self, description: str) -> List[Tuple[ElementHandle, float]]:
        """Find elements matching a description using heuristics"""
        # This is a simplified implementation
        # Production would use actual vision model integration
        results = []
        
        # Try to find by text content
        text_elements = await self.page.query_selector_all(f"text=/{description}/i")
        for elem in text_elements[:3]:  # Limit to top 3
            results.append((elem, 0.6))
        
        # Try to find by aria-label
        aria_elements = await self.page.query_selector_all(f"[aria-label*='{description}' i]")
        for elem in aria_elements[:2]:
            results.append((elem, 0.7))
        
        # Try to find by placeholder
        placeholder_elements = await self.page.query_selector_all(f"[placeholder*='{description}' i]")
        for elem in placeholder_elements[:2]:
            results.append((elem, 0.65))
        
        return results


class VisualSimilarityStrategy(DetectionStrategy):
    """Visual similarity based detection"""
    
    def __init__(self, page: BrowserPage, config: ResilientLocatorConfig):
        super().__init__(page, config)
        self._element_signatures: Dict[str, ElementSignature] = {}
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        if not self.config.enable_visual_similarity:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.VISUAL_SIMILARITY
            )
        
        reference_signature = context.get("element_signature")
        if not reference_signature:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.VISUAL_SIMILARITY
            )
        
        cache_key = self._get_cache_key(f"visual:{reference_signature}", context)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Find all elements of the same type
            tag_name = context.get("tag_name", "*")
            elements = await self.page.query_selector_all(tag_name)
            
            best_match = None
            best_score = 0.0
            
            for element in elements[:50]:  # Limit search for performance
                score = await self._calculate_similarity(element, reference_signature)
                if score > best_score and score >= self.config.similarity_threshold:
                    best_score = score
                    best_match = element
            
            if best_match:
                bbox = await best_match.bounding_box()
                result = StrategyResult(
                    element=best_match,
                    confidence=best_score,
                    strategy_type=StrategyType.VISUAL_SIMILARITY,
                    bounding_box=bbox,
                    metadata={"similarity_score": best_score}
                )
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.debug(f"Visual similarity strategy failed: {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.VISUAL_SIMILARITY
        )
    
    async def _calculate_similarity(self, element: ElementHandle, reference: ElementSignature) -> float:
        """Calculate similarity score between element and reference signature"""
        try:
            # Get element properties
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            text_content = await element.evaluate("el => el.textContent?.trim() || ''")
            bbox = await element.bounding_box()
            
            # Simple similarity scoring
            score = 0.0
            
            # Tag name match
            if tag_name == reference.tag_name:
                score += 0.3
            
            # Text content similarity
            if text_content and reference.text_content:
                text_similarity = self._text_similarity(text_content, reference.text_content)
                score += 0.4 * text_similarity
            
            # Bounding box similarity (if available)
            if bbox and reference.bounding_box:
                bbox_similarity = self._bbox_similarity(bbox, reference.bounding_box)
                score += 0.3 * bbox_similarity
            
            return min(score, 1.0)
            
        except Exception:
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified)"""
        if text1 == text2:
            return 1.0
        
        # Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _bbox_similarity(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate bounding box similarity"""
        # Simple IoU (Intersection over Union) calculation
        x1 = max(bbox1["x"], bbox2["x"])
        y1 = max(bbox1["y"], bbox2["y"])
        x2 = min(bbox1["x"] + bbox1["width"], bbox2["x"] + bbox2["width"])
        y2 = min(bbox1["y"] + bbox1["height"], bbox2["y"] + bbox2["height"])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1["width"] * bbox1["height"]
        area2 = bbox2["width"] * bbox2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class CoordinateBasedStrategy(DetectionStrategy):
    """Coordinate-based detection (fallback)"""
    
    async def detect(self, original_selector: str, context: Dict[str, Any]) -> StrategyResult:
        coordinates = context.get("coordinates")
        if not coordinates or "x" not in coordinates or "y" not in coordinates:
            return StrategyResult(
                element=None,
                confidence=0.0,
                strategy_type=StrategyType.COORDINATE_BASED
            )
        
        try:
            # Get element at coordinates
            element = await self.page.evaluate_handle(
                f"document.elementFromPoint({coordinates['x']}, {coordinates['y']})"
            )
            
            if element:
                bbox = await element.bounding_box()
                result = StrategyResult(
                    element=element,
                    confidence=0.5,  # Low confidence as coordinates might be stale
                    strategy_type=StrategyType.COORDINATE_BASED,
                    bounding_box=bbox,
                    metadata={"coordinates": coordinates}
                )
                return result
                
        except Exception as e:
            logger.debug(f"Coordinate-based strategy failed: {e}")
        
        return StrategyResult(
            element=None,
            confidence=0.0,
            strategy_type=StrategyType.COORDINATE_BASED
        )


class StrategyFactory:
    """Factory for creating detection strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: StrategyType, 
                       page: BrowserPage, 
                       config: ResilientLocatorConfig) -> DetectionStrategy:
        strategies = {
            StrategyType.CSS_SELECTOR: CssSelectorStrategy,
            StrategyType.XPATH: XPathStrategy,
            StrategyType.TEXT_CONTENT: TextContentStrategy,
            StrategyType.ACCESSIBILITY: AccessibilityStrategy,
            StrategyType.AI_VISION: AIVisionStrategy,
            StrategyType.VISUAL_SIMILARITY: VisualSimilarityStrategy,
            StrategyType.COORDINATE_BASED: CoordinateBasedStrategy
        }
        
        strategy_class = strategies.get(strategy_type)
        if strategy_class:
            return strategy_class(page, config)
        
        raise ValueError(f"Unknown strategy type: {strategy_type}")


class ResilientLocator:
    """
    Self-healing element locator that tries multiple detection strategies
    with automatic fallback and retry. Adapts to DOM changes.
    """
    
    def __init__(self, 
                 page: BrowserPage,
                 original_selector: str,
                 element_description: str = "",
                 config: Optional[ResilientLocatorConfig] = None):
        self.page = page
        self.original_selector = original_selector
        self.element_description = element_description
        self.config = config or ResilientLocatorConfig()
        
        self._strategies: List[DetectionStrategy] = []
        self._successful_strategy: Optional[DetectionStrategy] = None
        self._last_successful_selector: Optional[str] = None
        self._element_signature: Optional[ElementSignature] = None
        self._failure_count: int = 0
        self._last_attempt_time: float = 0
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize detection strategies based on config"""
        for strategy_type in self.config.strategies_order:
            try:
                strategy = StrategyFactory.create_strategy(
                    strategy_type, self.page, self.config
                )
                self._strategies.append(strategy)
            except Exception as e:
                logger.warning(f"Failed to initialize {strategy_type} strategy: {e}")
    
    async def locate(self, 
                    timeout_ms: Optional[int] = None,
                    force_retry: bool = False) -> Optional[ElementHandle]:
        """
        Locate element using resilient strategies with fallback.
        
        Args:
            timeout_ms: Timeout in milliseconds
            force_retry: Force retry even if cached strategy exists
            
        Returns:
            ElementHandle if found, None otherwise
        """
        timeout = timeout_ms or self.config.timeout_ms
        start_time = time.time() * 1000
        
        # Build context for strategies
        context = await self._build_context()
        
        # Try cached successful strategy first (if not forcing retry)
        if not force_retry and self._successful_strategy:
            try:
                result = await asyncio.wait_for(
                    self._successful_strategy.detect(self.original_selector, context),
                    timeout=timeout/1000
                )
                
                if result.success:
                    self._update_from_result(result)
                    return result.element
                else:
                    # Cached strategy failed, increment failure count
                    self._failure_count += 1
                    if self._failure_count >= self.config.max_retries:
                        # Too many failures, clear cached strategy
                        self._successful_strategy = None
                        self._failure_count = 0
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Cached strategy failed: {e}")
        
        # Try all strategies in order
        for strategy in self._strategies:
            if time.time() * 1000 - start_time > timeout:
                break
            
            try:
                result = await asyncio.wait_for(
                    strategy.detect(self.original_selector, context),
                    timeout=timeout/1000
                )
                
                if result.success:
                    self._update_from_result(result)
                    self._successful_strategy = strategy
                    self._failure_count = 0
                    
                    # Update selector if enabled
                    if self.config.update_selectors and result.selector_used:
                        await self._update_selector(result.selector_used)
                    
                    return result.element
                    
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Strategy {strategy.__class__.__name__} failed: {e}")
                continue
        
        # All strategies failed
        self._failure_count += 1
        return None
    
    async def _build_context(self) -> Dict[str, Any]:
        """Build context dictionary for detection strategies"""
        context = {
            "element_description": self.element_description,
            "timestamp": time.time()
        }
        
        # Try to extract additional context from original selector
        if self.original_selector:
            # Extract text content if selector contains text
            if "text=" in self.original_selector or "text*" in self.original_selector:
                text_part = self.original_selector.split("=")[1] if "=" in self.original_selector else ""
                context["text_content"] = text_part.strip("'\"")
            
            # Extract accessible name if present
            if "aria-label" in self.original_selector:
                # Simplified extraction
                context["accessible_name"] = self.element_description
        
        # Get element signature if we have a previous successful detection
        if self._element_signature:
            context["element_signature"] = self._element_signature.to_dict()
            context["tag_name"] = self._element_signature.tag_name
        
        # Get current page state
        try:
            context["url"] = self.page.url
            context["viewport_size"] = await self.page.evaluate("() => ({width: window.innerWidth, height: window.innerHeight})")
        except:
            pass
        
        return context
    
    def _update_from_result(self, result: StrategyResult):
        """Update locator state from successful detection result"""
        self._last_attempt_time = time.time()
        
        if result.element:
            # Update element signature for future visual matching
            asyncio.create_task(self._update_element_signature(result.element))
            
            # Update last successful selector
            if result.selector_used:
                self._last_successful_selector = result.selector_used
    
    async def _update_element_signature(self, element: ElementHandle):
        """Update element signature for visual similarity matching"""
        try:
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            attributes = await element.evaluate("""el => {
                const attrs = {};
                for (const attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }""")
            text_content = await element.evaluate("el => el.textContent?.trim() || ''")
            bbox = await element.bounding_box()
            
            # Calculate visual hash (simplified)
            visual_hash = None
            try:
                # In production, would calculate actual visual hash
                visual_hash = hashlib.md5(f"{tag_name}:{json.dumps(attributes, sort_keys=True)}".encode()).hexdigest()
            except:
                pass
            
            self._element_signature = ElementSignature(
                tag_name=tag_name,
                attributes=attributes,
                text_content=text_content,
                bounding_box=bbox or {},
                visual_hash=visual_hash
            )
        except Exception as e:
            logger.debug(f"Failed to update element signature: {e}")
    
    async def _update_selector(self, new_selector: str):
        """Update the selector when a better one is found"""
        if new_selector != self.original_selector:
            logger.info(f"Updating selector from '{self.original_selector}' to '{new_selector}'")
            self.original_selector = new_selector
            
            # In production, this would persist the updated selector
            # to a database or configuration file
    
    async def locate_with_retry(self, 
                               max_attempts: int = 3,
                               delay_ms: int = 1000) -> Optional[ElementHandle]:
        """
        Locate element with retry logic.
        
        Args:
            max_attempts: Maximum number of attempts
            delay_ms: Delay between attempts in milliseconds
            
        Returns:
            ElementHandle if found, None otherwise
        """
        for attempt in range(max_attempts):
            element = await self.locate(force_retry=(attempt > 0))
            if element:
                return element
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay_ms / 1000)
        
        return None
    
    async def wait_for_stable(self, 
                             stability_ms: int = 500,
                             timeout_ms: int = 10000) -> Optional[ElementHandle]:
        """
        Wait for element to become stable (not moving/changing).
        
        Args:
            stability_ms: Time element must be stable in milliseconds
            timeout_ms: Maximum wait time in milliseconds
            
        Returns:
            Stable ElementHandle if found, None otherwise
        """
        start_time = time.time() * 1000
        last_bbox = None
        stable_since = None
        
        while time.time() * 1000 - start_time < timeout_ms:
            element = await self.locate()
            if not element:
                await asyncio.sleep(0.1)
                continue
            
            try:
                current_bbox = await element.bounding_box()
                
                if last_bbox and self._bboxes_equal(last_bbox, current_bbox):
                    if stable_since is None:
                        stable_since = time.time() * 1000
                    elif time.time() * 1000 - stable_since >= stability_ms:
                        return element
                else:
                    stable_since = None
                
                last_bbox = current_bbox
                
            except Exception:
                stable_since = None
            
            await asyncio.sleep(0.1)
        
        return None
    
    def _bboxes_equal(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
        """Check if two bounding boxes are equal (within tolerance)"""
        tolerance = 2.0  # pixels
        return (
            abs(bbox1.get("x", 0) - bbox2.get("x", 0)) < tolerance and
            abs(bbox1.get("y", 0) - bbox2.get("y", 0)) < tolerance and
            abs(bbox1.get("width", 0) - bbox2.get("width", 0)) < tolerance and
            abs(bbox1.get("height", 0) - bbox2.get("height", 0)) < tolerance
        )
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get statistics about strategy performance"""
        stats = {
            "successful_strategy": self._successful_strategy.__class__.__name__ if self._successful_strategy else None,
            "last_successful_selector": self._last_successful_selector,
            "failure_count": self._failure_count,
            "has_signature": self._element_signature is not None,
            "strategies_available": [s.__class__.__name__ for s in self._strategies]
        }
        return stats
    
    def reset(self):
        """Reset locator state"""
        self._successful_strategy = None
        self._last_successful_selector = None
        self._element_signature = None
        self._failure_count = 0
        self._last_attempt_time = 0


class ResilientElement(Element):
    """
    Enhanced Element class with self-healing capabilities.
    Wraps standard Element with resilient locator.
    """
    
    def __init__(self, 
                 page: BrowserPage,
                 selector: str,
                 description: str = "",
                 config: Optional[ResilientLocatorConfig] = None):
        super().__init__(page, selector)
        self.description = description
        self.resilient_locator = ResilientLocator(
            page, selector, description, config
        )
        self._current_handle: Optional[ElementHandle] = None
    
    async def locate(self, timeout_ms: Optional[int] = None) -> Optional[ElementHandle]:
        """Locate element using resilient strategies"""
        handle = await self.resilient_locator.locate(timeout_ms)
        if handle:
            self._current_handle = handle
            self._element_handle = handle  # Update parent class handle
        return handle
    
    async def click(self, **kwargs) -> None:
        """Click element with self-healing"""
        handle = await self.locate()
        if not handle:
            raise TimeoutError(f"Could not locate element: {self.selector}")
        
        await handle.click(**kwargs)
    
    async def fill(self, text: str, **kwargs) -> None:
        """Fill element with self-healing"""
        handle = await self.locate()
        if not handle:
            raise TimeoutError(f"Could not locate element: {self.selector}")
        
        await handle.fill(text, **kwargs)
    
    async def get_attribute(self, name: str) -> Optional[str]:
        """Get attribute with self-healing"""
        handle = await self.locate()
        if not handle:
            return None
        
        return await handle.get_attribute(name)
    
    async def is_visible(self, timeout_ms: int = 5000) -> bool:
        """Check if element is visible with self-healing"""
        try:
            handle = await self.locate(timeout_ms)
            if not handle:
                return False
            
            return await handle.is_visible()
        except:
            return False
    
    async def wait_for_state(self, 
                            state: str = "visible",
                            timeout_ms: int = 10000) -> bool:
        """Wait for element to reach a specific state"""
        start_time = time.time() * 1000
        
        while time.time() * 1000 - start_time < timeout_ms:
            handle = await self.locate()
            if not handle:
                await asyncio.sleep(0.1)
                continue
            
            try:
                if state == "visible":
                    if await handle.is_visible():
                        return True
                elif state == "hidden":
                    if not await handle.is_visible():
                        return True
                elif state == "enabled":
                    if await handle.is_enabled():
                        return True
                elif state == "disabled":
                    if not await handle.is_enabled():
                        return True
            except:
                pass
            
            await asyncio.sleep(0.1)
        
        return False
    
    def get_locator_stats(self) -> Dict[str, Any]:
        """Get statistics about the resilient locator"""
        return self.resilient_locator.get_strategy_stats()


class ResilientLocatorManager:
    """
    Manager for multiple resilient locators.
    Provides centralized management and caching.
    """
    
    def __init__(self, page: BrowserPage):
        self.page = page
        self._locators: Dict[str, ResilientLocator] = {}
        self._global_cache: Dict[str, Any] = {}
    
    def create_locator(self, 
                      selector: str,
                      description: str = "",
                      config: Optional[ResilientLocatorConfig] = None) -> ResilientLocator:
        """Create or retrieve a resilient locator"""
        key = f"{selector}:{description}"
        
        if key not in self._locators:
            self._locators[key] = ResilientLocator(
                self.page, selector, description, config
            )
        
        return self._locators[key]
    
    def create_element(self,
                      selector: str,
                      description: str = "",
                      config: Optional[ResilientLocatorConfig] = None) -> ResilientElement:
        """Create a resilient element"""
        return ResilientElement(self.page, selector, description, config)
    
    async def locate_all(self, 
                        selectors: List[str],
                        timeout_ms: int = 10000) -> Dict[str, Optional[ElementHandle]]:
        """Locate multiple elements in parallel"""
        tasks = []
        for selector in selectors:
            locator = self.create_locator(selector)
            tasks.append(locator.locate(timeout_ms))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            selector: result if not isinstance(result, Exception) else None
            for selector, result in zip(selectors, results)
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self._locators.clear()
        self._global_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about all locators"""
        stats = {
            "total_locators": len(self._locators),
            "locators": {}
        }
        
        for key, locator in self._locators.items():
            stats["locators"][key] = locator.get_strategy_stats()
        
        return stats


# Export main classes
__all__ = [
    "ResilientLocator",
    "ResilientElement", 
    "ResilientLocatorManager",
    "ResilientLocatorConfig",
    "StrategyType",
    "DetectionStrategy",
    "StrategyFactory"
]