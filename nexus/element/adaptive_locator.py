"""
Adaptive Element Location Engine - Multi-strategy adaptive system for robust element location.

This module replaces static element selectors with a multi-strategy adaptive system
combining CSS/XPath, visual recognition, accessibility tree analysis, and LLM-guided exploration.
Includes automatic fallback chains and self-healing selectors that adapt to UI changes.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import re

from playwright.async_api import Page, ElementHandle, Locator

from nexus.actor.element import Element
from nexus.agent.service import AgentService
from nexus.agent.views import AgentResponse

logger = logging.getLogger(__name__)


class LocatorStrategy(Enum):
    """Available element location strategies."""
    CSS = "css"
    XPATH = "xpath"
    VISUAL = "visual"
    ACCESSIBILITY = "accessibility"
    LLM = "llm"
    TEXT = "text"
    ROLE = "role"
    TEST_ID = "test_id"
    COMBINED = "combined"


@dataclass
class LocationResult:
    """Result of an element location attempt."""
    element: Optional[Element]
    strategy: LocatorStrategy
    confidence: float = 1.0
    selector_used: Optional[str] = None
    fallback_used: bool = False
    time_taken: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorScore:
    """Robustness score for a selector."""
    specificity: float = 0.0  # 0-1, how specific the selector is
    stability: float = 0.0    # 0-1, how stable across page changes
    uniqueness: float = 0.0   # 0-1, how unique on current page
    overall: float = 0.0      # Combined score
    reasons: List[str] = field(default_factory=list)


class BaseLocator(ABC):
    """Abstract base class for element locators."""
    
    def __init__(self, page: Page):
        self.page = page
        self._cache: Dict[str, Element] = {}
    
    @abstractmethod
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements using this strategy."""
        pass
    
    @abstractmethod
    def score_selector(self, selector: str) -> SelectorScore:
        """Score the robustness of a selector."""
        pass
    
    async def clear_cache(self):
        """Clear the element cache."""
        self._cache.clear()


class CssLocator(BaseLocator):
    """CSS selector based locator."""
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements using CSS selectors."""
        start_time = time.time()
        results = []
        
        try:
            elements = await self.page.query_selector_all(selector)
            for element in elements:
                elem = Element(element, self.page)
                results.append(LocationResult(
                    element=elem,
                    strategy=LocatorStrategy.CSS,
                    confidence=0.9,
                    selector_used=selector,
                    time_taken=time.time() - start_time
                ))
        except Exception as e:
            logger.debug(f"CSS selector failed: {selector} - {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score CSS selector robustness."""
        score = SelectorScore()
        
        # Calculate specificity
        id_count = selector.count('#')
        class_count = selector.count('.')
        attr_count = selector.count('[')
        element_count = len(re.findall(r'[a-zA-Z][\w-]*', selector))
        
        if id_count > 0:
            score.specificity = min(1.0, 0.3 + (id_count * 0.2))
            score.reasons.append("Contains ID selector (high specificity)")
        elif class_count > 0:
            score.specificity = min(1.0, 0.2 + (class_count * 0.15))
            score.reasons.append("Contains class selector")
        elif attr_count > 0:
            score.specificity = min(1.0, 0.1 + (attr_count * 0.1))
            score.reasons.append("Contains attribute selector")
        else:
            score.specificity = 0.1
            score.reasons.append("Element selector only (low specificity)")
        
        # Stability assessment
        if ':nth-child' in selector or ':nth-of-type' in selector:
            score.stability = 0.3
            score.reasons.append("Contains positional selectors (unstable)")
        elif '>' in selector or '+' in selector or '~' in selector:
            score.stability = 0.5
            score.reasons.append("Contains combinators (moderately stable)")
        else:
            score.stability = 0.8
            score.reasons.append("No positional or combinator selectors (stable)")
        
        # Overall score
        score.overall = (score.specificity * 0.4 + score.stability * 0.6)
        
        return score


class XPathLocator(BaseLocator):
    """XPath based locator."""
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements using XPath."""
        start_time = time.time()
        results = []
        
        try:
            elements = await self.page.query_selector_all(f"xpath={selector}")
            for element in elements:
                elem = Element(element, self.page)
                results.append(LocationResult(
                    element=elem,
                    strategy=LocatorStrategy.XPATH,
                    confidence=0.85,
                    selector_used=selector,
                    time_taken=time.time() - start_time
                ))
        except Exception as e:
            logger.debug(f"XPath selector failed: {selector} - {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score XPath selector robustness."""
        score = SelectorScore()
        
        # Check for absolute paths
        if selector.startswith('/'):
            score.stability = 0.2
            score.reasons.append("Absolute XPath (very unstable)")
        elif selector.startswith('//'):
            score.stability = 0.6
            score.reasons.append("Relative XPath (moderately stable)")
        
        # Check for positional indexes
        if re.search(r'\[\d+\]', selector):
            score.stability *= 0.7
            score.reasons.append("Contains positional indexes")
        
        # Check for text() or contains()
        if 'text()' in selector or 'contains(' in selector:
            score.specificity = 0.7
            score.reasons.append("Uses text content (good for uniqueness)")
        else:
            score.specificity = 0.5
        
        score.overall = (score.specificity * 0.5 + score.stability * 0.5)
        
        return score


class TextLocator(BaseLocator):
    """Text content based locator."""
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements by text content."""
        start_time = time.time()
        results = []
        
        try:
            # Try exact text match first
            locator = self.page.get_by_text(selector, exact=True)
            elements = await locator.all()
            
            if not elements:
                # Try partial text match
                locator = self.page.get_by_text(selector, exact=False)
                elements = await locator.all()
            
            for element in elements:
                elem = Element(element, self.page)
                results.append(LocationResult(
                    element=elem,
                    strategy=LocatorStrategy.TEXT,
                    confidence=0.8,
                    selector_used=f"text={selector}",
                    time_taken=time.time() - start_time
                ))
        except Exception as e:
            logger.debug(f"Text locator failed: {selector} - {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score text selector robustness."""
        score = SelectorScore()
        
        # Longer text is more specific
        text_length = len(selector)
        if text_length > 20:
            score.specificity = 0.9
            score.reasons.append("Long text (highly specific)")
        elif text_length > 10:
            score.specificity = 0.7
            score.reasons.append("Medium text")
        else:
            score.specificity = 0.3
            score.reasons.append("Short text (may match multiple elements)")
        
        # Text can change with localization
        score.stability = 0.5
        score.reasons.append("Text content may change with UI updates")
        
        score.overall = (score.specificity * 0.6 + score.stability * 0.4)
        
        return score


class RoleLocator(BaseLocator):
    """ARIA role based locator."""
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements by ARIA role."""
        start_time = time.time()
        results = []
        
        try:
            locator = self.page.get_by_role(selector)
            elements = await locator.all()
            
            for element in elements:
                elem = Element(element, self.page)
                results.append(LocationResult(
                    element=elem,
                    strategy=LocatorStrategy.ROLE,
                    confidence=0.95,
                    selector_used=f"role={selector}",
                    time_taken=time.time() - start_time
                ))
        except Exception as e:
            logger.debug(f"Role locator failed: {selector} - {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score role selector robustness."""
        score = SelectorScore()
        
        # ARIA roles are semantic and stable
        score.specificity = 0.8
        score.stability = 0.9
        score.reasons.append("ARIA role (semantically stable)")
        
        score.overall = (score.specificity * 0.3 + score.stability * 0.7)
        
        return score


class TestIdLocator(BaseLocator):
    """Test ID based locator."""
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements by test ID."""
        start_time = time.time()
        results = []
        
        try:
            locator = self.page.get_by_test_id(selector)
            elements = await locator.all()
            
            for element in elements:
                elem = Element(element, self.page)
                results.append(LocationResult(
                    element=elem,
                    strategy=LocatorStrategy.TEST_ID,
                    confidence=0.98,
                    selector_used=f"testid={selector}",
                    time_taken=time.time() - start_time
                ))
        except Exception as e:
            logger.debug(f"Test ID locator failed: {selector} - {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score test ID selector robustness."""
        score = SelectorScore()
        
        # Test IDs are designed for stability
        score.specificity = 0.95
        score.stability = 0.95
        score.reasons.append("Test ID (designed for test stability)")
        
        score.overall = (score.specificity * 0.2 + score.stability * 0.8)
        
        return score


class AccessibilityLocator(BaseLocator):
    """Accessibility tree based locator."""
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements using accessibility tree."""
        start_time = time.time()
        results = []
        
        try:
            # Parse selector as JSON for accessibility properties
            if selector.startswith('{'):
                acc_props = json.loads(selector)
            else:
                # Simple name-based lookup
                acc_props = {"name": selector}
            
            # Use Playwright's accessibility tree
            snapshot = await self.page.accessibility.snapshot()
            elements = self._find_in_tree(snapshot, acc_props)
            
            for element in elements:
                results.append(LocationResult(
                    element=element,
                    strategy=LocatorStrategy.ACCESSIBILITY,
                    confidence=0.85,
                    selector_used=selector,
                    time_taken=time.time() - start_time,
                    metadata={"accessibility_properties": acc_props}
                ))
        except Exception as e:
            logger.debug(f"Accessibility locator failed: {selector} - {e}")
        
        return results
    
    def _find_in_tree(self, node: Dict, props: Dict) -> List[Element]:
        """Recursively find elements in accessibility tree."""
        elements = []
        
        # Check if current node matches
        matches = True
        for key, value in props.items():
            if key not in node or node[key] != value:
                matches = False
                break
        
        if matches and 'element' in node:
            try:
                # Convert to Element if possible
                elements.append(Element(node['element'], self.page))
            except:
                pass
        
        # Recurse through children
        for child in node.get('children', []):
            elements.extend(self._find_in_tree(child, props))
        
        return elements
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score accessibility selector robustness."""
        score = SelectorScore()
        
        # Accessibility properties are semantic
        score.specificity = 0.7
        score.stability = 0.8
        score.reasons.append("Accessibility properties (semantic and stable)")
        
        score.overall = (score.specificity * 0.4 + score.stability * 0.6)
        
        return score


class VisualLocator(BaseLocator):
    """Visual recognition based locator."""
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        super().__init__(page)
        self.agent_service = agent_service
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements using visual recognition."""
        start_time = time.time()
        results = []
        
        if not self.agent_service:
            logger.warning("Visual locator requires AgentService")
            return results
        
        try:
            # Take screenshot for visual analysis
            screenshot = await self.page.screenshot()
            
            # Use LLM to identify element in screenshot
            prompt = f"""
            Analyze this webpage screenshot and find the element described as: "{selector}"
            
            Return a JSON object with:
            - "found": boolean
            - "coordinates": [x, y] if found (center of element)
            - "confidence": 0-1 confidence score
            - "description": brief description of what you found
            
            Only return the JSON, no other text.
            """
            
            response = await self.agent_service.run_with_screenshot(
                prompt=prompt,
                screenshot=screenshot
            )
            
            if response and response.text:
                try:
                    data = json.loads(response.text)
                    if data.get('found'):
                        # Get element at coordinates
                        x, y = data['coordinates']
                        element = await self.page.evaluate(
                            f"document.elementFromPoint({x}, {y})"
                        )
                        
                        if element:
                            elem = Element(element, self.page)
                            results.append(LocationResult(
                                element=elem,
                                strategy=LocatorStrategy.VISUAL,
                                confidence=data.get('confidence', 0.7),
                                selector_used=selector,
                                time_taken=time.time() - start_time,
                                metadata={
                                    "coordinates": [x, y],
                                    "visual_description": data.get('description', '')
                                }
                            ))
                except json.JSONDecodeError:
                    logger.debug("Failed to parse visual locator response")
        
        except Exception as e:
            logger.debug(f"Visual locator failed: {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score visual selector robustness."""
        score = SelectorScore()
        
        # Visual recognition is less specific but can handle dynamic UIs
        score.specificity = 0.6
        score.stability = 0.7  # Can adapt to visual changes
        score.reasons.append("Visual recognition (adaptable but less precise)")
        
        score.overall = (score.specificity * 0.3 + score.stability * 0.7)
        
        return score


class LLMGuidedLocator(BaseLocator):
    """LLM-guided element discovery locator."""
    
    def __init__(self, page: Page, agent_service: AgentService):
        super().__init__(page)
        self.agent_service = agent_service
    
    async def locate(self, selector: str, **kwargs) -> List[LocationResult]:
        """Locate elements using LLM-guided exploration."""
        start_time = time.time()
        results = []
        
        try:
            # Get page content for context
            page_content = await self.page.content()
            
            # Ask LLM to generate selectors
            prompt = f"""
            I need to find an element on a webpage described as: "{selector}"
            
            Here's a portion of the page HTML (truncated):
            {page_content[:5000]}...
            
            Generate 3 different CSS selectors that could locate this element.
            Consider:
            1. ID-based selector if available
            2. Class-based selector
            3. Attribute-based selector
            
            Return a JSON array of selectors, most reliable first.
            Example: ["#submit-button", ".primary-button", "button[type='submit']"]
            
            Only return the JSON array, no other text.
            """
            
            response = await self.agent_service.run(prompt=prompt)
            
            if response and response.text:
                try:
                    selectors = json.loads(response.text)
                    
                    for i, css_selector in enumerate(selectors[:3]):
                        try:
                            element = await self.page.query_selector(css_selector)
                            if element:
                                elem = Element(element, self.page)
                                confidence = 0.9 - (i * 0.1)  # Decrease confidence for fallbacks
                                results.append(LocationResult(
                                    element=elem,
                                    strategy=LocatorStrategy.LLM,
                                    confidence=confidence,
                                    selector_used=css_selector,
                                    time_taken=time.time() - start_time,
                                    metadata={"llm_generated": True, "rank": i + 1}
                                ))
                        except:
                            continue
                except json.JSONDecodeError:
                    logger.debug("Failed to parse LLM locator response")
        
        except Exception as e:
            logger.debug(f"LLM locator failed: {e}")
        
        return results
    
    def score_selector(self, selector: str) -> SelectorScore:
        """Score LLM-generated selector robustness."""
        score = SelectorScore()
        
        # LLM can generate context-aware selectors
        score.specificity = 0.8
        score.stability = 0.6  # Depends on LLM quality
        score.reasons.append("LLM-generated selector (context-aware)")
        
        score.overall = (score.specificity * 0.5 + score.stability * 0.5)
        
        return score


class AdaptiveLocator:
    """
    Adaptive element location engine with multi-strategy support.
    
    Combines multiple location strategies with automatic fallback chains
    and self-healing selectors that adapt to UI changes.
    """
    
    def __init__(
        self,
        page: Page,
        agent_service: Optional[AgentService] = None,
        strategies: Optional[List[LocatorStrategy]] = None,
        max_fallbacks: int = 3,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the adaptive locator.
        
        Args:
            page: Playwright page object
            agent_service: Optional agent service for LLM-based strategies
            strategies: List of strategies to use (default: all available)
            max_fallbacks: Maximum number of fallback attempts
            confidence_threshold: Minimum confidence to accept a result
        """
        self.page = page
        self.agent_service = agent_service
        self.max_fallbacks = max_fallbacks
        self.confidence_threshold = confidence_threshold
        
        # Initialize available strategies
        self._strategies: Dict[LocatorStrategy, BaseLocator] = {
            LocatorStrategy.CSS: CssLocator(page),
            LocatorStrategy.XPATH: XPathLocator(page),
            LocatorStrategy.TEXT: TextLocator(page),
            LocatorStrategy.ROLE: RoleLocator(page),
            LocatorStrategy.TEST_ID: TestIdLocator(page),
            LocatorStrategy.ACCESSIBILITY: AccessibilityLocator(page),
        }
        
        # Add LLM-based strategies if agent service is available
        if agent_service:
            self._strategies[LocatorStrategy.VISUAL] = VisualLocator(page, agent_service)
            self._strategies[LocatorStrategy.LLM] = LLMGuidedLocator(page, agent_service)
        
        # Set active strategies
        if strategies:
            self.active_strategies = [s for s in strategies if s in self._strategies]
        else:
            # Default strategy order (most reliable first)
            self.active_strategies = [
                LocatorStrategy.TEST_ID,
                LocatorStrategy.ROLE,
                LocatorStrategy.ACCESSIBILITY,
                LocatorStrategy.CSS,
                LocatorStrategy.XPATH,
                LocatorStrategy.TEXT,
                LocatorStrategy.VISUAL,
                LocatorStrategy.LLM
            ]
        
        # Cache for successful selectors
        self._selector_cache: Dict[str, Dict[str, Any]] = {}
        
        # History for self-healing
        self._location_history: List[Dict[str, Any]] = []
    
    async def locate(
        self,
        selector: str,
        strategy: Optional[LocatorStrategy] = None,
        **kwargs
    ) -> Optional[Element]:
        """
        Locate an element using adaptive strategies.
        
        Args:
            selector: Element selector or description
            strategy: Specific strategy to use (optional)
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Located Element or None if not found
        """
        start_time = time.time()
        results: List[LocationResult] = []
        
        # Try specific strategy if provided
        if strategy and strategy in self._strategies:
            locator = self._strategies[strategy]
            strategy_results = await locator.locate(selector, **kwargs)
            results.extend(strategy_results)
        
        # Try all active strategies with fallback chain
        fallback_count = 0
        for strat in self.active_strategies:
            if fallback_count >= self.max_fallbacks:
                break
            
            if strat not in self._strategies:
                continue
            
            locator = self._strategies[strat]
            strategy_results = await locator.locate(selector, **kwargs)
            
            for result in strategy_results:
                if result.confidence >= self.confidence_threshold:
                    results.append(result)
                    
                    # Cache successful selector
                    self._cache_successful_selector(
                        selector=selector,
                        strategy=strat,
                        result=result
                    )
                    
                    # Record in history for self-healing
                    self._record_location_attempt(
                        selector=selector,
                        strategy=strat,
                        success=True,
                        time_taken=result.time_taken
                    )
                    
                    return result.element
            
            fallback_count += 1
        
        # If no results found, try LLM-based self-healing
        if not results and self.agent_service:
            healed_result = await self._self_healing_locate(selector, **kwargs)
            if healed_result:
                results.append(healed_result)
        
        # Record failed attempt
        if not results:
            self._record_location_attempt(
                selector=selector,
                strategy=None,
                success=False,
                time_taken=time.time() - start_time
            )
        
        # Return best result if any
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            return best_result.element
        
        return None
    
    async def locate_all(
        self,
        selector: str,
        strategy: Optional[LocatorStrategy] = None,
        **kwargs
    ) -> List[Element]:
        """
        Locate all matching elements.
        
        Args:
            selector: Element selector or description
            strategy: Specific strategy to use (optional)
            **kwargs: Additional arguments
            
        Returns:
            List of located Elements
        """
        results: List[Element] = []
        
        # Try specific strategy if provided
        if strategy and strategy in self._strategies:
            locator = self._strategies[strategy]
            strategy_results = await locator.locate(selector, **kwargs)
            results.extend([r.element for r in strategy_results if r.element])
        
        # Try all active strategies
        for strat in self.active_strategies:
            if strat not in self._strategies:
                continue
            
            locator = self._strategies[strat]
            strategy_results = await locator.locate(selector, **kwargs)
            
            for result in strategy_results:
                if result.element and result.element not in results:
                    results.append(result.element)
        
        return results
    
    async def _self_healing_locate(
        self,
        selector: str,
        **kwargs
    ) -> Optional[LocationResult]:
        """
        Attempt to locate element using self-healing techniques.
        
        Uses historical data and LLM to generate new selectors when
        existing ones fail.
        """
        if not self.agent_service:
            return None
        
        try:
            # Get historical successful selectors for similar elements
            similar_selectors = self._get_similar_successful_selectors(selector)
            
            # Take screenshot for visual context
            screenshot = await self.page.screenshot()
            
            # Ask LLM to generate new selectors based on context
            prompt = f"""
            I'm trying to find an element described as: "{selector}"
            
            Previous successful selectors for similar elements:
            {json.dumps(similar_selectors[:5], indent=2)}
            
            The current page has changed and old selectors no longer work.
            Based on the screenshot and description, generate 3 new CSS selectors
            that might locate this element on the updated page.
            
            Return a JSON array of selectors, most likely to work first.
            Only return the JSON array.
            """
            
            response = await self.agent_service.run_with_screenshot(
                prompt=prompt,
                screenshot=screenshot
            )
            
            if response and response.text:
                try:
                    new_selectors = json.loads(response.text)
                    
                    for i, new_selector in enumerate(new_selectors[:3]):
                        try:
                            element = await self.page.query_selector(new_selector)
                            if element:
                                elem = Element(element, self.page)
                                return LocationResult(
                                    element=elem,
                                    strategy=LocatorStrategy.LLM,
                                    confidence=0.7 - (i * 0.1),
                                    selector_used=new_selector,
                                    fallback_used=True,
                                    metadata={
                                        "self_healed": True,
                                        "original_selector": selector,
                                        "generated_selectors": new_selectors
                                    }
                                )
                        except:
                            continue
                except json.JSONDecodeError:
                    pass
        
        except Exception as e:
            logger.debug(f"Self-healing failed: {e}")
        
        return None
    
    def _cache_successful_selector(
        self,
        selector: str,
        strategy: LocatorStrategy,
        result: LocationResult
    ):
        """Cache a successful selector for future use."""
        if selector not in self._selector_cache:
            self._selector_cache[selector] = {
                "strategy": strategy,
                "last_used": time.time(),
                "success_count": 1,
                "selector_used": result.selector_used,
                "metadata": result.metadata
            }
        else:
            cache_entry = self._selector_cache[selector]
            cache_entry["success_count"] += 1
            cache_entry["last_used"] = time.time()
    
    def _record_location_attempt(
        self,
        selector: str,
        strategy: Optional[LocatorStrategy],
        success: bool,
        time_taken: float
    ):
        """Record location attempt for analysis."""
        self._location_history.append({
            "selector": selector,
            "strategy": strategy.value if strategy else None,
            "success": success,
            "time_taken": time_taken,
            "timestamp": time.time()
        })
        
        # Keep history limited
        if len(self._location_history) > 1000:
            self._location_history = self._location_history[-500:]
    
    def _get_similar_successful_selectors(self, selector: str) -> List[str]:
        """Get previously successful selectors for similar elements."""
        similar = []
        
        # Simple similarity: check if selector contains similar keywords
        selector_keywords = set(re.findall(r'\w+', selector.lower()))
        
        for cached_selector, data in self._selector_cache.items():
            cached_keywords = set(re.findall(r'\w+', cached_selector.lower()))
            
            # Calculate similarity
            if selector_keywords and cached_keywords:
                similarity = len(selector_keywords & cached_keywords) / len(selector_keywords | cached_keywords)
                if similarity > 0.3:  # 30% similarity threshold
                    similar.append({
                        "selector": cached_selector,
                        "similarity": similarity,
                        "success_count": data["success_count"]
                    })
        
        # Sort by similarity and success count
        similar.sort(key=lambda x: (x["similarity"], x["success_count"]), reverse=True)
        return [s["selector"] for s in similar[:10]]
    
    async def generate_robust_selector(
        self,
        element: Element,
        preferred_strategies: Optional[List[LocatorStrategy]] = None
    ) -> Dict[str, Any]:
        """
        Generate a robust selector for an element.
        
        Args:
            element: Element to generate selector for
            preferred_strategies: Strategies to prefer
            
        Returns:
            Dictionary with selector information and scores
        """
        if not preferred_strategies:
            preferred_strategies = self.active_strategies
        
        best_selectors = []
        
        for strategy in preferred_strategies:
            if strategy not in self._strategies:
                continue
            
            locator = self._strategies[strategy]
            
            # Try to generate selector based on strategy
            try:
                if strategy == LocatorStrategy.CSS:
                    selector = await self._generate_css_selector(element)
                elif strategy == LocatorStrategy.XPATH:
                    selector = await self._generate_xpath_selector(element)
                elif strategy == LocatorStrategy.TEXT:
                    selector = await self._get_element_text(element)
                elif strategy == LocatorStrategy.ROLE:
                    selector = await self._get_element_role(element)
                elif strategy == LocatorStrategy.TEST_ID:
                    selector = await self._get_test_id(element)
                else:
                    continue
                
                if selector:
                    score = locator.score_selector(selector)
                    best_selectors.append({
                        "strategy": strategy.value,
                        "selector": selector,
                        "score": score.overall,
                        "specificity": score.specificity,
                        "stability": score.stability,
                        "reasons": score.reasons
                    })
            except Exception as e:
                logger.debug(f"Failed to generate {strategy.value} selector: {e}")
        
        # Sort by overall score
        best_selectors.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "element_info": {
                "tag": await element.get_tag_name(),
                "text": (await element.get_text())[:100] if await element.get_text() else "",
                "attributes": await element.get_attributes()
            },
            "selectors": best_selectors[:5],  # Top 5 selectors
            "recommendation": best_selectors[0] if best_selectors else None
        }
    
    async def _generate_css_selector(self, element: Element) -> Optional[str]:
        """Generate CSS selector for element."""
        try:
            # Try to get ID first
            element_id = await element.get_attribute("id")
            if element_id:
                return f"#{element_id}"
            
            # Try data-testid
            test_id = await element.get_attribute("data-testid")
            if test_id:
                return f"[data-testid='{test_id}']"
            
            # Try unique class combination
            class_name = await element.get_attribute("class")
            if class_name:
                classes = class_name.split()
                if len(classes) <= 3:  # Don't use too many classes
                    return "." + ".".join(classes)
            
            # Generate path-based selector
            return await self._generate_css_path(element)
        
        except Exception as e:
            logger.debug(f"CSS selector generation failed: {e}")
            return None
    
    async def _generate_css_path(self, element: Element) -> str:
        """Generate CSS path selector."""
        try:
            # Use Playwright's built-in method if available
            handle = element.handle
            if hasattr(handle, "generate_locator"):
                return await handle.generate_locator()
            
            # Fallback to manual generation
            path = []
            current = handle
            
            while current:
                tag = await current.evaluate("el => el.tagName.toLowerCase()")
                parent = await current.evaluate("el => el.parentElement")
                
                if not parent:
                    break
                
                # Get siblings of same tag
                siblings = await parent.evaluate(f"""
                    el => Array.from(el.children).filter(c => c.tagName.toLowerCase() === '{tag}')
                """)
                
                if len(siblings) > 1:
                    index = siblings.index(current) + 1
                    path.append(f"{tag}:nth-of-type({index})")
                else:
                    path.append(tag)
                
                current = parent
            
            path.reverse()
            return " > ".join(path)
        
        except Exception as e:
            logger.debug(f"CSS path generation failed: {e}")
            return ""
    
    async def _generate_xpath_selector(self, element: Element) -> Optional[str]:
        """Generate XPath selector for element."""
        try:
            handle = element.handle
            
            # Try to get unique attributes
            element_id = await handle.evaluate("el => el.id")
            if element_id:
                return f"//*[@id='{element_id}']"
            
            # Generate XPath path
            return await handle.evaluate("""
                el => {
                    function getXPath(el) {
                        if (el.id) return `//*[@id="${el.id}"]`;
                        if (el === document.body) return '/html/body';
                        
                        let ix = 0;
                        let siblings = el.parentNode.childNodes;
                        for (let i = 0; i < siblings.length; i++) {
                            let sibling = siblings[i];
                            if (sibling === el) {
                                return getXPath(el.parentNode) + '/' + el.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                            }
                            if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {
                                ix++;
                            }
                        }
                    }
                    return getXPath(el);
                }
            """)
        
        except Exception as e:
            logger.debug(f"XPath generation failed: {e}")
            return None
    
    async def _get_element_text(self, element: Element) -> Optional[str]:
        """Get element text for text-based selector."""
        try:
            text = await element.get_text()
            if text and len(text.strip()) > 0:
                # Return first 50 chars for selector
                return text.strip()[:50]
        except:
            pass
        return None
    
    async def _get_element_role(self, element: Element) -> Optional[str]:
        """Get ARIA role for element."""
        try:
            role = await element.get_attribute("role")
            if role:
                return role
            
            # Try to infer role from tag
            tag = await element.get_tag_name()
            role_map = {
                "button": "button",
                "a": "link",
                "input": "textbox",
                "textarea": "textbox",
                "select": "combobox",
                "img": "img",
                "h1": "heading",
                "h2": "heading",
                "h3": "heading",
                "h4": "heading",
                "h5": "heading",
                "h6": "heading"
            }
            return role_map.get(tag.lower())
        except:
            return None
    
    async def _get_test_id(self, element: Element) -> Optional[str]:
        """Get test ID for element."""
        try:
            # Try common test ID attributes
            test_attrs = ["data-testid", "data-test-id", "data-test", "test-id"]
            for attr in test_attrs:
                value = await element.get_attribute(attr)
                if value:
                    return value
        except:
            pass
        return None
    
    async def heal_broken_selector(
        self,
        old_selector: str,
        new_selector: str,
        element_description: str
    ):
        """
        Record a healed selector for future reference.
        
        Args:
            old_selector: Selector that stopped working
            new_selector: New working selector
            element_description: Description of the element
        """
        # Update cache
        self._selector_cache[element_description] = {
            "strategy": LocatorStrategy.CSS,  # Assume CSS for now
            "last_used": time.time(),
            "success_count": 1,
            "selector_used": new_selector,
            "metadata": {
                "healed_from": old_selector,
                "healing_time": time.time()
            }
        }
        
        logger.info(f"Healed selector for '{element_description}': {old_selector} -> {new_selector}")
    
    async def clear_caches(self):
        """Clear all caches."""
        self._selector_cache.clear()
        self._location_history.clear()
        
        for locator in self._strategies.values():
            await locator.clear_cache()
    
    def get_location_stats(self) -> Dict[str, Any]:
        """Get statistics about location attempts."""
        if not self._location_history:
            return {}
        
        total_attempts = len(self._location_history)
        successful_attempts = sum(1 for h in self._location_history if h["success"])
        
        strategy_stats = {}
        for strategy in LocatorStrategy:
            attempts = [h for h in self._location_history if h["strategy"] == strategy.value]
            if attempts:
                success = sum(1 for a in attempts if a["success"])
                strategy_stats[strategy.value] = {
                    "attempts": len(attempts),
                    "success": success,
                    "success_rate": success / len(attempts) if attempts else 0
                }
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts if total_attempts else 0,
            "strategy_stats": strategy_stats,
            "cache_size": len(self._selector_cache)
        }


# Convenience function for quick element location
async def find_element(
    page: Page,
    selector: str,
    strategy: Optional[LocatorStrategy] = None,
    agent_service: Optional[AgentService] = None,
    **kwargs
) -> Optional[Element]:
    """
    Quick function to find an element using adaptive location.
    
    Args:
        page: Playwright page object
        selector: Element selector or description
        strategy: Optional specific strategy
        agent_service: Optional agent service for LLM strategies
        **kwargs: Additional arguments
        
    Returns:
        Located Element or None
    """
    locator = AdaptiveLocator(page, agent_service=agent_service)
    return await locator.locate(selector, strategy=strategy, **kwargs)


# Export main classes and functions
__all__ = [
    'AdaptiveLocator',
    'LocatorStrategy',
    'LocationResult',
    'SelectorScore',
    'BaseLocator',
    'CssLocator',
    'XPathLocator',
    'TextLocator',
    'RoleLocator',
    'TestIdLocator',
    'AccessibilityLocator',
    'VisualLocator',
    'LLMGuidedLocator',
    'find_element'
]