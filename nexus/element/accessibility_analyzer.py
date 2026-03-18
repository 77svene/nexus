"""
Adaptive Element Location Engine for nexus
Implements multi-strategy element location with self-healing selectors
"""

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from playwright.async_api import ElementHandle, Page, Locator

from nexus.actor.element import Element
from nexus.agent.service import AgentService

logger = logging.getLogger(__name__)


class LocatorStrategy(Enum):
    """Available element location strategies"""
    CSS = "css"
    XPATH = "xpath"
    ACCESSIBILITY = "accessibility"
    VISUAL = "visual"
    TEXT = "text"
    ROLE = "role"
    TEST_ID = "test_id"
    LLM_GUIDED = "llm_guided"


@dataclass
class ElementSignature:
    """Signature of an element for identification and tracking"""
    tag_name: str
    attributes: Dict[str, str]
    text_content: Optional[str] = None
    accessibility_properties: Dict[str, Any] = field(default_factory=dict)
    bounding_box: Optional[Dict[str, float]] = None
    visual_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tag_name": self.tag_name,
            "attributes": self.attributes,
            "text_content": self.text_content,
            "accessibility_properties": self.accessibility_properties,
            "bounding_box": self.bounding_box,
            "visual_hash": self.visual_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElementSignature':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SelectorCandidate:
    """A candidate selector with metadata"""
    selector: str
    strategy: LocatorStrategy
    confidence: float = 0.0
    robustness_score: float = 0.0
    last_success: Optional[float] = None
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseElementLocator(ABC):
    """Base class for element locators"""
    
    def __init__(self, page: Page):
        self.page = page
        self.strategy: LocatorStrategy = None
    
    @abstractmethod
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Locate element using this strategy"""
        pass
    
    @abstractmethod
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """Generate selector for given element using this strategy"""
        pass


class CSSLocator(BaseElementLocator):
    """CSS selector based locator"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.strategy = LocatorStrategy.CSS
    
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Locate element using CSS selector"""
        if isinstance(target, str):
            # Direct CSS selector
            try:
                element = await self.page.query_selector(target)
                return element
            except Exception as e:
                logger.debug(f"CSS selector failed: {target}, error: {e}")
                return None
        elif isinstance(target, dict) and "selector" in target:
            # Selector candidate from dict
            try:
                element = await self.page.query_selector(target["selector"])
                return element
            except Exception:
                return None
        return None
    
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """Generate CSS selector for element"""
        try:
            # Get element properties
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            element_id = await element.evaluate("el => el.id")
            class_list = await element.evaluate("el => Array.from(el.classList)")
            
            # Try ID first (most specific)
            if element_id:
                selector = f"#{element_id}"
                return SelectorCandidate(
                    selector=selector,
                    strategy=LocatorStrategy.CSS,
                    confidence=0.95,
                    robustness_score=0.9
                )
            
            # Try unique class combination
            if class_list:
                class_selector = "." + ".".join(class_list[:3])  # Limit to first 3 classes
                count = await self.page.evaluate(f"document.querySelectorAll('{class_selector}').length")
                if count == 1:
                    return SelectorCandidate(
                        selector=class_selector,
                        strategy=LocatorStrategy.CSS,
                        confidence=0.85,
                        robustness_score=0.7
                    )
            
            # Generate path-based selector
            selector = await element.evaluate("""el => {
                const path = [];
                while (el && el.nodeType === Node.ELEMENT_NODE) {
                    let selector = el.tagName.toLowerCase();
                    if (el.id) {
                        selector += '#' + el.id;
                        path.unshift(selector);
                        break;
                    } else {
                        let sibling = el;
                        let nth = 1;
                        while (sibling.previousElementSibling) {
                            sibling = sibling.previousElementSibling;
                            if (sibling.tagName === el.tagName) nth++;
                        }
                        if (nth !== 1) selector += ':nth-of-type(' + nth + ')';
                    }
                    path.unshift(selector);
                    el = el.parentElement;
                }
                return path.join(' > ');
            }""")
            
            return SelectorCandidate(
                selector=selector,
                strategy=LocatorStrategy.CSS,
                confidence=0.7,
                robustness_score=0.5
            )
            
        except Exception as e:
            logger.error(f"Failed to generate CSS selector: {e}")
            return None


class XPathLocator(BaseElementLocator):
    """XPath based locator"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.strategy = LocatorStrategy.XPATH
    
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Locate element using XPath"""
        xpath = target if isinstance(target, str) else target.get("selector", "")
        if not xpath:
            return None
        
        try:
            # Use Playwright's locator with xpath
            locator = self.page.locator(f"xpath={xpath}")
            element = await locator.element_handle()
            return element
        except Exception as e:
            logger.debug(f"XPath failed: {xpath}, error: {e}")
            return None
    
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """Generate XPath for element"""
        try:
            xpath = await element.evaluate("""el => {
                const idx = (sib, name) => 
                    sib ? idx(sib.previousElementSibling, name || sib.tagName) + (sib.tagName === name) : 1;
                const segs = el => !el || el.nodeType !== 1 ? 
                    [''] : el.id ? 
                        [`//*[@id="${el.id}"]`] : 
                        [...segs(el.parentNode), `${el.tagName.toLowerCase()}[${idx(el)}]`];
                return segs(el).join('/');
            }""")
            
            return SelectorCandidate(
                selector=xpath,
                strategy=LocatorStrategy.XPATH,
                confidence=0.75,
                robustness_score=0.6
            )
        except Exception as e:
            logger.error(f"Failed to generate XPath: {e}")
            return None


class AccessibilityLocator(BaseElementLocator):
    """Accessibility tree based locator"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.strategy = LocatorStrategy.ACCESSIBILITY
        self._accessibility_tree = None
    
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Locate element using accessibility properties"""
        if isinstance(target, ElementSignature):
            # Use accessibility properties from signature
            props = target.accessibility_properties
            role = props.get("role")
            name = props.get("name")
            value = props.get("value")
            
            if role:
                # Build accessibility selector
                selector_parts = [f'role={role}']
                if name:
                    selector_parts.append(f'name="{name}"')
                if value:
                    selector_parts.append(f'value="{value}"')
                
                selector = " >> ".join(selector_parts)
                try:
                    locator = self.page.locator(selector)
                    return await locator.element_handle()
                except Exception:
                    pass
        
        elif isinstance(target, dict):
            # Direct accessibility query
            role = target.get("role")
            name = target.get("name")
            
            if role:
                try:
                    # Use Playwright's built-in role selectors
                    options = {}
                    if name:
                        options["name"] = name
                    
                    locator = self.page.get_by_role(role, **options)
                    return await locator.element_handle()
                except Exception as e:
                    logger.debug(f"Accessibility locator failed: {e}")
        
        return None
    
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """Generate accessibility-based selector"""
        try:
            # Get accessibility properties
            props = await element.evaluate("""el => {
                const role = el.getAttribute('role') || 
                             (el.tagName === 'BUTTON' ? 'button' : 
                              el.tagName === 'A' ? 'link' : 
                              el.tagName === 'INPUT' ? el.type || 'textbox' : 
                              el.tagName.toLowerCase());
                
                const name = el.getAttribute('aria-label') || 
                            el.getAttribute('title') || 
                            el.textContent?.trim().substring(0, 50) || 
                            el.getAttribute('alt') || 
                            el.getAttribute('placeholder');
                
                const value = el.value || el.getAttribute('aria-valuenow');
                
                return { role, name, value };
            }""")
            
            if props.get("role"):
                selector_parts = [f'role={props["role"]}']
                if props.get("name"):
                    selector_parts.append(f'name="{props["name"]}"')
                
                selector = " >> ".join(selector_parts)
                
                return SelectorCandidate(
                    selector=selector,
                    strategy=LocatorStrategy.ACCESSIBILITY,
                    confidence=0.8,
                    robustness_score=0.85,
                    metadata={"accessibility_props": props}
                )
        except Exception as e:
            logger.error(f"Failed to generate accessibility selector: {e}")
        
        return None


class VisualLocator(BaseElementLocator):
    """Visual recognition based locator (placeholder for ML integration)"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.strategy = LocatorStrategy.VISUAL
    
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Locate element using visual recognition"""
        # This is a placeholder - in production, integrate with visual AI service
        # For now, return None to fall back to other strategies
        logger.debug("Visual locator not yet implemented")
        return None
    
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """Generate visual signature for element"""
        try:
            # Get bounding box and visual properties
            box = await element.bounding_box()
            if box:
                # Create visual hash based on element properties
                props = await element.evaluate("""el => {
                    const style = window.getComputedStyle(el);
                    return {
                        backgroundColor: style.backgroundColor,
                        color: style.color,
                        fontSize: style.fontSize,
                        fontWeight: style.fontWeight,
                        borderRadius: style.borderRadius,
                        boxShadow: style.boxShadow
                    };
                }""")
                
                # Create hash from visual properties
                visual_str = json.dumps(props, sort_keys=True)
                visual_hash = hashlib.md5(visual_str.encode()).hexdigest()
                
                return SelectorCandidate(
                    selector=f"visual:{visual_hash}",
                    strategy=LocatorStrategy.VISUAL,
                    confidence=0.6,
                    robustness_score=0.4,
                    metadata={
                        "bounding_box": box,
                        "visual_properties": props,
                        "visual_hash": visual_hash
                    }
                )
        except Exception as e:
            logger.error(f"Failed to generate visual signature: {e}")
        
        return None


class TextLocator(BaseElementLocator):
    """Text content based locator"""
    
    def __init__(self, page: Page):
        super().__init__(page)
        self.strategy = LocatorStrategy.TEXT
    
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Locate element by text content"""
        text = target if isinstance(target, str) else target.get("text", "")
        if not text:
            return None
        
        try:
            # Use Playwright's text locator
            locator = self.page.get_by_text(text, exact=kwargs.get("exact", False))
            return await locator.element_handle()
        except Exception as e:
            logger.debug(f"Text locator failed: {text}, error: {e}")
            return None
    
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """Generate text-based selector"""
        try:
            text = await element.evaluate("el => el.textContent?.trim().substring(0, 100)")
            if text:
                # Escape quotes for selector
                escaped_text = text.replace('"', '\\"').replace("'", "\\'")
                return SelectorCandidate(
                    selector=f'text="{escaped_text}"',
                    strategy=LocatorStrategy.TEXT,
                    confidence=0.7,
                    robustness_score=0.5,
                    metadata={"text_content": text}
                )
        except Exception as e:
            logger.error(f"Failed to generate text selector: {e}")
        
        return None


class LLMGuidedLocator(BaseElementLocator):
    """LLM-guided element discovery for complex cases"""
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        super().__init__(page)
        self.strategy = LocatorStrategy.LLM_GUIDED
        self.agent_service = agent_service or AgentService()
    
    async def locate(self, target: Union[str, Dict, ElementSignature], **kwargs) -> Optional[ElementHandle]:
        """Use LLM to discover element when other methods fail"""
        if not self.agent_service:
            logger.warning("LLM locator requires AgentService")
            return None
        
        description = target if isinstance(target, str) else str(target)
        
        try:
            # Get page context for LLM
            page_content = await self.page.content()
            page_text = await self.page.evaluate("document.body.innerText")
            
            # Create prompt for LLM
            prompt = f"""
            Find the element on this webpage that matches: {description}
            
            Page text content (truncated):
            {page_text[:2000]}
            
            Return a JSON object with:
            - strategy: one of [css, xpath, text, role, test_id]
            - selector: the selector to find the element
            - confidence: 0-1 confidence score
            - reasoning: brief explanation
            
            Only return the JSON, no other text.
            """
            
            # Get LLM response
            response = await self.agent_service.generate_response(prompt)
            
            # Parse response
            try:
                result = json.loads(response)
                strategy = LocatorStrategy(result.get("strategy", "css"))
                selector = result.get("selector")
                
                if selector:
                    # Try the suggested selector
                    locator_map = {
                        LocatorStrategy.CSS: lambda: self.page.query_selector(selector),
                        LocatorStrategy.XPATH: lambda: self.page.locator(f"xpath={selector}").element_handle(),
                        LocatorStrategy.TEXT: lambda: self.page.get_by_text(selector).element_handle(),
                        LocatorStrategy.ROLE: lambda: self.page.get_by_role(selector).element_handle(),
                        LocatorStrategy.TEST_ID: lambda: self.page.get_by_test_id(selector).element_handle(),
                    }
                    
                    if strategy in locator_map:
                        element = await locator_map[strategy]()
                        if element:
                            return element
            except json.JSONDecodeError:
                logger.warning(f"LLM returned invalid JSON: {response}")
            
        except Exception as e:
            logger.error(f"LLM locator failed: {e}")
        
        return None
    
    async def generate_selector(self, element: ElementHandle) -> Optional[SelectorCandidate]:
        """LLM doesn't generate selectors, only locates"""
        return None


class SelectorRobustnessScorer:
    """Scores selector robustness and generates fallbacks"""
    
    @staticmethod
    def score_selector(selector: str, strategy: LocatorStrategy) -> float:
        """Score selector robustness (0-1, higher is more robust)"""
        score = 0.5  # Base score
        
        if strategy == LocatorStrategy.CSS:
            # ID-based selectors are most robust
            if selector.startswith("#"):
                score = 0.9
            # Class-based selectors are moderately robust
            elif selector.startswith("."):
                score = 0.7
            # Attribute selectors
            elif "[" in selector and "]" in selector:
                score = 0.6
            # Path-based selectors are fragile
            elif ">" in selector or "nth" in selector:
                score = 0.3
        
        elif strategy == LocatorStrategy.ACCESSIBILITY:
            # Accessibility selectors are generally robust
            score = 0.85
            if "role=" in selector and "name=" in selector:
                score = 0.9
        
        elif strategy == LocatorStrategy.XPATH:
            # Absolute XPaths are fragile
            if selector.startswith("/html"):
                score = 0.2
            # Relative XPaths with IDs are better
            elif "id=" in selector:
                score = 0.7
            else:
                score = 0.4
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def generate_fallbacks(selector: str, strategy: LocatorStrategy) -> List[SelectorCandidate]:
        """Generate fallback selectors for a given selector"""
        fallbacks = []
        
        if strategy == LocatorStrategy.CSS:
            # Try to extract useful parts from CSS selector
            if "[" in selector and "]" in selector:
                # Extract attribute selectors
                attr_pattern = r'\[([^\]]+)\]'
                matches = re.findall(attr_pattern, selector)
                for attr in matches[:2]:  # Limit to first 2 attributes
                    fallbacks.append(SelectorCandidate(
                        selector=f"[{attr}]",
                        strategy=LocatorStrategy.CSS,
                        confidence=0.5,
                        robustness_score=0.4
                    ))
        
        elif strategy == LocatorStrategy.ACCESSIBILITY:
            # Try with just role if name is present
            if "role=" in selector and "name=" in selector:
                role_match = re.search(r'role=([^\s]+)', selector)
                if role_match:
                    fallbacks.append(SelectorCandidate(
                        selector=f"role={role_match.group(1)}",
                        strategy=LocatorStrategy.ACCESSIBILITY,
                        confidence=0.6,
                        robustness_score=0.7
                    ))
        
        return fallbacks


class AdaptiveElementLocator:
    """
    Main adaptive element location engine
    Combines multiple strategies with fallback chains
    """
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        self.page = page
        self.agent_service = agent_service
        
        # Initialize all locators
        self.locators: Dict[LocatorStrategy, BaseElementLocator] = {
            LocatorStrategy.CSS: CSSLocator(page),
            LocatorStrategy.XPATH: XPathLocator(page),
            LocatorStrategy.ACCESSIBILITY: AccessibilityLocator(page),
            LocatorStrategy.VISUAL: VisualLocator(page),
            LocatorStrategy.TEXT: TextLocator(page),
            LocatorStrategy.LLM_GUIDED: LLMGuidedLocator(page, agent_service),
        }
        
        # Default strategy order (most to least reliable)
        self.strategy_order = [
            LocatorStrategy.ACCESSIBILITY,
            LocatorStrategy.CSS,
            LocatorStrategy.XPATH,
            LocatorStrategy.TEXT,
            LocatorStrategy.VISUAL,
            LocatorStrategy.LLM_GUIDED,
        ]
        
        # Cache for successful selectors
        self._selector_cache: Dict[str, List[SelectorCandidate]] = {}
        self.scorer = SelectorRobustnessScorer()
    
    async def locate_element(
        self,
        target: Union[str, Dict, ElementSignature],
        strategies: Optional[List[LocatorStrategy]] = None,
        **kwargs
    ) -> Optional[ElementHandle]:
        """
        Locate element using adaptive strategy with fallbacks
        
        Args:
            target: Element description (selector, text, or signature)
            strategies: Ordered list of strategies to try (None for default)
            **kwargs: Additional arguments for specific strategies
        
        Returns:
            ElementHandle if found, None otherwise
        """
        strategies = strategies or self.strategy_order
        
        for strategy in strategies:
            locator = self.locators.get(strategy)
            if not locator:
                continue
            
            try:
                logger.debug(f"Trying strategy: {strategy.value}")
                element = await locator.locate(target, **kwargs)
                
                if element:
                    # Verify element is visible and interactable
                    is_visible = await element.is_visible()
                    if is_visible:
                        logger.info(f"Element found with strategy: {strategy.value}")
                        
                        # Generate and cache selector for future use
                        candidate = await locator.generate_selector(element)
                        if candidate:
                            cache_key = self._get_cache_key(target)
                            if cache_key not in self._selector_cache:
                                self._selector_cache[cache_key] = []
                            self._selector_cache[cache_key].append(candidate)
                        
                        return element
            except Exception as e:
                logger.debug(f"Strategy {strategy.value} failed: {e}")
                continue
        
        logger.warning(f"Element not found with any strategy: {target}")
        return None
    
    async def locate_with_self_healing(
        self,
        target: Union[str, Dict, ElementSignature],
        known_selectors: Optional[List[SelectorCandidate]] = None,
        **kwargs
    ) -> Optional[ElementHandle]:
        """
        Locate element with self-healing capabilities
        
        Args:
            target: Element description
            known_selectors: Previously successful selectors for this element
            **kwargs: Additional arguments
        
        Returns:
            ElementHandle if found, None otherwise
        """
        # Try known selectors first
        if known_selectors:
            # Sort by robustness score
            known_selectors.sort(key=lambda x: x.robustness_score, reverse=True)
            
            for candidate in known_selectors:
                locator = self.locators.get(candidate.strategy)
                if locator:
                    try:
                        element = await locator.locate({"selector": candidate.selector})
                        if element and await element.is_visible():
                            # Update success timestamp
                            candidate.last_success = asyncio.get_event_loop().time()
                            candidate.failure_count = 0
                            return element
                        else:
                            candidate.failure_count += 1
                    except Exception:
                        candidate.failure_count += 1
        
        # Fall back to adaptive search
        element = await self.locate_element(target, **kwargs)
        
        if element:
            # Generate new selectors and add to known selectors
            new_candidates = await self.generate_all_selectors(element)
            if known_selectors is not None:
                known_selectors.extend(new_candidates)
        
        return element
    
    async def generate_all_selectors(self, element: ElementHandle) -> List[SelectorCandidate]:
        """Generate selectors using all available strategies"""
        candidates = []
        
        for strategy, locator in self.locators.items():
            if strategy == LocatorStrategy.LLM_GUIDED:
                continue  # LLM doesn't generate selectors
            
            try:
                candidate = await locator.generate_selector(element)
                if candidate:
                    # Score the selector
                    candidate.robustness_score = self.scorer.score_selector(
                        candidate.selector, candidate.strategy
                    )
                    candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Failed to generate {strategy.value} selector: {e}")
        
        return candidates
    
    def _get_cache_key(self, target: Union[str, Dict, ElementSignature]) -> str:
        """Generate cache key for target"""
        if isinstance(target, str):
            return f"str:{target}"
        elif isinstance(target, dict):
            return f"dict:{json.dumps(target, sort_keys=True)}"
        elif isinstance(target, ElementSignature):
            return f"sig:{hashlib.md5(json.dumps(target.to_dict(), sort_keys=True).encode()).hexdigest()}"
        return f"unknown:{str(target)}"
    
    async def heal_broken_selector(
        self,
        broken_selector: str,
        strategy: LocatorStrategy,
        context: Optional[Dict] = None
    ) -> Optional[SelectorCandidate]:
        """
        Attempt to heal a broken selector
        
        Args:
            broken_selector: The selector that no longer works
            strategy: Strategy of the broken selector
            context: Additional context about the element
        
        Returns:
            New selector candidate if healing successful
        """
        logger.info(f"Attempting to heal broken selector: {broken_selector}")
        
        # Generate fallbacks
        fallbacks = self.scorer.generate_fallbacks(broken_selector, strategy)
        
        for fallback in fallbacks:
            locator = self.locators.get(fallback.strategy)
            if locator:
                try:
                    element = await locator.locate({"selector": fallback.selector})
                    if element:
                        logger.info(f"Healed selector with fallback: {fallback.selector}")
                        return fallback
                except Exception:
                    continue
        
        # Try other strategies
        if context and "description" in context:
            element = await self.locate_element(context["description"])
            if element:
                candidates = await self.generate_all_selectors(element)
                if candidates:
                    # Return the most robust candidate
                    candidates.sort(key=lambda x: x.robustness_score, reverse=True)
                    return candidates[0]
        
        return None
    
    async def get_element_signature(self, element: ElementHandle) -> ElementSignature:
        """Create a signature for an element"""
        try:
            # Get basic properties
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            attributes = await element.evaluate("""el => {
                const attrs = {};
                for (const attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }""")
            
            text_content = await element.evaluate("el => el.textContent?.trim().substring(0, 200)")
            
            # Get accessibility properties
            accessibility_props = await element.evaluate("""el => {
                const props = {};
                props.role = el.getAttribute('role');
                props.name = el.getAttribute('aria-label') || el.getAttribute('title');
                props.value = el.value || el.getAttribute('aria-valuenow');
                props.description = el.getAttribute('aria-describedby');
                return props;
            }""")
            
            # Get bounding box
            bounding_box = await element.bounding_box()
            
            # Get visual hash (simplified)
            visual_hash = await element.evaluate("""el => {
                const style = window.getComputedStyle(el);
                const hashParts = [
                    style.backgroundColor,
                    style.color,
                    style.fontSize,
                    el.offsetWidth,
                    el.offsetHeight
                ];
                return hashParts.join('|');
            }""")
            
            return ElementSignature(
                tag_name=tag_name,
                attributes=attributes,
                text_content=text_content,
                accessibility_properties=accessibility_props,
                bounding_box=bounding_box,
                visual_hash=visual_hash
            )
        except Exception as e:
            logger.error(f"Failed to create element signature: {e}")
            return ElementSignature(tag_name="unknown", attributes={})


# Integration with existing Element class
def enhance_element_with_adaptive_locator(element: Element, page: Page, agent_service: Optional[AgentService] = None):
    """Enhance existing Element class with adaptive locator capabilities"""
    locator = AdaptiveElementLocator(page, agent_service)
    
    # Store original methods
    original_find = element.find if hasattr(element, 'find') else None
    
    async def enhanced_find(selector: str = None, **kwargs) -> Optional[ElementHandle]:
        """Enhanced find method with adaptive location"""
        if selector:
            # Try original method first
            if original_find:
                try:
                    result = await original_find(selector, **kwargs)
                    if result:
                        return result
                except Exception:
                    pass
            
            # Fall back to adaptive locator
            return await locator.locate_element(selector, **kwargs)
        return None
    
    # Replace find method
    element.find = enhanced_find
    element._adaptive_locator = locator
    
    return element


# Export public API
__all__ = [
    'AdaptiveElementLocator',
    'LocatorStrategy',
    'ElementSignature',
    'SelectorCandidate',
    'SelectorRobustnessScorer',
    'enhance_element_with_adaptive_locator',
]