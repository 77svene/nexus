"""
Self-Healing Element Detection - Resilient Locators with Multi-Strategy Fallback

Implements a Strategy pattern for element detection that adapts to DOM changes.
Tries multiple approaches: original selector, AI vision, text proximity, and visual similarity.
Caches successful strategies and automatically updates selectors when patterns change.
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import io
import base64

from ..element import Element
from ..page import Page
from ..utils import (
    get_text_similarity,
    calculate_iou,
    take_element_screenshot,
    encode_image_base64,
)
from ...agent.service import AgentService


class LocatorStrategy(Enum):
    """Available locator strategies in priority order."""
    ORIGINAL_CSS = "original_css"
    ORIGINAL_XPATH = "original_xpath"
    AI_VISION = "ai_vision"
    TEXT_PROXIMITY = "text_proximity"
    VISUAL_SIMILARITY = "visual_similarity"
    ACCESSIBILITY_TREE = "accessibility_tree"
    DOM_MUTATION = "dom_mutation"


@dataclass
class ElementSignature:
    """Signature for element identification across DOM changes."""
    tag_name: str
    text_content: str
    attributes: Dict[str, str]
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    visual_hash: Optional[str] = None
    parent_signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_name": self.tag_name,
            "text_content": self.text_content[:100],  # Truncate for storage
            "attributes": {k: v[:50] for k, v in self.attributes.items()},
            "bounding_box": self.bounding_box,
            "visual_hash": self.visual_hash,
            "parent_signature": self.parent_signature,
        }
    
    def similarity_score(self, other: 'ElementSignature') -> float:
        """Calculate similarity score between two signatures (0-1)."""
        score = 0.0
        weights = {
            "tag_name": 0.2,
            "text": 0.3,
            "attributes": 0.2,
            "position": 0.2,
            "visual": 0.1,
        }
        
        # Tag name match
        if self.tag_name == other.tag_name:
            score += weights["tag_name"]
        
        # Text similarity
        text_sim = get_text_similarity(self.text_content, other.text_content)
        score += weights["text"] * text_sim
        
        # Attribute similarity
        common_attrs = set(self.attributes.keys()) & set(other.attributes.keys())
        if common_attrs:
            attr_matches = sum(
                1 for attr in common_attrs 
                if self.attributes[attr] == other.attributes[attr]
            )
            attr_score = attr_matches / len(common_attrs)
            score += weights["attributes"] * attr_score
        
        # Position similarity (using IoU of bounding boxes)
        if self.bounding_box and other.bounding_box:
            iou = calculate_iou(self.bounding_box, other.bounding_box)
            score += weights["position"] * iou
        
        # Visual hash similarity
        if self.visual_hash and other.visual_hash:
            if self.visual_hash == other.visual_hash:
                score += weights["visual"]
        
        return min(score, 1.0)


@dataclass
class LocatorResult:
    """Result of a locator strategy attempt."""
    strategy: LocatorStrategy
    element: Optional[Element]
    confidence: float
    selector_used: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResilientLocatorState:
    """State tracking for a resilient locator."""
    original_selector: str
    selector_type: str  # 'css' or 'xpath'
    last_known_signature: Optional[ElementSignature] = None
    successful_strategies: List[Tuple[LocatorStrategy, str, float]] = field(default_factory=list)
    failure_count: int = 0
    last_success_time: float = 0.0
    strategy_cache: Dict[str, LocatorResult] = field(default_factory=dict)
    
    def update_success(self, strategy: LocatorStrategy, selector: Optional[str], confidence: float):
        """Update state after successful element location."""
        self.failure_count = 0
        self.last_success_time = time.time()
        
        # Update successful strategies list (keep top 5)
        self.successful_strategies.insert(0, (strategy, selector, confidence))
        self.successful_strategies = self.successful_strategies[:5]
    
    def update_failure(self):
        """Update state after failed element location."""
        self.failure_count += 1


class LocatorStrategyBase(ABC):
    """Base class for all locator strategies."""
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        self.page = page
        self.agent_service = agent_service
        self.timeout_ms = 2000  # Default timeout per strategy
    
    @abstractmethod
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        """Attempt to locate element using this strategy."""
        pass
    
    def _create_cache_key(self, state: ResilientLocatorState, **kwargs) -> str:
        """Create a cache key for this strategy attempt."""
        key_data = {
            "strategy": self.__class__.__name__,
            "selector": state.original_selector,
            **kwargs
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


class OriginalSelectorStrategy(LocatorStrategyBase):
    """Try the original CSS or XPath selector."""
    
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        start_time = time.time()
        
        try:
            # Try original selector
            elements = await self.page.query_selector_all(state.original_selector)
            
            if elements:
                element = elements[0]
                execution_time = (time.time() - start_time) * 1000
                
                return LocatorResult(
                    strategy=LocatorStrategy.ORIGINAL_CSS if state.selector_type == 'css' 
                             else LocatorStrategy.ORIGINAL_XPATH,
                    element=element,
                    confidence=1.0,
                    selector_used=state.original_selector,
                    execution_time_ms=execution_time,
                    metadata={"elements_found": len(elements)}
                )
        except Exception as e:
            pass
        
        execution_time = (time.time() - start_time) * 1000
        return LocatorResult(
            strategy=LocatorStrategy.ORIGINAL_CSS,
            element=None,
            confidence=0.0,
            execution_time_ms=execution_time,
            metadata={"error": "Selector not found"}
        )


class AIVisionStrategy(LocatorStrategyBase):
    """Use AI vision to locate element by description or appearance."""
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        super().__init__(page, agent_service)
        self.timeout_ms = 5000  # AI vision takes longer
    
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        start_time = time.time()
        
        if not self.agent_service:
            return LocatorResult(
                strategy=LocatorStrategy.AI_VISION,
                element=None,
                confidence=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "Agent service not available"}
            )
        
        try:
            # Take screenshot of the page
            screenshot = await self.page.screenshot()
            
            # Create prompt for AI to locate element
            prompt = self._create_vision_prompt(state)
            
            # Use agent service to analyze screenshot
            # Note: This is a simplified integration - actual implementation
            # would depend on the agent service's capabilities
            if hasattr(self.agent_service, 'locate_element_in_screenshot'):
                element_info = await self.agent_service.locate_element_in_screenshot(
                    screenshot=screenshot,
                    prompt=prompt,
                    page_url=self.page.url
                )
                
                if element_info and 'selector' in element_info:
                    # Try the suggested selector
                    elements = await self.page.query_selector_all(element_info['selector'])
                    if elements:
                        execution_time = (time.time() - start_time) * 1000
                        return LocatorResult(
                            strategy=LocatorStrategy.AI_VISION,
                            element=elements[0],
                            confidence=element_info.get('confidence', 0.8),
                            selector_used=element_info['selector'],
                            execution_time_ms=execution_time,
                            metadata={
                                "ai_suggestion": element_info,
                                "screenshot_hash": hashlib.md5(screenshot).hexdigest()
                            }
                        )
        except Exception as e:
            pass
        
        execution_time = (time.time() - start_time) * 1000
        return LocatorResult(
            strategy=LocatorStrategy.AI_VISION,
            element=None,
            confidence=0.0,
            execution_time_ms=execution_time,
            metadata={"error": "AI vision failed to locate element"}
        )
    
    def _create_vision_prompt(self, state: ResilientLocatorState) -> str:
        """Create prompt for AI vision model."""
        base_prompt = "Locate the web element that matches this description:\n"
        
        if state.last_known_signature:
            sig = state.last_known_signature
            details = []
            
            if sig.tag_name:
                details.append(f"- Element type: <{sig.tag_name}>")
            if sig.text_content:
                details.append(f"- Text content: '{sig.text_content[:100]}'")
            if sig.attributes:
                attrs = ", ".join([f"{k}='{v}'" for k, v in list(sig.attributes.items())[:5]])
                details.append(f"- Attributes: {attrs}")
            
            return base_prompt + "\n".join(details) + "\n\nProvide the best CSS selector for this element."
        
        return base_prompt + f"Original selector was: {state.original_selector}\n\nFind the most similar element."


class TextProximityStrategy(LocatorStrategyBase):
    """Locate element by text content and proximity to original location."""
    
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        start_time = time.time()
        
        if not state.last_known_signature:
            return LocatorResult(
                strategy=LocatorStrategy.TEXT_PROXIMITY,
                element=None,
                confidence=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "No signature available for text proximity"}
            )
        
        try:
            # Find all elements with similar text
            all_elements = await self.page.query_selector_all("*")
            candidates = []
            
            for element in all_elements:
                try:
                    text = await element.text_content()
                    if not text or not text.strip():
                        continue
                    
                    # Check text similarity
                    text_similarity = get_text_similarity(
                        state.last_known_signature.text_content,
                        text
                    )
                    
                    if text_similarity > 0.7:  # Threshold for text match
                        # Get bounding box
                        bbox = await element.bounding_box()
                        if bbox:
                            # Calculate distance from original location
                            original_bbox = state.last_known_signature.bounding_box
                            distance = self._calculate_distance(
                                (bbox['x'], bbox['y']),
                                (original_bbox[0], original_bbox[1])
                            )
                            
                            candidates.append({
                                'element': element,
                                'text_similarity': text_similarity,
                                'distance': distance,
                                'bbox': (bbox['x'], bbox['y'], bbox['width'], bbox['height'])
                            })
                except:
                    continue
            
            if candidates:
                # Sort by combined score (text similarity and proximity)
                for candidate in candidates:
                    # Normalize distance (closer is better)
                    max_distance = 1000  # pixels
                    distance_score = max(0, 1 - (candidate['distance'] / max_distance))
                    
                    # Combined score
                    candidate['score'] = (
                        0.7 * candidate['text_similarity'] + 
                        0.3 * distance_score
                    )
                
                candidates.sort(key=lambda x: x['score'], reverse=True)
                best = candidates[0]
                
                execution_time = (time.time() - start_time) * 1000
                return LocatorResult(
                    strategy=LocatorStrategy.TEXT_PROXIMITY,
                    element=best['element'],
                    confidence=best['score'],
                    execution_time_ms=execution_time,
                    metadata={
                        "text_similarity": best['text_similarity'],
                        "distance": best['distance'],
                        "candidates_count": len(candidates)
                    }
                )
        
        except Exception as e:
            pass
        
        execution_time = (time.time() - start_time) * 1000
        return LocatorResult(
            strategy=LocatorStrategy.TEXT_PROXIMITY,
            element=None,
            confidence=0.0,
            execution_time_ms=execution_time,
            metadata={"error": "No elements with similar text found"}
        )
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


class VisualSimilarityStrategy(LocatorStrategyBase):
    """Locate element by visual similarity using image comparison."""
    
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        start_time = time.time()
        
        if not state.last_known_signature or not state.last_known_signature.visual_hash:
            return LocatorResult(
                strategy=LocatorStrategy.VISUAL_SIMILARITY,
                element=None,
                confidence=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": "No visual hash available"}
            )
        
        try:
            # Get all interactive elements
            interactive_selectors = [
                "button", "a", "input", "select", "textarea",
                "[role='button']", "[role='link']", "[role='checkbox']",
                "[onclick]", "[tabindex]"
            ]
            
            all_elements = []
            for selector in interactive_selectors:
                elements = await self.page.query_selector_all(selector)
                all_elements.extend(elements)
            
            # Remove duplicates
            unique_elements = list({await el.evaluate('el => el.outerHTML'): el for el in all_elements}.values())
            
            candidates = []
            
            for element in unique_elements[:50]:  # Limit to 50 elements for performance
                try:
                    # Take screenshot of element
                    screenshot = await take_element_screenshot(element)
                    if not screenshot:
                        continue
                    
                    # Calculate visual hash
                    current_hash = self._calculate_visual_hash(screenshot)
                    
                    # Compare with original hash
                    similarity = self._compare_visual_hashes(
                        state.last_known_signature.visual_hash,
                        current_hash
                    )
                    
                    if similarity > 0.6:  # Threshold for visual similarity
                        bbox = await element.bounding_box()
                        candidates.append({
                            'element': element,
                            'similarity': similarity,
                            'bbox': bbox
                        })
                except:
                    continue
            
            if candidates:
                candidates.sort(key=lambda x: x['similarity'], reverse=True)
                best = candidates[0]
                
                execution_time = (time.time() - start_time) * 1000
                return LocatorResult(
                    strategy=LocatorStrategy.VISUAL_SIMILARITY,
                    element=best['element'],
                    confidence=best['similarity'],
                    execution_time_ms=execution_time,
                    metadata={
                        "visual_similarity": best['similarity'],
                        "candidates_count": len(candidates)
                    }
                )
        
        except Exception as e:
            pass
        
        execution_time = (time.time() - start_time) * 1000
        return LocatorResult(
            strategy=LocatorStrategy.VISUAL_SIMILARITY,
            element=None,
            confidence=0.0,
            execution_time_ms=execution_time,
            metadata={"error": "No visually similar elements found"}
        )
    
    def _calculate_visual_hash(self, image_data: bytes) -> str:
        """Calculate perceptual hash for an image."""
        try:
            image = Image.open(io.BytesIO(image_data))
            # Resize to 8x8 for simple hash
            image = image.resize((8, 8), Image.Resampling.LANCZOS)
            # Convert to grayscale
            image = image.convert('L')
            # Get pixel values
            pixels = list(image.getdata())
            # Calculate average
            avg = sum(pixels) / len(pixels)
            # Create hash (1 for above average, 0 for below)
            hash_bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
            return hash_bits
        except:
            return ""
    
    def _compare_visual_hashes(self, hash1: str, hash2: str) -> float:
        """Compare two visual hashes (Hamming distance)."""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        max_distance = len(hash1)
        
        # Convert to similarity (0-1)
        return 1.0 - (distance / max_distance)


class AccessibilityTreeStrategy(LocatorStrategyBase):
    """Locate element using the accessibility tree."""
    
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        start_time = time.time()
        
        try:
            # Get accessibility tree
            # Note: This requires CDP (Chrome DevTools Protocol) access
            # Implementation depends on the browser automation library
            
            # For Playwright, we can use the accessibility tree
            snapshot = await self.page.accessibility.snapshot()
            
            if not snapshot:
                return LocatorResult(
                    strategy=LocatorStrategy.ACCESSIBILITY_TREE,
                    element=None,
                    confidence=0.0,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    metadata={"error": "No accessibility snapshot available"}
                )
            
            # Search for element in accessibility tree
            target_name = None
            if state.last_known_signature:
                # Try to find by accessible name
                target_name = state.last_known_signature.attributes.get('aria-label') or \
                             state.last_known_signature.attributes.get('title') or \
                             state.last_known_signature.text_content
            
            if target_name:
                # Find node with matching name
                node = self._find_node_by_name(snapshot, target_name)
                if node and 'role' in node:
                    # Try to find element by role and name
                    selector = f"[role='{node['role']}']"
                    if 'name' in node:
                        # Try to find by accessible name
                        elements = await self.page.query_selector_all(
                            f"{selector}[aria-label='{node['name']}']"
                        )
                        if not elements:
                            elements = await self.page.query_selector_all(
                                f"{selector}[title='{node['name']}']"
                            )
                        
                        if elements:
                            execution_time = (time.time() - start_time) * 1000
                            return LocatorResult(
                                strategy=LocatorStrategy.ACCESSIBILITY_TREE,
                                element=elements[0],
                                confidence=0.8,
                                selector_used=selector,
                                execution_time_ms=execution_time,
                                metadata={
                                    "accessible_name": node.get('name'),
                                    "role": node.get('role')
                                }
                            )
        
        except Exception as e:
            pass
        
        execution_time = (time.time() - start_time) * 1000
        return LocatorResult(
            strategy=LocatorStrategy.ACCESSIBILITY_TREE,
            element=None,
            confidence=0.0,
            execution_time_ms=execution_time,
            metadata={"error": "Accessibility tree search failed"}
        )
    
    def _find_node_by_name(self, node: Dict[str, Any], target_name: str) -> Optional[Dict[str, Any]]:
        """Recursively search accessibility tree for node with matching name."""
        if 'name' in node and node['name'] and target_name.lower() in node['name'].lower():
            return node
        
        if 'children' in node:
            for child in node['children']:
                result = self._find_node_by_name(child, target_name)
                if result:
                    return result
        
        return None


class DOMMutationStrategy(LocatorStrategyBase):
    """Monitor DOM mutations to track element changes."""
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        super().__init__(page, agent_service)
        self.mutation_observers: Dict[str, Any] = {}
    
    async def locate(
        self, 
        state: ResilientLocatorState,
        context_elements: Optional[List[Element]] = None
    ) -> LocatorResult:
        start_time = time.time()
        
        # This strategy doesn't actively locate elements but monitors changes
        # It's used to update selectors when mutations are detected
        
        # Check if we have mutation data for this selector
        cache_key = self._create_cache_key(state, strategy="dom_mutation")
        
        if cache_key in state.strategy_cache:
            cached = state.strategy_cache[cache_key]
            if cached.element and time.time() - cached.metadata.get('timestamp', 0) < 60:
                # Use cached result if recent
                execution_time = (time.time() - start_time) * 1000
                return LocatorResult(
                    strategy=LocatorStrategy.DOM_MUTATION,
                    element=cached.element,
                    confidence=cached.confidence * 0.9,  # Slightly lower confidence for cached
                    selector_used=cached.selector_used,
                    execution_time_ms=execution_time,
                    metadata={"source": "cached_mutation", **cached.metadata}
                )
        
        # If no cached result, return failure
        execution_time = (time.time() - start_time) * 1000
        return LocatorResult(
            strategy=LocatorStrategy.DOM_MUTATION,
            element=None,
            confidence=0.0,
            execution_time_ms=execution_time,
            metadata={"error": "No DOM mutation data available"}
        )
    
    async def setup_mutation_observer(self, selector: str):
        """Set up mutation observer for a selector."""
        # This would inject JavaScript to observe DOM changes
        # Implementation depends on browser automation capabilities
        pass


class ResilientLocator:
    """
    Self-healing element locator that tries multiple strategies.
    
    Automatically adapts to DOM changes by:
    1. Trying original selector first
    2. Falling back to AI vision, text proximity, and visual similarity
    3. Caching successful strategies
    4. Updating selectors when patterns change
    """
    
    def __init__(
        self, 
        page: Page,
        selector: str,
        selector_type: str = 'css',
        agent_service: Optional[AgentService] = None,
        strategies: Optional[List[LocatorStrategy]] = None,
        max_retries: int = 3,
        retry_delay_ms: int = 500
    ):
        self.page = page
        self.agent_service = agent_service
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        
        # Initialize state
        self.state = ResilientLocatorState(
            original_selector=selector,
            selector_type=selector_type
        )
        
        # Initialize strategies
        self.strategies = strategies or [
            LocatorStrategy.ORIGINAL_CSS,
            LocatorStrategy.AI_VISION,
            LocatorStrategy.TEXT_PROXIMITY,
            LocatorStrategy.VISUAL_SIMILARITY,
            LocatorStrategy.ACCESSIBILITY_TREE,
            LocatorStrategy.DOM_MUTATION,
        ]
        
        # Strategy instances
        self._strategy_instances = {
            LocatorStrategy.ORIGINAL_CSS: OriginalSelectorStrategy(page, agent_service),
            LocatorStrategy.ORIGINAL_XPATH: OriginalSelectorStrategy(page, agent_service),
            LocatorStrategy.AI_VISION: AIVisionStrategy(page, agent_service),
            LocatorStrategy.TEXT_PROXIMITY: TextProximityStrategy(page, agent_service),
            LocatorStrategy.VISUAL_SIMILARITY: VisualSimilarityStrategy(page, agent_service),
            LocatorStrategy.ACCESSIBILITY_TREE: AccessibilityTreeStrategy(page, agent_service),
            LocatorStrategy.DOM_MUTATION: DOMMutationStrategy(page, agent_service),
        }
        
        # Statistics
        self.stats = {
            "total_attempts": 0,
            "successful_finds": 0,
            "strategy_successes": {s.value: 0 for s in LocatorStrategy},
            "average_find_time_ms": 0.0,
            "last_find_time_ms": 0.0,
        }
    
    async def locate(
        self, 
        timeout_ms: int = 10000,
        context_elements: Optional[List[Element]] = None
    ) -> Optional[Element]:
        """
        Locate element using resilient strategies.
        
        Args:
            timeout_ms: Maximum time to spend trying all strategies
            context_elements: Optional context elements to help with location
            
        Returns:
            Located element or None if all strategies fail
        """
        start_time = time.time()
        deadline = start_time + (timeout_ms / 1000)
        
        self.stats["total_attempts"] += 1
        
        # Try cached successful strategies first
        for strategy, selector, confidence in self.state.successful_strategies[:2]:
            if time.time() > deadline:
                break
            
            result = await self._try_cached_strategy(strategy, selector, confidence)
            if result and result.element:
                await self._update_state_on_success(result)
                self._update_stats(result, start_time)
                return result.element
        
        # Try each strategy in order
        for strategy_enum in self.strategies:
            if time.time() > deadline:
                break
            
            # Skip if we've tried this strategy recently and it failed
            cache_key = self._strategy_instances[strategy_enum]._create_cache_key(self.state)
            if cache_key in self.state.strategy_cache:
                cached = self.state.strategy_cache[cache_key]
                if cached.element is None and time.time() - cached.metadata.get('timestamp', 0) < 30:
                    continue  # Skip recently failed strategies
            
            result = await self._try_strategy(strategy_enum, context_elements)
            
            if result and result.element:
                await self._update_state_on_success(result)
                self._update_stats(result, start_time)
                return result.element
            
            # Cache the failure
            if result:
                cache_key = self._strategy_instances[strategy_enum]._create_cache_key(self.state)
                result.metadata['timestamp'] = time.time()
                self.state.strategy_cache[cache_key] = result
        
        # All strategies failed
        self.state.update_failure()
        return None
    
    async def _try_strategy(
        self, 
        strategy_enum: LocatorStrategy,
        context_elements: Optional[List[Element]] = None
    ) -> Optional[LocatorResult]:
        """Try a specific strategy with retries."""
        strategy = self._strategy_instances.get(strategy_enum)
        if not strategy:
            return None
        
        for attempt in range(self.max_retries):
            try:
                result = await strategy.locate(self.state, context_elements)
                
                if result.element:
                    return result
                
                # If this is not the last attempt, wait before retrying
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay_ms / 1000)
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    return LocatorResult(
                        strategy=strategy_enum,
                        element=None,
                        confidence=0.0,
                        metadata={"error": str(e), "attempts": attempt + 1}
                    )
                await asyncio.sleep(self.retry_delay_ms / 1000)
        
        return None
    
    async def _try_cached_strategy(
        self, 
        strategy: LocatorStrategy, 
        selector: str, 
        confidence: float
    ) -> Optional[LocatorResult]:
        """Try a cached successful strategy."""
        try:
            if strategy in [LocatorStrategy.ORIGINAL_CSS, LocatorStrategy.ORIGINAL_XPATH]:
                elements = await self.page.query_selector_all(selector)
                if elements:
                    return LocatorResult(
                        strategy=strategy,
                        element=elements[0],
                        confidence=confidence,
                        selector_used=selector,
                        metadata={"source": "cached_strategy"}
                    )
        except:
            pass
        
        return None
    
    async def _update_state_on_success(self, result: LocatorResult):
        """Update locator state after successful element location."""
        # Update signature
        if result.element:
            try:
                signature = await self._create_element_signature(result.element)
                self.state.last_known_signature = signature
            except:
                pass
        
        # Update successful strategies
        self.state.update_success(
            result.strategy, 
            result.selector_used, 
            result.confidence
        )
        
        # If we found a new selector, update the original if confidence is high
        if (result.selector_used and 
            result.selector_used != self.state.original_selector and 
            result.confidence > 0.9):
            self.state.original_selector = result.selector_used
    
    async def _create_element_signature(self, element: Element) -> ElementSignature:
        """Create a signature for an element."""
        try:
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            text_content = await element.text_content() or ""
            
            # Get attributes
            attributes = await element.evaluate('''el => {
                const attrs = {};
                for (const attr of el.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }''')
            
            # Get bounding box
            bbox = await element.bounding_box()
            bounding_box = (bbox['x'], bbox['y'], bbox['width'], bbox['height']) if bbox else (0, 0, 0, 0)
            
            # Get visual hash
            visual_hash = None
            try:
                screenshot = await take_element_screenshot(element)
                if screenshot:
                    visual_hash = self._strategy_instances[LocatorStrategy.VISUAL_SIMILARITY]._calculate_visual_hash(screenshot)
            except:
                pass
            
            # Get parent signature (simplified)
            parent_signature = None
            try:
                parent = await element.evaluate('el => el.parentElement')
                if parent:
                    parent_tag = await element.evaluate('el => el.parentElement.tagName.toLowerCase()')
                    parent_id = await element.evaluate('el => el.parentElement.id')
                    parent_signature = f"{parent_tag}#{parent_id}" if parent_id else parent_tag
            except:
                pass
            
            return ElementSignature(
                tag_name=tag_name,
                text_content=text_content,
                attributes=attributes,
                bounding_box=bounding_box,
                visual_hash=visual_hash,
                parent_signature=parent_signature
            )
        except:
            # Return minimal signature
            return ElementSignature(
                tag_name="unknown",
                text_content="",
                attributes={},
                bounding_box=(0, 0, 0, 0)
            )
    
    def _update_stats(self, result: LocatorResult, start_time: float):
        """Update locator statistics."""
        self.stats["successful_finds"] += 1
        self.stats["strategy_successes"][result.strategy.value] += 1
        
        find_time = (time.time() - start_time) * 1000
        self.stats["last_find_time_ms"] = find_time
        
        # Update rolling average
        total = self.stats["successful_finds"]
        current_avg = self.stats["average_find_time_ms"]
        self.stats["average_find_time_ms"] = (
            (current_avg * (total - 1) + find_time) / total
        )
    
    async def update_selector(self, new_selector: str):
        """Manually update the selector (e.g., after manual inspection)."""
        self.state.original_selector = new_selector
        # Clear successful strategies cache to force re-evaluation
        self.state.successful_strategies.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get locator statistics."""
        return {
            **self.stats,
            "current_selector": self.state.original_selector,
            "failure_count": self.state.failure_count,
            "successful_strategies_count": len(self.state.successful_strategies),
            "cache_size": len(self.state.strategy_cache),
        }
    
    async def clear_cache(self):
        """Clear the strategy cache."""
        self.state.strategy_cache.clear()
    
    def __repr__(self) -> str:
        return (
            f"ResilientLocator(selector='{self.state.original_selector}', "
            f"strategies={len(self.strategies)}, "
            f"success_rate={self.stats['successful_finds']}/{self.stats['total_attempts']})"
        )


# Factory function for easy creation
def create_resilient_locator(
    page: Page,
    selector: str,
    selector_type: str = 'css',
    agent_service: Optional[AgentService] = None,
    **kwargs
) -> ResilientLocator:
    """Create a resilient locator with sensible defaults."""
    return ResilientLocator(
        page=page,
        selector=selector,
        selector_type=selector_type,
        agent_service=agent_service,
        **kwargs
    )


# Integration with existing Element class
async def make_element_resilient(
    element: Element,
    page: Page,
    agent_service: Optional[AgentService] = None
) -> ResilientLocator:
    """Convert an existing element to use resilient location."""
    # Try to get a selector for the element
    selector = None
    selector_type = 'css'
    
    try:
        # Try to generate a CSS selector
        selector = await element.evaluate('''el => {
            function getCssPath(el) {
                if (!(el instanceof Element)) return;
                const path = [];
                while (el.nodeType === Node.ELEMENT_NODE) {
                    let selector = el.nodeName.toLowerCase();
                    if (el.id) {
                        selector += '#' + el.id;
                        path.unshift(selector);
                        break;
                    } else {
                        let sib = el, nth = 1;
                        while (sib = sib.previousElementSibling) {
                            if (sib.nodeName.toLowerCase() == selector) nth++;
                        }
                        if (nth != 1) selector += ":nth-of-type("+nth+")";
                    }
                    path.unshift(selector);
                    el = el.parentNode;
                }
                return path.join(' > ');
            }
            return getCssPath(el);
        }''')
    except:
        # Fallback to XPath
        try:
            selector = await element.evaluate('''el => {
                function getXPath(el) {
                    if (el.id !== '') return '//*[@id="' + el.id + '"]';
                    if (el === document.body) return '/html/body';
                    
                    let ix = 0;
                    const siblings = el.parentNode.childNodes;
                    for (let i = 0; i < siblings.length; i++) {
                        const sibling = siblings[i];
                        if (sibling === el) {
                            return getXPath(el.parentNode) + '/' + el.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        }
                        if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {
                            ix++;
                        }
                    }
                }
                return getXPath(el);
            }''')
            selector_type = 'xpath'
        except:
            # Last resort: use tag name and text
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            text = await element.text_content() or ""
            selector = f"{tag_name}:has-text('{text[:50]}')"
    
    if selector:
        locator = ResilientLocator(
            page=page,
            selector=selector,
            selector_type=selector_type,
            agent_service=agent_service
        )
        
        # Set initial signature from the element
        try:
            signature = await locator._create_element_signature(element)
            locator.state.last_known_signature = signature
        except:
            pass
        
        return locator
    
    raise ValueError("Could not create resilient locator for element")