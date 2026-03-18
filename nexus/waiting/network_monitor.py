"""Smart Wait & Timing System for browser automation.

This module implements intelligent waiting mechanisms that replace fixed sleeps with
dynamic waiting based on network idle detection, element stability analysis, and
LLM-predicted load times. Reduces flakiness and improves execution speed.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from urllib.parse import urlparse

from playwright.async_api import Page, Request, Response, ElementHandle


class WaitStrategy(Enum):
    """Available wait strategies for different action types."""
    NETWORK_IDLE = "network_idle"
    DOM_STABLE = "dom_stable"
    ELEMENT_STABLE = "element_stable"
    LLM_PREDICTED = "llm_predicted"
    HYBRID = "hybrid"
    NONE = "none"


@dataclass
class NetworkRequest:
    """Represents a network request for monitoring."""
    url: str
    method: str
    resource_type: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_finished(self) -> bool:
        return self.end_time is not None


@dataclass
class ElementStability:
    """Tracks element stability metrics."""
    element_id: str
    last_bounding_box: Optional[Dict[str, float]] = None
    last_position: Optional[Dict[str, float]] = None
    last_size: Optional[Dict[str, float]] = None
    stable_since: Optional[float] = None
    change_count: int = 0


@dataclass
class WaitConfig:
    """Configuration for wait strategies."""
    network_idle_timeout: float = 5000.0  # ms
    network_idle_threshold: int = 0  # No requests for this many ms
    dom_mutation_timeout: float = 1000.0  # ms
    element_stability_timeout: float = 500.0  # ms
    element_stability_threshold: int = 3  # Stable for N checks
    max_wait_time: float = 30000.0  # ms
    llm_timeout_buffer: float = 1.2  # Multiply LLM prediction by this factor
    
    # Strategy mappings for different actions
    action_strategies: Dict[str, WaitStrategy] = field(default_factory=lambda: {
        "click": WaitStrategy.HYBRID,
        "fill": WaitStrategy.ELEMENT_STABLE,
        "navigate": WaitStrategy.NETWORK_IDLE,
        "screenshot": WaitStrategy.DOM_STABLE,
        "extract": WaitStrategy.DOM_STABLE,
        "default": WaitStrategy.HYBRID
    })


class NetworkMonitor:
    """Monitors network activity to detect idle states."""
    
    def __init__(self, page: Page, config: WaitConfig):
        self.page = page
        self.config = config
        self.active_requests: Dict[str, NetworkRequest] = {}
        self.request_history: List[NetworkRequest] = []
        self._request_lock = asyncio.Lock()
        self._idle_event = asyncio.Event()
        self._last_activity_time: float = time.time() * 1000
        self._monitoring = False
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up Playwright request/response handlers."""
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)
        self.page.on("requestfailed", self._on_request_failed)
    
    async def _on_request(self, request: Request) -> None:
        """Handle new network request."""
        async with self._request_lock:
            req_id = f"{request.url}_{id(request)}"
            self.active_requests[req_id] = NetworkRequest(
                url=request.url,
                method=request.method,
                resource_type=request.resource_type,
                start_time=time.time() * 1000
            )
            self._last_activity_time = time.time() * 1000
            self._idle_event.clear()
    
    async def _on_response(self, response: Response) -> None:
        """Handle network response."""
        async with self._request_lock:
            request = response.request
            req_id = f"{request.url}_{id(request)}"
            if req_id in self.active_requests:
                self.active_requests[req_id].end_time = time.time() * 1000
                self.active_requests[req_id].status_code = response.status
                self.request_history.append(self.active_requests[req_id])
                del self.active_requests[req_id]
                
                if not self.active_requests:
                    self._last_activity_time = time.time() * 1000
    
    async def _on_request_failed(self, request: Request) -> None:
        """Handle failed network request."""
        async with self._request_lock:
            req_id = f"{request.url}_{id(request)}"
            if req_id in self.active_requests:
                self.active_requests[req_id].end_time = time.time() * 1000
                self.request_history.append(self.active_requests[req_id])
                del self.active_requests[req_id]
    
    async def wait_for_network_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait for network to become idle."""
        timeout = timeout or self.config.network_idle_timeout
        start_time = time.time() * 1000
        
        while (time.time() * 1000 - start_time) < timeout:
            async with self._request_lock:
                if not self.active_requests:
                    idle_duration = (time.time() * 1000 - self._last_activity_time)
                    if idle_duration >= self.config.network_idle_threshold:
                        return True
            
            await asyncio.sleep(0.1)
        
        return False
    
    def get_active_request_count(self) -> int:
        """Get count of active network requests."""
        return len(self.active_requests)
    
    def get_recent_requests(self, time_window_ms: float = 1000) -> List[NetworkRequest]:
        """Get requests within the specified time window."""
        current_time = time.time() * 1000
        return [
            req for req in self.request_history
            if req.end_time and (current_time - req.end_time) <= time_window_ms
        ]
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring network activity."""
        self._monitoring = False
        self.page.remove_listener("request", self._on_request)
        self.page.remove_listener("response", self._on_response)
        self.page.remove_listener("requestfailed", self._on_request_failed)


class DOMStabilityMonitor:
    """Monitors DOM mutations to detect page stability."""
    
    def __init__(self, page: Page, config: WaitConfig):
        self.page = page
        self.config = config
        self._mutation_count = 0
        self._last_mutation_time: float = time.time() * 1000
        self._observer_handle: Optional[str] = None
        self._monitoring = False
    
    async def start_monitoring(self) -> None:
        """Start monitoring DOM mutations."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._mutation_count = 0
        
        # Inject MutationObserver
        script = """
        () => {
            window.__mutationCount = 0;
            window.__lastMutationTime = Date.now();
            
            const observer = new MutationObserver((mutations) => {
                window.__mutationCount += mutations.length;
                window.__lastMutationTime = Date.now();
            });
            
            observer.observe(document.body, {
                childList: true,
                attributes: true,
                characterData: true,
                subtree: true
            });
            
            window.__mutationObserver = observer;
            return true;
        }
        """
        
        try:
            await self.page.evaluate(script)
        except Exception:
            # Fallback if document.body not available yet
            pass
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring DOM mutations."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        
        script = """
        () => {
            if (window.__mutationObserver) {
                window.__mutationObserver.disconnect();
                delete window.__mutationObserver;
            }
            return true;
        }
        """
        
        try:
            await self.page.evaluate(script)
        except Exception:
            pass
    
    async def get_mutation_stats(self) -> Dict[str, Any]:
        """Get current DOM mutation statistics."""
        script = """
        () => {
            return {
                count: window.__mutationCount || 0,
                lastMutationTime: window.__lastMutationTime || 0,
                currentTime: Date.now()
            };
        }
        """
        
        try:
            stats = await self.page.evaluate(script)
            return stats
        except Exception:
            return {"count": 0, "lastMutationTime": 0, "currentTime": time.time() * 1000}
    
    async def wait_for_dom_stable(self, timeout: Optional[float] = None) -> bool:
        """Wait for DOM to become stable (no mutations)."""
        timeout = timeout or self.config.dom_mutation_timeout
        start_time = time.time() * 1000
        
        await self.start_monitoring()
        
        while (time.time() * 1000 - start_time) < timeout:
            stats = await self.get_mutation_stats()
            current_time = stats["currentTime"]
            last_mutation = stats["lastMutationTime"]
            
            # Check if enough time has passed since last mutation
            if (current_time - last_mutation) >= self.config.dom_mutation_timeout:
                return True
            
            await asyncio.sleep(0.1)
        
        return False
    
    async def get_recent_mutations(self, time_window_ms: float = 1000) -> int:
        """Get number of mutations in the specified time window."""
        stats = await self.get_mutation_stats()
        current_time = stats["currentTime"]
        last_mutation = stats["lastMutationTime"]
        
        if (current_time - last_mutation) <= time_window_ms:
            return stats["count"]
        return 0


class ElementStabilityMonitor:
    """Monitors element stability (position, size) over time."""
    
    def __init__(self, page: Page, config: WaitConfig):
        self.page = page
        self.config = config
        self._tracked_elements: Dict[str, ElementStability] = {}
        self._monitoring = False
    
    async def track_element(self, element: ElementHandle, element_id: Optional[str] = None) -> str:
        """Start tracking an element's stability."""
        element_id = element_id or f"element_{id(element)}"
        
        try:
            bounding_box = await element.bounding_box()
            if bounding_box:
                self._tracked_elements[element_id] = ElementStability(
                    element_id=element_id,
                    last_bounding_box=bounding_box,
                    last_position={"x": bounding_box["x"], "y": bounding_box["y"]},
                    last_size={"width": bounding_box["width"], "height": bounding_box["height"]},
                    stable_since=time.time() * 1000
                )
        except Exception:
            # Element might not be attached to DOM
            pass
        
        return element_id
    
    async def check_element_stability(self, element_id: str) -> bool:
        """Check if an element is stable (no position/size changes)."""
        if element_id not in self._tracked_elements:
            return False
        
        stability = self._tracked_elements[element_id]
        
        try:
            # Find element by ID or re-query if needed
            element = await self.page.query_selector(f'[data-stability-id="{element_id}"]')
            if not element:
                # Try to find by other means or assume unstable
                return False
            
            current_box = await element.bounding_box()
            if not current_box:
                return False
            
            # Compare with last known state
            if stability.last_bounding_box:
                last = stability.last_bounding_box
                tolerance = 1.0  # pixels
                
                position_changed = (
                    abs(current_box["x"] - last["x"]) > tolerance or
                    abs(current_box["y"] - last["y"]) > tolerance
                )
                
                size_changed = (
                    abs(current_box["width"] - last["width"]) > tolerance or
                    abs(current_box["height"] - last["height"]) > tolerance
                )
                
                if position_changed or size_changed:
                    stability.change_count += 1
                    stability.stable_since = time.time() * 1000
                    stability.last_bounding_box = current_box
                    stability.last_position = {"x": current_box["x"], "y": current_box["y"]}
                    stability.last_size = {"width": current_box["width"], "height": current_box["height"]}
                    return False
                else:
                    # Element hasn't changed
                    stable_duration = time.time() * 1000 - stability.stable_since
                    return stable_duration >= self.config.element_stability_timeout
            
        except Exception as e:
            # Element might be detached or not interactable
            return False
        
        return False
    
    async def wait_for_element_stable(self, element: ElementHandle, timeout: Optional[float] = None) -> bool:
        """Wait for an element to become stable."""
        timeout = timeout or self.config.element_stability_timeout
        element_id = await self.track_element(element)
        start_time = time.time() * 1000
        
        stable_checks = 0
        
        while (time.time() * 1000 - start_time) < timeout:
            is_stable = await self.check_element_stability(element_id)
            
            if is_stable:
                stable_checks += 1
                if stable_checks >= self.config.element_stability_threshold:
                    return True
            else:
                stable_checks = 0
            
            await asyncio.sleep(0.05)  # Check every 50ms
        
        return False
    
    async def stop_tracking(self, element_id: str) -> None:
        """Stop tracking an element."""
        if element_id in self._tracked_elements:
            del self._tracked_elements[element_id]


class LLMTimingPredictor:
    """Predicts load times using LLM based on page complexity."""
    
    def __init__(self, agent_service: Any = None):
        self.agent_service = agent_service
        self._prediction_cache: Dict[str, float] = {}
    
    async def predict_load_time(self, page: Page, action_type: str) -> float:
        """Predict load time for an action using LLM."""
        cache_key = f"{page.url}_{action_type}"
        
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        try:
            # Get page complexity metrics
            complexity = await self._analyze_page_complexity(page)
            
            # If we have an agent service, use it for prediction
            if self.agent_service and hasattr(self.agent_service, 'predict_timing'):
                prediction = await self.agent_service.predict_timing(
                    page.url,
                    action_type,
                    complexity
                )
                self._prediction_cache[cache_key] = prediction
                return prediction
            
            # Fallback: heuristic-based prediction
            return self._heuristic_prediction(complexity, action_type)
            
        except Exception:
            # Default fallback
            return 2000.0  # 2 seconds
    
    async def _analyze_page_complexity(self, page: Page) -> Dict[str, Any]:
        """Analyze page complexity metrics."""
        script = """
        () => {
            const complexity = {
                domElements: document.querySelectorAll('*').length,
                scripts: document.scripts.length,
                images: document.images.length,
                forms: document.forms.length,
                iframes: document.querySelectorAll('iframe').length,
                eventListeners: 0,  // Hard to measure accurately
                cssRules: 0,
                jsHeapSize: performance.memory ? performance.memory.usedJSHeapSize : 0
            };
            
            try {
                complexity.cssRules = Array.from(document.styleSheets)
                    .reduce((count, sheet) => {
                        try {
                            return count + (sheet.cssRules ? sheet.cssRules.length : 0);
                        } catch (e) {
                            return count;
                        }
                    }, 0);
            } catch (e) {
                // CORS restrictions
            }
            
            return complexity;
        }
        """
        
        try:
            return await page.evaluate(script)
        except Exception:
            return {"domElements": 0, "scripts": 0, "images": 0}
    
    def _heuristic_prediction(self, complexity: Dict[str, Any], action_type: str) -> float:
        """Heuristic-based timing prediction."""
        base_time = 1000.0  # 1 second base
        
        # Adjust based on complexity
        dom_factor = min(complexity.get("domElements", 0) / 1000, 3.0)
        script_factor = min(complexity.get("scripts", 0) / 10, 2.0)
        image_factor = min(complexity.get("images", 0) / 20, 2.0)
        
        complexity_multiplier = 1 + (dom_factor + script_factor + image_factor) / 3
        
        # Adjust based on action type
        action_multipliers = {
            "navigate": 2.0,
            "click": 1.0,
            "fill": 0.8,
            "extract": 1.2,
            "screenshot": 1.5
        }
        
        action_multiplier = action_multipliers.get(action_type, 1.0)
        
        prediction = base_time * complexity_multiplier * action_multiplier
        return min(prediction, 10000.0)  # Cap at 10 seconds


class SmartWaiter:
    """Main class for intelligent waiting strategies."""
    
    def __init__(self, page: Page, config: Optional[WaitConfig] = None, agent_service: Any = None):
        self.page = page
        self.config = config or WaitConfig()
        self.agent_service = agent_service
        
        # Initialize monitors
        self.network_monitor = NetworkMonitor(page, self.config)
        self.dom_monitor = DOMStabilityMonitor(page, self.config)
        self.element_monitor = ElementStabilityMonitor(page, self.config)
        self.timing_predictor = LLMTimingPredictor(agent_service)
        
        self._active = True
    
    async def wait_for_action(self, action_type: str, element: Optional[ElementHandle] = None, 
                            custom_strategy: Optional[WaitStrategy] = None) -> bool:
        """Wait intelligently for an action to complete."""
        if not self._active:
            return True
        
        strategy = custom_strategy or self.config.action_strategies.get(
            action_type, 
            self.config.action_strategies["default"]
        )
        
        if strategy == WaitStrategy.NONE:
            return True
        
        start_time = time.time() * 1000
        max_wait = self.config.max_wait_time
        
        try:
            if strategy == WaitStrategy.NETWORK_IDLE:
                return await self.network_monitor.wait_for_network_idle(max_wait)
            
            elif strategy == WaitStrategy.DOM_STABLE:
                return await self.dom_monitor.wait_for_dom_stable(max_wait)
            
            elif strategy == WaitStrategy.ELEMENT_STABLE and element:
                return await self.element_monitor.wait_for_element_stable(element, max_wait)
            
            elif strategy == WaitStrategy.LLM_PREDICTED:
                predicted_time = await self.timing_predictor.predict_load_time(self.page, action_type)
                adjusted_time = predicted_time * self.config.llm_timeout_buffer
                await asyncio.sleep(adjusted_time / 1000)
                return True
            
            elif strategy == WaitStrategy.HYBRID:
                return await self._hybrid_wait(action_type, element, max_wait)
            
        except Exception as e:
            # Log error but don't fail the action
            print(f"SmartWait error for {action_type}: {e}")
            return False
        
        return True
    
    async def _hybrid_wait(self, action_type: str, element: Optional[ElementHandle], 
                          timeout: float) -> bool:
        """Hybrid waiting strategy combining multiple approaches."""
        start_time = time.time() * 1000
        remaining_time = timeout
        
        # Phase 1: Wait for network idle (with timeout)
        network_timeout = min(remaining_time * 0.3, 2000)  # Max 2 seconds for network
        if await self.network_monitor.wait_for_network_idle(network_timeout):
            # Network is idle, now check DOM stability
            elapsed = (time.time() * 1000 - start_time)
            remaining_time = timeout - elapsed
            
            if remaining_time > 0:
                dom_timeout = min(remaining_time * 0.5, 1000)  # Max 1 second for DOM
                await self.dom_monitor.wait_for_dom_stable(dom_timeout)
        
        # Phase 2: If we have an element, wait for it to be stable
        if element and remaining_time > 0:
            elapsed = (time.time() * 1000 - start_time)
            remaining_time = timeout - elapsed
            
            if remaining_time > 0:
                element_timeout = min(remaining_time, 500)  # Max 500ms for element
                await self.element_monitor.wait_for_element_stable(element, element_timeout)
        
        # Phase 3: Final check with LLM prediction if available
        if self.agent_service and remaining_time > 0:
            predicted_time = await self.timing_predictor.predict_load_time(self.page, action_type)
            elapsed = (time.time() * 1000 - start_time)
            
            if elapsed < predicted_time:
                sleep_time = (predicted_time - elapsed) / 1000
                await asyncio.sleep(min(sleep_time, 1.0))  # Max 1 second additional wait
        
        return True
    
    async def wait_for_page_load(self, timeout: Optional[float] = None) -> bool:
        """Wait for complete page load."""
        timeout = timeout or self.config.max_wait_time
        return await self.network_monitor.wait_for_network_idle(timeout)
    
    async def wait_for_element(self, selector: str, timeout: Optional[float] = None) -> Optional[ElementHandle]:
        """Wait for an element to appear and be stable."""
        timeout = timeout or self.config.max_wait_time
        start_time = time.time() * 1000
        
        while (time.time() * 1000 - start_time) < timeout:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    # Wait for element to be stable
                    if await self.element_monitor.wait_for_element_stable(element, 500):
                        return element
            except Exception:
                pass
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def adaptive_sleep(self, base_ms: float, action_type: str = "default") -> None:
        """Adaptive sleep based on page complexity and action type."""
        if not self._active:
            await asyncio.sleep(base_ms / 1000)
            return
        
        try:
            predicted_time = await self.timing_predictor.predict_load_time(self.page, action_type)
            # Use the larger of base time or predicted time
            sleep_time = max(base_ms, predicted_time) / 1000
            await asyncio.sleep(sleep_time)
        except Exception:
            await asyncio.sleep(base_ms / 1000)
    
    async def stop(self) -> None:
        """Stop all monitoring and clean up."""
        self._active = False
        await self.network_monitor.stop_monitoring()
        await self.dom_monitor.stop_monitoring()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get waiting statistics."""
        return {
            "active_requests": self.network_monitor.get_active_request_count(),
            "tracked_elements": len(self.element_monitor._tracked_elements),
            "config": {
                "network_idle_timeout": self.config.network_idle_timeout,
                "dom_mutation_timeout": self.config.dom_mutation_timeout,
                "element_stability_timeout": self.config.element_stability_timeout,
                "max_wait_time": self.config.max_wait_time
            }
        }


# Integration helper for existing codebase
async def create_smart_waiter(page: Page, agent_service: Any = None, 
                            config: Optional[WaitConfig] = None) -> SmartWaiter:
    """Factory function to create a SmartWaiter instance."""
    waiter = SmartWaiter(page, config, agent_service)
    # Start DOM monitoring by default
    await waiter.dom_monitor.start_monitoring()
    return waiter


# Example usage with existing Page class
async def enhanced_click(page: Page, selector: str, waiter: SmartWaiter) -> None:
    """Example of enhanced click with smart waiting."""
    element = await waiter.wait_for_element(selector)
    if element:
        await element.click()
        # Smart wait after click
        await waiter.wait_for_action("click", element)


async def enhanced_navigate(page: Page, url: str, waiter: SmartWaiter) -> None:
    """Example of enhanced navigation with smart waiting."""
    await page.goto(url)
    # Wait for page to load
    await waiter.wait_for_page_load()


# Configuration presets
QUICK_WAIT_CONFIG = WaitConfig(
    network_idle_timeout=2000,
    dom_mutation_timeout=500,
    element_stability_timeout=300,
    max_wait_time=10000,
    action_strategies={
        "click": WaitStrategy.ELEMENT_STABLE,
        "fill": WaitStrategy.ELEMENT_STABLE,
        "navigate": WaitStrategy.NETWORK_IDLE,
        "screenshot": WaitStrategy.DOM_STABLE,
        "extract": WaitStrategy.DOM_STABLE,
        "default": WaitStrategy.ELEMENT_STABLE
    }
)

THOROUGH_WAIT_CONFIG = WaitConfig(
    network_idle_timeout=10000,
    dom_mutation_timeout=2000,
    element_stability_timeout=1000,
    max_wait_time=60000,
    action_strategies={
        "click": WaitStrategy.HYBRID,
        "fill": WaitStrategy.HYBRID,
        "navigate": WaitStrategy.HYBRID,
        "screenshot": WaitStrategy.HYBRID,
        "extract": WaitStrategy.HYBRID,
        "default": WaitStrategy.HYBRID
    }
)