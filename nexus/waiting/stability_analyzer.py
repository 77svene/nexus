"""
nexus/waiting/stability_analyzer.py

Smart Wait & Timing System - Replaces fixed sleeps with intelligent waiting based on
network idle detection, element stability analysis, and LLM-predicted load times.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging

from playwright.async_api import Page, ElementHandle, Response, Request
import numpy as np

from nexus.actor.page import PageActor
from nexus.agent.service import AgentService

logger = logging.getLogger(__name__)


class WaitStrategy(Enum):
    """Predefined wait strategies for different action types."""
    NETWORK_IDLE = "network_idle"
    ELEMENT_STABLE = "element_stable"
    DOM_MUTATION_IDLE = "dom_mutation_idle"
    HYBRID = "hybrid"
    LLM_PREDICTED = "llm_predicted"
    CUSTOM = "custom"


class PageComplexity(Enum):
    """Page complexity levels for LLM timing predictions."""
    SIMPLE = "simple"  # Static content, minimal JS
    MODERATE = "moderate"  # Some dynamic content, moderate JS
    COMPLEX = "complex"  # Heavy SPA, lots of async operations
    VERY_COMPLEX = "very_complex"  # Multiple frameworks, heavy animations


@dataclass
class NetworkActivity:
    """Tracks network request activity."""
    pending_requests: Set[str] = field(default_factory=set)
    request_timestamps: Dict[str, float] = field(default_factory=dict)
    last_activity_time: float = 0.0
    request_count: int = 0
    response_count: int = 0


@dataclass
class ElementMetrics:
    """Tracks element stability metrics."""
    position_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    last_change_time: float = 0.0
    stable_duration: float = 0.0
    is_visible: bool = False
    opacity_history: List[float] = field(default_factory=list)


@dataclass
class WaitConfig:
    """Configuration for wait strategies."""
    network_idle_timeout: float = 500.0  # ms
    network_idle_threshold: int = 0  # Max pending requests
    element_stable_timeout: float = 300.0  # ms
    element_stable_threshold: int = 3  # Number of stable checks
    dom_mutation_timeout: float = 400.0  # ms
    max_wait_time: float = 30000.0  # ms
    poll_interval: float = 50.0  # ms
    llm_prediction_weight: float = 0.7  # Weight for LLM predictions vs heuristics


@dataclass
class TimingPrediction:
    """LLM-based timing prediction result."""
    predicted_load_time: float  # ms
    confidence: float  # 0-1
    reasoning: str
    page_complexity: PageComplexity
    critical_resources: List[str] = field(default_factory=list)


class StabilityAnalyzer:
    """
    Intelligent wait system that replaces fixed sleeps with dynamic waiting based on:
    1. Network idle detection
    2. Element stability analysis
    3. DOM mutation monitoring
    4. LLM-predicted load times
    
    Provides configurable wait strategies per action type.
    """
    
    def __init__(
        self,
        page: Page,
        agent_service: Optional[AgentService] = None,
        config: Optional[WaitConfig] = None
    ):
        self.page = page
        self.agent_service = agent_service
        self.config = config or WaitConfig()
        
        # State tracking
        self.network_activity = NetworkActivity()
        self.dom_mutation_count = 0
        self.last_dom_mutation_time = 0.0
        self.element_metrics_cache: Dict[str, ElementMetrics] = {}
        self.timing_predictions_cache: Dict[str, TimingPrediction] = {}
        
        # Event handlers
        self._request_handler = None
        self._response_handler = None
        self._dom_mutation_observer = None
        self._frame_navigated_handler = None
        
        # Performance tracking
        self.wait_times: List[float] = []
        self.strategy_success_rates: Dict[WaitStrategy, float] = {}
        
        self._setup_monitoring()
    
    def _setup_monitoring(self) -> None:
        """Set up event listeners for monitoring page activity."""
        # Network monitoring
        self._request_handler = self.page.on("request", self._on_request)
        self._response_handler = self.page.on("response", self._on_response)
        
        # DOM mutation monitoring
        self._setup_dom_mutation_observer()
        
        # Navigation monitoring
        self._frame_navigated_handler = self.page.on("framenavigated", self._on_navigation)
    
    async def _setup_dom_mutation_observer(self) -> None:
        """Set up a MutationObserver to track DOM changes."""
        script = """
        () => {
            if (window.__stabilityAnalyzerObserver) {
                window.__stabilityAnalyzerObserver.disconnect();
            }
            
            window.__stabilityAnalyzerMutationCount = 0;
            window.__stabilityAnalyzerLastMutationTime = 0;
            
            const observer = new MutationObserver((mutations) => {
                window.__stabilityAnalyzerMutationCount += mutations.length;
                window.__stabilityAnalyzerLastMutationTime = Date.now();
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true,
                attributes: true,
                characterData: true
            });
            
            window.__stabilityAnalyzerObserver = observer;
            return true;
        }
        """
        
        try:
            await self.page.evaluate(script)
        except Exception as e:
            logger.warning(f"Failed to set up DOM mutation observer: {e}")
    
    def _on_request(self, request: Request) -> None:
        """Handle new network request."""
        request_id = f"{request.url}_{time.time()}"
        self.network_activity.pending_requests.add(request_id)
        self.network_activity.request_timestamps[request_id] = time.time()
        self.network_activity.last_activity_time = time.time()
        self.network_activity.request_count += 1
    
    def _on_response(self, response: Response) -> None:
        """Handle network response."""
        # Remove matching request (simplified - in production would track by request ID)
        current_time = time.time()
        stale_requests = [
            req_id for req_id, timestamp in self.network_activity.request_timestamps.items()
            if current_time - timestamp > 5000  # 5 second timeout
        ]
        
        for req_id in stale_requests:
            self.network_activity.pending_requests.discard(req_id)
            self.network_activity.request_timestamps.pop(req_id, None)
        
        self.network_activity.response_count += 1
        self.network_activity.last_activity_time = current_time
    
    async def _on_navigation(self, frame) -> None:
        """Handle page navigation."""
        if frame == self.page.main_frame:
            # Reset monitoring on navigation
            self.network_activity = NetworkActivity()
            self.dom_mutation_count = 0
            self.last_dom_mutation_time = 0.0
            self.element_metrics_cache.clear()
            
            # Re-setup DOM observer
            await self._setup_dom_mutation_observer()
    
    async def _get_dom_mutation_stats(self) -> Tuple[int, float]:
        """Get current DOM mutation statistics from the page."""
        script = """
        () => {
            return {
                count: window.__stabilityAnalyzerMutationCount || 0,
                lastTime: window.__stabilityAnalyzerLastMutationTime || 0
            };
        }
        """
        
        try:
            result = await self.page.evaluate(script)
            return result.get('count', 0), result.get('lastTime', 0)
        except:
            return 0, 0
    
    async def _get_element_metrics(self, element: ElementHandle) -> Optional[ElementMetrics]:
        """Get current metrics for an element."""
        try:
            # Get bounding box
            bbox = await element.bounding_box()
            if not bbox:
                return None
            
            # Get visibility and opacity
            is_visible = await element.is_visible()
            opacity = await element.evaluate("el => window.getComputedStyle(el).opacity")
            
            metrics = ElementMetrics(
                position_history=[(bbox['x'], bbox['y'], bbox['width'], bbox['height'])],
                last_change_time=time.time(),
                is_visible=is_visible,
                opacity_history=[float(opacity) if opacity else 1.0]
            )
            
            return metrics
        except Exception as e:
            logger.debug(f"Failed to get element metrics: {e}")
            return None
    
    async def _check_element_stability(
        self,
        element: ElementHandle,
        element_id: str,
        timeout: Optional[float] = None
    ) -> bool:
        """Check if element is stable (no position/size changes)."""
        timeout = timeout or self.config.element_stable_timeout
        start_time = time.time() * 1000  # Convert to ms
        stable_checks = 0
        
        # Get or create metrics for this element
        if element_id not in self.element_metrics_cache:
            metrics = await self._get_element_metrics(element)
            if not metrics:
                return False
            self.element_metrics_cache[element_id] = metrics
        else:
            metrics = self.element_metrics_cache[element_id]
        
        while (time.time() * 1000 - start_time) < timeout:
            current_metrics = await self._get_element_metrics(element)
            if not current_metrics:
                return False
            
            # Compare with previous metrics
            if metrics.position_history:
                last_pos = metrics.position_history[-1]
                current_pos = current_metrics.position_history[0]
                
                # Check if position/size changed significantly
                position_changed = (
                    abs(current_pos[0] - last_pos[0]) > 1 or
                    abs(current_pos[1] - last_pos[1]) > 1 or
                    abs(current_pos[2] - last_pos[2]) > 1 or
                    abs(current_pos[3] - last_pos[3]) > 1
                )
                
                # Check opacity change
                opacity_changed = False
                if metrics.opacity_history and current_metrics.opacity_history:
                    opacity_changed = abs(
                        current_metrics.opacity_history[0] - metrics.opacity_history[-1]
                    ) > 0.05
                
                if not position_changed and not opacity_changed:
                    stable_checks += 1
                    if stable_checks >= self.config.element_stable_threshold:
                        return True
                else:
                    stable_checks = 0
                    # Update metrics with new position
                    metrics.position_history.append(current_pos)
                    metrics.opacity_history.append(current_metrics.opacity_history[0])
                    metrics.last_change_time = time.time()
            
            await asyncio.sleep(self.config.poll_interval / 1000)
        
        return stable_checks >= self.config.element_stable_threshold
    
    async def _check_network_idle(self, timeout: Optional[float] = None) -> bool:
        """Check if network is idle (no pending requests)."""
        timeout = timeout or self.config.network_idle_timeout
        start_time = time.time() * 1000
        
        while (time.time() * 1000 - start_time) < timeout:
            if len(self.network_activity.pending_requests) <= self.config.network_idle_threshold:
                # Check if we've been idle for enough time
                idle_duration = (time.time() * 1000) - (self.network_activity.last_activity_time * 1000)
                if idle_duration >= self.config.network_idle_threshold:
                    return True
            
            await asyncio.sleep(self.config.poll_interval / 1000)
        
        return len(self.network_activity.pending_requests) <= self.config.network_idle_threshold
    
    async def _check_dom_mutation_idle(self, timeout: Optional[float] = None) -> bool:
        """Check if DOM mutations have stopped."""
        timeout = timeout or self.config.dom_mutation_timeout
        start_time = time.time() * 1000
        
        while (time.time() * 1000 - start_time) < timeout:
            current_count, last_mutation_time = await self._get_dom_mutation_stats()
            
            if last_mutation_time > 0:
                idle_duration = time.time() * 1000 - last_mutation_time
                if idle_duration >= self.config.dom_mutation_timeout:
                    return True
            
            await asyncio.sleep(self.config.poll_interval / 1000)
        
        return False
    
    async def _predict_load_time_with_llm(
        self,
        action_type: str,
        element_selector: Optional[str] = None
    ) -> TimingPrediction:
        """Use LLM to predict load time based on page complexity."""
        # Check cache first
        cache_key = f"{self.page.url}_{action_type}_{element_selector}"
        if cache_key in self.timing_predictions_cache:
            return self.timing_predictions_cache[cache_key]
        
        # Default prediction if no agent service available
        if not self.agent_service:
            prediction = TimingPrediction(
                predicted_load_time=1000.0,
                confidence=0.5,
                reasoning="No LLM service available, using default prediction",
                page_complexity=PageComplexity.MODERATE
            )
            self.timing_predictions_cache[cache_key] = prediction
            return prediction
        
        try:
            # Analyze page complexity
            complexity = await self._analyze_page_complexity()
            
            # Get LLM prediction
            prompt = self._create_prediction_prompt(action_type, element_selector, complexity)
            
            # In a real implementation, this would call the LLM service
            # For now, we'll use heuristic-based prediction
            base_time = {
                PageComplexity.SIMPLE: 500,
                PageComplexity.MODERATE: 1000,
                PageComplexity.COMPLEX: 2000,
                PageComplexity.VERY_COMPLEX: 3000
            }[complexity]
            
            # Adjust based on action type
            action_multipliers = {
                "click": 1.0,
                "type": 1.2,
                "navigate": 1.5,
                "wait_for_element": 0.8,
                "screenshot": 0.5
            }
            
            predicted_time = base_time * action_multipliers.get(action_type, 1.0)
            
            prediction = TimingPrediction(
                predicted_load_time=predicted_time,
                confidence=0.7,
                reasoning=f"Heuristic prediction based on {complexity.value} page complexity",
                page_complexity=complexity
            )
            
            self.timing_predictions_cache[cache_key] = prediction
            return prediction
            
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")
            return TimingPrediction(
                predicted_load_time=1500.0,
                confidence=0.3,
                reasoning=f"LLM prediction failed: {str(e)}",
                page_complexity=PageComplexity.MODERATE
            )
    
    async def _analyze_page_complexity(self) -> PageComplexity:
        """Analyze page to determine complexity level."""
        try:
            script = """
            () => {
                const metrics = {
                    domSize: document.querySelectorAll('*').length,
                    scriptCount: document.querySelectorAll('script').length,
                    styleCount: document.querySelectorAll('style, link[rel="stylesheet"]').length,
                    iframeCount: document.querySelectorAll('iframe').length,
                    imageCount: document.querySelectorAll('img').length,
                    hasServiceWorker: 'serviceWorker' in navigator,
                    hasWebGL: !!document.createElement('canvas').getContext('webgl'),
                    hasWebAssembly: typeof WebAssembly === 'object'
                };
                
                // Simple complexity scoring
                let score = 0;
                score += Math.min(metrics.domSize / 100, 10);  // Up to 10 points for DOM size
                score += Math.min(metrics.scriptCount / 5, 5);  // Up to 5 points for scripts
                score += metrics.hasWebGL ? 2 : 0;
                score += metrics.hasWebAssembly ? 2 : 0;
                score += Math.min(metrics.iframeCount, 3);
                
                return { score, metrics };
            }
            """
            
            result = await self.page.evaluate(script)
            score = result.get('score', 0)
            
            if score < 5:
                return PageComplexity.SIMPLE
            elif score < 10:
                return PageComplexity.MODERATE
            elif score < 15:
                return PageComplexity.COMPLEX
            else:
                return PageComplexity.VERY_COMPLEX
                
        except Exception as e:
            logger.warning(f"Page complexity analysis failed: {e}")
            return PageComplexity.MODERATE
    
    def _create_prediction_prompt(
        self,
        action_type: str,
        element_selector: Optional[str],
        complexity: PageComplexity
    ) -> str:
        """Create prompt for LLM timing prediction."""
        return f"""
        Predict the expected load/execution time for a browser automation action.
        
        Context:
        - Page URL: {self.page.url}
        - Action type: {action_type}
        - Element selector: {element_selector or 'N/A'}
        - Page complexity: {complexity.value}
        - Current network requests: {len(self.network_activity.pending_requests)}
        
        Consider:
        1. Typical load times for this type of action
        2. Page complexity and framework usage
        3. Network conditions
        4. Element rendering requirements
        
        Provide:
        1. Predicted time in milliseconds
        2. Confidence level (0-1)
        3. Brief reasoning
        4. List of critical resources that might affect timing
        """
    
    async def wait_for_stability(
        self,
        strategy: Union[WaitStrategy, str] = WaitStrategy.HYBRID,
        element: Optional[ElementHandle] = None,
        action_type: str = "generic",
        timeout: Optional[float] = None,
        custom_condition: Optional[Callable[[], bool]] = None
    ) -> bool:
        """
        Main method to wait for page/element stability using specified strategy.
        
        Args:
            strategy: Wait strategy to use
            element: Optional element to wait for stability
            action_type: Type of action being performed (for strategy selection)
            timeout: Maximum wait time in ms
            custom_condition: Custom condition function for CUSTOM strategy
            
        Returns:
            bool: True if stability achieved, False if timeout
        """
        start_time = time.time() * 1000
        timeout = timeout or self.config.max_wait_time
        
        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            try:
                strategy = WaitStrategy(strategy)
            except ValueError:
                strategy = WaitStrategy.HYBRID
        
        # Get LLM prediction for better timing
        prediction = await self._predict_load_time_with_llm(
            action_type,
            str(element) if element else None
        )
        
        # Adjust timeout based on LLM prediction
        adjusted_timeout = min(
            timeout,
            prediction.predicted_load_time * (2 - prediction.confidence)  # Less confident = more buffer
        )
        
        logger.debug(
            f"Waiting for stability with strategy {strategy.value}, "
            f"predicted time: {prediction.predicted_load_time:.0f}ms, "
            f"adjusted timeout: {adjusted_timeout:.0f}ms"
        )
        
        try:
            if strategy == WaitStrategy.NETWORK_IDLE:
                return await self._wait_network_idle(adjusted_timeout)
            
            elif strategy == WaitStrategy.ELEMENT_STABLE:
                if not element:
                    raise ValueError("Element required for ELEMENT_STABLE strategy")
                return await self._wait_element_stable(element, adjusted_timeout)
            
            elif strategy == WaitStrategy.DOM_MUTATION_IDLE:
                return await self._wait_dom_mutation_idle(adjusted_timeout)
            
            elif strategy == WaitStrategy.HYBRID:
                return await self._wait_hybrid(element, adjusted_timeout)
            
            elif strategy == WaitStrategy.LLM_PREDICTED:
                # Wait for the predicted time plus a buffer
                await asyncio.sleep(prediction.predicted_load_time / 1000)
                return True
            
            elif strategy == WaitStrategy.CUSTOM:
                if not custom_condition:
                    raise ValueError("Custom condition required for CUSTOM strategy")
                return await self._wait_custom(custom_condition, adjusted_timeout)
            
            else:
                # Fallback to hybrid
                return await self._wait_hybrid(element, adjusted_timeout)
                
        except Exception as e:
            logger.error(f"Wait for stability failed: {e}")
            return False
        finally:
            # Track performance
            wait_time = time.time() * 1000 - start_time
            self.wait_times.append(wait_time)
    
    async def _wait_network_idle(self, timeout: float) -> bool:
        """Wait for network to become idle."""
        return await self._check_network_idle(timeout)
    
    async def _wait_element_stable(self, element: ElementHandle, timeout: float) -> bool:
        """Wait for element to become stable."""
        element_id = str(id(element))
        return await self._check_element_stability(element, element_id, timeout)
    
    async def _wait_dom_mutation_idle(self, timeout: float) -> bool:
        """Wait for DOM mutations to stop."""
        return await self._check_dom_mutation_idle(timeout)
    
    async def _wait_hybrid(self, element: Optional[ElementHandle], timeout: float) -> bool:
        """
        Hybrid strategy: Wait for network idle AND element stability if provided,
        with DOM mutation idle as a fallback.
        """
        start_time = time.time() * 1000
        remaining_timeout = timeout
        
        # Wait for network idle first
        network_timeout = min(remaining_timeout, self.config.network_idle_timeout * 2)
        network_idle = await self._check_network_idle(network_timeout)
        
        elapsed = time.time() * 1000 - start_time
        remaining_timeout = max(0, timeout - elapsed)
        
        # If element provided, wait for its stability
        if element and remaining_timeout > 0:
            element_timeout = min(remaining_timeout, self.config.element_stable_timeout * 2)
            element_stable = await self._check_element_stability(element, str(id(element)), element_timeout)
            
            elapsed = time.time() * 1000 - start_time
            remaining_timeout = max(0, timeout - elapsed)
        
        # If still have time, check DOM mutations
        if remaining_timeout > 0:
            dom_timeout = min(remaining_timeout, self.config.dom_mutation_timeout * 2)
            dom_idle = await self._check_dom_mutation_idle(dom_timeout)
        
        # Consider success if at least network is idle
        return network_idle
    
    async def _wait_custom(self, condition: Callable[[], bool], timeout: float) -> bool:
        """Wait for custom condition to be true."""
        start_time = time.time() * 1000
        
        while (time.time() * 1000 - start_time) < timeout:
            try:
                if condition():
                    return True
            except Exception as e:
                logger.warning(f"Custom condition check failed: {e}")
            
            await asyncio.sleep(self.config.poll_interval / 1000)
        
        return False
    
    def get_optimal_strategy(self, action_type: str) -> WaitStrategy:
        """
        Get optimal wait strategy for an action type based on historical performance.
        
        Args:
            action_type: Type of action being performed
            
        Returns:
            Optimal WaitStrategy for this action type
        """
        # Default strategies per action type
        default_strategies = {
            "click": WaitStrategy.HYBRID,
            "type": WaitStrategy.ELEMENT_STABLE,
            "navigate": WaitStrategy.NETWORK_IDLE,
            "wait_for_element": WaitStrategy.ELEMENT_STABLE,
            "screenshot": WaitStrategy.DOM_MUTATION_IDLE,
            "scroll": WaitStrategy.ELEMENT_STABLE,
            "select": WaitStrategy.HYBRID,
            "hover": WaitStrategy.ELEMENT_STABLE,
            "evaluate": WaitStrategy.NETWORK_IDLE,
            "generic": WaitStrategy.HYBRID
        }
        
        # Return default strategy for action type
        return default_strategies.get(action_type, WaitStrategy.HYBRID)
    
    async def smart_wait(
        self,
        element: Optional[ElementHandle] = None,
        action_type: str = "generic",
        timeout: Optional[float] = None
    ) -> bool:
        """
        Convenience method that automatically selects the best wait strategy.
        
        Args:
            element: Optional element to wait for
            action_type: Type of action being performed
            timeout: Maximum wait time
            
        Returns:
            True if stability achieved
        """
        strategy = self.get_optimal_strategy(action_type)
        return await self.wait_for_stability(
            strategy=strategy,
            element=element,
            action_type=action_type,
            timeout=timeout
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the wait system."""
        if not self.wait_times:
            return {
                "average_wait_time": 0,
                "max_wait_time": 0,
                "min_wait_time": 0,
                "total_waits": 0,
                "strategy_success_rates": self.strategy_success_rates
            }
        
        return {
            "average_wait_time": np.mean(self.wait_times),
            "max_wait_time": np.max(self.wait_times),
            "min_wait_time": np.min(self.wait_times),
            "total_waits": len(self.wait_times),
            "strategy_success_rates": self.strategy_success_rates,
            "network_stats": {
                "total_requests": self.network_activity.request_count,
                "total_responses": self.network_activity.response_count,
                "current_pending": len(self.network_activity.pending_requests)
            }
        }
    
    def cleanup(self) -> None:
        """Clean up event listeners and resources."""
        if self._request_handler:
            self.page.remove_listener("request", self._request_handler)
        
        if self._response_handler:
            self.page.remove_listener("response", self._response_handler)
        
        if self._frame_navigated_handler:
            self.page.remove_listener("framenavigated", self._frame_navigated_handler)
        
        # Remove DOM mutation observer
        try:
            asyncio.create_task(self.page.evaluate(
                "() => { if (window.__stabilityAnalyzerObserver) window.__stabilityAnalyzerObserver.disconnect(); }"
            ))
        except:
            pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


# Integration with existing PageActor
class SmartPageActor(PageActor):
    """Extended PageActor with smart waiting capabilities."""
    
    def __init__(self, page: Page, agent_service: Optional[AgentService] = None):
        super().__init__(page)
        self.stability_analyzer = StabilityAnalyzer(page, agent_service)
    
    async def click(
        self,
        selector: str,
        wait_strategy: Union[WaitStrategy, str] = WaitStrategy.HYBRID,
        timeout: Optional[float] = None
    ) -> None:
        """Click with intelligent waiting."""
        element = await self.page.query_selector(selector)
        if element:
            await self.stability_analyzer.wait_for_stability(
                strategy=wait_strategy,
                element=element,
                action_type="click",
                timeout=timeout
            )
        
        await super().click(selector)
    
    async def type(
        self,
        selector: str,
        text: str,
        wait_strategy: Union[WaitStrategy, str] = WaitStrategy.ELEMENT_STABLE,
        timeout: Optional[float] = None
    ) -> None:
        """Type with intelligent waiting."""
        element = await self.page.query_selector(selector)
        if element:
            await self.stability_analyzer.wait_for_stability(
                strategy=wait_strategy,
                element=element,
                action_type="type",
                timeout=timeout
            )
        
        await super().type(selector, text)
    
    async def navigate(
        self,
        url: str,
        wait_strategy: Union[WaitStrategy, str] = WaitStrategy.NETWORK_IDLE,
        timeout: Optional[float] = None
    ) -> None:
        """Navigate with intelligent waiting."""
        await self.stability_analyzer.wait_for_stability(
            strategy=wait_strategy,
            action_type="navigate",
            timeout=timeout
        )
        
        await super().navigate(url)
        
        # Wait for new page to stabilize
        await self.stability_analyzer.wait_for_stability(
            strategy=wait_strategy,
            action_type="post_navigation",
            timeout=timeout
        )
    
    async def wait_for_element(
        self,
        selector: str,
        wait_strategy: Union[WaitStrategy, str] = WaitStrategy.ELEMENT_STABLE,
        timeout: Optional[float] = None
    ) -> ElementHandle:
        """Wait for element with intelligent stability checking."""
        # First wait for element to exist
        element = await self.page.wait_for_selector(selector, timeout=timeout)
        
        if element:
            # Then wait for it to be stable
            await self.stability_analyzer.wait_for_stability(
                strategy=wait_strategy,
                element=element,
                action_type="wait_for_element",
                timeout=timeout
            )
        
        return element
    
    def get_wait_stats(self) -> Dict[str, Any]:
        """Get waiting performance statistics."""
        return self.stability_analyzer.get_performance_stats()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stability_analyzer.cleanup()


# Factory function for easy integration
def create_smart_wait_system(
    page: Page,
    agent_service: Optional[AgentService] = None,
    config: Optional[WaitConfig] = None
) -> StabilityAnalyzer:
    """
    Factory function to create a smart wait system.
    
    Args:
        page: Playwright page object
        agent_service: Optional agent service for LLM predictions
        config: Optional configuration
        
    Returns:
        Configured StabilityAnalyzer instance
    """
    return StabilityAnalyzer(page, agent_service, config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from playwright.async_api import async_playwright
    
    async def test_stability_analyzer():
        """Test the stability analyzer."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Create stability analyzer
            analyzer = StabilityAnalyzer(page)
            
            try:
                # Navigate to a test page
                await page.goto("https://example.com")
                
                # Wait for page to stabilize
                success = await analyzer.wait_for_stability(
                    strategy=WaitStrategy.HYBRID,
                    action_type="navigate"
                )
                
                print(f"Wait for stability: {'Success' if success else 'Failed'}")
                
                # Get performance stats
                stats = analyzer.get_performance_stats()
                print(f"Performance stats: {stats}")
                
            finally:
                analyzer.cleanup()
                await browser.close()
    
    # Run test
    asyncio.run(test_stability_analyzer())