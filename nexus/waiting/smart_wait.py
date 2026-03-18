"""Smart Wait & Timing System for nexus.

Replaces fixed sleeps with intelligent waiting based on network idle detection,
element stability analysis, and LLM-predicted load times. Reduces flakiness
and improves speed.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, List, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from ..actor.page import Page
from ..actor.element import Element
from ..agent.service import AgentService

logger = logging.getLogger(__name__)


class WaitStrategy(Enum):
    """Configurable wait strategies for different action types."""
    NETWORK_IDLE = "network_idle"
    ELEMENT_STABLE = "element_stable"
    DOM_STABLE = "dom_stable"
    LLM_PREDICTED = "llm_predicted"
    HYBRID = "hybrid"


@dataclass
class WaitConfig:
    """Configuration for smart waiting."""
    strategy: WaitStrategy = WaitStrategy.HYBRID
    network_idle_timeout: float = 2.0  # seconds
    network_idle_threshold: int = 0  # max concurrent requests
    element_stability_timeout: float = 1.0  # seconds to wait for stability
    element_stability_interval: float = 0.1  # check interval
    dom_mutation_timeout: float = 1.0  # seconds without mutations
    llm_timeout_multiplier: float = 1.5  # safety margin for LLM predictions
    min_wait_time: float = 0.1  # minimum wait time
    max_wait_time: float = 30.0  # maximum wait time
    action_overrides: Dict[str, 'WaitConfig'] = field(default_factory=dict)


class NetworkMonitor:
    """Monitors network activity to detect idle state."""
    
    def __init__(self, page: Page):
        self.page = page
        self._pending_requests: Set[str] = set()
        self._request_counter = 0
        self._network_idle_event = asyncio.Event()
        self._network_idle_event.set()  # Initially idle
        self._setup_listeners()
    
    def _setup_listeners(self):
        """Set up network event listeners."""
        if hasattr(self.page, '_page'):
            playwright_page = self.page._page
            
            def on_request(request):
                self._request_counter += 1
                request_id = f"req_{self._request_counter}"
                self._pending_requests.add(request_id)
                self._network_idle_event.clear()
                logger.debug(f"Network request started: {request.url} (ID: {request_id})")
            
            def on_response(response):
                request_id = self._find_request_id_for_response(response)
                if request_id and request_id in self._pending_requests:
                    self._pending_requests.remove(request_id)
                    logger.debug(f"Network response received: {response.url} (ID: {request_id})")
                    if not self._pending_requests:
                        self._network_idle_event.set()
            
            def on_request_failed(request):
                request_id = self._find_request_id_for_request(request)
                if request_id and request_id in self._pending_requests:
                    self._pending_requests.remove(request_id)
                    logger.debug(f"Network request failed: {request.url} (ID: {request_id})")
                    if not self._pending_requests:
                        self._network_idle_event.set()
            
            playwright_page.on("request", on_request)
            playwright_page.on("response", on_response)
            playwright_page.on("requestfailed", on_request_failed)
    
    def _find_request_id_for_request(self, request) -> Optional[str]:
        """Find request ID for a given request (simplified)."""
        # In a real implementation, we'd track request IDs more carefully
        for req_id in self._pending_requests:
            if hasattr(request, '_id') and str(request._id) in req_id:
                return req_id
        return None
    
    def _find_request_id_for_response(self, response) -> Optional[str]:
        """Find request ID for a given response (simplified)."""
        return self._find_request_id_for_request(response.request)
    
    async def wait_for_network_idle(self, timeout: float = 2.0, 
                                   threshold: int = 0) -> bool:
        """Wait for network to become idle.
        
        Args:
            timeout: Maximum time to wait
            threshold: Maximum number of concurrent requests to consider idle
            
        Returns:
            True if network became idle, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if len(self._pending_requests) <= threshold:
                # Double-check after a short delay
                await asyncio.sleep(0.1)
                if len(self._pending_requests) <= threshold:
                    logger.debug(f"Network idle detected ({len(self._pending_requests)} pending requests)")
                    return True
            
            # Wait for network idle event with remaining timeout
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                break
            
            try:
                await asyncio.wait_for(
                    self._network_idle_event.wait(),
                    timeout=min(remaining, 0.5)
                )
                if self._network_idle_event.is_set():
                    return True
            except asyncio.TimeoutError:
                continue
        
        logger.warning(f"Network idle timeout after {timeout}s ({len(self._pending_requests)} pending requests)")
        return False
    
    def get_pending_requests_count(self) -> int:
        """Get current number of pending requests."""
        return len(self._pending_requests)


class DOMStabilityMonitor:
    """Monitors DOM mutations and element stability."""
    
    def __init__(self, page: Page):
        self.page = page
        self._mutation_count = 0
        self._last_mutation_time = 0
        self._mutation_event = asyncio.Event()
        self._mutation_event.set()  # Initially stable
        self._element_metrics: Dict[str, Dict[str, Any]] = {}
        self._setup_mutation_observer()
    
    def _setup_mutation_observer(self):
        """Set up mutation observer on the page."""
        if hasattr(self.page, '_page'):
            playwright_page = self.page._page
            
            # Inject mutation observer script
            script = """
            () => {
                if (window.__browserUseMutationObserver) {
                    return; // Already set up
                }
                
                window.__browserUseMutationCount = 0;
                window.__browserUseLastMutationTime = Date.now();
                
                const observer = new MutationObserver((mutations) => {
                    window.__browserUseMutationCount += mutations.length;
                    window.__browserUseLastMutationTime = Date.now();
                    
                    // Notify Python side
                    if (window.__browserUseMutationCallback) {
                        window.__browserUseMutationCallback(mutations.length);
                    }
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    characterData: true
                });
                
                window.__browserUseMutationObserver = observer;
                return () => observer.disconnect();
            }
            """
            
            # We'll set up the callback separately
            asyncio.create_task(self._setup_mutation_callback(playwright_page))
    
    async def _setup_mutation_callback(self, playwright_page):
        """Set up callback for mutation events."""
        try:
            # Expose a function that the page can call
            def on_mutation(count: int):
                self._mutation_count += count
                self._last_mutation_time = time.time()
                self._mutation_event.clear()
                logger.debug(f"DOM mutation detected (count: {count})")
            
            await playwright_page.expose_function(
                "__browserUseMutationCallback",
                on_mutation
            )
            
            # Re-run the observer setup with the callback
            await playwright_page.evaluate("""
            () => {
                if (window.__browserUseMutationObserver) {
                    window.__browserUseMutationCallback = (count) => {
                        // This will be replaced by the exposed function
                    };
                }
            }
            """)
        except Exception as e:
            logger.warning(f"Could not set up mutation callback: {e}")
    
    async def wait_for_dom_stability(self, timeout: float = 1.0) -> bool:
        """Wait for DOM to become stable (no mutations).
        
        Args:
            timeout: Time without mutations to consider stable
            
        Returns:
            True if DOM became stable, False if timeout
        """
        start_time = time.time()
        last_mutation_time = self._last_mutation_time
        
        while time.time() - start_time < timeout:
            current_time = time.time()
            
            # Check if we've had a period without mutations
            if current_time - last_mutation_time >= timeout:
                logger.debug(f"DOM stable for {timeout}s")
                return True
            
            # Wait for next mutation or timeout
            remaining = timeout - (current_time - last_mutation_time)
            if remaining <= 0:
                return True
            
            try:
                await asyncio.wait_for(
                    self._mutation_event.wait(),
                    timeout=min(remaining, 0.1)
                )
                # If we get here, a mutation occurred
                last_mutation_time = self._last_mutation_time
                self._mutation_event.set()  # Reset for next wait
            except asyncio.TimeoutError:
                # No mutation in this interval
                if time.time() - last_mutation_time >= timeout:
                    return True
        
        logger.warning(f"DOM stability timeout after {timeout}s")
        return False
    
    async def check_element_stability(self, element: Element, 
                                    check_interval: float = 0.1,
                                    stable_duration: float = 0.5) -> bool:
        """Check if an element is stable (no position/size changes).
        
        Args:
            element: Element to check
            check_interval: Time between stability checks
            stable_duration: Time element must remain stable
            
        Returns:
            True if element is stable
        """
        if not element or not hasattr(element, 'bounding_box'):
            return False
        
        try:
            previous_box = await element.bounding_box()
            if not previous_box:
                return False
            
            stable_start = None
            
            while True:
                await asyncio.sleep(check_interval)
                current_box = await element.bounding_box()
                
                if not current_box:
                    return False
                
                # Check if bounding box changed
                if (abs(current_box['x'] - previous_box['x']) > 1 or
                    abs(current_box['y'] - previous_box['y']) > 1 or
                    abs(current_box['width'] - previous_box['width']) > 1 or
                    abs(current_box['height'] - previous_box['height']) > 1):
                    # Element moved/resized
                    stable_start = None
                    previous_box = current_box
                    continue
                
                # Element is stable
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_duration:
                    logger.debug(f"Element stable for {stable_duration}s")
                    return True
                    
        except Exception as e:
            logger.warning(f"Error checking element stability: {e}")
            return False


class LLMLoadPredictor:
    """Predicts load times using LLM based on page complexity."""
    
    def __init__(self, agent_service: Optional[AgentService] = None):
        self.agent_service = agent_service
        self._predictions_cache: Dict[str, float] = {}
    
    async def predict_load_time(self, page: Page, action_type: str, 
                              context: Optional[Dict[str, Any]] = None) -> float:
        """Predict load time for an action using LLM.
        
        Args:
            page: Page instance
            action_type: Type of action (click, type, navigate, etc.)
            context: Additional context for prediction
            
        Returns:
            Predicted load time in seconds
        """
        if not self.agent_service:
            return 1.0  # Default fallback
        
        try:
            # Gather page complexity metrics
            complexity = await self._analyze_page_complexity(page)
            
            # Create prompt for LLM
            prompt = self._create_prediction_prompt(
                action_type, complexity, context
            )
            
            # Get prediction from LLM (simplified - actual implementation
            # would use the agent service)
            prediction = await self._get_llm_prediction(prompt)
            
            # Cache the prediction
            cache_key = f"{action_type}_{hash(str(complexity))}"
            self._predictions_cache[cache_key] = prediction
            
            logger.debug(f"LLM predicted {prediction:.2f}s for {action_type}")
            return prediction
            
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")
            return 1.0  # Fallback
    
    async def _analyze_page_complexity(self, page: Page) -> Dict[str, Any]:
        """Analyze page complexity metrics."""
        if not hasattr(page, '_page'):
            return {}
        
        try:
            playwright_page = page._page
            
            # Get various complexity metrics
            metrics = await playwright_page.evaluate("""
            () => {
                return {
                    domElements: document.querySelectorAll('*').length,
                    scripts: document.querySelectorAll('script').length,
                    iframes: document.querySelectorAll('iframe').length,
                    images: document.querySelectorAll('img').length,
                    forms: document.querySelectorAll('form').length,
                    eventListeners: window.getEventListeners ? 
                        Object.keys(window.getEventListeners(document)).length : 0,
                    domDepth: (() => {
                        let maxDepth = 0;
                        const walk = (node, depth) => {
                            if (depth > maxDepth) maxDepth = depth;
                            Array.from(node.children).forEach(child => 
                                walk(child, depth + 1));
                        };
                        walk(document.body, 0);
                        return maxDepth;
                    })()
                };
            }
            """)
            
            # Add network activity if available
            if hasattr(page, '_network_monitor'):
                metrics['pendingRequests'] = page._network_monitor.get_pending_requests_count()
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not analyze page complexity: {e}")
            return {}
    
    def _create_prediction_prompt(self, action_type: str, 
                                complexity: Dict[str, Any],
                                context: Optional[Dict[str, Any]] = None) -> str:
        """Create prompt for LLM prediction."""
        prompt = f"""Predict the expected load time in seconds for a browser automation action.

Action Type: {action_type}
Page Complexity Metrics:
- DOM Elements: {complexity.get('domElements', 'unknown')}
- Scripts: {complexity.get('scripts', 'unknown')}
- IFrames: {complexity.get('iframes', 'unknown')}
- Images: {complexity.get('images', 'unknown')}
- Forms: {complexity.get('forms', 'unknown')}
- DOM Depth: {complexity.get('domDepth', 'unknown')}
- Pending Network Requests: {complexity.get('pendingRequests', 'unknown')}

Additional Context: {json.dumps(context) if context else 'None'}

Consider typical load times for this type of action on pages of similar complexity.
Return ONLY a number representing seconds (e.g., 1.5)."""
        
        return prompt
    
    async def _get_llm_prediction(self, prompt: str) -> float:
        """Get prediction from LLM (stub implementation)."""
        # In a real implementation, this would call the agent service
        # For now, use heuristics based on action type
        
        # Simple heuristic fallback
        if "navigate" in prompt.lower():
            return 2.0
        elif "click" in prompt.lower():
            return 1.0
        elif "type" in prompt.lower():
            return 0.5
        else:
            return 1.0


class SmartWait:
    """Intelligent waiting system that replaces fixed sleeps.
    
    Combines network idle detection, element stability analysis,
    and LLM-predicted load times for optimal waiting.
    """
    
    def __init__(self, page: Page, config: Optional[WaitConfig] = None,
                 agent_service: Optional[AgentService] = None):
        """Initialize SmartWait.
        
        Args:
            page: Page instance to monitor
            config: Wait configuration
            agent_service: Optional agent service for LLM predictions
        """
        self.page = page
        self.config = config or WaitConfig()
        self.agent_service = agent_service
        
        # Initialize monitors
        self.network_monitor = NetworkMonitor(page)
        self.dom_monitor = DOMStabilityMonitor(page)
        self.llm_predictor = LLMLoadPredictor(agent_service)
        
        # Store monitors on page for access by other components
        if not hasattr(page, '_network_monitor'):
            page._network_monitor = self.network_monitor
        if not hasattr(page, '_dom_monitor'):
            page._dom_monitor = self.dom_monitor
        
        # Statistics
        self._wait_stats = defaultdict(list)
    
    async def wait_for_action(self, action_type: str, 
                            element: Optional[Element] = None,
                            context: Optional[Dict[str, Any]] = None) -> bool:
        """Wait intelligently for an action to complete.
        
        Args:
            action_type: Type of action (click, type, navigate, etc.)
            element: Optional element involved in the action
            context: Additional context for waiting
            
        Returns:
            True if wait conditions met, False if timeout
        """
        start_time = time.time()
        
        # Get config for this action type (with overrides)
        config = self._get_config_for_action(action_type)
        
        logger.info(f"Smart wait for {action_type} (strategy: {config.strategy.value})")
        
        try:
            # Execute wait strategy
            success = await self._execute_wait_strategy(
                config.strategy, element, context, config
            )
            
            # Record statistics
            elapsed = time.time() - start_time
            self._wait_stats[action_type].append({
                'success': success,
                'elapsed': elapsed,
                'strategy': config.strategy.value
            })
            
            # Ensure minimum wait time
            if elapsed < config.min_wait_time:
                await asyncio.sleep(config.min_wait_time - elapsed)
            
            return success
            
        except Exception as e:
            logger.error(f"Smart wait failed for {action_type}: {e}")
            return False
    
    def _get_config_for_action(self, action_type: str) -> WaitConfig:
        """Get wait configuration for a specific action type."""
        if action_type in self.config.action_overrides:
            return self.config.action_overrides[action_type]
        return self.config
    
    async def _execute_wait_strategy(self, strategy: WaitStrategy,
                                   element: Optional[Element],
                                   context: Optional[Dict[str, Any]],
                                   config: WaitConfig) -> bool:
        """Execute a specific wait strategy."""
        if strategy == WaitStrategy.NETWORK_IDLE:
            return await self._wait_network_idle(config)
        
        elif strategy == WaitStrategy.ELEMENT_STABLE:
            if not element:
                logger.warning("Element stability requested but no element provided")
                return True
            return await self._wait_element_stable(element, config)
        
        elif strategy == WaitStrategy.DOM_STABLE:
            return await self._wait_dom_stable(config)
        
        elif strategy == WaitStrategy.LLM_PREDICTED:
            return await self._wait_llm_predicted(element, context, config)
        
        elif strategy == WaitStrategy.HYBRID:
            return await self._wait_hybrid(element, context, config)
        
        else:
            logger.warning(f"Unknown wait strategy: {strategy}")
            return True
    
    async def _wait_network_idle(self, config: WaitConfig) -> bool:
        """Wait for network to become idle."""
        return await self.network_monitor.wait_for_network_idle(
            timeout=config.network_idle_timeout,
            threshold=config.network_idle_threshold
        )
    
    async def _wait_element_stable(self, element: Element, 
                                 config: WaitConfig) -> bool:
        """Wait for element to become stable."""
        return await self.dom_monitor.check_element_stability(
            element,
            check_interval=config.element_stability_interval,
            stable_duration=config.element_stability_timeout
        )
    
    async def _wait_dom_stable(self, config: WaitConfig) -> bool:
        """Wait for DOM to become stable."""
        return await self.dom_monitor.wait_for_dom_stability(
            timeout=config.dom_mutation_timeout
        )
    
    async def _wait_llm_predicted(self, element: Optional[Element],
                                context: Optional[Dict[str, Any]],
                                config: WaitConfig) -> bool:
        """Wait for LLM-predicted duration."""
        action_type = context.get('action_type', 'unknown') if context else 'unknown'
        
        # Get prediction
        predicted_time = await self.llm_predictor.predict_load_time(
            self.page, action_type, context
        )
        
        # Apply multiplier and bounds
        wait_time = min(
            max(predicted_time * config.llm_timeout_multiplier, config.min_wait_time),
            config.max_wait_time
        )
        
        logger.debug(f"LLM predicted {predicted_time:.2f}s, waiting {wait_time:.2f}s")
        await asyncio.sleep(wait_time)
        return True
    
    async def _wait_hybrid(self, element: Optional[Element],
                         context: Optional[Dict[str, Any]],
                         config: WaitConfig) -> bool:
        """Hybrid strategy combining multiple approaches."""
        start_time = time.time()
        action_type = context.get('action_type', 'unknown') if context else 'unknown'
        
        # Get LLM prediction for overall timeout
        predicted_time = await self.llm_predictor.predict_load_time(
            self.page, action_type, context
        )
        overall_timeout = min(
            max(predicted_time * config.llm_timeout_multiplier, config.min_wait_time),
            config.max_wait_time
        )
        
        # Start with network idle (most important for page loads)
        network_success = await self._wait_network_idle(config)
        
        # If we have an element, wait for it to be stable
        if element:
            element_success = await self._wait_element_stable(element, config)
            if not element_success:
                logger.warning("Element did not stabilize")
        
        # Wait for DOM stability if we have time
        elapsed = time.time() - start_time
        remaining = overall_timeout - elapsed
        
        if remaining > 0.1:  # Only if we have meaningful time left
            dom_success = await self.dom_monitor.wait_for_dom_stability(
                timeout=min(remaining, config.dom_mutation_timeout)
            )
        
        # Ensure we've waited at least the predicted time
        elapsed = time.time() - start_time
        if elapsed < overall_timeout:
            await asyncio.sleep(overall_timeout - elapsed)
        
        return network_success
    
    async def wait_for_page_load(self, timeout: Optional[float] = None) -> bool:
        """Wait for full page load (convenience method)."""
        config = WaitConfig(
            strategy=WaitStrategy.HYBRID,
            network_idle_timeout=timeout or 5.0,
            dom_mutation_timeout=timeout or 2.0
        )
        return await self._wait_hybrid(None, {'action_type': 'navigate'}, config)
    
    async def wait_for_element_ready(self, element: Element,
                                   timeout: Optional[float] = None) -> bool:
        """Wait for element to be ready for interaction."""
        config = WaitConfig(
            strategy=WaitStrategy.ELEMENT_STABLE,
            element_stability_timeout=timeout or 1.0
        )
        return await self._wait_element_stable(element, config)
    
    def get_wait_statistics(self) -> Dict[str, Any]:
        """Get statistics about wait operations."""
        stats = {}
        for action_type, waits in self._wait_stats.items():
            if waits:
                avg_time = sum(w['elapsed'] for w in waits) / len(waits)
                success_rate = sum(1 for w in waits if w['success']) / len(waits)
                stats[action_type] = {
                    'count': len(waits),
                    'avg_wait_time': avg_time,
                    'success_rate': success_rate,
                    'strategies_used': list(set(w['strategy'] for w in waits))
                }
        return stats
    
    def clear_statistics(self):
        """Clear wait statistics."""
        self._wait_stats.clear()


# Convenience functions for easy integration
async def smart_wait(page: Page, action_type: str, 
                    element: Optional[Element] = None,
                    config: Optional[WaitConfig] = None,
                    agent_service: Optional[AgentService] = None,
                    **context) -> bool:
    """Convenience function for smart waiting.
    
    Args:
        page: Page instance
        action_type: Type of action
        element: Optional element
        config: Wait configuration
        agent_service: Optional agent service
        **context: Additional context
        
    Returns:
        True if wait successful
    """
    waiter = SmartWait(page, config, agent_service)
    context_dict = {'action_type': action_type, **context}
    return await waiter.wait_for_action(action_type, element, context_dict)


# Default configurations for common action types
DEFAULT_ACTION_CONFIGS = {
    'click': WaitConfig(
        strategy=WaitStrategy.HYBRID,
        network_idle_timeout=1.0,
        element_stability_timeout=0.5
    ),
    'type': WaitConfig(
        strategy=WaitStrategy.ELEMENT_STABLE,
        element_stability_timeout=0.3
    ),
    'navigate': WaitConfig(
        strategy=WaitStrategy.HYBRID,
        network_idle_timeout=3.0,
        dom_mutation_timeout=2.0
    ),
    'screenshot': WaitConfig(
        strategy=WaitStrategy.DOM_STABLE,
        dom_mutation_timeout=1.0
    ),
    'extract': WaitConfig(
        strategy=WaitStrategy.HYBRID,
        network_idle_timeout=1.0,
        dom_mutation_timeout=0.5
    )
}


def create_default_config() -> WaitConfig:
    """Create default wait configuration with action overrides."""
    config = WaitConfig()
    config.action_overrides = DEFAULT_ACTION_CONFIGS.copy()
    return config


# Integration helper for existing code
class SmartWaitMixin:
    """Mixin class to add smart waiting capabilities to existing classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._smart_wait: Optional[SmartWait] = None
        self._wait_config = create_default_config()
    
    def init_smart_wait(self, page: Page, 
                       agent_service: Optional[AgentService] = None):
        """Initialize smart waiting for this instance."""
        self._smart_wait = SmartWait(page, self._wait_config, agent_service)
    
    async def smart_wait(self, action_type: str, 
                        element: Optional[Element] = None,
                        **context) -> bool:
        """Perform smart wait for an action."""
        if not self._smart_wait:
            logger.warning("Smart wait not initialized, using fallback")
            await asyncio.sleep(1.0)  # Fallback
            return True
        
        return await self._smart_wait.wait_for_action(
            action_type, element, context
        )
    
    def configure_wait_strategy(self, action_type: str, 
                              config: WaitConfig):
        """Configure wait strategy for a specific action type."""
        self._wait_config.action_overrides[action_type] = config
        if self._smart_wait:
            self._smart_wait.config = self._wait_config
    
    def get_wait_statistics(self) -> Dict[str, Any]:
        """Get wait statistics."""
        if self._smart_wait:
            return self._smart_wait.get_wait_statistics()
        return {}