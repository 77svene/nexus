import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import re

from playwright.async_api import Page, BrowserContext, Response, Request

logger = logging.getLogger(__name__)


class PageState(Enum):
    """Predicted states of page loading"""
    INITIAL = "initial"
    DOM_LOADING = "dom_loading"
    DOM_READY = "dom_ready"
    RESOURCES_LOADING = "resources_loading"
    JAVASCRIPT_EXECUTING = "javascript_executing"
    STABLE = "stable"
    INTERACTIVE = "interactive"


@dataclass
class NetworkPattern:
    """Pattern analysis for network requests"""
    url_pattern: str
    avg_response_time: float
    success_rate: float
    last_accessed: float
    resource_type: str  # 'document', 'script', 'xhr', 'fetch', etc.
    is_critical: bool = True


@dataclass
class PrefetchCandidate:
    """Candidate URL for prefetching"""
    url: str
    confidence: float
    predicted_next_actions: List[str]
    estimated_load_time: float
    priority: int = 1


@dataclass
class DOMMutationStats:
    """Statistics about DOM mutations"""
    mutation_count: int
    mutation_rate: float  # mutations per second
    last_mutation_time: float
    element_types_changed: Dict[str, int] = field(default_factory=dict)


class PredictivePrefetcher:
    """Handles predictive prefetching of likely next pages"""
    
    def __init__(self, context: BrowserContext):
        self.context = context
        self.prefetched_pages: Dict[str, Page] = {}
        self.navigation_history: deque = deque(maxlen=100)
        self.url_patterns: Dict[str, NetworkPattern] = {}
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_worker_task: Optional[asyncio.Task] = None
        self.max_prefetch_pages = 3
        self.confidence_threshold = 0.7
        
    async def start(self):
        """Start the prefetch worker"""
        if not self.prefetch_worker_task:
            self.prefetch_worker_task = asyncio.create_task(self._prefetch_worker())
    
    async def stop(self):
        """Stop the prefetch worker and cleanup"""
        if self.prefetch_worker_task:
            self.prefetch_worker_task.cancel()
            try:
                await self.prefetch_worker_task
            except asyncio.CancelledError:
                pass
        
        # Close all prefetched pages
        for page in self.prefetched_pages.values():
            try:
                await page.close()
            except:
                pass
        self.prefetched_pages.clear()
    
    def record_navigation(self, from_url: str, to_url: str):
        """Record a navigation for pattern analysis"""
        self.navigation_history.append((from_url, to_url, time.time()))
        self._update_patterns(from_url, to_url)
    
    def _update_patterns(self, from_url: str, to_url: str):
        """Update navigation patterns"""
        # Extract pattern from URL
        pattern = self._extract_url_pattern(to_url)
        
        if pattern not in self.url_patterns:
            self.url_patterns[pattern] = NetworkPattern(
                url_pattern=pattern,
                avg_response_time=0.0,
                success_rate=1.0,
                last_accessed=time.time(),
                resource_type="document"
            )
        
        # Update pattern statistics
        pattern_obj = self.url_patterns[pattern]
        pattern_obj.last_accessed = time.time()
    
    def _extract_url_pattern(self, url: str) -> str:
        """Extract pattern from URL for grouping similar URLs"""
        # Remove query parameters and fragments
        clean_url = url.split('?')[0].split('#')[0]
        
        # Replace numbers with placeholders for pattern matching
        pattern = re.sub(r'/\d+', '/{id}', clean_url)
        pattern = re.sub(r'/[a-f0-9]{8,}', '/{hash}', pattern)
        
        return pattern
    
    async def predict_next_urls(self, current_url: str, current_state: Dict) -> List[PrefetchCandidate]:
        """Predict likely next URLs based on patterns and current state"""
        candidates = []
        
        # Analyze navigation history for patterns
        for from_url, to_url, timestamp in self.navigation_history:
            if self._urls_match_pattern(from_url, current_url):
                pattern = self._extract_url_pattern(to_url)
                if pattern in self.url_patterns:
                    confidence = self._calculate_confidence(pattern, current_state)
                    if confidence >= self.confidence_threshold:
                        candidates.append(PrefetchCandidate(
                            url=to_url,
                            confidence=confidence,
                            predicted_next_actions=self._predict_actions(to_url),
                            estimated_load_time=self.url_patterns[pattern].avg_response_time,
                            priority=int(confidence * 10)
                        ))
        
        # Analyze current page for links
        link_candidates = await self._analyze_page_links(current_url)
        candidates.extend(link_candidates)
        
        # Sort by confidence and priority
        candidates.sort(key=lambda x: (x.confidence, x.priority), reverse=True)
        
        return candidates[:self.max_prefetch_pages]
    
    def _urls_match_pattern(self, url1: str, url2: str) -> bool:
        """Check if two URLs match the same pattern"""
        pattern1 = self._extract_url_pattern(url1)
        pattern2 = self._extract_url_pattern(url2)
        return pattern1 == pattern2
    
    def _calculate_confidence(self, pattern: str, current_state: Dict) -> float:
        """Calculate confidence score for a navigation pattern"""
        if pattern not in self.url_patterns:
            return 0.0
        
        pattern_obj = self.url_patterns[pattern]
        
        # Base confidence on recency and success rate
        recency = 1.0 / (time.time() - pattern_obj.last_accessed + 1)
        confidence = recency * pattern_obj.success_rate
        
        # Adjust based on current page state
        if current_state.get('page_state') == PageState.STABLE:
            confidence *= 1.2
        elif current_state.get('page_state') == PageState.INTERACTIVE:
            confidence *= 1.5
        
        return min(confidence, 1.0)
    
    def _predict_actions(self, url: str) -> List[str]:
        """Predict likely actions on a page"""
        # Simple heuristic based on URL patterns
        actions = []
        
        if '/search' in url or '/find' in url:
            actions.extend(['type', 'click_search', 'click_result'])
        elif '/product' in url or '/item' in url:
            actions.extend(['click_add_to_cart', 'click_buy_now', 'scroll_to_reviews'])
        elif '/checkout' in url or '/cart' in url:
            actions.extend(['click_proceed', 'fill_form', 'click_confirm'])
        
        return actions
    
    async def _analyze_page_links(self, current_url: str) -> List[PrefetchCandidate]:
        """Analyze current page for likely next links"""
        candidates = []
        
        # This would be implemented to analyze the current page's DOM
        # For now, return empty list - would be integrated with DOM analysis
        
        return candidates
    
    async def _prefetch_worker(self):
        """Background worker for prefetching pages"""
        while True:
            try:
                candidate = await self.prefetch_queue.get()
                
                if len(self.prefetched_pages) >= self.max_prefetch_pages:
                    # Remove oldest prefetched page
                    oldest_url = next(iter(self.prefetched_pages))
                    old_page = self.prefetched_pages.pop(oldest_url)
                    await old_page.close()
                
                # Create new page for prefetching
                page = await self.context.new_page()
                
                # Prefetch with low priority
                try:
                    await page.goto(candidate.url, wait_until='domcontentloaded', timeout=10000)
                    self.prefetched_pages[candidate.url] = page
                    logger.debug(f"Prefetched: {candidate.url}")
                except Exception as e:
                    logger.warning(f"Failed to prefetch {candidate.url}: {e}")
                    await page.close()
                
                self.prefetch_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                await asyncio.sleep(1)
    
    async def prefetch_url(self, candidate: PrefetchCandidate):
        """Add URL to prefetch queue"""
        if candidate.url not in self.prefetched_pages:
            await self.prefetch_queue.put(candidate)
    
    def get_prefetched_page(self, url: str) -> Optional[Page]:
        """Get a prefetched page if available"""
        return self.prefetched_pages.pop(url, None)


class SmartWaiter:
    """Intelligent waiting mechanism that predicts page load completion"""
    
    def __init__(self, page: Page):
        self.page = page
        self.dom_mutation_stats = DOMMutationStats(0, 0.0, 0.0)
        self.network_requests: Dict[str, float] = {}  # request_id -> start_time
        self.critical_resources: Set[str] = set()
        self.javascript_executing = False
        self.last_activity_time = time.time()
        self.load_start_time = time.time()
        
        # Thresholds for stability detection
        self.dom_stable_threshold = 0.5  # seconds
        self.network_idle_threshold = 0.5  # seconds
        self.javascript_idle_threshold = 0.3  # seconds
        
        # Historical data for learning
        self.load_times: List[float] = []
        self.stability_patterns: Dict[str, float] = {}
        
    async def setup(self):
        """Setup event listeners for monitoring"""
        # Network monitoring
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)
        self.page.on("requestfailed", self._on_request_failed)
        
        # DOM mutation monitoring
        await self.page.evaluate("""
            () => {
                window._mutationCount = 0;
                window._lastMutationTime = Date.now();
                
                const observer = new MutationObserver((mutations) => {
                    window._mutationCount += mutations.length;
                    window._lastMutationTime = Date.now();
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    characterData: true
                });
            }
        """)
        
        # JavaScript execution monitoring
        await self.page.evaluate("""
            () => {
                window._jsExecutionCount = 0;
                window._lastJsExecution = Date.now();
                
                // Monitor setTimeout/setInterval
                const originalSetTimeout = window.setTimeout;
                window.setTimeout = function(fn, delay) {
                    window._jsExecutionCount++;
                    window._lastJsExecution = Date.now();
                    return originalSetTimeout(fn, delay);
                };
                
                const originalSetInterval = window.setInterval;
                window.setInterval = function(fn, delay) {
                    window._jsExecutionCount++;
                    window._lastJsExecution = Date.now();
                    return originalSetInterval(fn, delay);
                };
            }
        """)
    
    def _on_request(self, request: Request):
        """Handle new network request"""
        self.network_requests[request.url] = time.time()
        self.last_activity_time = time.time()
        
        # Identify critical resources
        if request.resource_type in ['document', 'script', 'xhr', 'fetch']:
            self.critical_resources.add(request.url)
    
    def _on_response(self, response: Response):
        """Handle network response"""
        url = response.url
        if url in self.network_requests:
            load_time = time.time() - self.network_requests[url]
            del self.network_requests[url]
            
            # Update pattern statistics
            if url in self.critical_resources:
                self.critical_resources.remove(url)
    
    def _on_request_failed(self, request: Request):
        """Handle failed network request"""
        url = request.url
        if url in self.network_requests:
            del self.network_requests[url]
        if url in self.critical_resources:
            self.critical_resources.remove(url)
    
    async def _get_dom_mutation_stats(self) -> DOMMutationStats:
        """Get current DOM mutation statistics"""
        try:
            stats = await self.page.evaluate("""
                () => {
                    const now = Date.now();
                    const timeDiff = (now - window._lastMutationTime) / 1000;
                    const rate = window._mutationCount / Math.max(timeDiff, 1);
                    
                    return {
                        count: window._mutationCount,
                        rate: rate,
                        lastMutation: window._lastMutationTime
                    };
                }
            """)
            
            return DOMMutationStats(
                mutation_count=stats['count'],
                mutation_rate=stats['rate'],
                last_mutation_time=stats['lastMutation'] / 1000
            )
        except:
            return DOMMutationStats(0, 0.0, time.time())
    
    async def _get_javascript_activity(self) -> Tuple[bool, float]:
        """Check JavaScript execution activity"""
        try:
            activity = await self.page.evaluate("""
                () => {
                    const now = Date.now();
                    const timeSinceLastJs = (now - window._lastJsExecution) / 1000;
                    return {
                        executing: timeSinceLastJs < 0.1,
                        lastExecution: window._lastJsExecution
                    };
                }
            """)
            return activity['executing'], activity['lastExecution'] / 1000
        except:
            return False, time.time()
    
    async def predict_page_state(self) -> PageState:
        """Predict current page state based on multiple signals"""
        # Get current metrics
        dom_stats = await self._get_dom_mutation_stats()
        js_executing, last_js_time = await self._get_javascript_activity()
        
        network_idle = len(self.network_requests) == 0
        critical_resources_loaded = len(self.critical_resources) == 0
        
        time_since_last_activity = time.time() - self.last_activity_time
        
        # Determine page state
        if time_since_last_activity < 0.1:
            return PageState.INITIAL
        
        if dom_stats.mutation_rate > 5:
            return PageState.DOM_LOADING
        
        if not critical_resources_loaded:
            return PageState.RESOURCES_LOADING
        
        if js_executing:
            return PageState.JAVASCRIPT_EXECUTING
        
        if (network_idle and 
            dom_stats.mutation_rate < 1 and 
            time.time() - dom_stats.last_mutation_time > self.dom_stable_threshold):
            return PageState.STABLE
        
        if (time.time() - last_js_time > self.javascript_idle_threshold and
            time.time() - dom_stats.last_mutation_time > self.dom_stable_threshold):
            return PageState.INTERACTIVE
        
        return PageState.DOM_READY
    
    async def wait_until_ready(self, timeout: float = 30.0, 
                              required_state: PageState = PageState.STABLE) -> bool:
        """Wait until page reaches required state"""
        start_time = time.time()
        last_state = PageState.INITIAL
        state_start_time = start_time
        
        while time.time() - start_time < timeout:
            current_state = await self.predict_page_state()
            
            # State transition tracking
            if current_state != last_state:
                state_duration = time.time() - state_start_time
                self.stability_patterns[f"{last_state.value}_duration"] = state_duration
                last_state = current_state
                state_start_time = time.time()
            
            # Check if we've reached required state
            if self._state_meets_requirement(current_state, required_state):
                total_time = time.time() - start_time
                self.load_times.append(total_time)
                
                # Learn from this load
                self._update_learning(current_state, total_time)
                
                logger.debug(f"Page ready in {total_time:.2f}s (state: {current_state.value})")
                return True
            
            # Adaptive sleep based on activity
            if current_state in [PageState.DOM_LOADING, PageState.RESOURCES_LOADING]:
                await asyncio.sleep(0.05)  # Check frequently during loading
            elif current_state == PageState.JAVASCRIPT_EXECUTING:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.2)
        
        logger.warning(f"Timeout waiting for page ready after {timeout}s")
        return False
    
    def _state_meets_requirement(self, current: PageState, required: PageState) -> bool:
        """Check if current state meets or exceeds required state"""
        state_order = [
            PageState.INITIAL,
            PageState.DOM_LOADING,
            PageState.DOM_READY,
            PageState.RESOURCES_LOADING,
            PageState.JAVASCRIPT_EXECUTING,
            PageState.STABLE,
            PageState.INTERACTIVE
        ]
        
        current_idx = state_order.index(current)
        required_idx = state_order.index(required)
        
        return current_idx >= required_idx
    
    def _update_learning(self, final_state: PageState, load_time: float):
        """Update learning model based on load experience"""
        # Simple exponential moving average for load times
        if len(self.load_times) > 10:
            # Keep only recent history
            self.load_times = self.load_times[-10:]
        
        # Update stability patterns
        pattern_key = f"{final_state.value}_time"
        if pattern_key in self.stability_patterns:
            # Exponential moving average
            self.stability_patterns[pattern_key] = (
                0.7 * self.stability_patterns[pattern_key] + 0.3 * load_time
            )
        else:
            self.stability_patterns[pattern_key] = load_time
    
    def get_estimated_load_time(self, url_pattern: str) -> float:
        """Get estimated load time for a URL pattern"""
        # Default to median of historical load times
        if self.load_times:
            return statistics.median(self.load_times)
        return 3.0  # Default 3 seconds
    
    async def cleanup(self):
        """Cleanup monitoring"""
        try:
            await self.page.evaluate("""
                () => {
                    // Restore original functions if needed
                    delete window._mutationCount;
                    delete window._lastMutationTime;
                    delete window._jsExecutionCount;
                    delete window._lastJsExecution;
                }
            """)
        except:
            pass


class SpeculativeExecutor:
    """Handles speculative execution of likely next actions"""
    
    def __init__(self, predictor: 'Predictor'):
        self.predictor = predictor
        self.speculative_tasks: Dict[str, asyncio.Task] = {}
        self.action_predictions: Dict[str, List[str]] = {}
        self.confidence_threshold = 0.8
        
    async def start_speculative_execution(self, current_url: str, 
                                        current_state: Dict) -> None:
        """Start speculative execution based on predictions"""
        # Get predictions for next URLs
        candidates = await self.predictor.prefetcher.predict_next_urls(
            current_url, current_state
        )
        
        for candidate in candidates:
            if candidate.confidence >= self.confidence_threshold:
                # Prefetch the page
                await self.predictor.prefetcher.prefetch_url(candidate)
                
                # Start preparing for likely actions
                task_key = f"speculative_{candidate.url}"
                if task_key not in self.speculative_tasks:
                    task = asyncio.create_task(
                        self._prepare_for_actions(candidate)
                    )
                    self.speculative_tasks[task_key] = task
    
    async def _prepare_for_actions(self, candidate: PrefetchCandidate):
        """Prepare for likely actions on a page"""
        try:
            # Wait a bit for prefetch to complete
            await asyncio.sleep(0.5)
            
            # Get prefetched page
            page = self.predictor.prefetcher.get_prefetched_page(candidate.url)
            if not page:
                return
            
            # Pre-execute common actions based on predictions
            for action in candidate.predicted_next_actions:
                if action == 'type':
                    await self._prepare_input_fields(page)
                elif action == 'click_search':
                    await self._prepare_search_buttons(page)
                elif action == 'scroll_to_reviews':
                    await self._prepare_scroll_position(page)
            
            # Return page to prefetch cache
            self.predictor.prefetcher.prefetched_pages[candidate.url] = page
            
        except Exception as e:
            logger.debug(f"Speculative preparation failed: {e}")
    
    async def _prepare_input_fields(self, page: Page):
        """Prepare input fields for quick typing"""
        try:
            # Focus first input field
            await page.evaluate("""
                () => {
                    const inputs = document.querySelectorAll('input[type="text"], input[type="search"], textarea');
                    if (inputs.length > 0) {
                        inputs[0].focus();
                    }
                }
            """)
        except:
            pass
    
    async def _prepare_search_buttons(self, page: Page):
        """Prepare search buttons for quick clicking"""
        try:
            # Ensure search button is visible and clickable
            await page.evaluate("""
                () => {
                    const buttons = document.querySelectorAll('button[type="submit"], input[type="submit"]');
                    buttons.forEach(btn => {
                        btn.style.transition = 'none'; // Disable transitions for instant click
                    });
                }
            """)
        except:
            pass
    
    async def _prepare_scroll_position(self, page: Page):
        """Prepare scroll position for quick navigation"""
        try:
            # Scroll to reviews section if it exists
            await page.evaluate("""
                () => {
                    const reviews = document.querySelector('#reviews, .reviews, [data-section="reviews"]');
                    if (reviews) {
                        reviews.scrollIntoView({behavior: 'instant', block: 'start'});
                    }
                }
            """)
        except:
            pass
    
    async def cancel_speculative_tasks(self):
        """Cancel all speculative tasks"""
        for task in self.speculative_tasks.values():
            task.cancel()
        
        # Wait for cancellation
        if self.speculative_tasks:
            await asyncio.gather(
                *self.speculative_tasks.values(),
                return_exceptions=True
            )
        
        self.speculative_tasks.clear()


class Predictor:
    """Main predictor class that coordinates all predictive features"""
    
    def __init__(self, context: BrowserContext):
        self.context = context
        self.prefetcher = PredictivePrefetcher(context)
        self.speculative_executor = SpeculativeExecutor(self)
        self.smart_waiters: Dict[str, SmartWaiter] = {}  # page_id -> waiter
        self.navigation_graph: Dict[str, Set[str]] = defaultdict(set)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.enable_prefetching = True
        self.enable_speculative_execution = True
        self.enable_smart_waiting = True
        
    async def initialize_page(self, page: Page) -> SmartWaiter:
        """Initialize predictor for a page"""
        waiter = SmartWaiter(page)
        await waiter.setup()
        
        page_id = id(page)
        self.smart_waiters[page_id] = waiter
        
        return waiter
    
    async def on_navigation(self, page: Page, from_url: str, to_url: str):
        """Handle page navigation"""
        # Record navigation for pattern learning
        self.prefetcher.record_navigation(from_url, to_url)
        
        # Update navigation graph
        self.navigation_graph[from_url].add(to_url)
        
        # Start prefetching for likely next pages
        if self.enable_prefetching:
            page_state = await self.get_page_state(page)
            await self.prefetcher.predict_next_urls(to_url, page_state)
        
        # Start speculative execution
        if self.enable_speculative_execution:
            page_state = await self.get_page_state(page)
            await self.speculative_executor.start_speculative_execution(
                to_url, page_state
            )
    
    async def get_page_state(self, page: Page) -> Dict:
        """Get current page state for predictions"""
        page_id = id(page)
        waiter = self.smart_waiters.get(page_id)
        
        if not waiter:
            return {}
        
        state = await waiter.predict_page_state()
        
        return {
            'page_state': state,
            'url': page.url,
            'timestamp': time.time()
        }
    
    async def wait_for_page_ready(self, page: Page, timeout: float = 30.0,
                                 required_state: PageState = PageState.STABLE) -> bool:
        """Smart wait for page to be ready"""
        if not self.enable_smart_waiting:
            # Fallback to standard Playwright waiting
            try:
                await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
                return True
            except:
                return False
        
        page_id = id(page)
        waiter = self.smart_waiters.get(page_id)
        
        if not waiter:
            waiter = await self.initialize_page(page)
        
        return await waiter.wait_until_ready(timeout, required_state)
    
    async def get_prefetched_page(self, url: str) -> Optional[Page]:
        """Get a prefetched page if available"""
        return self.prefetcher.get_prefetched_page(url)
    
    def record_performance(self, url: str, metric: str, value: float):
        """Record performance metric for learning"""
        key = f"{url}:{metric}"
        self.performance_metrics[key].append(value)
        
        # Keep only recent metrics
        if len(self.performance_metrics[key]) > 100:
            self.performance_metrics[key] = self.performance_metrics[key][-100:]
    
    def get_performance_stats(self, url: str, metric: str) -> Dict[str, float]:
        """Get performance statistics for a URL and metric"""
        key = f"{url}:{metric}"
        values = self.performance_metrics.get(key, [])
        
        if not values:
            return {}
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
    
    async def optimize_for_url(self, url: str) -> Dict[str, Any]:
        """Get optimization recommendations for a URL"""
        recommendations = {
            'prefetch_urls': [],
            'preload_resources': [],
            'predicted_load_time': 3.0,
            'suggested_wait_state': PageState.STABLE
        }
        
        # Analyze navigation patterns
        if url in self.navigation_graph:
            next_urls = list(self.navigation_graph[url])
            recommendations['prefetch_urls'] = next_urls[:3]  # Top 3 likely next URLs
        
        # Get performance predictions
        load_stats = self.get_performance_stats(url, 'load_time')
        if load_stats:
            recommendations['predicted_load_time'] = load_stats['median']
            
            # Adjust wait state based on historical performance
            if load_stats['median'] > 5.0:
                recommendations['suggested_wait_state'] = PageState.INTERACTIVE
            elif load_stats['median'] > 2.0:
                recommendations['suggested_wait_state'] = PageState.STABLE
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup all resources"""
        # Stop prefetcher
        await self.prefetcher.stop()
        
        # Cancel speculative tasks
        await self.speculative_executor.cancel_speculative_tasks()
        
        # Cleanup smart waiters
        for waiter in self.smart_waiters.values():
            await waiter.cleanup()
        
        self.smart_waiters.clear()


# Integration with existing Page class
class PredictivePage:
    """Wrapper that adds predictive capabilities to Playwright Page"""
    
    def __init__(self, page: Page, predictor: Predictor):
        self.page = page
        self.predictor = predictor
        self.smart_waiter: Optional[SmartWaiter] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize predictive features"""
        if not self._initialized:
            self.smart_waiter = await self.predictor.initialize_page(self.page)
            self._initialized = True
    
    async def goto(self, url: str, **kwargs) -> Optional[Response]:
        """Navigate with predictive features"""
        from_url = self.page.url
        
        # Use smart waiting if enabled
        if self.predictor.enable_smart_waiting:
            # Get optimization recommendations
            optimizations = await self.predictor.optimize_for_url(url)
            
            # Set timeout based on prediction
            timeout = kwargs.get('timeout', 30000)
            predicted_load = optimizations['predicted_load_time'] * 1000
            kwargs['timeout'] = max(timeout, predicted_load * 1.5)
        
        # Perform navigation
        response = await self.page.goto(url, **kwargs)
        
        # Record navigation for learning
        await self.predictor.on_navigation(self.page, from_url, url)
        
        # Record performance
        if response:
            load_time = (time.time() - self.predictor.smart_waiters[id(self.page)].load_start_time)
            self.predictor.record_performance(url, 'load_time', load_time)
        
        return response
    
    async def wait_for_load_state(self, state: str = 'load', **kwargs):
        """Smart wait for load state"""
        if self.predictor.enable_smart_waiting and self.smart_waiter:
            # Map Playwright states to our states
            state_map = {
                'load': PageState.DOM_READY,
                'domcontentloaded': PageState.DOM_READY,
                'networkidle': PageState.STABLE
            }
            
            required_state = state_map.get(state, PageState.STABLE)
            timeout = kwargs.get('timeout', 30000) / 1000  # Convert to seconds
            
            success = await self.smart_waiter.wait_until_ready(
                timeout=timeout,
                required_state=required_state
            )
            
            if not success:
                # Fallback to standard waiting
                await self.page.wait_for_load_state(state, **kwargs)
        else:
            await self.page.wait_for_load_state(state, **kwargs)
    
    async def click(self, selector: str, **kwargs):
        """Click with speculative preparation"""
        # Record click for pattern learning
        current_url = self.page.url
        
        # Perform click
        await self.page.click(selector, **kwargs)
        
        # Update navigation patterns if this was a link
        try:
            href = await self.page.evaluate(f"""
                () => {{
                    const el = document.querySelector('{selector}');
                    return el ? el.href : null;
                }}
            """)
            
            if href:
                await self.predictor.on_navigation(self.page, current_url, href)
        except:
            pass
    
    def __getattr__(self, name):
        """Delegate other methods to underlying page"""
        return getattr(self.page, name)


# Factory function for easy integration
async def create_predictive_page(page: Page, context: BrowserContext) -> PredictivePage:
    """Create a predictive page wrapper"""
    predictor = Predictor(context)
    await predictor.prefetcher.start()
    
    predictive_page = PredictivePage(page, predictor)
    await predictive_page.initialize()
    
    return predictive_page


# Example usage in existing codebase
"""
# In nexus/actor/page.py, modify the Page class:

from nexus.browser.predictor import Predictor, PredictivePage

class Page:
    def __init__(self, page: PlaywrightPage, context: BrowserContext):
        self.page = page
        self.context = context
        self.predictor = Predictor(context)
        self.predictive_page = None
        
    async def initialize(self):
        await self.predictor.prefetcher.start()
        self.predictive_page = PredictivePage(self.page, self.predictor)
        await self.predictive_page.initialize()
        
    async def goto(self, url: str, **kwargs):
        if self.predictive_page:
            return await self.predictive_page.goto(url, **kwargs)
        return await self.page.goto(url, **kwargs)
        
    async def wait_for_load_state(self, state: str = 'load', **kwargs):
        if self.predictive_page:
            await self.predictive_page.wait_for_load_state(state, **kwargs)
        else:
            await self.page.wait_for_load_state(state, **kwargs)
"""