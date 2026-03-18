"""Page class for page-level operations."""

from typing import TYPE_CHECKING, TypeVar, Dict, List, Optional, Tuple, Any, Set, Callable
import asyncio
import json
import re
import time
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
from collections import defaultdict, deque
from statistics import mean, stdev
import hashlib

from pydantic import BaseModel

from nexus import logger
from nexus.actor.utils import get_key_info
from nexus.dom.serializer.serializer import DOMTreeSerializer
from nexus.dom.service import DomService
from nexus.llm.messages import SystemMessage, UserMessage

T = TypeVar('T', bound=BaseModel)

if TYPE_CHECKING:
	from cdp_use.cdp.dom.commands import (
		DescribeNodeParameters,
		QuerySelectorAllParameters,
	)
	from cdp_use.cdp.emulation.commands import SetDeviceMetricsOverrideParameters
	from cdp_use.cdp.input.commands import (
		DispatchKeyEventParameters,
	)
	from cdp_use.cdp.page.commands import CaptureScreenshotParameters, NavigateParameters, NavigateToHistoryEntryParameters
	from cdp_use.cdp.runtime.commands import EvaluateParameters
	from cdp_use.cdp.target.commands import (
		AttachToTargetParameters,
		GetTargetInfoParameters,
	)
	from cdp_use.cdp.target.types import TargetInfo

	from nexus.browser.session import BrowserSession
	from nexus.llm.base import BaseChatModel

	from .element import Element
	from .mouse import Mouse


class LocatorStrategy(Enum):
	"""Strategies for locating elements."""
	ORIGINAL_SELECTOR = "original_selector"
	CSS_SELECTOR = "css_selector"
	XPATH = "xpath"
	TEXT_CONTENT = "text_content"
	AI_VISION = "ai_vision"
	VISUAL_SIMILARITY = "visual_similarity"
	ACCESSIBILITY_TREE = "accessibility_tree"


@dataclass
class LocatorResult:
	"""Result from a locator strategy."""
	strategy: LocatorStrategy
	backend_node_id: Optional[int] = None
	confidence: float = 1.0
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}


@dataclass
class LoadMetrics:
	"""Metrics for predicting page load completion."""
	network_requests: int = 0
	completed_requests: int = 0
	failed_requests: int = 0
	dom_mutations: int = 0
	js_execution_time: float = 0.0
	last_activity: float = field(default_factory=time.time)
	load_start: float = field(default_factory=time.time)
	
	@property
	def network_idle(self) -> bool:
		return self.completed_requests + self.failed_requests >= self.network_requests
	
	@property
	def dom_stable(self) -> bool:
		return time.time() - self.last_activity > 0.5 and self.dom_mutations < 5


class SmartWaiter:
	"""Intelligent waiting that predicts page load completion."""
	
	def __init__(self, page: 'Page'):
		self.page = page
		self.metrics = LoadMetrics()
		self._monitoring = False
		self._mutation_observer_id = None
		self._network_listeners = []
		self._js_execution_tracker = []
		self._prediction_model = self._init_prediction_model()
		self._prefetch_queue = asyncio.Queue()
		self._prefetch_task = None
		
	def _init_prediction_model(self) -> Dict[str, Any]:
		"""Initialize prediction model based on historical patterns."""
		return {
			'avg_load_time': 2.0,
			'network_patterns': defaultdict(list),
			'dom_stability_threshold': 0.3,
			'js_execution_patterns': defaultdict(float),
			'page_type_patterns': defaultdict(dict)
		}
	
	async def start_monitoring(self):
		"""Start monitoring page activity for intelligent waiting."""
		if self._monitoring:
			return
			
		self._monitoring = True
		self.metrics = LoadMetrics()
		
		# Set up network monitoring
		await self._setup_network_monitoring()
		
		# Set up DOM mutation monitoring
		await self._setup_mutation_monitoring()
		
		# Set up JS execution monitoring
		await self._setup_js_monitoring()
		
		# Start prefetch worker
		self._prefetch_task = asyncio.create_task(self._prefetch_worker())
		
		logger.debug("Smart waiter monitoring started")
	
	async def stop_monitoring(self):
		"""Stop monitoring page activity."""
		self._monitoring = False
		
		# Clean up listeners
		for listener in self._network_listeners:
			try:
				await self.page._client.send.Network.disable()
			except:
				pass
				
		if self._mutation_observer_id:
			try:
				await self.page.evaluate(f"""
					if (window._mutationObserver) {{
						window._mutationObserver.disconnect();
						delete window._mutationObserver;
					}}
				""")
			except:
				pass
				
		if self._prefetch_task:
			self._prefetch_task.cancel()
			
		logger.debug("Smart waiter monitoring stopped")
	
	async def _setup_network_monitoring(self):
		"""Set up network request monitoring."""
		try:
			session_id = await self.page.session_id
			
			# Enable network domain
			await self.page._client.send.Network.enable({}, session_id=session_id)
			
			# Track request counts
			self.metrics.network_requests = 0
			self.metrics.completed_requests = 0
			self.metrics.failed_requests = 0
			
			# Listen for network events
			async def on_request_will_be_sent(event):
				self.metrics.network_requests += 1
				self.metrics.last_activity = time.time()
				
			async def on_loading_finished(event):
				self.metrics.completed_requests += 1
				self.metrics.last_activity = time.time()
				
			async def on_loading_failed(event):
				self.metrics.failed_requests += 1
				self.metrics.last_activity = time.time()
			
			# Add listeners (simplified - in real implementation would use CDP event listeners)
			self._network_listeners.extend([
				on_request_will_be_sent,
				on_loading_finished,
				on_loading_failed
			])
			
		except Exception as e:
			logger.debug(f"Failed to setup network monitoring: {e}")
	
	async def _setup_mutation_monitoring(self):
		"""Set up DOM mutation monitoring."""
		try:
			js_code = """
			(() => {
				if (window._mutationObserver) {
					window._mutationObserver.disconnect();
				}
				
				let mutationCount = 0;
				const observer = new MutationObserver((mutations) => {
					mutationCount += mutations.length;
					window._mutationCount = mutationCount;
					window._lastMutationTime = Date.now();
					
					// Notify Python side
					if (window._mutationCallback) {
						window._mutationCallback(mutations.length);
					}
				});
				
				observer.observe(document.body, {
					childList: true,
					subtree: true,
					attributes: true,
					characterData: true
				});
				
				window._mutationObserver = observer;
				window._mutationCount = 0;
				window._lastMutationTime = Date.now();
				
				return true;
			})()
			"""
			
			await self.page.evaluate(js_code)
			
			# Set up callback to track mutations
			def mutation_callback(count):
				self.metrics.dom_mutations += count
				self.metrics.last_activity = time.time()
				
			# In real implementation, would set up proper callback mechanism
			
		except Exception as e:
			logger.debug(f"Failed to setup mutation monitoring: {e}")
	
	async def _setup_js_monitoring(self):
		"""Set up JavaScript execution monitoring."""
		try:
			js_code = """
			(() => {
				// Override setTimeout and setInterval to track JS execution
				const originalSetTimeout = window.setTimeout;
				const originalSetInterval = window.setInterval;
				
				window.setTimeout = function(callback, delay, ...args) {
					const startTime = performance.now();
					const wrappedCallback = function() {
						const endTime = performance.now();
						window._jsExecutionTime = (window._jsExecutionTime || 0) + (endTime - startTime);
						return callback.apply(this, args);
					};
					return originalSetTimeout(wrappedCallback, delay);
				};
				
				window.setInterval = function(callback, delay, ...args) {
					const startTime = performance.now();
					const wrappedCallback = function() {
						const endTime = performance.now();
						window._jsExecutionTime = (window._jsExecutionTime || 0) + (endTime - startTime);
						return callback.apply(this, args);
					};
					return originalSetInterval(wrappedCallback, delay);
				};
				
				window._jsExecutionTime = 0;
				return true;
			})()
			"""
			
			await self.page.evaluate(js_code)
			
		except Exception as e:
			logger.debug(f"Failed to setup JS monitoring: {e}")
	
	async def _prefetch_worker(self):
		"""Worker for prefetching likely next resources."""
		while self._monitoring:
			try:
				# Wait for prefetch requests
				url = await asyncio.wait_for(self._prefetch_queue.get(), timeout=1.0)
				
				# Prefetch the resource
				await self.page.evaluate(f"""
					const link = document.createElement('link');
					link.rel = 'prefetch';
					link.href = '{url}';
					document.head.appendChild(link);
				""")
				
				self._prefetch_queue.task_done()
				
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				logger.debug(f"Prefetch worker error: {e}")
	
	async def predict_load_completion(self) -> float:
		"""Predict when page load will complete."""
		if self.metrics.network_idle and self.metrics.dom_stable:
			return 0.0
		
		# Use historical patterns to predict
		elapsed = time.time() - self.metrics.load_start
		
		# Calculate remaining time based on network and DOM activity
		network_remaining = max(0, self.metrics.network_requests - 
							   self.metrics.completed_requests - 
							   self.metrics.failed_requests) * 0.1
		
		dom_remaining = max(0, 5 - self.metrics.dom_mutations) * 0.05
		
		js_remaining = max(0, 1.0 - self.metrics.js_execution_time) * 0.2
		
		predicted_remaining = network_remaining + dom_remaining + js_remaining
		
		# Apply historical patterns
		page_hash = self._get_page_hash()
		if page_hash in self._prediction_model['page_type_patterns']:
			pattern = self._prediction_model['page_type_patterns'][page_hash]
			if 'avg_load_time' in pattern:
				historical_avg = pattern['avg_load_time']
				predicted_remaining = (predicted_remaining + historical_avg) / 2
		
		return max(0.1, predicted_remaining)
	
	def _get_page_hash(self) -> str:
		"""Generate a hash for the current page for pattern matching."""
		url = self.page.url or ""
		title = self.page.title or ""
		content = f"{url}:{title}"
		return hashlib.md5(content.encode()).hexdigest()
	
	async def wait_for_load(self, timeout: float = 30.0) -> bool:
		"""Intelligently wait for page load to complete."""
		start_time = time.time()
		
		while time.time() - start_time < timeout:
			predicted_remaining = await self.predict_load_completion()
			
			if predicted_remaining <= 0.1:
				# Update historical patterns
				page_hash = self._get_page_hash()
				elapsed = time.time() - self.metrics.load_start
				
				if page_hash not in self._prediction_model['page_type_patterns']:
					self._prediction_model['page_type_patterns'][page_hash] = {
						'count': 0,
						'avg_load_time': elapsed
					}
				else:
					pattern = self._prediction_model['page_type_patterns'][page_hash]
					pattern['count'] += 1
					pattern['avg_load_time'] = (
						(pattern['avg_load_time'] * (pattern['count'] - 1) + elapsed) / 
						pattern['count']
					)
				
				return True
			
			# Wait for predicted remaining time or a minimum interval
			wait_time = min(predicted_remaining, 0.5)
			await asyncio.sleep(wait_time)
		
		return False


class LLMOptimizedDOMSerializer:
	"""DOM serializer optimized for LLM consumption with hierarchical prioritization."""
	
	# Element priority levels
	PRIORITY_CRITICAL = 1  # Interactive elements (buttons, inputs, links)
	PRIORITY_VISIBLE = 2   # Visible content
	PRIORITY_STRUCTURAL = 3  # Structural context
	PRIORITY_IGNORED = 4   # Decorative elements
	
	# Interactive element tags
	INTERACTIVE_TAGS = {
		'button', 'input', 'select', 'textarea', 'a', 'area',
		'details', 'embed', 'iframe', 'label', 'object',
		'summary', 'video', 'audio', 'img', 'svg'
	}
	
	# Structural tags
	STRUCTURAL_TAGS = {
		'nav', 'main', 'header', 'footer', 'aside', 'section',
		'article', 'form', 'table', 'ul', 'ol', 'dl',
		'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
	}
	
	# Decorative tags to ignore
	DECORATIVE_TAGS = {
		'style', 'script', 'noscript', 'template', 'link',
		'meta', 'base', 'br', 'hr', 'wbr'
	}
	
	# Attributes to preserve
	PRESERVE_ATTRIBUTES = {
		'id', 'class', 'name', 'type', 'value', 'placeholder',
		'href', 'src', 'alt', 'title', 'role', 'aria-label',
		'aria-describedby', 'aria-hidden', 'disabled', 'readonly',
		'required', 'checked', 'selected', 'tabindex', 'onclick',
		'onchange', 'onsubmit', 'action', 'method'
	}
	
	def __init__(self, task_context: Optional[str] = None, 
				 max_tokens: int = 4000,
				 compression_ratio: float = 0.3):
		"""
		Initialize the LLM-optimized DOM serializer.
		
		Args:
			task_context: Current task context for relevance scoring
			max_tokens: Maximum tokens to generate
			compression_ratio: Target compression ratio (0.3 = 70% reduction)
		"""
		self.task_context = task_context
		self.max_tokens = max_tokens
		self.compression_ratio = compression_ratio
		self._relevance_scores = {}
		self._element_cache = {}
		
	def serialize_dom_tree(self, dom_tree: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Serialize DOM tree with LLM optimization.
		
		Args:
			dom_tree: Original DOM tree from DOMTreeSerializer
			
		Returns:
			Optimized DOM tree with hierarchical prioritization
		"""
		if not dom_tree:
			return {}
		
		# Phase 1: Score all elements
		self._score_elements(dom_tree)
		
		# Phase 2: Filter and prioritize
		optimized_tree = self._filter_and_prioritize(dom_tree)
		
		# Phase 3: Compress representation
		compressed_tree = self._compress_representation(optimized_tree)
		
		# Phase 4: Ensure token limit
		final_tree = self._enforce_token_limit(compressed_tree)
		
		return final_tree
	
	def _score_elements(self, node: Dict[str, Any], depth: int = 0, 
					   parent_score: float = 0.0):
		"""Recursively score all elements in the DOM tree."""
		node_id = self._get_node_id(node)
		
		# Calculate base score based on element type
		base_score = self._calculate_base_score(node)
		
		# Adjust score based on visibility
		visibility_score = self._calculate_visibility_score(node)
		
		# Adjust score based on interactivity
		interactivity_score = self._calculate_interactivity_score(node)
		
		# Adjust score based on task relevance
		relevance_score = self._calculate_relevance_score(node)
		
		# Adjust score based on depth (prefer shallower elements)
		depth_score = max(0, 10 - depth) / 10
		
		# Combine scores with weights
		total_score = (
			base_score * 0.3 +
			visibility_score * 0.2 +
			interactivity_score * 0.3 +
			relevance_score * 0.15 +
			depth_score * 0.05
		)
		
		# Boost score if parent has high score (contextual importance)
		if parent_score > 7:
			total_score *= 1.2
		
		self._relevance_scores[node_id] = total_score
		
		# Recursively score children
		children = node.get('children', [])
		for child in children:
			self._score_elements(child, depth + 1, total_score)
	
	def _calculate_base_score(self, node: Dict[str, Any]) -> float:
		"""Calculate base score based on element type."""
		tag_name = node.get('tagName', '').lower()
		
		if tag_name in self.INTERACTIVE_TAGS:
			return 10.0
		elif tag_name in self.STRUCTURAL_TAGS:
			return 7.0
		elif tag_name in self.DECORATIVE_TAGS:
			return 1.0
		elif tag_name == 'div' or tag_name == 'span':
			# Check if contains meaningful content
			text_content = self._get_text_content(node)
			if text_content and len(text_content.strip()) > 0:
				return 5.0
			return 3.0
		else:
			return 4.0
	
	def _calculate_visibility_score(self, node: Dict[str, Any]) -> float:
		"""Calculate visibility score (0-10)."""
		attributes = node.get('attributes', {})
		
		# Check for hidden attributes
		if attributes.get('hidden') == 'true' or attributes.get('aria-hidden') == 'true':
			return 0.0
		
		# Check for display:none in style (simplified check)
		style = attributes.get('style', '')
		if 'display: none' in style or 'display:none' in style:
			return 0.0
		if 'visibility: hidden' in style or 'visibility:hidden' in style:
			return 2.0
		
		# Check for zero dimensions
		if attributes.get('width') == '0' or attributes.get('height') == '0':
			return 1.0
		
		# Default to visible
		return 8.0
	
	def _calculate_interactivity_score(self, node: Dict[str, Any]) -> float:
		"""Calculate interactivity score (0-10)."""
		tag_name = node.get('tagName', '').lower()
		attributes = node.get('attributes', {})
		
		# Check if element is interactive
		if tag_name in self.INTERACTIVE_TAGS:
			# Check if disabled
			if attributes.get('disabled') == 'true' or attributes.get('aria-disabled') == 'true':
				return 3.0
			return 10.0
		
		# Check for event handlers
		for attr in attributes:
			if attr.startswith('on'):
				return 7.0
		
		# Check for role attribute
		role = attributes.get('role', '')
		if role in ['button', 'link', 'checkbox', 'radio', 'textbox', 'combobox']:
			return 9.0
		
		return 2.0
	
	def _calculate_relevance_score(self, node: Dict[str, Any]) -> float:
		"""Calculate relevance score based on task context."""
		if not self.task_context:
			return 5.0
		
		node_id = self._get_node_id(node)
		
		# Check cache
		if node_id in self._element_cache:
			return self._element_cache[node_id]
		
		# Get text content and attributes
		text_content = self._get_text_content(node).lower()
		attributes = node.get('attributes', {})
		attribute_values = ' '.join(str(v) for v in attributes.values()).lower()
		
		# Simple keyword matching (in production, use embedding similarity)
		task_words = set(self.task_context.lower().split())
		node_content = f"{text_content} {attribute_values}".lower()
		
		if not task_words:
			return 5.0
		
		# Calculate word overlap
		matches = sum(1 for word in task_words if word in node_content)
		match_ratio = matches / len(task_words) if task_words else 0
		
		# Convert to score (0-10)
		score = min(10.0, match_ratio * 15)
		
		self._element_cache[node_id] = score
		return score
	
	def _filter_and_prioritize(self, dom_tree: Dict[str, Any]) -> Dict[str, Any]:
		"""Filter elements based on priority and relevance scores."""
		# Collect all nodes with their scores
		nodes_with_scores = []
		self._collect_nodes_with_scores(dom_tree, nodes_with_scores)
		
		# Sort by score (descending)
		nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
		
		# Select top elements based on compression ratio
		total_nodes = len(nodes_with_scores)
		target_nodes = max(1, int(total_nodes * self.compression_ratio))
		
		selected_nodes = set()
		for node_id, score in nodes_with_scores[:target_nodes]:
			selected_nodes.add(node_id)
		
		# Build optimized tree with selected nodes and their context
		optimized_tree = self._build_optimized_tree(dom_tree, selected_nodes)
		
		return optimized_tree
	
	def _collect_nodes_with_scores(self, node: Dict[str, Any], 
								  result: List[Tuple[str, float]]):
		"""Collect all nodes with their relevance scores."""
		node_id = self._get_node_id(node)
		score = self._relevance_scores.get(node_id, 0.0)
		result.append((node_id, score))
		
		for child in node.get('children', []):
			self._collect_nodes_with_scores(child, result)
	
	def _build_optimized_tree(self, node: Dict[str, Any], 
							 selected_nodes: Set[str],
							 depth: int = 0) -> Optional[Dict[str, Any]]:
		"""Build optimized tree containing only selected nodes and their context."""
		node_id = self._get_node_id(node)
		
		# Check if this node or any descendant is selected
		if not self._has_selected_descendant(node, selected_nodes):
			return None
		
		# Create optimized node
		optimized_node = {
			'tagName': node.get('tagName', ''),
			'nodeId': node.get('nodeId'),
			'backendNodeId': node.get('backendNodeId'),
			'attributes': self._filter_attributes(node.get('attributes', {})),
			'children': []
		}
		
		# Add text content if present and meaningful
		text_content = self._get_text_content(node)
		if text_content and len(text_content.strip()) > 0:
			# Truncate long text content
			if len(text_content) > 200:
				text_content = text_content[:197] + "..."
			optimized_node['textContent'] = text_content
		
		# Add priority level
		optimized_node['priority'] = self._get_priority_level(node)
		
		# Add relevance score
		optimized_node['relevanceScore'] = self._relevance_scores.get(node_id, 0.0)
		
		# Process children
		for child in node.get('children', []):
			optimized_child = self._build_optimized_tree(child, selected_nodes, depth + 1)
			if optimized_child:
				optimized_node['children'].append(optimized_child)
		
		# If no children were added and node isn't selected, return None
		if not optimized_node['children'] and node_id not in selected_nodes:
			return None
		
		return optimized_node
	
	def _has_selected_descendant(self, node: Dict[str, Any], 
								selected_nodes: Set[str]) -> bool:
		"""Check if node or any descendant is in selected nodes."""
		node_id = self._get_node_id(node)
		if node_id in selected_nodes:
			return True
		
		for child in node.get('children', []):
			if self._has_selected_descendant(child, selected_nodes):
				return True
		
		return False
	
	def _filter_attributes(self, attributes: Dict[str, str]) -> Dict[str, str]:
		"""Filter attributes to preserve only important ones."""
		filtered = {}
		for key, value in attributes.items():
			if key in self.PRESERVE_ATTRIBUTES:
				# Truncate long attribute values
				if isinstance(value, str) and len(value) > 100:
					value = value[:97] + "..."
				filtered[key] = value
		return filtered
	
	def _get_priority_level(self, node: Dict[str, Any]) -> int:
		"""Get priority level for node."""
		tag_name = node.get('tagName', '').lower()
		
		if tag_name in self.INTERACTIVE_TAGS:
			return self.PRIORITY_CRITICAL
		elif tag_name in self.STRUCTURAL_TAGS:
			return self.PRIORITY_STRUCTURAL
		elif tag_name in self.DECORATIVE_TAGS:
			return self.PRIORITY_IGNORED
		else:
			# Check if has visible text
			text_content = self._get_text_content(node)
			if text_content and len(text_content.strip()) > 0:
				return self.PRIORITY_VISIBLE
			return self.PRIORITY_STRUCTURAL
	
	def _compress_representation(self, tree: Dict[str, Any]) -> Dict[str, Any]:
		"""Compress the representation for token efficiency."""
		if not tree:
			return {}
		
		compressed = {
			'tag': tree['tagName'].lower(),
			'id': tree['attributes'].get('id'),
			'class': tree['attributes'].get('class'),
			'role': tree['attributes'].get('role'),
			'p': tree['priority'],  # priority
			'r': round(tree['relevanceScore'], 1),  # relevance
			'c': []  # children
		}
		
		# Add important attributes
		important_attrs = ['href', 'src', 'type', 'name', 'value', 'placeholder']
		for attr in important_attrs:
			if attr in tree['attributes']:
				compressed[attr] = tree['attributes'][attr]
		
		# Add text content if present
		if 'textContent' in tree:
			compressed['t'] = tree['textContent']
		
		# Add node IDs for reference
		if 'nodeId' in tree:
			compressed['nid'] = tree['nodeId']
		if 'backendNodeId' in tree:
			compressed['bid'] = tree['backendNodeId']
		
		# Process children
		for child in tree.get('children', []):
			compressed_child = self._compress_representation(child)
			if compressed_child:
				compressed['c'].append(compressed_child)
		
		# Remove empty children array
		if not compressed['c']:
			del compressed['c']
		
		return compressed
	
	def _enforce_token_limit(self, tree: Dict[str, Any]) -> Dict[str, Any]:
		"""Ensure the serialized tree stays within token limit."""
		# Estimate token count (simplified - in production use tiktoken)
		serialized = json.dumps(tree, separators=(',', ':'))
		estimated_tokens = len(serialized) // 4  # Rough estimate
		
		if estimated_tokens <= self.max_tokens:
			return tree
		
		# If over limit, remove lowest priority elements
		logger.debug(f"DOM tree exceeds token limit ({estimated_tokens} > {self.max_tokens}), pruning...")
		
		# Flatten tree and sort by priority and relevance
		nodes = []
		self._flatten_tree(tree, nodes)
		
		# Sort by priority (ascending) and relevance (descending)
		nodes.sort(key=lambda x: (x['p'], -x.get('r', 0)))
		
		# Remove nodes until under limit
		while estimated_tokens > self.max_tokens and nodes:
			removed = nodes.pop(0)  # Remove lowest priority
			estimated_tokens -= len(json.dumps(removed, separators=(',', ':'))) // 4
		
		# Rebuild tree from remaining nodes
		return self._rebuild_tree_from_nodes(nodes)
	
	def _flatten_tree(self, node: Dict[str, Any], result: List[Dict[str, Any]]):
		"""Flatten tree into list of nodes."""
		result.append(node)
		for child in node.get('c', []):
			self._flatten_tree(child, result)
	
	def _rebuild_tree_from_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Rebuild tree structure from flat list of nodes."""
		if not nodes:
			return {}
		
		# Group nodes by parent-child relationships
		node_map = {node.get('nid'): node for node in nodes if 'nid' in node}
		
		# Build tree structure
		root = None
		for node in nodes:
			if 'nid' not in node:
				continue
			
			# Try to find parent
			parent_found = False
			for potential_parent in nodes:
				if 'c' in potential_parent:
					for child in potential_parent['c']:
						if child.get('nid') == node['nid']:
							parent_found = True
							break
			
			if not parent_found and root is None:
				root = node
		
		return root or nodes[0] if nodes else {}
	
	def _get_node_id(self, node: Dict[str, Any]) -> str:
		"""Generate unique ID for node."""
		node_id = node.get('nodeId')
		backend_node_id = node.get('backendNodeId')
		
		if node_id:
			return f"n{node_id}"
		elif backend_node_id:
			return f"b{backend_node_id}"
		else:
			# Generate from content
			content = json.dumps(node, sort_keys=True)
			return f"h{hashlib.md5(content.encode()).hexdigest()[:8]}"
	
	def _get_text_content(self, node: Dict[str, Any]) -> str:
		"""Extract text content from node and its children."""
		text_parts = []
		
		# Get direct text content
		if 'textContent' in node:
			text_parts.append(node['textContent'])
		
		# Get text from children
		for child in node.get('children', []):
			child_text = self._get_text_content(child)
			if child_text:
				text_parts.append(child_text)
		
		return ' '.join(text_parts).strip()


class Page:
	"""Represents a browser page with advanced automation capabilities."""
	
	def __init__(self, browser_session: 'BrowserSession', target_id: str):
		self.browser_session = browser_session
		self.target_id = target_id
		self._client = browser_session.client
		self._dom_service = DomService(self)
		self._smart_waiter = SmartWaiter(self)
		self._element_cache = {}
		self._screenshot_cache = {}
		self._navigation_history = deque(maxlen=50)
		self._current_url = None
		self._page_load_time = None
		self._dom_tree_cache = {}
		self._dom_tree_cache_time = 0
		self._llm_serializer = None
		self._task_context = None
		
		# Performance monitoring
		self._performance_metrics = {
			'dom_serialization_time': [],
			'llm_token_usage': [],
			'element_location_time': []
		}
	
	async def set_task_context(self, task_context: str):
		"""Set the current task context for LLM optimization."""
		self._task_context = task_context
		self._llm_serializer = LLMOptimizedDOMSerializer(
			task_context=task_context,
			max_tokens=4000,
			compression_ratio=0.3
		)
	
	async def get_dom_tree(self, use_cache: bool = True, 
						  use_llm_optimized: bool = False) -> Dict[str, Any]:
		"""
		Get DOM tree representation.
		
		Args:
			use_cache: Whether to use cached DOM tree
			use_llm_optimized: Whether to use LLM-optimized serialization
			
		Returns:
			DOM tree representation
		"""
		start_time = time.time()
		
		# Check cache
		cache_key = f"dom_tree_{use_llm_optimized}"
		if use_cache and cache_key in self._dom_tree_cache:
			cache_age = time.time() - self._dom_tree_cache_time
			if cache_age < 2.0:  # Cache for 2 seconds
				logger.debug(f"Using cached DOM tree (age: {cache_age:.2f}s)")
				return self._dom_tree_cache[cache_key]
		
		try:
			# Get raw DOM tree
			raw_tree = await self._dom_service.get_dom_tree()
			
			if use_llm_optimized and self._llm_serializer:
				# Use LLM-optimized serialization
				optimized_tree = self._llm_serializer.serialize_dom_tree(raw_tree)
				
				# Track token usage
				serialized = json.dumps(optimized_tree, separators=(',', ':'))
				token_estimate = len(serialized) // 4
				self._performance_metrics['llm_token_usage'].append(token_estimate)
				
				logger.debug(f"LLM-optimized DOM tree: {token_estimate} estimated tokens")
				
				result = optimized_tree
			else:
				# Use standard serialization
				result = raw_tree
			
			# Update cache
			self._dom_tree_cache[cache_key] = result
			self._dom_tree_cache_time = time.time()
			
			# Track performance
			elapsed = time.time() - start_time
			self._performance_metrics['dom_serialization_time'].append(elapsed)
			
			logger.debug(f"DOM tree serialization took {elapsed:.3f}s")
			
			return result
			
		except Exception as e:
			logger.error(f"Failed to get DOM tree: {e}")
			return {}
	
	async def get_simplified_dom_tree(self, include_attributes: Optional[List[str]] = None,
									 max_depth: int = 10,
									 task_description: Optional[str] = None) -> Dict[str, Any]:
		"""
		Get simplified DOM tree optimized for LLM consumption.
		
		Args:
			include_attributes: Attributes to include
			max_depth: Maximum tree depth
			task_description: Current task description for relevance scoring
			
		Returns:
			Simplified DOM tree
		"""
		# Update task context if provided
		if task_description:
			await self.set_task_context(task_description)
		
		# Use LLM-optimized serialization
		return await self.get_dom_tree(use_cache=True, use_llm_optimized=True)
	
	async def get_critical_elements(self) -> List[Dict[str, Any]]:
		"""Get only critical interactive elements for quick LLM processing."""
		dom_tree = await self.get_dom_tree(use_llm_optimized=True)
		
		critical_elements = []
		self._extract_critical_elements(dom_tree, critical_elements)
		
		return critical_elements
	
	def _extract_critical_elements(self, node: Dict[str, Any], 
								  result: List[Dict[str, Any]]):
		"""Extract critical interactive elements from DOM tree."""
		if not node:
			return
		
		# Check if this is a critical element
		priority = node.get('p', 4)  # Default to ignored
		if priority == LLMOptimizedDOMSerializer.PRIORITY_CRITICAL:
			# Create simplified representation
			element = {
				'tag': node.get('tag', ''),
				'id': node.get('id'),
				'class': node.get('class'),
				'text': node.get('t', ''),
				'role': node.get('role'),
				'nodeId': node.get('nid'),
				'backendNodeId': node.get('bid')
			}
			
			# Add important attributes
			for attr in ['href', 'src', 'type', 'name', 'value', 'placeholder']:
				if attr in node:
					element[attr] = node[attr]
			
			result.append(element)
		
		# Process children
		for child in node.get('c', []):
			self._extract_critical_elements(child, result)
	
	async def get_performance_metrics(self) -> Dict[str, Any]:
		"""Get performance metrics for DOM serialization."""
		metrics = {}
		
		for key, values in self._performance_metrics.items():
			if values:
				metrics[key] = {
					'avg': mean(values),
					'std': stdev(values) if len(values) > 1 else 0,
					'min': min(values),
					'max': max(values),
					'count': len(values)
				}
		
		# Add token savings estimate
		if 'llm_token_usage' in metrics and metrics['llm_token_usage']['count'] > 0:
			avg_tokens = metrics['llm_token_usage']['avg']
			# Estimate original token count (rough estimate)
			estimated_original = avg_tokens / 0.3  # Assuming 70% reduction
			savings = estimated_original - avg_tokens
			savings_percent = (savings / estimated_original) * 100
			
			metrics['token_savings'] = {
				'estimated_original': estimated_original,
				'optimized': avg_tokens,
				'savings': savings,
				'savings_percent': savings_percent
			}
		
		return metrics
	
	# ... [rest of the existing Page class methods remain unchanged] ...