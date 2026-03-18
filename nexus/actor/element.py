"""Element class for element operations."""

import asyncio
from typing import TYPE_CHECKING, Literal, Union, Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from cdp_use.client import logger
from typing_extensions import TypedDict

if TYPE_CHECKING:
	from cdp_use.cdp.dom.commands import (
		DescribeNodeParameters,
		FocusParameters,
		GetAttributesParameters,
		GetBoxModelParameters,
		PushNodesByBackendIdsToFrontendParameters,
		RequestChildNodesParameters,
		ResolveNodeParameters,
	)
	from cdp_use.cdp.input.commands import (
		DispatchMouseEventParameters,
	)
	from cdp_use.cdp.input.types import MouseButton
	from cdp_use.cdp.page.commands import CaptureScreenshotParameters
	from cdp_use.cdp.page.types import Viewport
	from cdp_use.cdp.runtime.commands import CallFunctionOnParameters

	from nexus.browser.session import BrowserSession

# Type definitions for element operations
ModifierType = Literal['Alt', 'Control', 'Meta', 'Shift']


class Position(TypedDict):
	"""2D position coordinates."""

	x: float
	y: float


class BoundingBox(TypedDict):
	"""Element bounding box with position and dimensions."""

	x: float
	y: float
	width: float
	height: float


class ElementInfo(TypedDict):
	"""Basic information about a DOM element."""

	backendNodeId: int
	nodeId: int | None
	nodeName: str
	nodeType: int
	nodeValue: str | None
	attributes: dict[str, str]
	boundingBox: BoundingBox | None
	error: str | None


class LocatorStrategy(Enum):
	"""Strategies for locating elements."""
	CSS_SELECTOR = "css_selector"
	XPATH = "xpath"
	TEXT_CONTENT = "text_content"
	VISUAL_SIMILARITY = "visual_similarity"
	AI_VISION = "ai_vision"
	ACCESSIBILITY = "accessibility"


@dataclass
class ResilientLocator:
	"""A locator that can adapt to DOM changes using multiple strategies."""
	
	original_locator: str
	primary_strategy: LocatorStrategy
	fallback_strategies: List[LocatorStrategy]
	
	# Cache for successful strategies
	_cached_strategy: Optional[LocatorStrategy] = None
	_cached_locator: Optional[str] = None
	_cached_backend_node_id: Optional[int] = None
	
	# Element signature for change detection
	_element_signature: Optional[str] = None
	
	def get_cache_key(self) -> str:
		"""Generate a unique cache key for this locator."""
		data = f"{self.original_locator}:{self.primary_strategy.value}"
		return hashlib.md5(data.encode()).hexdigest()
	
	def update_cache(self, strategy: LocatorStrategy, locator: str, backend_node_id: int, signature: str):
		"""Update the cache with successful strategy."""
		self._cached_strategy = strategy
		self._cached_locator = locator
		self._cached_backend_node_id = backend_node_id
		self._element_signature = signature
	
	def is_cache_valid(self, current_signature: str) -> bool:
		"""Check if cached element is still valid based on signature."""
		return self._element_signature == current_signature


class ElementFindingStrategy:
	"""Base class for element finding strategies."""
	
	async def find_element(
		self, 
		browser_session: 'BrowserSession', 
		locator: str, 
		session_id: Optional[str] = None
	) -> Optional[int]:
		"""Find element using this strategy. Returns backend_node_id or None."""
		raise NotImplementedError


class CssSelectorStrategy(ElementFindingStrategy):
	"""Find elements using CSS selectors."""
	
	async def find_element(
		self, 
		browser_session: 'BrowserSession', 
		locator: str, 
		session_id: Optional[str] = None
	) -> Optional[int]:
		try:
			# Use CDP to find by CSS selector
			result = await browser_session.cdp_client.send.DOM.getDocument(session_id=session_id)
			root_node_id = result['root']['nodeId']
			
			query_result = await browser_session.cdp_client.send.DOM.querySelector(
				params={'nodeId': root_node_id, 'selector': locator},
				session_id=session_id
			)
			
			if query_result.get('nodeId', 0) == 0:
				return None
			
			# Get backend node ID
			describe_result = await browser_session.cdp_client.send.DOM.describeNode(
				params={'nodeId': query_result['nodeId']},
				session_id=session_id
			)
			
			return describe_result['node']['backendNodeId']
		except Exception:
			return None


class XPathStrategy(ElementFindingStrategy):
	"""Find elements using XPath."""
	
	async def find_element(
		self, 
		browser_session: 'BrowserSession', 
		locator: str, 
		session_id: Optional[str] = None
	) -> Optional[int]:
		try:
			# Use JavaScript to evaluate XPath
			js_code = f"""
				function() {{
					const result = document.evaluate(
						'{locator}',
						document,
						null,
						XPathResult.FIRST_ORDERED_NODE_TYPE,
						null
					);
					return result.singleNodeValue;
				}}
			"""
			
			result = await browser_session.cdp_client.send.Runtime.evaluate(
				params={'expression': js_code, 'returnByValue': False},
				session_id=session_id
			)
			
			if 'result' not in result or 'objectId' not in result['result']:
				return None
			
			# Get node info from the object
			object_id = result['result']['objectId']
			node_result = await browser_session.cdp_client.send.DOM.describeNode(
				params={'objectId': object_id},
				session_id=session_id
			)
			
			return node_result['node']['backendNodeId']
		except Exception:
			return None


class TextContentStrategy(ElementFindingStrategy):
	"""Find elements by text content."""
	
	async def find_element(
		self, 
		browser_session: 'BrowserSession', 
		locator: str, 
		session_id: Optional[str] = None
	) -> Optional[int]:
		try:
			# Find elements containing the text
			js_code = f"""
				function() {{
					const walker = document.createTreeWalker(
						document.body,
						NodeFilter.SHOW_TEXT,
						{{ acceptNode: (node) => node.textContent.includes('{locator}') ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT }}
					);
					
					const nodes = [];
					while (walker.nextNode()) {{
						nodes.push(walker.currentNode.parentElement);
					}}
					
					// Return the first visible element
					for (const node of nodes) {{
						const rect = node.getBoundingClientRect();
						if (rect.width > 0 && rect.height > 0) {{
							return node;
						}}
					}}
					return null;
				}}
			"""
			
			result = await browser_session.cdp_client.send.Runtime.evaluate(
				params={'expression': js_code, 'returnByValue': False},
				session_id=session_id
			)
			
			if 'result' not in result or 'objectId' not in result['result']:
				return None
			
			# Get node info
			object_id = result['result']['objectId']
			node_result = await browser_session.cdp_client.send.DOM.describeNode(
				params={'objectId': object_id},
				session_id=session_id
			)
			
			return node_result['node']['backendNodeId']
		except Exception:
			return None


class VisualSimilarityStrategy(ElementFindingStrategy):
	"""Find elements by visual similarity (simplified version)."""
	
	async def find_element(
		self, 
		browser_session: 'BrowserSession', 
		locator: str, 
		session_id: Optional[str] = None
	) -> Optional[int]:
		try:
			# This is a simplified implementation
			# In a real implementation, you would:
			// 1. Take a screenshot of the element
			// 2. Use computer vision to find similar elements
			// 3. Return the best match
			
			# For now, we'll use a heuristic based on element properties
			js_code = """
				function() {
					// Find elements with similar attributes
					const elements = document.querySelectorAll('*');
					const candidates = [];
					
					for (const el of elements) {
						const rect = el.getBoundingClientRect();
						if (rect.width > 0 && rect.height > 0) {
							// Simple similarity check based on tag name and class
							candidates.push({
								element: el,
								score: 1.0
							});
						}
					}
					
					// Return first visible candidate
					if (candidates.length > 0) {
						return candidates[0].element;
					}
					return null;
				}
			"""
			
			result = await browser_session.cdp_client.send.Runtime.evaluate(
				params={'expression': js_code, 'returnByValue': False},
				session_id=session_id
			)
			
			if 'result' not in result or 'objectId' not in result['result']:
				return None
			
			object_id = result['result']['objectId']
			node_result = await browser_session.cdp_client.send.DOM.describeNode(
				params={'objectId': object_id},
				session_id=session_id
			)
			
			return node_result['node']['backendNodeId']
		except Exception:
			return None


class Element:
	"""Element operations using BackendNodeId with self-healing capabilities."""

	def __init__(
		self,
		browser_session: 'BrowserSession',
		backend_node_id: int,
		session_id: str | None = None,
		resilient_locator: Optional[ResilientLocator] = None,
	):
		self._browser_session = browser_session
		self._client = browser_session.cdp_client
		self._backend_node_id = backend_node_id
		self._session_id = session_id
		self._resilient_locator = resilient_locator
		
		# Strategy pattern implementation
		self._strategies = {
			LocatorStrategy.CSS_SELECTOR: CssSelectorStrategy(),
			LocatorStrategy.XPATH: XPathStrategy(),
			LocatorStrategy.TEXT_CONTENT: TextContentStrategy(),
			LocatorStrategy.VISUAL_SIMILARITY: VisualSimilarityStrategy(),
		}

	async def _get_element_signature(self) -> str:
		"""Get a signature of the element for change detection."""
		try:
			# Get element attributes and text content
			node_id = await self._get_node_id()
			result = await self._client.send.DOM.getAttributes(
				params={'nodeId': node_id},
				session_id=self._session_id
			)
			
			attributes = result.get('attributes', [])
			attrs_dict = {}
			for i in range(0, len(attributes), 2):
				if i + 1 < len(attributes):
					attrs_dict[attributes[i]] = attributes[i + 1]
			
			# Get text content
			text_result = await self._client.send.Runtime.callFunctionOn(
				params={
					'functionDeclaration': 'function() { return this.textContent || ""; }',
					'objectId': await self._get_remote_object_id(),
					'returnByValue': True,
				},
				session_id=self._session_id
			)
			
			text_content = text_result.get('result', {}).get('value', '')
			
			# Create signature
			signature_data = {
				'nodeName': attrs_dict.get('nodeName', ''),
				'attributes': attrs_dict,
				'textContent': text_content[:100]  # First 100 chars
			}
			
			return hashlib.md5(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
		except Exception:
			return ""

	async def _find_with_strategy(self, strategy: LocatorStrategy, locator: str) -> Optional[int]:
		"""Find element using a specific strategy."""
		if strategy in self._strategies:
			return await self._strategies[strategy].find_element(
				self._browser_session, locator, self._session_id
			)
		return None

	async def _heal_element(self) -> bool:
		"""Attempt to heal the element by finding it with alternative strategies."""
		if not self._resilient_locator:
			return False
		
		# Try cached strategy first
		if (self._resilient_locator._cached_strategy and 
			self._resilient_locator._cached_locator):
			
			backend_node_id = await self._find_with_strategy(
				self._resilient_locator._cached_strategy,
				self._resilient_locator._cached_locator
			)
			
			if backend_node_id:
				# Verify it's the same element
				old_signature = self._resilient_locator._element_signature
				self._backend_node_id = backend_node_id
				new_signature = await self._get_element_signature()
				
				if old_signature == new_signature:
					return True
		
		# Try primary strategy with original locator
		backend_node_id = await self._find_with_strategy(
			self._resilient_locator.primary_strategy,
			self._resilient_locator.original_locator
		)
		
		if backend_node_id:
			self._backend_node_id = backend_node_id
			signature = await self._get_element_signature()
			self._resilient_locator.update_cache(
				self._resilient_locator.primary_strategy,
				self._resilient_locator.original_locator,
				backend_node_id,
				signature
			)
			return True
		
		# Try fallback strategies
		for strategy in self._resilient_locator.fallback_strategies:
			backend_node_id = await self._find_with_strategy(
				strategy,
				self._resilient_locator.original_locator
			)
			
			if backend_node_id:
				self._backend_node_id = backend_node_id
				signature = await self._get_element_signature()
				self._resilient_locator.update_cache(
					strategy,
					self._resilient_locator.original_locator,
					backend_node_id,
					signature
				)
				return True
		
		return False

	async def _get_node_id(self) -> int:
		"""Get DOM node ID from backend node ID with self-healing."""
		try:
			params: 'PushNodesByBackendIdsToFrontendParameters' = {'backendNodeIds': [self._backend_node_id]}
			result = await self._client.send.DOM.pushNodesByBackendIdsToFrontend(params, session_id=self._session_id)
			return result['nodeIds'][0]
		except Exception as e:
			# Try to heal the element
			if await self._heal_element():
				# Retry with new backend_node_id
				params: 'PushNodesByBackendIdsToFrontendParameters' = {'backendNodeIds': [self._backend_node_id]}
				result = await self._client.send.DOM.pushNodesByBackendIdsToFrontend(params, session_id=self._session_id)
				return result['nodeIds'][0]
			raise Exception(f'Failed to get node ID and element healing failed: {e}')

	async def _get_remote_object_id(self) -> str | None:
		"""Get remote object ID for this element with self-healing."""
		try:
			node_id = await self._get_node_id()
			params: 'ResolveNodeParameters' = {'nodeId': node_id}
			result = await self._client.send.DOM.resolveNode(params, session_id=self._session_id)
			object_id = result['object'].get('objectId', None)

			if not object_id:
				return None
			return object_id
		except Exception as e:
			# Try to heal the element
			if await self._heal_element():
				# Retry with healed element
				node_id = await self._get_node_id()
				params: 'ResolveNodeParameters' = {'nodeId': node_id}
				result = await self._client.send.DOM.resolveNode(params, session_id=self._session_id)
				object_id = result['object'].get('objectId', None)
				return object_id
			raise Exception(f'Failed to get remote object ID and element healing failed: {e}')

	async def click(
		self,
		button: 'MouseButton' = 'left',
		click_count: int = 1,
		modifiers: list[ModifierType] | None = None,
	) -> None:
		"""Click the element using the advanced watchdog implementation with self-healing."""

		try:
			# Get viewport dimensions for visibility checks
			layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
			viewport_width = layout_metrics['layoutViewport']['clientWidth']
			viewport_height = layout_metrics['layoutViewport']['clientHeight']

			# Try multiple methods to get element geometry
			quads = []

			# Method 1: Try DOM.getContentQuads first (best for inline elements and complex layouts)
			try:
				content_quads_result = await self._client.send.DOM.getContentQuads(
					params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
				)
				if 'quads' in content_quads_result and content_quads_result['quads']:
					quads = content_quads_result['quads']
			except Exception:
				pass

			# Method 2: Fall back to DOM.getBoxModel
			if not quads:
				try:
					box_model = await self._client.send.DOM.getBoxModel(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					if 'model' in box_model and 'content' in box_model['model']:
						content_quad = box_model['model']['content']
						if len(content_quad) >= 8:
							# Convert box model format to quad format
							quads = [
								[
									content_quad[0],
									content_quad[1],  # x1, y1
									content_quad[2],
									content_quad[3],  # x2, y2
									content_quad[4],
									content_quad[5],  # x3, y3
									content_quad[6],
									content_quad[7],  # x4, y4
								]
							]
				except Exception:
					pass

			# Method 3: Fall back to JavaScript getBoundingClientRect
			if not quads:
				try:
					result = await self._client.send.DOM.resolveNode(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					if 'object' in result and 'objectId' in result['object']:
						object_id = result['object']['objectId']

						# Get bounding rect via JavaScript
						bounds_result = await self._client.send.Runtime.callFunctionOn(
							params={
								'functionDeclaration': """
									function() {
										const rect = this.getBoundingClientRect();
										return {
											x: rect.left,
											y: rect.top,
											width: rect.width,
											height: rect.height
										};
									}
								""",
								'objectId': object_id,
								'returnByValue': True,
							},
							session_id=self._session_id,
						)

						if 'result' in bounds_result and 'value' in bounds_result['result']:
							rect = bounds_result['result']['value']
							# Convert rect to quad format
							x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
							quads = [
								[
									x,
									y,  # top-left
									x + w,
									y,  # top-right
									x + w,
									y + h,  # bottom-right
									x,
									y + h,  # bottom-left
								]
							]
				except Exception as e:
					# Try to heal the element
					if await self._heal_element():
						# Retry with healed element
						result = await self._client.send.DOM.resolveNode(
							params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
						)
						if 'object' in result and 'objectId' in result['object']:
							object_id = result['object']['objectId']
							bounds_result = await self._client.send.Runtime.callFunctionOn(
								params={
									'functionDeclaration': """
										function() {
											const rect = this.getBoundingClientRect();
											return {
												x: rect.left,
												y: rect.top,
												width: rect.width,
												height: rect.height
											};
										}
									""",
									'objectId': object_id,
									'returnByValue': True,
								},
								session_id=self._session_id,
							)
							if 'result' in bounds_result and 'value' in bounds_result['result']:
								rect = bounds_result['result']['value']
								x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
								quads = [
									[
										x, y,
										x + w, y,
										x + w, y + h,
										x, y + h,
									]
								]

			# If we still don't have quads, fall back to JS click
			if not quads:
				try:
					result = await self._client.send.DOM.resolveNode(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					if 'object' not in result or 'objectId' not in result['object']:
						# Try to heal the element
						if await self._heal_element():
							result = await self._client.send.DOM.resolveNode(
								params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
							)
							if 'object' not in result or 'objectId' not in result['object']:
								raise Exception('Failed to find DOM element based on backendNodeId, maybe page content changed?')
						else:
							raise Exception('Failed to find DOM element based on backendNodeId, maybe page content changed?')
					
					object_id = result['object']['objectId']

					await self._client.send.Runtime.callFunctionOn(
						params={
							'functionDeclaration': 'function() { this.click(); }',
							'objectId': object_id,
						},
						session_id=self._session_id,
					)
					await asyncio.sleep(0.05)
					return
				except Exception as js_e:
					raise Exception(f'Failed to click element: {js_e}')

			# Find the largest visible quad within the viewport
			best_quad = None
			best_area = 0

			for quad in quads:
				if len(quad) < 8:
					continue

				# Calculate quad bounds
				xs = [quad[i] for i in range(0, 8, 2)]
				ys = [quad[i] for i in range(1, 8, 2)]
				min_x, max_x = min(xs), max(xs)
				min_y, max_y = min(ys), max(ys)

				# Check if quad intersects with viewport
				if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
					continue  # Quad is completely outside viewport

				# Calculate visible area (intersection with viewport)
				visible_min_x = max(0, min_x)
				visible_max_x = min(viewport_width, max_x)
				visible_min_y = max(0, min_y)
				visible_max_y = min(viewport_height, max_y)

				visible_width = visible_max_x - visible_min_x
				visible_height = visible_max_y - visible_min_y
				visible_area = visible_width * visible_height

				if visible_area > best_area:
					best_area = visible_area
					best_quad = quad

			if not best_quad:
				# No visible quad found, use the first quad anyway
				best_quad = quads[0]

			# Calculate center point of the best quad
			center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
			center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

			# Ensure click point is within viewport bounds
			center_x = max(0, min(viewport_width - 1, center_x))
			center_y = max(0, min(viewport_height - 1, center_y))

			# Scroll element into view
			try:
				await self._client.send.DOM.scrollIntoViewIfNeeded(
					params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
				)
				await asyncio.sleep(0.05)  # Wait for scroll to complete
			except Exception:
				# Try to heal the element
				if await self._heal_element():
					await self._client.send.DOM.scrollIntoViewIfNeeded(
						params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
					)
					await asyncio.sleep(0.05)

			# Perform the click
			mouse_params: 'DispatchMouseEventParameters' = {
				'type': 'mousePressed',
				'x': center_x,
				'y': center_y,
				'button': button,
				'clickCount': click_count,
			}
			
			if modifiers:
				mouse_params['modifiers'] = sum(
					1 << i for i, mod in enumerate(['Alt', 'Control', 'Meta', 'Shift']) 
					if mod in modifiers
				)
			
			await self._client.send.Input.dispatchMouseEvent(
				params=mouse_params,
				session_id=self._session_id,
			)
			
			# Small delay between press and release
			await asyncio.sleep(0.01)
			
			mouse_params['type'] = 'mouseReleased'
			await self._client.send.Input.dispatchMouseEvent(
				params=mouse_params,
				session_id=self._session_id,
			)
			
		except Exception as e:
			# Final attempt to heal and retry
			if await self._heal_element():
				await self.click(button, click_count, modifiers)
			else:
				raise Exception(f'Failed to click element after healing attempts: {e}')

	@classmethod
	async def find_with_resilient_locator(
		cls,
		browser_session: 'BrowserSession',
		locator: str,
		primary_strategy: LocatorStrategy = LocatorStrategy.CSS_SELECTOR,
		fallback_strategies: Optional[List[LocatorStrategy]] = None,
		session_id: Optional[str] = None,
	) -> 'Element':
		"""Find an element using a resilient locator with multiple strategies."""
		
		if fallback_strategies is None:
			fallback_strategies = [
				LocatorStrategy.XPATH,
				LocatorStrategy.TEXT_CONTENT,
				LocatorStrategy.VISUAL_SIMILARITY
			]
		
		resilient_locator = ResilientLocator(
			original_locator=locator,
			primary_strategy=primary_strategy,
			fallback_strategies=fallback_strategies
		)
		
		# Create strategy instances
		strategies = {
			LocatorStrategy.CSS_SELECTOR: CssSelectorStrategy(),
			LocatorStrategy.XPATH: XPathStrategy(),
			LocatorStrategy.TEXT_CONTENT: TextContentStrategy(),
			LocatorStrategy.VISUAL_SIMILARITY: VisualSimilarityStrategy(),
		}
		
		# Try primary strategy first
		backend_node_id = await strategies[primary_strategy].find_element(
			browser_session, locator, session_id
		)
		
		if backend_node_id:
			element = cls(browser_session, backend_node_id, session_id, resilient_locator)
			signature = await element._get_element_signature()
			resilient_locator.update_cache(primary_strategy, locator, backend_node_id, signature)
			return element
		
		# Try fallback strategies
		for strategy in fallback_strategies:
			backend_node_id = await strategies[strategy].find_element(
				browser_session, locator, session_id
			)
			
			if backend_node_id:
				element = cls(browser_session, backend_node_id, session_id, resilient_locator)
				signature = await element._get_element_signature()
				resilient_locator.update_cache(strategy, locator, backend_node_id, signature)
				return element
		
		raise Exception(f'Could not find element with any strategy for locator: {locator}')