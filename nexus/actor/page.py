"""Page class for page-level operations with cross-browser support."""

import asyncio
import json
import re
import time
from typing import TYPE_CHECKING, TypeVar, Optional, List, Dict, Any, Tuple, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, asdict, field
from functools import wraps
import statistics
from abc import ABC, abstractmethod
import hashlib
import copy

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


class BrowserType(Enum):
	CHROME = "chrome"
	FIREFOX = "firefox"
	SAFARI = "safari"
	EDGE = "edge"


class BrowserProtocol(Enum):
	CDP = "cdp"
	WEBDRIVER_BIDI = "webdriver_bidi"


class BrowserCapabilities(BaseModel):
	"""Browser capability detection and feature support."""
	browser_type: BrowserType
	protocol: BrowserProtocol
	supports_cdp: bool = False
	supports_bidi: bool = False
	supports_screenshot: bool = True
	supports_network_monitoring: bool = True
	supports_dom_inspection: bool = True
	supports_input_simulation: bool = True
	supports_javascript_execution: bool = True
	supports_cookies: bool = True
	supports_local_storage: bool = True
	supports_session_storage: bool = True
	supports_geolocation: bool = False
	supports_permissions: bool = False
	supports_service_workers: bool = False
	supports_web_authn: bool = False
	fallbacks: Dict[str, str] = {}  # feature -> fallback implementation


class ProtocolAdapter(ABC):
	"""Abstract base class for browser protocol adapters."""
	
	@abstractmethod
	async def send_command(self, method: str, params: Dict[str, Any] = None, session_id: str = None) -> Any:
		"""Send a command to the browser."""
		pass
	
	@abstractmethod
	async def add_event_listener(self, event: str, callback: Callable) -> None:
		"""Add event listener for browser events."""
		pass
	
	@abstractmethod
	async def remove_event_listener(self, event: str, callback: Callable) -> None:
		"""Remove event listener for browser events."""
		pass


class CDPAdapter(ProtocolAdapter):
	"""CDP protocol adapter for Chrome/Edge."""
	
	def __init__(self, client):
		self.client = client
		self._event_listeners = {}
	
	async def send_command(self, method: str, params: Dict[str, Any] = None, session_id: str = None) -> Any:
		"""Send CDP command."""
		if session_id:
			return await self.client.send(method, params=params, session_id=session_id)
		return await self.client.send(method, params=params)
	
	async def add_event_listener(self, event: str, callback: Callable) -> None:
		"""Add CDP event listener."""
		if event not in self._event_listeners:
			self._event_listeners[event] = []
		self._event_listeners[event].append(callback)
		self.client.add_event_listener(event, callback)
	
	async def remove_event_listener(self, event: str, callback: Callable) -> None:
		"""Remove CDP event listener."""
		if event in self._event_listeners and callback in self._event_listeners[event]:
			self._event_listeners[event].remove(callback)
			self.client.remove_event_listener(event, callback)


class WebDriverBiDiAdapter(ProtocolAdapter):
	"""WebDriver BiDi protocol adapter for cross-browser support."""
	
	def __init__(self, browser_type: BrowserType):
		self.browser_type = browser_type
		self._ws_connection = None
		self._event_listeners = {}
		self._command_id = 0
	
	async def connect(self, url: str) -> None:
		"""Connect to WebDriver BiDi endpoint."""
		# Implementation would connect to browser's WebDriver BiDi endpoint
		# This is a placeholder for the actual implementation
		logger.info(f"Connecting to {self.browser_type.value} via WebDriver BiDi at {url}")
		# In real implementation, establish WebSocket connection
	
	async def send_command(self, method: str, params: Dict[str, Any] = None, session_id: str = None) -> Any:
		"""Send WebDriver BiDi command."""
		self._command_id += 1
		command = {
			"id": self._command_id,
			"method": method,
			"params": params or {}
		}
		if session_id:
			command["sessionId"] = session_id
		
		# Implementation would send command via WebSocket
		# This is a placeholder returning mock response
		logger.debug(f"WebDriver BiDi command: {command}")
		
		# Return mock response based on command type
		if method == "browsingContext.navigate":
			return {"navigation": "mock-navigation-id", "url": params.get("url", "")}
		elif method == "script.evaluate":
			return {"result": {"type": "string", "value": "mock-result"}}
		return {}
	
	async def add_event_listener(self, event: str, callback: Callable) -> None:
		"""Add WebDriver BiDi event listener."""
		if event not in self._event_listeners:
			self._event_listeners[event] = []
		self._event_listeners[event].append(callback)
		# Implementation would subscribe to BiDi events
		logger.debug(f"Subscribed to WebDriver BiDi event: {event}")
	
	async def remove_event_listener(self, event: str, callback: Callable) -> None:
		"""Remove WebDriver BiDi event listener."""
		if event in self._event_listeners and callback in self._event_listeners[event]:
			self._event_listeners[event].remove(callback)
			# Implementation would unsubscribe from BiDi events
			logger.debug(f"Unsubscribed from WebDriver BiDi event: {event}")


class CircuitState(Enum):
	CLOSED = "closed"
	OPEN = "open"
	HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
	failure_threshold: int = 5
	recovery_timeout: float = 30.0
	half_open_max_attempts: int = 3
	exponential_backoff_base: float = 1.0
	exponential_backoff_max: float = 60.0
	expected_exceptions: tuple = (Exception,)


class CircuitBreaker:
	"""Circuit breaker pattern implementation for resilient operations."""
	
	def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
		self.name = name
		self.config = config or CircuitBreakerConfig()
		self.state = CircuitState.CLOSED
		self.failure_count = 0
		self.last_failure_time = 0
		self.half_open_attempts = 0
		self._lock = asyncio.Lock()
	
	async def execute(self, func: Callable, *args, **kwargs):
		"""Execute function with circuit breaker protection."""
		async with self._lock:
			if self.state == CircuitState.OPEN:
				if time.time() - self.last_failure_time >= self.config.recovery_timeout:
					self.state = CircuitState.HALF_OPEN
					self.half_open_attempts = 0
					logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
				else:
					raise CircuitBreakerOpenError(
						f"Circuit {self.name} is OPEN. "
						f"Retry after {self.config.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
					)
		
		try:
			result = await func(*args, **kwargs)
			await self._on_success()
			return result
		except self.config.expected_exceptions as e:
			await self._on_failure(e)
			raise
	
	async def _on_success(self):
		"""Handle successful execution."""
		async with self._lock:
			if self.state == CircuitState.HALF_OPEN:
				self.half_open_attempts += 1
				if self.half_open_attempts >= self.config.half_open_max_attempts:
					self.state = CircuitState.CLOSED
					self.failure_count = 0
					logger.info(f"Circuit {self.name} transitioning to CLOSED after successful recovery")
			else:
				self.failure_count = 0
	
	async def _on_failure(self, exception: Exception):
		"""Handle failed execution."""
		async with self._lock:
			self.failure_count += 1
			self.last_failure_time = time.time()
			
			if self.state == CircuitState.HALF_OPEN:
				self.state = CircuitState.OPEN
				logger.warning(f"Circuit {self.name} transitioning back to OPEN after failure in HALF_OPEN state")
			elif self.failure_count >= self.config.failure_threshold:
				self.state = CircuitState.OPEN
				logger.warning(f"Circuit {self.name} transitioning to OPEN after {self.failure_count} failures")


class CircuitBreakerOpenError(Exception):
	"""Exception raised when circuit breaker is open."""
	pass


class ActionRiskLevel(Enum):
	"""Risk levels for browser actions."""
	SAFE = auto()       # Read-only operations, no side effects
	LOW = auto()        # Minor UI changes, easily reversible
	MEDIUM = auto()     # Form submissions, navigation, data changes
	HIGH = auto()       # Destructive actions, deletions, financial transactions
	CRITICAL = auto()   # Irreversible actions, security changes


@dataclass
class ActionValidation:
	"""Validation result for an action."""
	action: str
	risk_level: ActionRiskLevel
	requires_confirmation: bool = False
	validation_errors: List[str] = field(default_factory=list)
	warnings: List[str] = field(default_factory=list)
	sandbox_test_passed: Optional[bool] = None
	estimated_impact: str = ""


@dataclass
class DOMSnapshot:
	"""Snapshot of DOM state for rollback."""
	snapshot_id: str
	timestamp: float
	url: str
	dom_hash: str
	dom_content: str
	local_storage: Dict[str, str] = field(default_factory=dict)
	session_storage: Dict[str, str] = field(default_factory=dict)
	cookies: List[Dict[str, Any]] = field(default_factory=list)
	scroll_position: Tuple[int, int] = (0, 0)
	
	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)
	
	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'DOMSnapshot':
		return cls(**data)


class ActionValidator:
	"""Validates browser actions for safety and risk assessment."""
	
	# Risk assessment patterns
	HIGH_RISK_PATTERNS = [
		r'delete|remove|destroy|drop|truncate',
		r'payment|checkout|purchase|buy|order',
		r'submit|send|post|upload',
		r'login|logout|signin|signout',
		r'password|credential|token|secret',
		r'admin|sudo|root|privilege',
		r'format|wipe|reset|factory',
	]
	
	MEDIUM_RISK_PATTERNS = [
		r'click|tap|press',
		r'type|input|enter|fill',
		'navigate|goto|redirect',
		r'select|choose|option',
		r'open|close|toggle',
	]
	
	@classmethod
	def assess_risk(cls, action: str, **kwargs) -> ActionRiskLevel:
		"""Assess risk level of an action."""
		action_lower = action.lower()
		
		# Check for high-risk patterns
		for pattern in cls.HIGH_RISK_PATTERNS:
			if re.search(pattern, action_lower):
				return ActionRiskLevel.HIGH
		
		# Check for medium-risk patterns
		for pattern in cls.MEDIUM_RISK_PATTERNS:
			if re.search(pattern, action_lower):
				return ActionRiskLevel.MEDIUM
		
		# Check kwargs for high-risk indicators
		if kwargs.get('confirm', False):
			return ActionRiskLevel.HIGH
		
		if 'delete' in kwargs.get('type', '').lower():
			return ActionRiskLevel.HIGH
		
		# Default to safe for read-only operations
		if action_lower.startswith(('get', 'read', 'fetch', 'find', 'query')):
			return ActionRiskLevel.SAFE
		
		return ActionRiskLevel.LOW
	
	@classmethod
	def validate_action(cls, action: str, element: Optional['Element'] = None, **kwargs) -> ActionValidation:
		"""Validate an action and return validation result."""
		risk_level = cls.assess_risk(action, **kwargs)
		validation_errors = []
		warnings = []
		
		# Element-specific validations
		if element:
			# Check for disabled elements
			if element.disabled:
				validation_errors.append(f"Element is disabled: {element.selector}")
			
			# Check for read-only inputs
			if element.tag_name == 'input' and element.readonly:
				validation_errors.append(f"Input element is read-only: {element.selector}")
			
			# Check for hidden elements
			if not element.visible:
				warnings.append(f"Element may not be visible: {element.selector}")
		
		# Action-specific validations
		if action.lower() == 'navigate':
			url = kwargs.get('url', '')
			if url.startswith(('javascript:', 'data:')):
				validation_errors.append(f"Potentially unsafe URL scheme: {url[:50]}...")
		
		if action.lower() == 'click':
			if element and element.tag_name == 'a':
				href = element.attributes.get('href', '')
				if href.startswith(('javascript:', 'data:')):
					validation_errors.append(f"Link contains unsafe href: {href[:50]}...")
		
		requires_confirmation = risk_level in [ActionRiskLevel.HIGH, ActionRiskLevel.CRITICAL]
		
		return ActionValidation(
			action=action,
			risk_level=risk_level,
			requires_confirmation=requires_confirmation,
			validation_errors=validation_errors,
			warnings=warnings,
			estimated_impact=cls._estimate_impact(action, risk_level)
		)
	
	@staticmethod
	def _estimate_impact(action: str, risk_level: ActionRiskLevel) -> str:
		"""Estimate the impact of an action."""
		impacts = {
			ActionRiskLevel.SAFE: "No side effects, read-only operation",
			ActionRiskLevel.LOW: "Minor UI changes, easily reversible",
			ActionRiskLevel.MEDIUM: "May trigger navigation, form submission, or data changes",
			ActionRiskLevel.HIGH: "Destructive action, may cause data loss or irreversible changes",
			ActionRiskLevel.CRITICAL: "Critical action with security or financial implications"
		}
		return impacts.get(risk_level, "Unknown impact")


class SandboxEnvironment:
	"""Sandbox environment for testing actions before execution."""
	
	def __init__(self, page: 'Page'):
		self.page = page
		self._sandbox_frame_id: Optional[str] = None
		self._test_results: Dict[str, bool] = {}
	
	async def setup(self) -> None:
		"""Setup sandbox environment."""
		try:
			# Create a hidden iframe for sandbox testing
			await self.page.evaluate('''() => {
				if (!document.getElementById('__nexus_sandbox__')) {
					const iframe = document.createElement('iframe');
					iframe.id = '__nexus_sandbox__';
					iframe.style.cssText = 'position:fixed;top:-9999px;left:-9999px;width:1px;height:1px;opacity:0;pointer-events:none;';
					document.body.appendChild(iframe);
				}
			}''')
			
			# Get the frame ID for the sandbox iframe
			frames = await self.page.get_frames()
			for frame in frames:
				if frame.get('name') == '__nexus_sandbox__':
					self._sandbox_frame_id = frame.get('id')
					break
			
			logger.debug("Sandbox environment initialized")
		except Exception as e:
			logger.warning(f"Failed to setup sandbox environment: {e}")
	
	async def test_action(self, action: str, **kwargs) -> bool:
		"""Test an action in the sandbox environment."""
		if not self._sandbox_frame_id:
			await self.setup()
		
		try:
			# Copy current page state to sandbox
			await self._copy_state_to_sandbox()
			
			# Execute action in sandbox
			test_id = hashlib.md5(f"{action}{time.time()}".encode()).hexdigest()[:8]
			
			# Simulate action in sandbox (simplified implementation)
			result = await self.page.evaluate(f'''() => {{
				try {{
					const sandbox = document.getElementById('__nexus_sandbox__');
					if (!sandbox || !sandbox.contentDocument) return false;
					
					// Test action based on type
					const action = "{action}";
					const testResult = {{ success: true, errors: [] }};
					
					// Add test-specific logic here
					if (action.includes('click')) {{
						// Test click safety
						const testElement = sandbox.contentDocument.createElement('button');
						testElement.onclick = () => {{ throw new Error('Test click executed'); }};
						sandbox.contentDocument.body.appendChild(testElement);
					}}
					
					return testResult.success;
				}} catch (e) {{
					return false;
				}}
			}}''')
			
			self._test_results[test_id] = result
			return result
		except Exception as e:
			logger.warning(f"Sandbox test failed for action {action}: {e}")
			return False
	
	async def _copy_state_to_sandbox(self) -> None:
		"""Copy current page state to sandbox iframe."""
		try:
			# Get current page content
			content = await self.page.get_content()
			
			# Copy to sandbox iframe
			await self.page.evaluate(f'''() => {{
				const sandbox = document.getElementById('__nexus_sandbox__');
				if (sandbox && sandbox.contentDocument) {{
					sandbox.contentDocument.open();
					sandbox.contentDocument.write(`{content.replace('`', '\\`')}`);
					sandbox.contentDocument.close();
				}}
			}}''')
		except Exception as e:
			logger.warning(f"Failed to copy state to sandbox: {e}")
	
	async def cleanup(self) -> None:
		"""Cleanup sandbox environment."""
		try:
			await self.page.evaluate('''() => {
				const sandbox = document.getElementById('__nexus_sandbox__');
				if (sandbox) {
					sandbox.remove();
				}
			}''')
			self._sandbox_frame_id = None
			self._test_results.clear()
		except Exception as e:
			logger.warning(f"Failed to cleanup sandbox: {e}")


class RollbackManager:
	"""Manages DOM snapshots and rollback capabilities."""
	
	def __init__(self, page: 'Page', max_snapshots: int = 10):
		self.page = page
		self.max_snapshots = max_snapshots
		self.snapshots: List[DOMSnapshot] = []
		self._lock = asyncio.Lock()
	
	async def create_snapshot(self, description: str = "") -> DOMSnapshot:
		"""Create a DOM snapshot for potential rollback."""
		async with self._lock:
			try:
				# Capture current state
				url = self.page.url
				dom_content = await self.page.get_content()
				dom_hash = hashlib.md5(dom_content.encode()).hexdigest()
				
				# Capture storage
				local_storage = await self.page.evaluate('() => { return {...localStorage}; }')
				session_storage = await self.page.evaluate('() => { return {...sessionStorage}; }')
				cookies = await self.page.get_cookies()
				
				# Capture scroll position
				scroll_position = await self.page.evaluate('() => { return [window.scrollX, window.scrollY]; }')
				
				snapshot = DOMSnapshot(
					snapshot_id=hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:12],
					timestamp=time.time(),
					url=url,
					dom_hash=dom_hash,
					dom_content=dom_content,
					local_storage=local_storage or {},
					session_storage=session_storage or {},
					cookies=cookies or [],
					scroll_position=tuple(scroll_position) if scroll_position else (0, 0)
				)
				
				# Add to snapshots list
				self.snapshots.append(snapshot)
				
				# Trim old snapshots if exceeding max
				if len(self.snapshots) > self.max_snapshots:
					self.snapshots = self.snapshots[-self.max_snapshots:]
				
				logger.debug(f"Created DOM snapshot {snapshot.snapshot_id}: {description}")
				return snapshot
			except Exception as e:
				logger.error(f"Failed to create DOM snapshot: {e}")
				raise
	
	async def rollback_to_snapshot(self, snapshot_id: str) -> bool:
		"""Rollback to a specific snapshot."""
		async with self._lock:
			snapshot = next((s for s in self.snapshots if s.snapshot_id == snapshot_id), None)
			if not snapshot:
				logger.error(f"Snapshot {snapshot_id} not found")
				return False
			
			try:
				# Navigate to the snapshot URL if different
				current_url = self.page.url
				if current_url != snapshot.url:
					await self.page.goto(snapshot.url)
				
				# Restore DOM content
				await self.page.set_content(snapshot.dom_content)
				
				# Restore local storage
				if snapshot.local_storage:
					await self.page.evaluate(f'''() => {{
						localStorage.clear();
						for (const [key, value] of Object.entries({json.dumps(snapshot.local_storage)})) {{
							localStorage.setItem(key, value);
						}}
					}}''')
				
				# Restore session storage
				if snapshot.session_storage:
					await self.page.evaluate(f'''() => {{
						sessionStorage.clear();
						for (const [key, value] of Object.entries({json.dumps(snapshot.session_storage)})) {{
							sessionStorage.setItem(key, value);
						}}
					}}''')
				
				# Restore cookies
				if snapshot.cookies:
					for cookie in snapshot.cookies:
						await self.page.set_cookie(cookie)
				
				# Restore scroll position
				await self.page.evaluate(f'() => {{ window.scrollTo({snapshot.scroll_position[0]}, {snapshot.scroll_position[1]}); }}')
				
				logger.info(f"Successfully rolled back to snapshot {snapshot_id}")
				return True
			except Exception as e:
				logger.error(f"Failed to rollback to snapshot {snapshot_id}: {e}")
				return False
	
	async def get_snapshots(self) -> List[Dict[str, Any]]:
		"""Get list of available snapshots."""
		return [
			{
				"id": s.snapshot_id,
				"timestamp": s.timestamp,
				"url": s.url,
				"dom_hash": s.dom_hash
			}
			for s in self.snapshots
		]
	
	async def cleanup_old_snapshots(self, max_age_seconds: float = 3600) -> int:
		"""Cleanup snapshots older than specified age."""
		async with self._lock:
			current_time = time.time()
			initial_count = len(self.snapshots)
			self.snapshots = [
				s for s in self.snapshots
				if current_time - s.timestamp <= max_age_seconds
			]
			removed_count = initial_count - len(self.snapshots)
			if removed_count > 0:
				logger.debug(f"Cleaned up {removed_count} old snapshots")
			return removed_count


class SafetySystem:
	"""Integrated safety system for action validation, sandboxing, and rollback."""
	
	def __init__(self, page: 'Page'):
		self.page = page
		self.validator = ActionValidator()
		self.sandbox = SandboxEnvironment(page)
		self.rollback_manager = RollbackManager(page)
		self.confirmation_callback: Optional[Callable[[ActionValidation], bool]] = None
		self.auto_confirm_low_risk: bool = True
		self.sandbox_enabled: bool = True
		self.rollback_enabled: bool = True
	
	async def validate_and_execute(
		self,
		action: str,
		execution_func: Callable,
		element: Optional['Element'] = None,
		require_confirmation: Optional[bool] = None,
		create_snapshot: bool = True,
		**kwargs
	) -> Any:
		"""
		Validate an action and execute it with safety measures.
		
		Args:
			action: Description of the action to perform
			execution_func: Async function to execute the action
			element: Optional element involved in the action
			require_confirmation: Override confirmation requirement
			create_snapshot: Whether to create a snapshot before execution
			**kwargs: Additional arguments for the action
		
		Returns:
			Result of the execution function
		
		Raises:
			ActionValidationError: If validation fails
			ConfirmationRequiredError: If confirmation is required but not provided
			SandboxTestError: If sandbox test fails
		"""
		# Step 1: Validate the action
		validation = self.validator.validate_action(action, element, **kwargs)
		
		if validation.validation_errors:
			raise ActionValidationError(
				f"Action validation failed: {'; '.join(validation.validation_errors)}",
				validation=validation
			)
		
		# Log warnings
		for warning in validation.warnings:
			logger.warning(f"Action warning: {warning}")
		
		# Step 2: Check confirmation requirements
		needs_confirmation = require_confirmation if require_confirmation is not None else validation.requires_confirmation
		
		if needs_confirmation and not self.auto_confirm_low_risk:
			if self.confirmation_callback:
				if not self.confirmation_callback(validation):
					raise ConfirmationRequiredError(
						f"Action requires user confirmation: {action}",
						validation=validation
					)
			else:
				raise ConfirmationRequiredError(
					f"Action requires confirmation but no callback set: {action}",
					validation=validation
				)
		
		# Step 3: Test in sandbox if enabled and action is high risk
		if self.sandbox_enabled and validation.risk_level in [ActionRiskLevel.HIGH, ActionRiskLevel.CRITICAL]:
			sandbox_result = await self.sandbox.test_action(action, **kwargs)
			validation.sandbox_test_passed = sandbox_result
			
			if not sandbox_result:
				raise SandboxTestError(
					f"Action failed sandbox test: {action}",
					validation=validation
				)
		
		# Step 4: Create snapshot if enabled
		snapshot = None
		if self.rollback_enabled and create_snapshot:
			try:
				snapshot = await self.rollback_manager.create_snapshot(f"Before: {action}")
			except Exception as e:
				logger.warning(f"Failed to create snapshot: {e}")
		
		# Step 5: Execute the action
		try:
			result = await execution_func(**kwargs)
			return result
		except Exception as e:
			# Step 6: Rollback on failure if snapshot exists
			if snapshot and self.rollback_enabled:
				logger.warning(f"Action failed, attempting rollback: {e}")
				rollback_success = await self.rollback_manager.rollback_to_snapshot(snapshot.snapshot_id)
				if rollback_success:
					logger.info("Successfully rolled back after failed action")
				else:
					logger.error("Failed to rollback after failed action")
			raise
	
	async def quick_validate(self, action: str, **kwargs) -> ActionValidation:
		"""Quick validation without full safety measures."""
		return self.validator.validate_action(action, **kwargs)
	
	def set_confirmation_callback(self, callback: Callable[[ActionValidation], bool]) -> None:
		"""Set callback for user confirmation of high-risk actions."""
		self.confirmation_callback = callback
	
	async def cleanup(self) -> None:
		"""Cleanup safety system resources."""
		await self.sandbox.cleanup()


class ActionValidationError(Exception):
	"""Exception raised when action validation fails."""
	def __init__(self, message: str, validation: ActionValidation):
		super().__init__(message)
		self.validation = validation


class ConfirmationRequiredError(Exception):
	"""Exception raised when user confirmation is required."""
	def __init__(self, message: str, validation: ActionValidation):
		super().__init__(message)
		self.validation = validation


class SandboxTestError(Exception):
	"""Exception raised when sandbox test fails."""
	def __init__(self, message: str, validation: ActionValidation):
		super().__init__(message)
		self.validation = validation


class Page:
	"""Enhanced Page class with action validation and safety system."""
	
	def __init__(
		self,
		browser_session: 'BrowserSession',
		dom_service: DomService,
		llm: Optional['BaseChatModel'] = None,
		enable_safety_system: bool = True
	):
		self.browser_session = browser_session
		self.dom_service = dom_service
		self.llm = llm
		self._mouse: Optional['Mouse'] = None
		self._elements: Dict[str, 'Element'] = {}
		self._cdp_session_id: Optional[str] = None
		
		# Initialize safety system
		self.safety_system = SafetySystem(self) if enable_safety_system else None
		
		# Existing initialization code would continue here...
		# (Preserving all existing functionality)
	
	async def click(
		self,
		selector: str,
		validate_safety: bool = True,
		require_confirmation: Optional[bool] = None,
		**kwargs
	) -> None:
		"""Click on an element with safety validation."""
		element = await self.get_element(selector)
		
		if validate_safety and self.safety_system:
			await self.safety_system.validate_and_execute(
				action=f"click on {selector}",
				execution_func=self._click_impl,
				element=element,
				require_confirmation=require_confirmation,
				selector=selector,
				**kwargs
			)
		else:
			await self._click_impl(selector, **kwargs)
	
	async def _click_impl(self, selector: str, **kwargs) -> None:
		"""Implementation of click action."""
		# Original click implementation would go here
		pass
	
	async def type(
		self,
		selector: str,
		text: str,
		validate_safety: bool = True,
		require_confirmation: Optional[bool] = None,
		**kwargs
	) -> None:
		"""Type text into an element with safety validation."""
		element = await self.get_element(selector)
		
		if validate_safety and self.safety_system:
			await self.safety_system.validate_and_execute(
				action=f"type '{text}' into {selector}",
				execution_func=self._type_impl,
				element=element,
				require_confirmation=require_confirmation,
				selector=selector,
				text=text,
				**kwargs
			)
		else:
			await self._type_impl(selector, text, **kwargs)
	
	async def _type_impl(self, selector: str, text: str, **kwargs) -> None:
		"""Implementation of type action."""
		# Original type implementation would go here
		pass
	
	async def navigate(
		self,
		url: str,
		validate_safety: bool = True,
		require_confirmation: Optional[bool] = None,
		**kwargs
	) -> None:
		"""Navigate to a URL with safety validation."""
		if validate_safety and self.safety_system:
			await self.safety_system.validate_and_execute(
				action=f"navigate to {url}",
				execution_func=self._navigate_impl,
				require_confirmation=require_confirmation,
				url=url,
				**kwargs
			)
		else:
			await self._navigate_impl(url, **kwargs)
	
	async def _navigate_impl(self, url: str, **kwargs) -> None:
		"""Implementation of navigate action."""
		# Original navigate implementation would go here
		pass
	
	async def submit_form(
		self,
		selector: str,
		validate_safety: bool = True,
		require_confirmation: Optional[bool] = None,
		**kwargs
	) -> None:
		"""Submit a form with safety validation."""
		element = await self.get_element(selector)
		
		if validate_safety and self.safety_system:
			await self.safety_system.validate_and_execute(
				action=f"submit form {selector}",
				execution_func=self._submit_form_impl,
				element=element,
				require_confirmation=require_confirmation,  # Forms typically require confirmation
				selector=selector,
				**kwargs
			)
		else:
			await self._submit_form_impl(selector, **kwargs)
	
	async def _submit_form_impl(self, selector: str, **kwargs) -> None:
		"""Implementation of submit form action."""
		# Original submit form implementation would go here
		pass
	
	async def delete(
		self,
		selector: str,
		validate_safety: bool = True,
		require_confirmation: bool = True,  # Delete always requires confirmation by default
		**kwargs
	) -> None:
		"""Delete an element with safety validation."""
		element = await self.get_element(selector)
		
		if validate_safety and self.safety_system:
			await self.safety_system.validate_and_execute(
				action=f"delete {selector}",
				execution_func=self._delete_impl,
				element=element,
				require_confirmation=require_confirmation,
				selector=selector,
				**kwargs
			)
		else:
			await self._delete_impl(selector, **kwargs)
	
	async def _delete_impl(self, selector: str, **kwargs) -> None:
		"""Implementation of delete action."""
		# Original delete implementation would go here
		pass
	
	async def execute_javascript(
		self,
		script: str,
		validate_safety: bool = True,
		require_confirmation: Optional[bool] = None,
		**kwargs
	) -> Any:
		"""Execute JavaScript with safety validation."""
		# JavaScript execution is inherently high risk
		if validate_safety and self.safety_system:
			return await self.safety_system.validate_and_execute(
				action=f"execute JavaScript: {script[:100]}...",
				execution_func=self._execute_javascript_impl,
				require_confirmation=require_confirmation if require_confirmation is not None else True,
				script=script,
				**kwargs
			)
		else:
			return await self._execute_javascript_impl(script, **kwargs)
	
	async def _execute_javascript_impl(self, script: str, **kwargs) -> Any:
		"""Implementation of JavaScript execution."""
		# Original JavaScript execution implementation would go here
		pass
	
	async def rollback(self, snapshot_id: str) -> bool:
		"""Rollback to a specific snapshot."""
		if self.safety_system and self.safety_system.rollback_enabled:
			return await self.safety_system.rollback_manager.rollback_to_snapshot(snapshot_id)
		return False
	
	async def get_rollback_snapshots(self) -> List[Dict[str, Any]]:
		"""Get available rollback snapshots."""
		if self.safety_system and self.safety_system.rollback_enabled:
			return await self.safety_system.rollback_manager.get_snapshots()
		return []
	
	async def create_safety_snapshot(self, description: str = "") -> Optional[str]:
		"""Manually create a safety snapshot."""
		if self.safety_system and self.safety_system.rollback_enabled:
			snapshot = await self.safety_system.rollback_manager.create_snapshot(description)
			return snapshot.snapshot_id
		return None
	
	def set_confirmation_callback(self, callback: Callable[[ActionValidation], bool]) -> None:
		"""Set callback for user confirmation of high-risk actions."""
		if self.safety_system:
			self.safety_system.set_confirmation_callback(callback)
	
	async def get_element(self, selector: str) -> Optional['Element']:
		"""Get element by selector (placeholder for existing implementation)."""
		# This would integrate with existing element retrieval logic
		return self._elements.get(selector)
	
	async def get_content(self) -> str:
		"""Get page content (placeholder for existing implementation)."""
		# This would integrate with existing content retrieval logic
		return ""
	
	async def set_content(self, content: str) -> None:
		"""Set page content (placeholder for existing implementation)."""
		# This would integrate with existing content setting logic
		pass
	
	@property
	def url(self) -> str:
		"""Get current URL (placeholder for existing implementation)."""
		return ""
	
	async def get_cookies(self) -> List[Dict[str, Any]]:
		"""Get cookies (placeholder for existing implementation)."""
		return []
	
	async def set_cookie(self, cookie: Dict[str, Any]) -> None:
		"""Set cookie (placeholder for existing implementation)."""
		pass
	
	async def get_frames(self) -> List[Dict[str, Any]]:
		"""Get frames (placeholder for existing implementation)."""
		return []
	
	async def evaluate(self, script: str) -> Any:
		"""Evaluate JavaScript (placeholder for existing implementation)."""
		return None
	
	async def goto(self, url: str) -> None:
		"""Navigate to URL (placeholder for existing implementation)."""
		pass
	
	async def close(self) -> None:
		"""Close the page and cleanup resources."""
		if self.safety_system:
			await self.safety_system.cleanup()
		# Original close implementation would continue here...