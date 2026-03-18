import os
import asyncio
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import time
import json
import pickle

from nexus.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from nexus.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('nexus')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Circuit Breaker States
class CircuitState(Enum):
	CLOSED = "closed"  # Normal operation
	OPEN = "open"      # Circuit is open, failing fast
	HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
	"""Configuration for circuit breaker behavior"""
	failure_threshold: int = 5  # Number of failures before opening circuit
	reset_timeout: float = 30.0  # Seconds before trying half-open
	half_open_max_attempts: int = 1  # Max attempts in half-open state
	success_threshold: int = 2  # Successes needed to close circuit from half-open
	operation_timeout: float = 30.0  # Timeout for operations in seconds
	failure_exceptions: tuple = (Exception,)  # Exceptions that count as failures


@dataclass
class RetryConfig:
	"""Configuration for retry behavior"""
	max_retries: int = 3
	initial_delay: float = 1.0  # Initial delay in seconds
	max_delay: float = 30.0  # Maximum delay in seconds
	exponential_base: float = 2.0  # Base for exponential backoff
	jitter: bool = True  # Add randomness to delays


@dataclass
class ResiliencePolicy:
	"""Complete resilience policy for an operation type"""
	circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
	retry: RetryConfig = field(default_factory=RetryConfig)
	operation_name: str = "default"


class CircuitBreaker:
	"""Circuit breaker pattern implementation for CDP connections"""
	
	def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
		self.config = config
		self.name = name
		self.state = CircuitState.CLOSED
		self.failure_count = 0
		self.success_count = 0
		self.last_failure_time = 0
		self.half_open_attempts = 0
		self._lock = asyncio.Lock()
		logger.debug(f"Circuit breaker '{name}' initialized in {self.state.value} state")
	
	async def __call__(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
		"""Execute function through circuit breaker"""
		async with self._lock:
			if self.state == CircuitState.OPEN:
				if time.time() - self.last_failure_time >= self.config.reset_timeout:
					self.state = CircuitState.HALF_OPEN
					self.half_open_attempts = 0
					logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
				else:
					raise CircuitBreakerOpenError(
						f"Circuit breaker '{self.name}' is OPEN. "
						f"Failing fast. Last failure: {self.last_failure_time}"
					)
			
			if self.state == CircuitState.HALF_OPEN:
				if self.half_open_attempts >= self.config.half_open_max_attempts:
					self.state = CircuitState.OPEN
					self.last_failure_time = time.time()
					raise CircuitBreakerOpenError(
						f"Circuit breaker '{self.name}' HALF_OPEN attempts exceeded"
					)
				self.half_open_attempts += 1
		
		# Execute the function with timeout
		try:
			result = await asyncio.wait_for(
				func(*args, **kwargs),
				timeout=self.config.operation_timeout
			)
			await self._on_success()
			return result
		except self.config.failure_exceptions as e:
			await self._on_failure()
			raise
	
	async def _on_success(self):
		"""Handle successful execution"""
		async with self._lock:
			if self.state == CircuitState.HALF_OPEN:
				self.success_count += 1
				if self.success_count >= self.config.success_threshold:
					self.state = CircuitState.CLOSED
					self.failure_count = 0
					self.success_count = 0
					logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")
			else:
				self.failure_count = max(0, self.failure_count - 1)
	
	async def _on_failure(self):
		"""Handle failed execution"""
		async with self._lock:
			self.failure_count += 1
			self.last_failure_time = time.time()
			
			if self.state == CircuitState.HALF_OPEN:
				self.state = CircuitState.OPEN
				logger.warning(f"Circuit breaker '{self.name}' opened from HALF_OPEN after failure")
			elif self.failure_count >= self.config.failure_threshold:
				self.state = CircuitState.OPEN
				logger.warning(
					f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
				)
	
	def reset(self):
		"""Manually reset the circuit breaker"""
		self.state = CircuitState.CLOSED
		self.failure_count = 0
		self.success_count = 0
		self.half_open_attempts = 0
		logger.info(f"Circuit breaker '{self.name}' manually reset")
	
	@property
	def stats(self) -> Dict[str, Any]:
		"""Get circuit breaker statistics"""
		return {
			"name": self.name,
			"state": self.state.value,
			"failure_count": self.failure_count,
			"success_count": self.success_count,
			"last_failure_time": self.last_failure_time,
			"half_open_attempts": self.half_open_attempts,
			"config": {
				"failure_threshold": self.config.failure_threshold,
				"reset_timeout": self.config.reset_timeout,
				"operation_timeout": self.config.operation_timeout,
			}
		}


class CircuitBreakerOpenError(Exception):
	"""Raised when circuit breaker is open"""
	pass


class SessionState:
	"""Manages browser session state for persistence and recovery"""
	
	def __init__(self, session_id: str):
		self.session_id = session_id
		self.state: Dict[str, Any] = {}
		self.last_updated = time.time()
		self._lock = asyncio.Lock()
	
	async def update(self, key: str, value: Any):
		"""Update session state"""
		async with self._lock:
			self.state[key] = value
			self.last_updated = time.time()
	
	async def get(self, key: str, default: Any = None) -> Any:
		"""Get value from session state"""
		async with self._lock:
			return self.state.get(key, default)
	
	async def serialize(self) -> bytes:
		"""Serialize session state for persistence"""
		async with self._lock:
			data = {
				"session_id": self.session_id,
				"state": self.state,
				"last_updated": self.last_updated
			}
			return pickle.dumps(data)
	
	@classmethod
	async def deserialize(cls, data: bytes) -> 'SessionState':
		"""Deserialize session state"""
		try:
			parsed = pickle.loads(data)
			session = cls(parsed["session_id"])
			session.state = parsed["state"]
			session.last_updated = parsed["last_updated"]
			return session
		except Exception as e:
			logger.error(f"Failed to deserialize session state: {e}")
			raise
	
	async def clear(self):
		"""Clear session state"""
		async with self._lock:
			self.state.clear()
			self.last_updated = time.time()


class ResilienceLayer:
	"""Comprehensive resilience layer for browser operations"""
	
	def __init__(self):
		self.circuit_breakers: Dict[str, CircuitBreaker] = {}
		self.retry_configs: Dict[str, RetryConfig] = {}
		self.sessions: Dict[str, SessionState] = {}
		self.default_policy = ResiliencePolicy()
		self.health_check_interval = 60  # seconds
		self._health_check_task: Optional[asyncio.Task] = None
		self._browser_processes: Dict[str, Any] = {}  # Track browser processes
	
	def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
		"""Get or create circuit breaker for operation"""
		if operation_name not in self.circuit_breakers:
			config = self.default_policy.circuit_breaker
			self.circuit_breakers[operation_name] = CircuitBreaker(config, operation_name)
		return self.circuit_breakers[operation_name]
	
	def get_retry_config(self, operation_name: str) -> RetryConfig:
		"""Get retry configuration for operation"""
		return self.retry_configs.get(operation_name, self.default_policy.retry)
	
	async def with_resilience(
		self,
		operation_name: str,
		func: Callable[..., Awaitable[Any]],
		*args,
		session_id: Optional[str] = None,
		**kwargs
	) -> Any:
		"""Execute function with full resilience (circuit breaker + retry)"""
		circuit_breaker = self.get_circuit_breaker(operation_name)
		retry_config = self.get_retry_config(operation_name)
		
		last_exception = None
		
		for attempt in range(retry_config.max_retries + 1):
			try:
				# Execute through circuit breaker
				return await circuit_breaker(func, *args, **kwargs)
			except CircuitBreakerOpenError as e:
				# Circuit is open, fail fast
				logger.warning(f"Operation '{operation_name}' circuit breaker open: {e}")
				raise
			except Exception as e:
				last_exception = e
				
				if attempt == retry_config.max_retries:
					logger.error(
						f"Operation '{operation_name}' failed after {attempt + 1} attempts: {e}"
					)
					raise
				
				# Calculate delay with exponential backoff
				delay = min(
					retry_config.initial_delay * (retry_config.exponential_base ** attempt),
					retry_config.max_delay
				)
				
				if retry_config.jitter:
					import random
					delay *= (0.5 + random.random())
				
				logger.warning(
					f"Operation '{operation_name}' attempt {attempt + 1} failed: {e}. "
					f"Retrying in {delay:.2f}s"
				)
				
				await asyncio.sleep(delay)
		
		# This should never be reached, but just in case
		raise last_exception or Exception(f"Operation '{operation_name}' failed")
	
	async def create_session(self, session_id: str) -> SessionState:
		"""Create a new session for state persistence"""
		session = SessionState(session_id)
		self.sessions[session_id] = session
		logger.info(f"Created session: {session_id}")
		return session
	
	async def get_session(self, session_id: str) -> Optional[SessionState]:
		"""Get existing session"""
		return self.sessions.get(session_id)
	
	async def save_session(self, session_id: str, filepath: str):
		"""Save session to file"""
		session = self.sessions.get(session_id)
		if session:
			data = await session.serialize()
			with open(filepath, 'wb') as f:
				f.write(data)
			logger.debug(f"Saved session {session_id} to {filepath}")
	
	async def load_session(self, filepath: str) -> SessionState:
		"""Load session from file"""
		with open(filepath, 'rb') as f:
			data = f.read()
		session = await SessionState.deserialize(data)
		self.sessions[session.session_id] = session
		logger.info(f"Loaded session {session.session_id} from {filepath}")
		return session
	
	async def restart_browser_process(self, process_id: str):
		"""Restart a browser process (placeholder for actual implementation)"""
		logger.warning(f"Restarting browser process: {process_id}")
		# This would be implemented based on the actual browser management
		# For now, just log the action
		pass
	
	async def start_health_checks(self):
		"""Start periodic health checks"""
		if self._health_check_task is None:
			self._health_check_task = asyncio.create_task(self._health_check_loop())
			logger.info("Started health check loop")
	
	async def stop_health_checks(self):
		"""Stop health checks"""
		if self._health_check_task:
			self._health_check_task.cancel()
			try:
				await self._health_check_task
			except asyncio.CancelledError:
				pass
			self._health_check_task = None
			logger.info("Stopped health check loop")
	
	async def _health_check_loop(self):
		"""Periodic health check loop"""
		while True:
			try:
				await asyncio.sleep(self.health_check_interval)
				await self._perform_health_checks()
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Health check error: {e}")
	
	async def _perform_health_checks(self):
		"""Perform health checks on all circuit breakers and sessions"""
		logger.debug("Performing health checks")
		
		# Check circuit breakers
		for name, cb in self.circuit_breakers.items():
			if cb.state == CircuitState.OPEN:
				logger.warning(f"Circuit breaker '{name}' is OPEN")
		
		# Check sessions
		for session_id, session in self.sessions.items():
			if time.time() - session.last_updated > 3600:  # 1 hour
				logger.warning(f"Session {session_id} hasn't been updated in over an hour")
	
	def configure_policy(self, operation_name: str, policy: ResiliencePolicy):
		"""Configure resilience policy for specific operation"""
		self.retry_configs[operation_name] = policy.retry
		if operation_name not in self.circuit_breakers:
			self.circuit_breakers[operation_name] = CircuitBreaker(
				policy.circuit_breaker, operation_name
			)
		else:
			# Update existing circuit breaker config
			self.circuit_breakers[operation_name].config = policy.circuit_breaker
	
	def get_all_stats(self) -> Dict[str, Any]:
		"""Get statistics for all circuit breakers"""
		return {
			"circuit_breakers": {
				name: cb.stats for name, cb in self.circuit_breakers.items()
			},
			"sessions": {
				session_id: {
					"last_updated": session.last_updated,
					"state_keys": list(session.state.keys())
				}
				for session_id, session in self.sessions.items()
			}
		}


# Global resilience layer instance
_resilience_layer = ResilienceLayer()


def get_resilience_layer() -> ResilienceLayer:
	"""Get the global resilience layer instance"""
	return _resilience_layer


def resilient_operation(operation_name: str):
	"""Decorator for making operations resilient"""
	def decorator(func: Callable[..., Awaitable[Any]]):
		@wraps(func)
		async def wrapper(*args, **kwargs):
			return await _resilience_layer.with_resilience(
				operation_name, func, *args, **kwargs
			)
		return wrapper
	return decorator


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from nexus.agent.prompts import SystemPrompt
	from nexus.agent.service import Agent

	# from nexus.agent.service import Agent
	from nexus.agent.views import ActionModel, ActionResult, AgentHistoryList
	from nexus.browser import BrowserProfile, BrowserSession
	from nexus.browser import BrowserSession as Browser
	from nexus.code_use.service import CodeAgent
	from nexus.dom.service import DomService
	from nexus.llm import models
	from nexus.llm.anthropic.chat import ChatAnthropic
	from nexus.llm.azure.chat import ChatAzureOpenAI
	from nexus.llm.nexus.chat import ChatBrowserUse
	from nexus.llm.google.chat import ChatGoogle
	from nexus.llm.groq.chat import ChatGroq
	from nexus.llm.litellm.chat import ChatLiteLLM
	from nexus.llm.mistral.chat import ChatMistral
	from nexus.llm.oci_raw.chat import ChatOCIRaw
	from nexus.llm.ollama.chat import ChatOllama
	from nexus.llm.openai.chat import ChatOpenAI
	from nexus.llm.vercel.chat import ChatVercel
	from nexus.sandbox import sandbox
	from nexus.tools.service import Controller, Tools

	# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	# Agent service (heavy due to dependencies)
	# 'Agent': ('nexus.agent.service', 'Agent'),
	# Code-use agent (Jupyter notebook-like execution)
	'CodeAgent': ('nexus.code_use.service', 'CodeAgent'),
	'Agent': ('nexus.agent.service', 'Agent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('nexus.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('nexus.agent.views', 'ActionModel'),
	'ActionResult': ('nexus.agent.views', 'ActionResult'),
	'AgentHistoryList': ('nexus.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('nexus.browser', 'BrowserSession'),
	'Browser': ('nexus.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('nexus.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('nexus.tools.service', 'Tools'),
	'Controller': ('nexus.tools.service', 'Controller'),  # alias
	# DOM service (moderate weight)
	'DomService': ('nexus.dom.service', 'DomService'),
	# Chat models (very heavy imports)
	'ChatOpenAI': ('nexus.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('nexus.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('nexus.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatBrowserUse': ('nexus.llm.nexus.chat', 'ChatBrowserUse'),
	'ChatGroq': ('nexus.llm.groq.chat', 'ChatGroq'),
	'ChatLiteLLM': ('nexus.llm.litellm.chat', 'ChatLiteLLM'),
	'ChatMistral': ('nexus.llm.mistral.chat', 'ChatMistral'),
	'ChatAzureOpenAI': ('nexus.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOCIRaw': ('nexus.llm.oci_raw.chat', 'ChatOCIRaw'),
	'ChatOllama': ('nexus.llm.ollama.chat', 'ChatOllama'),
	'ChatVercel': ('nexus.llm.vercel.chat', 'ChatVercel'),
	# LLM models module
	'models': ('nexus.llm.models', None),
	# Sandbox execution
	'sandbox': ('nexus.sandbox', 'sandbox'),
}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'CodeAgent',
	# 'CodeAgent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatBrowserUse',
	'ChatGroq',
	'ChatLiteLLM',
	'ChatMistral',
	'ChatAzureOpenAI',
	'ChatOCIRaw',
	'ChatOllama',
	'ChatVercel',
	'Tools',
	'Controller',
	# LLM models module
	'models',
	# Sandbox execution
	'sandbox',
	# Resilience components
	'CircuitBreaker',
	'CircuitBreakerOpenError',
	'CircuitState',
	'CircuitBreakerConfig',
	'RetryConfig',
	'ResiliencePolicy',
	'ResilienceLayer',
	'SessionState',
	'get_resilience_layer',
	'resilient_operation',
]