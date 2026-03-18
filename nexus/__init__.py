import os
from typing import TYPE_CHECKING, Optional, Dict, Any, List
import asyncio
import logging
from enum import Enum

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


# Distributed task orchestration enums
class TaskStatus(Enum):
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	RETRYING = "retrying"


class TaskPriority(Enum):
	LOW = 0
	NORMAL = 1
	HIGH = 2
	CRITICAL = 3


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from nexus.agent.prompts import SystemPrompt
	from nexus.agent.service import Agent

	# from nexus.agent.service import Agent
	from nexus.agent.views import ActionModel, ActionResult, AgentHistoryList
	from nexus.browser import BrowserProfile, BrowserSession
	from nexus.browser import BrowserSession as Browser
	from nexus.browser.engines import BrowserEngine, ChromeEngine, FirefoxEngine, WebKitEngine
	from nexus.browser.mobile import MobileEmulator, DeviceProfile
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
	from nexus.distributed.coordinator import DistributedCoordinator
	from nexus.distributed.worker import DistributedWorker
	from nexus.distributed.task_queue import RedisTaskQueue, RabbitMQTaskQueue
	from nexus.distributed.checkpoint import TaskCheckpoint
	from nexus.distributed.models import DistributedTask, TaskResult, WorkerNode

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
	# Multi-browser engine support
	'BrowserEngine': ('nexus.browser.engines', 'BrowserEngine'),
	'ChromeEngine': ('nexus.browser.engines', 'ChromeEngine'),
	'FirefoxEngine': ('nexus.browser.engines', 'FirefoxEngine'),
	'WebKitEngine': ('nexus.browser.engines', 'WebKitEngine'),
	'MobileEmulator': ('nexus.browser.mobile', 'MobileEmulator'),
	'DeviceProfile': ('nexus.browser.mobile', 'DeviceProfile'),
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
	# Distributed orchestration components
	'DistributedCoordinator': ('nexus.distributed.coordinator', 'DistributedCoordinator'),
	'DistributedWorker': ('nexus.distributed.worker', 'DistributedWorker'),
	'RedisTaskQueue': ('nexus.distributed.task_queue', 'RedisTaskQueue'),
	'RabbitMQTaskQueue': ('nexus.distributed.task_queue', 'RabbitMQTaskQueue'),
	'TaskCheckpoint': ('nexus.distributed.checkpoint', 'TaskCheckpoint'),
	'DistributedTask': ('nexus.distributed.models', 'DistributedTask'),
	'TaskResult': ('nexus.distributed.models', 'TaskResult'),
	'WorkerNode': ('nexus.distributed.models', 'WorkerNode'),
	# Distributed Agent wrapper
	'DistributedAgent': ('nexus.distributed.agent', 'DistributedAgent'),
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


def create_browser_session(
	browser_engine: str = "chrome",
	profile: Optional['BrowserProfile'] = None,
	mobile_emulation: Optional[Dict[str, Any]] = None,
	**kwargs
) -> 'BrowserSession':
	"""
	Convenience function to create a browser session with multi-engine support.
	
	Args:
		browser_engine: Browser engine to use ("chrome", "firefox", "webkit")
		profile: Browser profile configuration
		mobile_emulation: Mobile device emulation settings
		**kwargs: Additional browser session configuration
	
	Returns:
		Configured BrowserSession instance with the specified engine
	"""
	from nexus.browser import BrowserSession
	from nexus.browser.engines import ChromeEngine, FirefoxEngine, WebKitEngine
	from nexus.browser.mobile import MobileEmulator
	
	# Map engine names to engine classes
	engine_map = {
		"chrome": ChromeEngine,
		"firefox": FirefoxEngine,
		"webkit": WebKitEngine,
	}
	
	engine_class = engine_map.get(browser_engine.lower())
	if not engine_class:
		raise ValueError(f"Unsupported browser engine: {browser_engine}. "
						 f"Supported engines: {list(engine_map.keys())}")
	
	# Create the browser engine instance
	engine = engine_class()
	
	# Apply mobile emulation if specified
	if mobile_emulation:
		emulator = MobileEmulator(**mobile_emulation)
		engine = emulator.wrap_engine(engine)
	
	# Create and return the browser session
	return BrowserSession(
		engine=engine,
		profile=profile,
		**kwargs
	)


def create_distributed_coordinator(
	backend: str = "redis",
	connection_url: Optional[str] = None,
	**kwargs
) -> 'DistributedCoordinator':
	"""
	Convenience function to create a distributed coordinator.
	
	Args:
		backend: Task queue backend ("redis" or "rabbitmq")
		connection_url: Connection URL for the backend
		**kwargs: Additional coordinator configuration
	
	Returns:
		Configured DistributedCoordinator instance
	"""
	from nexus.distributed.coordinator import DistributedCoordinator
	return DistributedCoordinator(
		backend=backend,
		connection_url=connection_url,
		**kwargs
	)


def create_distributed_worker(
	worker_id: Optional[str] = None,
	coordinator_url: Optional[str] = None,
	**kwargs
) -> 'DistributedWorker':
	"""
	Convenience function to create a distributed worker node.
	
	Args:
		worker_id: Unique identifier for the worker
		coordinator_url: URL of the coordinator to connect to
		**kwargs: Additional worker configuration
	
	Returns:
		Configured DistributedWorker instance
	"""
	from nexus.distributed.worker import DistributedWorker
	return DistributedWorker(
		worker_id=worker_id,
		coordinator_url=coordinator_url,
		**kwargs
	)