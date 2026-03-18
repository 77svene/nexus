import os
from typing import TYPE_CHECKING

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
	from nexus.farm import BrowserFarm, FarmConfig, FarmWorker
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
	# Distributed browser farm
	'BrowserFarm': ('nexus.farm', 'BrowserFarm'),
	'FarmConfig': ('nexus.farm', 'FarmConfig'),
	'FarmWorker': ('nexus.farm', 'FarmWorker'),
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
	# Distributed browser farm
	'BrowserFarm',
	'FarmConfig',
	'FarmWorker',
]
```

```python
# nexus/farm.py
"""
Distributed Browser Farm Implementation

Provides horizontal scaling of browser instances across local or cloud resources
with automatic load balancing, session persistence, and fault tolerance.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import json
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

from nexus.browser import BrowserSession, BrowserProfile
from nexus.agent.service import Agent

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
	"""Status of a farm worker"""
	IDLE = "idle"
	BUSY = "busy"
	HEALTHY = "healthy"
	UNHEALTHY = "unhealthy"
	OFFLINE = "offline"


class LoadBalancingStrategy(Enum):
	"""Load balancing strategies for distributing tasks"""
	ROUND_ROBIN = "round_robin"
	LEAST_CONNECTIONS = "least_connections"
	RESOURCE_BASED = "resource_based"
	RANDOM = "random"


@dataclass
class FarmConfig:
	"""Configuration for the browser farm"""
	# Worker pool configuration
	min_workers: int = 1
	max_workers: int = 10
	worker_timeout: int = 30  # seconds
	health_check_interval: int = 10  # seconds
	
	# Load balancing
	load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
	
	# Session persistence
	session_persistence: bool = True
	session_timeout: int = 300  # seconds
	
	# Cloud provider configuration (optional)
	cloud_provider: Optional[str] = None  # "aws", "gcp", "azure"
	cloud_config: Dict[str, Any] = field(default_factory=dict)
	
	# Local worker configuration
	local_worker_count: int = 3
	headless: bool = True
	
	# Fault tolerance
	max_retries: int = 3
	retry_delay: float = 1.0  # seconds
	circuit_breaker_threshold: int = 5
	circuit_breaker_timeout: int = 60  # seconds


@dataclass
class FarmWorker:
	"""Represents a single browser worker in the farm"""
	id: str
	status: WorkerStatus = WorkerStatus.IDLE
	browser_session: Optional[BrowserSession] = None
	current_task_id: Optional[str] = None
	last_health_check: float = field(default_factory=time.time)
	failure_count: int = 0
	created_at: float = field(default_factory=time.time)
	
	# Resource metrics
	cpu_usage: float = 0.0
	memory_usage: float = 0.0
	active_connections: int = 0
	
	# Cloud-specific attributes
	cloud_instance_id: Optional[str] = None
	cloud_region: Optional[str] = None


@dataclass
class FarmTask:
	"""Represents a task to be executed by the farm"""
	id: str
	priority: int = 0
	agent_config: Dict[str, Any] = field(default_factory=dict)
	browser_profile: Optional[BrowserProfile] = None
	created_at: float = field(default_factory=time.time)
	started_at: Optional[float] = None
	completed_at: Optional[float] = None
	worker_id: Optional[str] = None
	result: Any = None
	error: Optional[str] = None
	retry_count: int = 0


class BrowserFarm:
	"""
	Master-worker architecture for distributing browser tasks across multiple instances.
	
	Supports both local browser pools and cloud-based scaling with automatic
	load balancing, session persistence, and fault tolerance.
	"""
	
	def __init__(self, config: Optional[FarmConfig] = None):
		self.config = config or FarmConfig()
		self.workers: Dict[str, FarmWorker] = {}
		self.task_queue: asyncio.Queue[FarmTask] = asyncio.Queue()
		self.active_tasks: Dict[str, FarmTask] = {}
		self.completed_tasks: Dict[str, FarmTask] = {}
		
		# Load balancing state
		self._round_robin_index = 0
		self._worker_weights: Dict[str, float] = {}
		
		# Circuit breaker state
		self._circuit_open: bool = False
		self._circuit_opened_at: Optional[float] = None
		self._failure_count: int = 0
		
		# Session persistence
		self._sessions: Dict[str, Dict[str, Any]] = {}
		
		# Control flags
		self._running = False
		self._health_check_task: Optional[asyncio.Task] = None
		self._task_distributor_task: Optional[asyncio.Task] = None
		self._worker_monitor_task: Optional[asyncio.Task] = None
		
		# Thread pool for synchronous operations
		self._thread_pool = ThreadPoolExecutor(max_workers=4)
		
		# Event loop reference
		self._loop: Optional[asyncio.AbstractEventLoop] = None
	
	async def start(self):
		"""Start the browser farm"""
		if self._running:
			logger.warning("Browser farm is already running")
			return
		
		self._running = True
		self._loop = asyncio.get_running_loop()
		
		# Initialize workers
		await self._initialize_workers()
		
		# Start background tasks
		self._health_check_task = asyncio.create_task(self._health_check_loop())
		self._task_distributor_task = asyncio.create_task(self._task_distributor_loop())
		self._worker_monitor_task = asyncio.create_task(self._worker_monitor_loop())
		
		logger.info(f"Browser farm started with {len(self.workers)} workers")
	
	async def stop(self):
		"""Stop the browser farm and clean up resources"""
		self._running = False
		
		# Cancel background tasks
		if self._health_check_task:
			self._health_check_task.cancel()
		if self._task_distributor_task:
			self._task_distributor_task.cancel()
		if self._worker_monitor_task:
			self._worker_monitor_task.cancel()
		
		# Wait for tasks to complete
		await asyncio.sleep(0.1)
		
		# Close all browser sessions
		for worker in self.workers.values():
			if worker.browser_session:
				try:
					await worker.browser_session.close()
				except Exception as e:
					logger.error(f"Error closing worker {worker.id}: {e}")
		
		# Shutdown thread pool
		self._thread_pool.shutdown(wait=False)
		
		logger.info("Browser farm stopped")
	
	async def _initialize_workers(self):
		"""Initialize the initial pool of workers"""
		if self.config.cloud_provider:
			await self._initialize_cloud_workers()
		else:
			await self._initialize_local_workers()
	
	async def _initialize_local_workers(self):
		"""Initialize local browser workers"""
		for i in range(self.config.local_worker_count):
			worker_id = f"local-worker-{uuid.uuid4().hex[:8]}"
			
			try:
				# Create browser session
				browser_session = BrowserSession(
					headless=self.config.headless,
					profile=BrowserProfile()
				)
				await browser_session.start()
				
				# Create worker
				worker = FarmWorker(
					id=worker_id,
					status=WorkerStatus.IDLE,
					browser_session=browser_session
				)
				
				self.workers[worker_id] = worker
				logger.info(f"Initialized local worker: {worker_id}")
				
			except Exception as e:
				logger.error(f"Failed to initialize local worker {worker_id}: {e}")
	
	async def _initialize_cloud_workers(self):
		"""Initialize cloud-based workers"""
		if self.config.cloud_provider == "aws":
			await self._initialize_aws_workers()
		elif self.config.cloud_provider == "gcp":
			await self._initialize_gcp_workers()
		elif self.config.cloud_provider == "azure":
			await self._initialize_azure_workers()
		else:
			raise ValueError(f"Unsupported cloud provider: {self.config.cloud_provider}")
	
	async def _initialize_aws_workers(self):
		"""Initialize AWS-based browser workers using EC2 or ECS"""
		# This is a placeholder - actual implementation would use boto3
		logger.info("Initializing AWS workers (placeholder implementation)")
		
		# In a real implementation, you would:
		# 1. Launch EC2 instances or ECS tasks with browser containers
		# 2. Wait for instances to be ready
		# 3. Create FarmWorker objects with cloud_instance_id
		# 4. Set up SSH/remote connections to the browsers
		
		# For now, create mock cloud workers
		for i in range(self.config.local_worker_count):
			worker_id = f"aws-worker-{uuid.uuid4().hex[:8]}"
			worker = FarmWorker(
				id=worker_id,
				status=WorkerStatus.IDLE,
				cloud_instance_id=f"i-{uuid.uuid4().hex[:12]}",
				cloud_region=self.config.cloud_config.get("region", "us-east-1")
			)
			self.workers[worker_id] = worker
	
	async def _initialize_gcp_workers(self):
		"""Initialize GCP-based browser workers using Compute Engine or Cloud Run"""
		logger.info("Initializing GCP workers (placeholder implementation)")
		
		# Similar to AWS implementation
		for i in range(self.config.local_worker_count):
			worker_id = f"gcp-worker-{uuid.uuid4().hex[:8]}"
			worker = FarmWorker(
				id=worker_id,
				status=WorkerStatus.IDLE,
				cloud_instance_id=f"gcp-{uuid.uuid4().hex[:12]}",
				cloud_region=self.config.cloud_config.get("region", "us-central1")
			)
			self.workers[worker_id] = worker
	
	async def _initialize_azure_workers(self):
		"""Initialize Azure-based browser workers"""
		logger.info("Initializing Azure workers (placeholder implementation)")
		
		for i in range(self.config.local_worker_count):
			worker_id = f"azure-worker-{uuid.uuid4().hex[:8]}"
			worker = FarmWorker(
				id=worker_id,
				status=WorkerStatus.IDLE,
				cloud_instance_id=f"azure-{uuid.uuid4().hex[:12]}",
				cloud_region=self.config.cloud_config.get("region", "eastus")
			)
			self.workers[worker_id] = worker
	
	async def submit_task(self, 
						  agent_config: Dict[str, Any],
						  browser_profile: Optional[BrowserProfile] = None,
						  priority: int = 0) -> str:
		"""
		Submit a task to the browser farm.
		
		Args:
			agent_config: Configuration for the agent (LLM, tools, etc.)
			browser_profile: Optional browser profile configuration
			priority: Task priority (higher = more important)
			
		Returns:
			Task ID for tracking
		"""
		task_id = f"task-{uuid.uuid4().hex[:12]}"
		
		task = FarmTask(
			id=task_id,
			priority=priority,
			agent_config=agent_config,
			browser_profile=browser_profile
		)
		
		# Add to queue
		await self.task_queue.put(task)
		self.active_tasks[task_id] = task
		
		logger.info(f"Task {task_id} submitted to farm")
		return task_id
	
	async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
		"""
		Get the result of a submitted task.
		
		Args:
			task_id: ID of the task
			timeout: Maximum time to wait for result (None = wait indefinitely)
			
		Returns:
			Task result
			
		Raises:
			TimeoutError: If timeout is reached
			KeyError: If task not found
		"""
		start_time = time.time()
		
		while True:
			# Check if task is completed
			if task_id in self.completed_tasks:
				task = self.completed_tasks[task_id]
				if task.error:
					raise RuntimeError(f"Task failed: {task.error}")
				return task.result
			
			# Check timeout
			if timeout is not None and (time.time() - start_time) > timeout:
				raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
			
			# Wait before checking again
			await asyncio.sleep(0.1)
	
	async def _task_distributor_loop(self):
		"""Main loop for distributing tasks to workers"""
		while self._running:
			try:
				# Check circuit breaker
				if self._circuit_open:
					if time.time() - self._circuit_opened_at > self.config.circuit_breaker_timeout:
						logger.info("Circuit breaker reset")
						self._circuit_open = False
						self._failure_count = 0
					else:
						await asyncio.sleep(1)
						continue
				
				# Get next task
				try:
					task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
				except asyncio.TimeoutError:
					continue
				
				# Find available worker
				worker = await self._select_worker()
				if not worker:
					# No workers available, put task back in queue
					await self.task_queue.put(task)
					await asyncio.sleep(0.1)
					continue
				
				# Assign task to worker
				await self._assign_task_to_worker(task, worker)
				
			except Exception as e:
				logger.error(f"Error in task distributor loop: {e}")
				await asyncio.sleep(1)
	
	async def _select_worker(self) -> Optional[FarmWorker]:
		"""Select a worker based on load balancing strategy"""
		available_workers = [
			w for w in self.workers.values()
			if w.status == WorkerStatus.IDLE and w.failure_count < self.config.circuit_breaker_threshold
		]
		
		if not available_workers:
			return None
		
		if self.config.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
			return self._select_worker_round_robin(available_workers)
		elif self.config.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
			return self._select_worker_least_connections(available_workers)
		elif self.config.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_BASED:
			return self._select_worker_resource_based(available_workers)
		else:  # RANDOM
			import random
			return random.choice(available_workers)
	
	def _select_worker_round_robin(self, workers: List[FarmWorker]) -> FarmWorker:
		"""Select worker using round-robin strategy"""
		if not workers:
			raise ValueError("No workers available")
		
		worker = workers[self._round_robin_index % len(workers)]
		self._round_robin_index += 1
		return worker
	
	def _select_worker_least_connections(self, workers: List[FarmWorker]) -> FarmWorker:
		"""Select worker with least active connections"""
		return min(workers, key=lambda w: w.active_connections)
	
	def _select_worker_resource_based(self, workers: List[FarmWorker]) -> FarmWorker:
		"""Select worker based on resource usage"""
		# Simple scoring: lower CPU and memory usage is better
		def score(worker: FarmWorker) -> float:
			return worker.cpu_usage * 0.6 + worker.memory_usage * 0.4
		
		return min(workers, key=score)
	
	async def _assign_task_to_worker(self, task: FarmTask, worker: FarmWorker):
		"""Assign a task to a specific worker"""
		task.worker_id = worker.id
		task.started_at = time.time()
		
		worker.status = WorkerStatus.BUSY
		worker.current_task_id = task.id
		worker.active_connections += 1
		
		# Execute task in background
		asyncio.create_task(self._execute_task(task, worker))
		
		logger.info(f"Task {task.id} assigned to worker {worker.id}")
	
	async def _execute_task(self, task: FarmTask, worker: FarmWorker):
		"""Execute a task on a worker"""
		try:
			# Check if we need to restore a session
			session_key = None
			if self.config.session_persistence:
				session_key = self._get_session_key(task.agent_config)
				if session_key in self._sessions:
					# Restore previous session state
					logger.info(f"Restoring session {session_key} for task {task.id}")
			
			# Create or reuse browser session
			if not worker.browser_session:
				worker.browser_session = BrowserSession(
					headless=self.config.headless,
					profile=task.browser_profile or BrowserProfile()
				)
				await worker.browser_session.start()
			
			# Create agent with the worker's browser session
			agent = Agent(
				browser_session=worker.browser_session,
				**task.agent_config
			)
			
			# Execute the agent
			result = await agent.run()
			
			# Save session state if persistence is enabled
			if self.config.session_persistence and session_key:
				self._sessions[session_key] = {
					"last_used": time.time(),
					"worker_id": worker.id,
					"task_id": task.id
				}
			
			# Store result
			task.result = result
			task.completed_at = time.time()
			
			# Move to completed tasks
			self.completed_tasks[task.id] = task
			if task.id in self.active_tasks:
				del self.active_tasks[task.id]
			
			logger.info(f"Task {task.id} completed successfully on worker {worker.id}")
			
		except Exception as e:
			logger.error(f"Task {task.id} failed on worker {worker.id}: {e}")
			
			# Handle failure
			task.error = str(e)
			task.retry_count += 1
			
			if task.retry_count <= self.config.max_retries:
				# Retry the task
				logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
				await asyncio.sleep(self.config.retry_delay * task.retry_count)
				await self.task_queue.put(task)
			else:
				# Mark as failed
				task.completed_at = time.time()
				self.completed_tasks[task.id] = task
				if task.id in self.active_tasks:
					del self.active_tasks[task.id]
				
				# Update circuit breaker
				self._failure_count += 1
				if self._failure_count >= self.config.circuit_breaker_threshold:
					self._circuit_open = True
					self._circuit_opened_at = time.time()
					logger.warning("Circuit breaker opened due to high failure rate")
		
		finally:
			# Update worker status
			worker.status = WorkerStatus.IDLE
			worker.current_task_id = None
			worker.active_connections = max(0, worker.active_connections - 1)
			
			# Update failure count based on task outcome
			if task.error:
				worker.failure_count += 1
			else:
				worker.failure_count = max(0, worker.failure_count - 1)  # Decay failure count
	
	def _get_session_key(self, agent_config: Dict[str, Any]) -> str:
		"""Generate a session key from agent configuration"""
		# Use a hash of relevant config for session persistence
		import hashlib
		config_str = json.dumps(agent_config, sort_keys=True)
		return hashlib.md5(config_str.encode()).hexdigest()
	
	async def _health_check_loop(self):
		"""Periodically check health of all workers"""
		while self._running:
			try:
				for worker in list(self.workers.values()):
					await self._check_worker_health(worker)
				
				# Clean up old sessions
				if self.config.session_persistence:
					self._cleanup_old_sessions()
				
			except Exception as e:
				logger.error(f"Error in health check loop: {e}")
			
			await asyncio.sleep(self.config.health_check_interval)
	
	async def _check_worker_health(self, worker: FarmWorker):
		"""Check health of a single worker"""
		try:
			if worker.status == WorkerStatus.OFFLINE:
				# Try to recover offline worker
				if time.time() - worker.last_health_check > self.config.worker_timeout:
					await self._recover_worker(worker)
				return
			
			# Update last health check time
			worker.last_health_check = time.time()
			
			# Check browser session health
			if worker.browser_session:
				try:
					# Simple health check: try to get browser title
					await worker.browser_session.get_current_page_title()
					worker.status = WorkerStatus.HEALTHY
				except Exception:
					worker.status = WorkerStatus.UNHEALTHY
					worker.failure_count += 1
					
					# Try to restart browser session
					if worker.failure_count >= 3:
						await self._restart_worker_browser(worker)
			else:
				# No browser session, mark as unhealthy
				worker.status = WorkerStatus.UNHEALTHY
			
			# Update resource metrics (simulated)
			worker.cpu_usage = min(1.0, worker.cpu_usage * 0.9 + 0.1 * (1 if worker.status == WorkerStatus.BUSY else 0))
			worker.memory_usage = min(1.0, worker.memory_usage * 0.95 + 0.05)
			
		except Exception as e:
			logger.error(f"Error checking health of worker {worker.id}: {e}")
			worker.status = WorkerStatus.UNHEALTHY
	
	async def _recover_worker(self, worker: FarmWorker):
		"""Attempt to recover an offline worker"""
		logger.info(f"Attempting to recover worker {worker.id}")
		
		try:
			if worker.cloud_instance_id and self.config.cloud_provider:
				# Cloud worker recovery
				await self._recover_cloud_worker(worker)
			else:
				# Local worker recovery
				await self._recover_local_worker(worker)
			
			worker.status = WorkerStatus.IDLE
			worker.failure_count = 0
			logger.info(f"Worker {worker.id} recovered successfully")
			
		except Exception as e:
			logger.error(f"Failed to recover worker {worker.id}: {e}")
			worker.status = WorkerStatus.OFFLINE
	
	async def _recover_cloud_worker(self, worker: FarmWorker):
		"""Recover a cloud-based worker"""
		# This is a placeholder - actual implementation would:
		# 1. Check if cloud instance is still running
		# 2. Restart instance if needed
		# 3. Re-establish connection
		# 4. Create new browser session
		
		# For now, simulate recovery
		await asyncio.sleep(2)
		
		# Create new browser session
		worker.browser_session = BrowserSession(
			headless=self.config.headless,
			profile=BrowserProfile()
		)
		await worker.browser_session.start()
	
	async def _recover_local_worker(self, worker: FarmWorker):
		"""Recover a local worker"""
		# Close existing browser session if any
		if worker.browser_session:
			try:
				await worker.browser_session.close()
			except:
				pass
		
		# Create new browser session
		worker.browser_session = BrowserSession(
			headless=self.config.headless,
			profile=BrowserProfile()
		)
		await worker.browser_session.start()
	
	async def _restart_worker_browser(self, worker: FarmWorker):
		"""Restart a worker's browser session"""
		logger.info(f"Restarting browser for worker {worker.id}")
		
		try:
			# Close existing session
			if worker.browser_session:
				await worker.browser_session.close()
			
			# Create new session
			worker.browser_session = BrowserSession(
				headless=self.config.headless,
				profile=BrowserProfile()
			)
			await worker.browser_session.start()
			
			worker.failure_count = 0
			worker.status = WorkerStatus.IDLE
			
		except Exception as e:
			logger.error(f"Failed to restart browser for worker {worker.id}: {e}")
			worker.status = WorkerStatus.OFFLINE
	
	def _cleanup_old_sessions(self):
		"""Clean up old session data"""
		current_time = time.time()
		expired_sessions = []
		
		for session_key, session_data in self._sessions.items():
			if current_time - session_data["last_used"] > self.config.session_timeout:
				expired_sessions.append(session_key)
		
		for session_key in expired_sessions:
			del self._sessions[session_key]
		
		if expired_sessions:
			logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
	
	async def _worker_monitor_loop(self):
		"""Monitor worker pool and scale up/down as needed"""
		while self._running:
			try:
				# Check if we need more workers
				idle_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.IDLE)
				busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
				
				# Scale up if queue is backing up and we have capacity
				if (self.task_queue.qsize() > idle_workers * 2 and 
					len(self.workers) < self.config.max_workers):
					await self._scale_up()
				
				# Scale down if we have too many idle workers
				elif (idle_workers > self.config.min_workers and 
					  idle_workers > busy_workers * 2):
					await self._scale_down()
				
				# Check for unhealthy workers
				unhealthy_workers = [
					w for w in self.workers.values() 
					if w.status == WorkerStatus.UNHEALTHY and 
					time.time() - w.last_health_check > self.config.worker_timeout * 2
				]
				
				for worker in unhealthy_workers:
					logger.warning(f"Removing unhealthy worker {worker.id}")
					await self._remove_worker(worker)
				
			except Exception as e:
				logger.error(f"Error in worker monitor loop: {e}")
			
			await asyncio.sleep(5)  # Check every 5 seconds
	
	async def _scale_up(self):
		"""Add more workers to the pool"""
		workers_to_add = min(
			3,  # Add up to 3 workers at a time
			self.config.max_workers - len(self.workers)
		)
		
		if workers_to_add <= 0:
			return
		
		logger.info(f"Scaling up: adding {workers_to_add} workers")
		
		for i in range(workers_to_add):
			worker_id = f"worker-{uuid.uuid4().hex[:8]}"
			
			try:
				if self.config.cloud_provider:
					# Add cloud worker
					worker = FarmWorker(
						id=worker_id,
						status=WorkerStatus.IDLE,
						cloud_instance_id=f"{self.config.cloud_provider}-{uuid.uuid4().hex[:12]}",
						cloud_region=self.config.cloud_config.get("region", "default")
					)
				else:
					# Add local worker
					browser_session = BrowserSession(
						headless=self.config.headless,
						profile=BrowserProfile()
					)
					await browser_session.start()
					
					worker = FarmWorker(
						id=worker_id,
						status=WorkerStatus.IDLE,
						browser_session=browser_session
					)
				
				self.workers[worker_id] = worker
				logger.info(f"Added worker {worker_id}")
				
			except Exception as e:
				logger.error(f"Failed to add worker {worker_id}: {e}")
	
	async def _scale_down(self):
		"""Remove excess idle workers"""
		idle_workers = [
			w for w in self.workers.values() 
			if w.status == WorkerStatus.IDLE
		]
		
		# Keep at least min_workers
		workers_to_remove = max(0, len(idle_workers) - self.config.min_workers)
		
		if workers_to_remove <= 0:
			return
		
		logger.info(f"Scaling down: removing {workers_to_remove} workers")
		
		# Remove oldest idle workers first
		idle_workers.sort(key=lambda w: w.created_at)
		
		for i in range(min(workers_to_remove, len(idle_workers))):
			worker = idle_workers[i]
			await self._remove_worker(worker)
	
	async def _remove_worker(self, worker: FarmWorker):
		"""Remove a worker from the pool"""
		try:
			# Close browser session
			if worker.browser_session:
				await worker.browser_session.close()
			
			# Remove from workers dict
			if worker.id in self.workers:
				del self.workers[worker.id]
			
			logger.info(f"Removed worker {worker.id}")
			
		except Exception as e:
			logger.error(f"Error removing worker {worker.id}: {e}")
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get farm statistics"""
		idle_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.IDLE)
		busy_workers = sum(1 for w in self.workers.values() if w.status == WorkerStatus.BUSY)
		healthy_workers = sum(1 for w in self.workers.values() if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY])
		
		return {
			"total_workers": len(self.workers),
			"idle_workers": idle_workers,
			"busy_workers": busy_workers,
			"healthy_workers": healthy_workers,
			"queue_size": self.task_queue.qsize(),
			"active_tasks": len(self.active_tasks),
			"completed_tasks": len(self.completed_tasks),
			"sessions": len(self._sessions),
			"circuit_open": self._circuit_open,
			"uptime": time.time() - (min(w.created_at for w in self.workers.values()) if self.workers else time.time())
		}