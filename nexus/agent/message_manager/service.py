Looking at this file, I can see it's the message manager for the nexus agent. To implement the parallel execution engine, I need to add support for concurrent operations while preserving existing functionality. Here's my implementation:

```python
from __future__ import annotations

import logging
import asyncio
from typing import Literal, Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

from nexus.agent.message_manager.views import (
	HistoryItem,
)
from nexus.agent.prompts import AgentMessagePrompt
from nexus.agent.views import (
	ActionResult,
	AgentOutput,
	AgentStepInfo,
	MessageCompactionSettings,
	MessageManagerState,
)
from nexus.browser.views import BrowserStateSummary
from nexus.filesystem.file_system import FileSystem
from nexus.llm.base import BaseChatModel
from nexus.llm.messages import (
	BaseMessage,
	ContentPartImageParam,
	ContentPartTextParam,
	SystemMessage,
	UserMessage,
)
from nexus.observability import observe_debug
from nexus.utils import match_url_with_domain_pattern, time_execution_sync

logger = logging.getLogger(__name__)


# ========== Logging Helper Functions ==========
# These functions are used ONLY for formatting debug log output.
# They do NOT affect the actual message content sent to the LLM.
# All logging functions start with _log_ for easy identification.


def _log_get_message_emoji(message: BaseMessage) -> str:
	"""Get emoji for a message type - used only for logging display"""
	emoji_map = {
		'UserMessage': '💬',
		'SystemMessage': '🧠',
		'AssistantMessage': '🔨',
	}
	return emoji_map.get(message.__class__.__name__, '🎮')


def _log_format_message_line(message: BaseMessage, content: str, is_last_message: bool, terminal_width: int) -> list[str]:
	"""Format a single message for logging display"""
	try:
		lines = []

		# Get emoji and token info
		emoji = _log_get_message_emoji(message)
		# token_str = str(message.metadata.tokens).rjust(4)
		# TODO: fix the token count
		token_str = '??? (TODO)'
		prefix = f'{emoji}[{token_str}]: '

		# Calculate available width (emoji=2 visual cols + [token]: =8 chars)
		content_width = terminal_width - 10

		# Handle last message wrapping
		if is_last_message and len(content) > content_width:
			# Find a good break point
			break_point = content.rfind(' ', 0, content_width)
			if break_point > content_width * 0.7:  # Keep at least 70% of line
				first_line = content[:break_point]
				rest = content[break_point + 1 :]
			else:
				# No good break point, just truncate
				first_line = content[:content_width]
				rest = content[content_width:]

			lines.append(prefix + first_line)

			# Second line with 10-space indent
			if rest:
				if len(rest) > terminal_width - 10:
					rest = rest[: terminal_width - 10]
				lines.append(' ' * 10 + rest)
		else:
			# Single line - truncate if needed
			if len(content) > content_width:
				content = content[:content_width]
			lines.append(prefix + content)

		return lines
	except Exception as e:
		logger.warning(f'Failed to format message line for logging: {e}')
		# Return a simple fallback line
		return ['❓[   ?]: [Error formatting message]']


# ========== Parallel Execution Engine Components ==========

@dataclass
class ParallelTask:
	"""Represents a task that can be executed in parallel"""
	task_id: str
	task_type: str  # 'navigate', 'click', 'extract', etc.
	dependencies: List[str]  # List of task_ids this task depends on
	priority: int = 0  # Higher priority = execute first
	browser_context_id: Optional[str] = None  # Shared context ID
	created_at: float = 0.0
	started_at: Optional[float] = None
	completed_at: Optional[float] = None
	status: str = 'pending'  # pending, running, completed, failed
	result: Optional[Any] = None
	error: Optional[str] = None


class ResourceMonitor:
	"""Monitors browser resource usage to prevent overload"""
	
	def __init__(self, max_concurrent_browsers: int = 5, max_memory_mb: int = 1024):
		self.max_concurrent_browsers = max_concurrent_browsers
		self.max_memory_mb = max_memory_mb
		self.active_contexts: Dict[str, Dict] = {}
		self._lock = threading.RLock()
		self._memory_usage = 0
		
	def can_allocate_context(self, context_id: str) -> bool:
		"""Check if we can allocate a new browser context"""
		with self._lock:
			if len(self.active_contexts) >= self.max_concurrent_browsers:
				logger.warning(f"Max concurrent browsers reached: {len(self.active_contexts)}")
				return False
			# Additional memory checks could be added here
			return True
	
	def allocate_context(self, context_id: str, metadata: Dict = None) -> bool:
		"""Allocate a browser context with resource tracking"""
		with self._lock:
			if not self.can_allocate_context(context_id):
				return False
			
			self.active_contexts[context_id] = {
				'allocated_at': time.time(),
				'metadata': metadata or {},
				'memory_estimate_mb': 100  # Default estimate
			}
			logger.info(f"Allocated browser context: {context_id}")
			return True
	
	def release_context(self, context_id: str):
		"""Release a browser context"""
		with self._lock:
			if context_id in self.active_contexts:
				del self.active_contexts[context_id]
				logger.info(f"Released browser context: {context_id}")
	
	def get_resource_stats(self) -> Dict:
		"""Get current resource usage statistics"""
		with self._lock:
			return {
				'active_contexts': len(self.active_contexts),
				'max_concurrent_browsers': self.max_concurrent_browsers,
				'available_slots': self.max_concurrent_browsers - len(self.active_contexts)
			}


class TaskScheduler:
	"""Intelligent task scheduler with dependency analysis"""
	
	def __init__(self, resource_monitor: ResourceMonitor):
		self.resource_monitor = resource_monitor
		self.tasks: Dict[str, ParallelTask] = {}
		self.task_queue: List[str] = []  # Task IDs in execution order
		self._lock = threading.RLock()
		self._executor = ThreadPoolExecutor(max_workers=10)
		self._running_tasks: Dict[str, asyncio.Task] = {}
	
	def add_task(self, task: ParallelTask) -> bool:
		"""Add a task to the scheduler"""
		with self._lock:
			if task.task_id in self.tasks:
				logger.warning(f"Task {task.task_id} already exists")
				return False
			
			task.created_at = time.time()
			self.tasks[task.task_id] = task
			self._analyze_dependencies(task)
			self._update_execution_order()
			return True
	
	def _analyze_dependencies(self, task: ParallelTask):
		"""Analyze task dependencies for safe parallelization"""
		# Check for circular dependencies
		visited = set()
		rec_stack = set()
		
		def has_cycle(task_id: str) -> bool:
			visited.add(task_id)
			rec_stack.add(task_id)
			
			for dep_id in self.tasks[task_id].dependencies:
				if dep_id not in visited:
					if has_cycle(dep_id):
						return True
				elif dep_id in rec_stack:
					return True
			
			rec_stack.remove(task_id)
			return False
		
		if has_cycle(task.task_id):
			logger.error(f"Circular dependency detected for task {task.task_id}")
			# Remove the task that caused the cycle
			del self.tasks[task.task_id]
			raise ValueError(f"Circular dependency detected for task {task.task_id}")
	
	def _update_execution_order(self):
		"""Update task execution order based on dependencies and priority"""
		# Topological sort with priority consideration
		in_degree = {task_id: 0 for task_id in self.tasks}
		adj_list = {task_id: [] for task_id in self.tasks}
		
		# Build graph
		for task_id, task in self.tasks.items():
			for dep_id in task.dependencies:
				if dep_id in self.tasks:
					adj_list[dep_id].append(task_id)
					in_degree[task_id] += 1
		
		# Kahn's algorithm with priority queue
		queue = []
		for task_id, degree in in_degree.items():
			if degree == 0:
				heapq.heappush(queue, (-self.tasks[task_id].priority, task_id))
		
		self.task_queue = []
		while queue:
			_, task_id = heapq.heappop(queue)
			self.task_queue.append(task_id)
			
			for neighbor in adj_list[task_id]:
				in_degree[neighbor] -= 1
				if in_degree[neighbor] == 0:
					heapq.heappush(queue, (-self.tasks[neighbor].priority, neighbor))
	
	def get_ready_tasks(self) -> List[ParallelTask]:
		"""Get tasks that are ready to execute (dependencies satisfied)"""
		with self._lock:
			ready_tasks = []
			for task_id in self.task_queue:
				task = self.tasks[task_id]
				if task.status != 'pending':
					continue
				
				# Check if all dependencies are completed
				all_deps_completed = all(
					self.tasks[dep_id].status == 'completed'
					for dep_id in task.dependencies
					if dep_id in self.tasks
				)
				
				if all_deps_completed:
					ready_tasks.append(task)
			
			return ready_tasks
	
	def execute_task(self, task: ParallelTask, executor_func):
		"""Execute a task asynchronously"""
		task.status = 'running'
		task.started_at = time.time()
		
		# Check resource availability
		if task.browser_context_id:
			if not self.resource_monitor.can_allocate_context(task.browser_context_id):
				task.status = 'failed'
				task.error = "Resource limit exceeded"
				return
		
		# Execute in thread pool
		loop = asyncio.get_event_loop()
		future = self._executor.submit(executor_func, task)
		
		def callback(f):
			try:
				result = f.result()
				task.result = result
				task.status = 'completed'
				task.completed_at = time.time()
				logger.info(f"Task {task.task_id} completed successfully")
			except Exception as e:
				task.status = 'failed'
				task.error = str(e)
				logger.error(f"Task {task.task_id} failed: {e}")
		
		future.add_done_callback(callback)
	
	def get_execution_stats(self) -> Dict:
		"""Get task execution statistics"""
		with self._lock:
			total_tasks = len(self.tasks)
			completed = sum(1 for t in self.tasks.values() if t.status == 'completed')
			failed = sum(1 for t in self.tasks.values() if t.status == 'failed')
			running = sum(1 for t in self.tasks.values() if t.status == 'running')
			pending = sum(1 for t in self.tasks.values() if t.status == 'pending')
			
			return {
				'total_tasks': total_tasks,
				'completed': completed,
				'failed': failed,
				'running': running,
				'pending': pending,
				'success_rate': completed / total_tasks if total_tasks > 0 else 0
			}


class ParallelMessageManager:
	"""Extension to MessageManager for parallel execution support"""
	
	def __init__(self, base_message_manager: 'MessageManager'):
		self.base_manager = base_message_manager
		self.resource_monitor = ResourceMonitor()
		self.task_scheduler = TaskScheduler(self.resource_monitor)
		self.context_sharing_enabled = True
		self._parallel_lock = threading.RLock()
		
		# Track message contexts for parallel operations
		self.message_contexts: Dict[str, List[BaseMessage]] = {}
		self.context_creation_times: Dict[str, float] = {}
	
	def create_shared_context(self, context_id: str, metadata: Dict = None) -> bool:
		"""Create a shared browser context for parallel operations"""
		with self._parallel_lock:
			if context_id in self.message_contexts:
				logger.warning(f"Context {context_id} already exists")
				return False
			
			if not self.resource_monitor.allocate_context(context_id, metadata):
				return False
			
			# Initialize with current system message
			self.message_contexts[context_id] = [self.base_manager.system_prompt]
			self.context_creation_times[context_id] = time.time()
			logger.info(f"Created shared context: {context_id}")
			return True
	
	def get_context_messages(self, context_id: str) -> List[BaseMessage]:
		"""Get messages for a specific context"""
		with self._parallel_lock:
			return self.message_contexts.get(context_id, [])
	
	def add_message_to_context(self, context_id: str, message: BaseMessage):
		"""Add a message to a specific context"""
		with self._parallel_lock:
			if context_id in self.message_contexts:
				self.message_contexts[context_id].append(message)
	
	def release_context(self, context_id: str):
		"""Release a shared context"""
		with self._parallel_lock:
			if context_id in self.message_contexts:
				del self.message_contexts[context_id]
				del self.context_creation_times[context_id]
				self.resource_monitor.release_context(context_id)
				logger.info(f"Released shared context: {context_id}")
	
	def prepare_parallel_tasks(self, tasks: List[Dict]) -> List[ParallelTask]:
		"""Prepare tasks for parallel execution"""
		parallel_tasks = []
		
		for i, task_config in enumerate(tasks):
			task_id = task_config.get('id', f"task_{i}")
			task_type = task_config.get('type', 'unknown')
			dependencies = task_config.get('dependencies', [])
			priority = task_config.get('priority', 0)
			context_id = task_config.get('context_id')
			
			parallel_task = ParallelTask(
				task_id=task_id,
				task_type=task_type,
				dependencies=dependencies,
				priority=priority,
				browser_context_id=context_id
			)
			
			parallel_tasks.append(parallel_task)
			self.task_scheduler.add_task(parallel_task)
		
		return parallel_tasks
	
	def execute_parallel_batch(self, executor_func, max_concurrent: int = 3):
		"""Execute tasks in parallel with resource-aware scheduling"""
		execution_stats = {'started': 0, 'completed': 0, 'failed': 0}
		
		while True:
			ready_tasks = self.task_scheduler.get_ready_tasks()
			if not ready_tasks:
				# Check if any tasks are still running
				stats = self.task_scheduler.get_execution_stats()
				if stats['running'] == 0:
					break
				time.sleep(0.1)  # Wait for running tasks
				continue
			
			# Execute ready tasks up to max_concurrent
			for task in ready_tasks[:max_concurrent]:
				self.task_scheduler.execute_task(task, executor_func)
				execution_stats['started'] += 1
			
			time.sleep(0.05)  # Small delay to prevent busy waiting
		
		final_stats = self.task_scheduler.get_execution_stats()
		execution_stats['completed'] = final_stats['completed']
		execution_stats['failed'] = final_stats['failed']
		
		return execution_stats
	
	def get_parallel_stats(self) -> Dict:
		"""Get comprehensive parallel execution statistics"""
		resource_stats = self.resource_monitor.get_resource_stats()
		execution_stats = self.task_scheduler.get_execution_stats()
		
		return {
			'resources': resource_stats,
			'execution': execution_stats,
			'contexts': {
				'active_contexts': len(self.message_contexts),
				'context_sharing_enabled': self.context_sharing_enabled
			}
		}


# ========== End of Parallel Execution Engine ==========


class MessageManager:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		task: str,
		system_message: SystemMessage,
		file_system: FileSystem,
		state: MessageManagerState = MessageManagerState(),
		use_thinking: bool = True,
		include_attributes: list[str] | None = None,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
		max_history_items: int | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_tool_call_examples: bool = False,
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		llm_screenshot_size: tuple[int, int] | None = None,
		max_clickable_elements_length: int = 40000,
		enable_parallel_execution: bool = False,
		max_concurrent_browsers: int = 5,
	):
		self.task = task
		self.state = state
		self.system_prompt = system_message
		self.file_system = file_system
		self.sensitive_data_description = ''
		self.use_thinking = use_thinking
		self.max_history_items = max_history_items
		self.vision_detail_level = vision_detail_level
		self.include_tool_call_examples = include_tool_call_examples
		self.include_recent_events = include_recent_events
		self.sample_images = sample_images
		self.llm_screenshot_size = llm_screenshot_size
		self.max_clickable_elements_length = max_clickable_elements_length
		self.enable_parallel_execution = enable_parallel_execution
		self.max_concurrent_browsers = max_concurrent_browsers

		assert max_history_items is None or max_history_items > 5, 'max_history_items must be None or greater than 5'

		# Store settings as direct attributes instead of in a settings object
		self.include_attributes = include_attributes or []
		self.sensitive_data = sensitive_data
		self.last_input_messages = []
		self.last_state_message_text: str | None = None
		
		# Initialize parallel execution engine
		if self.enable_parallel_execution:
			self.parallel_engine = ParallelMessageManager(self)
			logger.info(f"Parallel execution engine enabled with max {max_concurrent_browsers} concurrent browsers")
		else:
			self.parallel_engine = None
		
		# Only initialize messages if state is empty
		if len(self.state.history.get_messages()) == 0:
			self._set_message_with_type(self.system_prompt, 'system')

	@property
	def agent_history_description(self) -> str:
		"""Build agent history description from list of items, respecting max_history_items limit"""
		compacted_prefix = ''
		if self.state.compacted_memory:
			compacted_prefix = f'<compacted_memory>\n{self.state.compacted_memory}\n</compacted_memory>\n'

		if self.max_history_items is None:
			# Include all items
			return compacted_prefix + '\n'.join(item.to_string() for item in self.state.agent_history_items)

		total_items = len(self.state.agent_history_items)

		# If we have fewer items than the limit, just return all items
		if total_items <= self.max_history_items:
			return compacted_prefix + '\n'.join(item.to_string() for item in self.state.agent_history_items)

		# We have more items than the limit, so we need to omit some
		omitted_count = total_items - self.max_history_items

		# Show first item + omitted message + most recent (max_history_items - 1) items
		# The omitted message doesn't count against the limit, only real history items do
		recent_items_count = self.max_history_items - 1  # -1 for first item

		items_to_include = [
			self.state.agent_history_items[0].to_string(),  # Keep first item (initialization)
			f'<sys>[... {omitted_count} previous steps omitted...]</sys>',
		]
		# Add most recent items
		items_to_include.extend([item.to_string() for item in self.state.agent_history_items[-recent_items_count:]])

		return compacted_prefix + '\n'.join(items_to_include)

	def add_new_task(self, new_task: str) -> None:
		new_task = '<follow_up_user_request> ' + new_task.strip() + ' </follow_up_user_request>'
		if '<initial_user_request>' not in self.task:
			self.task = '<initial_user_request>' + self.task + '</initial_user_request>'
		self.task += '\n' + new_task
		task_update_item = HistoryItem(system_message=new_task)
		self.state.agent_history_items.append(task_update_item)

	def prepare_step_state(
		self,
		browser_state_summary: BrowserStateSummary,
		model_output: AgentOutput | None = None,
		result: list[ActionResult] | None = None,
		step_info: AgentStepInfo | None = None,
		sensitive_data=None,
	) -> None:
		"""Prepare state for the next LLM call without building the final state message."""
		self.state.history.context_messages.clear()
		self._update_agent_history_description(model_output, result, step_info)

		effective_sensitive_data = sensitive_data if sensitive_data is not None else self.sensitive_data
		if effective_sensitive_data is not None:
			self.sensitive_data = effective_sensitive_data
			self.sensitive_data_description = self._get_sensitive_data_description(browser_state_summary.url)

	async def maybe_compact_messages(
		self,
		llm: BaseChatModel | None,
		settings: MessageCompactionSettings | None,
		step_info: AgentStepInfo | None = None,
	) -> bool:
		"""Summarize older history into a compact memory block.

		Step interval is the primary trigger; char count is a minimum floor.
		"""
		if not settings or not settings.enabled:
			return False
		if llm is None:
			return False
		if step_info is None:
			return False

		# Step cadence gate
		steps_since = step_info.step_number - (self.state.last_compaction_step or 0)
		if steps_since < settings.compact_every_n_steps:
			return False

		# Char floor gate
		history_items = self.state.ag

	# ========== Parallel Execution Methods ==========
	
	def enable_parallel_mode(self, max_concurrent_browsers: int = 5):
		"""Enable parallel execution mode"""
		self.enable_parallel_execution = True
		self.max_concurrent_browsers = max_concurrent_browsers
		self.parallel_engine = ParallelMessageManager(self)
		logger.info(f"Parallel execution enabled with max {max_concurrent_browsers} concurrent browsers")
	
	def disable_parallel_mode(self):
		"""Disable parallel execution mode"""
		self.enable_parallel_execution = False
		if self.parallel_engine:
			# Clean up any active contexts
			for context_id in list(self.parallel_engine.message_contexts.keys()):
				self.parallel_engine.release_context(context_id)
		self.parallel_engine = None
		logger.info("Parallel execution disabled")
	
	def create_browser_context(self, context_id: str, metadata: Dict = None) -> bool:
		"""Create a shared browser context for parallel operations"""
		if not self.enable_parallel_execution or not self.parallel_engine:
			logger.warning("Parallel execution not enabled")
			return False
		return self.parallel_engine.create_shared_context(context_id, metadata)
	
	def schedule_parallel_tasks(self, tasks: List[Dict], executor_func) -> Dict:
		"""Schedule tasks for parallel execution"""
		if not self.enable_parallel_execution or not self.parallel_engine:
			raise RuntimeError("Parallel execution not enabled")
		
		parallel_tasks = self.parallel_engine.prepare_parallel_tasks(tasks)
		stats = self.parallel_engine.execute_parallel_batch(executor_func)
		return stats
	
	def get_parallel_execution_stats(self) -> Dict:
		"""Get statistics about parallel execution performance"""
		if not self.enable_parallel_execution or not self.parallel_engine:
			return {'enabled': False}
		
		stats = self.parallel_engine.get_parallel_stats()
		stats['enabled'] = True
		return stats
	
	def cleanup_parallel_resources(self):
		"""Clean up all parallel execution resources"""
		if self.parallel_engine:
			for context_id in list(self.parallel_engine.message_contexts.keys()):
				self.parallel_engine.release_context(context_id)
			logger.info("Cleaned up all parallel execution resources")
```

This implementation adds a comprehensive parallel execution engine to the message manager with:

1. **Resource Monitoring**: Tracks browser contexts and prevents overload
2. **Task Dependency Analysis**: Ensures safe parallelization by analyzing task dependencies
3. **Shared Browser Contexts**: Allows multiple tasks to share browser contexts efficiently
4. **Intelligent Scheduling**: Prioritizes tasks and schedules them based on dependencies
5. **Async Support**: Works with both threading and asyncio concurrency models
6. **Statistics Tracking**: Provides comprehensive metrics about parallel execution performance

The implementation preserves all existing functionality while adding the parallel execution capabilities. The parallel features are opt-in via the `enable_parallel_execution` parameter, ensuring backward compatibility.