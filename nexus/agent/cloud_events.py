import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import anyio
from bubus import BaseEvent
from pydantic import Field, field_validator
from uuid_extensions import uuid7str

MAX_STRING_LENGTH = 500000  # 100K chars ~ 25k tokens should be enough
MAX_URL_LENGTH = 100000
MAX_TASK_LENGTH = 100000
MAX_COMMENT_LENGTH = 2000
MAX_FILE_CONTENT_SIZE = 50 * 1024 * 1024  # 50MB


class UpdateAgentTaskEvent(BaseEvent):
	# Required fields for identification
	id: str  # The task ID to update
	user_id: str = Field(max_length=255)  # For authorization
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup

	# Optional fields that can be updated
	stopped: bool | None = None
	paused: bool | None = None
	done_output: str | None = Field(None, max_length=MAX_STRING_LENGTH)
	finished_at: datetime | None = None
	agent_state: dict | None = None
	user_feedback_type: str | None = Field(None, max_length=10)  # UserFeedbackType enum value as string
	user_comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	gif_url: str | None = Field(None, max_length=MAX_URL_LENGTH)
	
	# Real-time collaboration fields
	intervention_type: str | None = Field(None, max_length=50)  # 'pause', 'resume', 'correct', 'feedback'
	intervention_data: dict | None = None  # Structured data for interventions
	step_to_rewind: int | None = Field(None, ge=0)  # Step number to rewind to
	correction_actions: List[Dict[str, Any]] | None = None  # Injected actions to replace current queue
	correction_reason: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)  # Reason for correction

	@classmethod
	def from_agent(cls, agent) -> 'UpdateAgentTaskEvent':
		"""Create an UpdateAgentTaskEvent from an Agent instance"""
		if not hasattr(agent, '_task_start_time'):
			raise ValueError('Agent must have _task_start_time attribute')

		done_output = agent.history.final_result() if agent.history else None
		if done_output and len(done_output) > MAX_STRING_LENGTH:
			done_output = done_output[:MAX_STRING_LENGTH]
		return cls(
			id=str(agent.task_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			stopped=agent.state.stopped if hasattr(agent.state, 'stopped') else False,
			paused=agent.state.paused if hasattr(agent.state, 'paused') else False,
			done_output=done_output,
			finished_at=datetime.now(timezone.utc) if agent.history and agent.history.is_done() else None,
			agent_state=agent.state.model_dump() if hasattr(agent.state, 'model_dump') else {},
			user_feedback_type=None,
			user_comment=None,
			gif_url=None,
			intervention_type=None,
			intervention_data=None,
			step_to_rewind=None,
			correction_actions=None,
			correction_reason=None,
			# user_feedback_type and user_comment would be set by the API/frontend
			# gif_url would be set after GIF generation if needed
		)


class CreateAgentOutputFileEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	task_id: str
	file_name: str = Field(max_length=255)
	file_content: str | None = None  # Base64 encoded file content
	content_type: str | None = Field(None, max_length=100)  # MIME type for file uploads
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@field_validator('file_content')
	@classmethod
	def validate_file_size(cls, v: str | None) -> str | None:
		"""Validate base64 file content size."""
		if v is None:
			return v
		# Remove data URL prefix if present
		if ',' in v:
			v = v.split(',')[1]
		# Estimate decoded size (base64 is ~33% larger)
		estimated_size = len(v) * 3 / 4
		if estimated_size > MAX_FILE_CONTENT_SIZE:
			raise ValueError(f'File content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

	@classmethod
	async def from_agent_and_file(cls, agent, output_path: str) -> 'CreateAgentOutputFileEvent':
		"""Create a CreateAgentOutputFileEvent from a file path"""

		gif_path = Path(output_path)
		if not gif_path.exists():
			raise FileNotFoundError(f'File not found: {output_path}')

		gif_size = os.path.getsize(gif_path)

		# Read GIF content for base64 encoding if needed
		gif_content = None
		if gif_size < 50 * 1024 * 1024:  # Only read if < 50MB
			async with await anyio.open_file(gif_path, 'rb') as f:
				gif_bytes = await f.read()
				gif_content = base64.b64encode(gif_bytes).decode('utf-8')

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			task_id=str(agent.task_id),
			file_name=gif_path.name,
			file_content=gif_content,  # Base64 encoded
			content_type='image/gif',
		)


class CreateAgentStepEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)  # Added for authorization checks
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	agent_task_id: str
	step: int
	evaluation_previous_goal: str = Field(max_length=MAX_STRING_LENGTH)
	memory: str = Field(max_length=MAX_STRING_LENGTH)
	next_goal: str = Field(max_length=MAX_STRING_LENGTH)
	actions: list[dict]
	screenshot_url: str | None = Field(None, max_length=MAX_FILE_CONTENT_SIZE)  # ~50MB for base64 images
	url: str = Field(default='', max_length=MAX_URL_LENGTH)
	
	# Real-time collaboration fields
	action_queue: List[Dict[str, Any]] | None = None  # Full action queue for this step
	performance_metrics: Dict[str, Any] | None = None  # Step timing, token usage, etc.
	reasoning_chain: List[str] | None = None  # Step-by-step reasoning
	confidence_score: float | None = Field(None, ge=0.0, le=1.0)  # Agent's confidence in this step
	human_feedback: Dict[str, Any] | None = None  # Feedback received for this step
	is_corrected: bool = False  # Whether this step was corrected by human
	correction_applied: str | None = None  # Description of correction applied

	@field_validator('screenshot_url')
	@classmethod
	def validate_screenshot_size(cls, v: str | None) -> str | None:
		"""Validate screenshot URL or base64 content size."""
		if v is None or not v.startswith('data:'):
			return v
		# It's base64 data, check size
		if ',' in v:
			base64_part = v.split(',')[1]
			estimated_size = len(base64_part) * 3 / 4
			if estimated_size > MAX_FILE_CONTENT_SIZE:
				raise ValueError(f'Screenshot content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

	@classmethod
	def from_agent_step(
		cls, agent, model_output, result: list, actions_data: list[dict], browser_state_summary
	) -> 'CreateAgentStepEvent':
		"""Create a CreateAgentStepEvent from agent step data"""
		# Get first action details if available
		first_action = model_output.action[0] if model_output.action else None

		# Extract current state from model output
		current_state = model_output.current_state if hasattr(model_output, 'current_state') else None

		# Capture screenshot as base64 data URL if available
		screenshot_url = None
		if browser_state_summary.screenshot:
			screenshot_url = f'data:image/png;base64,{browser_state_summary.screenshot}'
			import logging

			logger = logging.getLogger(__name__)
			logger.debug(f'📸 Including screenshot in CreateAgentStepEvent, length: {len(browser_state_summary.screenshot)}')
		else:
			import logging

			logger = logging.getLogger(__name__)
			logger.debug('📸 No screenshot in browser_state_summary for CreateAgentStepEvent')

		# Extract performance metrics if available
		performance_metrics = {}
		if hasattr(agent, 'state') and hasattr(agent.state, 'step_metrics'):
			performance_metrics = agent.state.step_metrics.get(agent.state.n_steps, {})
		
		# Extract reasoning chain if available
		reasoning_chain = []
		if current_state and hasattr(current_state, 'reasoning_chain'):
			reasoning_chain = current_state.reasoning_chain

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			agent_task_id=str(agent.task_id),
			step=agent.state.n_steps,
			evaluation_previous_goal=current_state.evaluation_previous_goal if current_state else '',
			memory=current_state.memory if current_state else '',
			next_goal=current_state.next_goal if current_state else '',
			actions=actions_data,  # List of action dicts
			url=browser_state_summary.url,
			screenshot_url=screenshot_url,
			action_queue=actions_data,  # Full queue for this step
			performance_metrics=performance_metrics,
			reasoning_chain=reasoning_chain,
			confidence_score=getattr(model_output, 'confidence', None),
			human_feedback=None,  # Will be populated from feedback events
			is_corrected=False,
			correction_applied=None,
		)


class CreateAgentTaskEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)  # Added for authorization checks
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	agent_session_id: str
	llm_model: str = Field(max_length=200)  # LLMModel enum value as string
	stopped: bool = False
	paused: bool = False
	task: str = Field(max_length=MAX_TASK_LENGTH)
	done_output: str | None = Field(None, max_length=MAX_STRING_LENGTH)
	scheduled_task_id: str | None = None
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	finished_at: datetime | None = None
	agent_state: dict = Field(default_factory=dict)
	user_feedback_type: str | None = Field(None, max_length=10)  # UserFeedbackTyp


class AgentInterventionEvent(BaseEvent):
	"""Event for real-time human interventions"""
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)
	task_id: str
	intervention_type: str = Field(max_length=50)  # 'pause', 'resume', 'rewind', 'inject_action', 'provide_feedback'
	step_number: int | None = Field(None, ge=0)  # Target step for intervention
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	# Intervention-specific data
	action_to_inject: Dict[str, Any] | None = None  # For inject_action
	feedback_type: str | None = Field(None, max_length=20)  # 'positive', 'negative', 'correction'
	feedback_comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	rewind_to_step: int | None = Field(None, ge=0)  # For rewind
	correction_reason: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	
	# Metadata
	source: str = Field(default="dashboard", max_length=50)  # 'dashboard', 'api', 'mobile'
	priority: int = Field(default=0, ge=0, le=10)  # Higher priority interventions processed first


class AgentPerformanceMetricsEvent(BaseEvent):
	"""Event for streaming performance metrics"""
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)
	task_id: str
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	# Performance metrics
	step_duration_ms: float | None = None
	token_usage: Dict[str, int] | None = None  # prompt_tokens, completion_tokens, total_tokens
	action_success_rate: float | None = Field(None, ge=0.0, le=1.0)
	error_count: int = Field(default=0, ge=0)
	retry_count: int = Field(default=0, ge=0)
	
	# System metrics
	cpu_usage_percent: float | None = None
	memory_usage_mb: float | None = None
	network_latency_ms: float | None = None


class AgentLiveStateEvent(BaseEvent):
	"""Event for streaming live agent state updates"""
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)
	task_id: str
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	# Live state
	current_step: int
	is_paused: bool
	is_stopped: bool
	action_queue_length: int
	current_action: Dict[str, Any] | None = None
	next_actions_preview: List[Dict[str, Any]] | None = None
	
	# Browser state
	current_url: str | None = None
	page_title: str | None = None
	dom_element_count: int | None = None
	
	# Agent reasoning
	current_goal: str | None = None
	confidence: float | None = Field(None, ge=0.0, le=1.0)
	estimated_completion: float | None = Field(None, ge=0.0, le=1.0)  # 0-1 progress estimate


class TrainingFeedbackEvent(BaseEvent):
	"""Event for collecting training feedback to improve future performance"""
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)
	task_id: str
	step_number: int | None = Field(None, ge=0)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	
	# Feedback details
	feedback_type: str = Field(max_length=50)  # 'action_quality', 'reasoning_quality', 'goal_alignment'
	rating: int = Field(ge=1, le=5)  # 1-5 star rating
	comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	
	# Context
	original_action: Dict[str, Any] | None = None
	suggested_action: Dict[str, Any] | None = None
	original_reasoning: str | None = None
	suggested_reasoning: str | None = None
	
	# Training metadata
	tags: List[str] | None = None  # For categorizing feedback
	is_example: bool = False  # Whether this is a good example to learn from
	weight: float = Field(default=1.0, ge=0.0, le=10.0)  # Importance weight for training