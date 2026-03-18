"""
Real-time Collaboration WebSocket Server for Browser-Use
Enables humans to observe, guide, and correct AI agents in real-time.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from collections import deque
import base64
import io
from threading import Lock

import websockets
from websockets.server import WebSocketServerProtocol
from PIL import Image

from nexus.agent.service import AgentService
from nexus.agent.views import AgentState, AgentStep, AgentAction
from nexus.actor.page import Page

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types"""
    # Server -> Client
    BROWSER_VIEW = "browser_view"
    AGENT_REASONING = "agent_reasoning"
    ACTION_QUEUE = "action_queue"
    PERFORMANCE_METRICS = "performance_metrics"
    STATE_UPDATE = "state_update"
    INTERVENTION_REQUEST = "intervention_request"
    TRAINING_DATA = "training_data"
    
    # Client -> Server
    PAUSE = "pause"
    RESUME = "resume"
    REWIND = "rewind"
    INJECT_ACTION = "inject_action"
    OVERRIDE_ACTION = "override_action"
    FEEDBACK = "feedback"
    INTERVENTION_RESPONSE = "intervention_response"
    SET_SPEED = "set_speed"
    REQUEST_STATE = "request_state"


class AgentStatus(str, Enum):
    """Agent execution status"""
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_INTERVENTION = "waiting_intervention"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class CollaborationState:
    """Current state of the collaboration session"""
    session_id: str
    agent_status: AgentStatus = AgentStatus.RUNNING
    current_step: int = 0
    total_steps: int = 0
    current_action: Optional[Dict] = None
    action_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    rewind_stack: List[Dict] = field(default_factory=list)
    max_rewind_steps: int = 50


@dataclass
class InterventionRequest:
    """Request for human intervention"""
    request_id: str
    step_index: int
    reason: str
    suggested_actions: List[Dict]
    context: Dict[str, Any]
    timeout: float = 30.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingFeedback:
    """Training feedback from human"""
    step_index: int
    action_index: int
    feedback_type: str  # "positive", "negative", "correction"
    score: float
    comment: Optional[str] = None
    corrected_action: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CollaborationWebSocketServer:
    """
    WebSocket server for real-time human-AI collaboration.
    
    Features:
    - Live browser view streaming
    - Agent reasoning visualization
    - Action queue management
    - Performance metrics dashboard
    - Human intervention controls
    - Training feedback loops
    """
    
    def __init__(
        self,
        agent_service: AgentService,
        host: str = "localhost",
        port: int = 8765,
        screenshot_quality: int = 75,
        stream_fps: int = 10,
        max_history: int = 1000,
        enable_training: bool = True
    ):
        self.agent_service = agent_service
        self.host = host
        self.port = port
        self.screenshot_quality = screenshot_quality
        self.stream_fps = stream_fps
        self.max_history = max_history
        self.enable_training = enable_training
        
        # Client management
        self.clients: Set[WebSocketServerProtocol] = set()
        self.client_sessions: Dict[str, CollaborationState] = {}
        
        # State management
        self.state_lock = Lock()
        self.intervention_requests: Dict[str, InterventionRequest] = {}
        self.training_data: List[TrainingFeedback] = []
        
        # Performance tracking
        self.metrics_history = deque(maxlen=max_history)
        self.action_history = deque(maxlen=max_history)
        
        # Callbacks
        self.on_intervention_requested: Optional[Callable] = None
        self.on_feedback_received: Optional[Callable] = None
        self.on_action_injected: Optional[Callable] = None
        
        # Streaming control
        self.streaming = False
        self.stream_task: Optional[asyncio.Task] = None
        
        # Setup agent service hooks
        self._setup_agent_hooks()
        
        logger.info(f"Collaboration server initialized on {host}:{port}")
    
    def _setup_agent_hooks(self):
        """Setup hooks into the agent service for state tracking"""
        original_step = self.agent_service.step
        
        async def hooked_step(state: AgentState) -> AgentStep:
            # Store pre-step state for rewind
            self._store_rewind_point(state)
            
            # Execute original step
            step = await original_step(state)
            
            # Track step completion
            self._track_step_completion(step, state)
            
            # Check for intervention triggers
            await self._check_intervention_triggers(step, state)
            
            return step
        
        self.agent_service.step = hooked_step
        
        # Hook into action execution
        if hasattr(self.agent_service, 'execute_action'):
            original_execute = self.agent_service.execute_action
            
            async def hooked_execute(action: AgentAction, page: Page) -> Any:
                # Broadcast action about to be executed
                await self._broadcast_action_preview(action)
                
                # Check for human override
                if self._should_wait_for_override(action):
                    await self._request_intervention_for_action(action)
                
                # Execute action
                result = await original_execute(action, page)
                
                # Track execution result
                self._track_action_result(action, result)
                
                return result
            
            self.agent_service.execute_action = hooked_execute
    
    def _store_rewind_point(self, state: AgentState):
        """Store current state for potential rewind"""
        with self.state_lock:
            rewind_data = {
                "timestamp": datetime.now().isoformat(),
                "step_index": state.current_step,
                "browser_state": self._serialize_browser_state(state.page),
                "agent_state": self._serialize_agent_state(state),
                "action_history": list(self.action_history)
            }
            
            for session_id, session_state in self.client_sessions.items():
                session_state.rewind_stack.append(rewind_data)
                if len(session_state.rewind_stack) > session_state.max_rewind_steps:
                    session_state.rewind_stack.pop(0)
    
    def _track_step_completion(self, step: AgentStep, state: AgentState):
        """Track step completion and update metrics"""
        step_data = {
            "step_index": state.current_step,
            "timestamp": datetime.now().isoformat(),
            "action": step.action.to_dict() if step.action else None,
            "reasoning": step.reasoning,
            "result": step.result,
            "success": step.success,
            "duration": step.duration
        }
        
        with self.state_lock:
            self.action_history.append(step_data)
            
            # Update session states
            for session_id, session_state in self.client_sessions.items():
                session_state.current_step = state.current_step
                session_state.total_steps = state.total_steps
                session_state.current_action = step_data
                session_state.action_history.append(step_data)
                session_state.last_update = datetime.now()
                
                # Update performance metrics
                self._update_performance_metrics(session_state, step_data)
    
    async def _check_intervention_triggers(self, step: AgentStep, state: AgentState):
        """Check if human intervention should be triggered"""
        triggers = [
            step.success == False,
            step.duration > 10.0,  # Long-running action
            "error" in str(step.result).lower(),
            state.confidence < 0.5 if hasattr(state, 'confidence') else False
        ]
        
        if any(triggers):
            await self._create_intervention_request(step, state)
    
    async def _create_intervention_request(self, step: AgentStep, state: AgentState):
        """Create and broadcast intervention request"""
        request_id = str(uuid.uuid4())
        
        # Generate suggested actions based on context
        suggested_actions = self._generate_suggested_actions(step, state)
        
        request = InterventionRequest(
            request_id=request_id,
            step_index=state.current_step,
            reason=f"Step failed or requires attention: {step.reasoning}",
            suggested_actions=suggested_actions,
            context={
                "current_url": state.page.url if state.page else None,
                "step_result": step.result,
                "error": step.error if hasattr(step, 'error') else None
            }
        )
        
        with self.state_lock:
            self.intervention_requests[request_id] = request
            
            # Update session states
            for session_id, session_state in self.client_sessions.items():
                session_state.agent_status = AgentStatus.WAITING_INTERVENTION
        
        # Broadcast intervention request
        await self._broadcast({
            "type": MessageType.INTERVENTION_REQUEST,
            "data": asdict(request)
        })
        
        # Notify callback
        if self.on_intervention_requested:
            await self.on_intervention_requested(request)
    
    def _generate_suggested_actions(self, step: AgentStep, state: AgentState) -> List[Dict]:
        """Generate suggested actions for human review"""
        suggestions = []
        
        # Add retry suggestion
        suggestions.append({
            "type": "retry",
            "description": "Retry the last action",
            "action": step.action.to_dict() if step.action else None
        })
        
        # Add alternative actions based on page state
        if state.page:
            interactive_elements = state.page.get_interactive_elements()
            for element in interactive_elements[:3]:  # Top 3 elements
                suggestions.append({
                    "type": "click_element",
                    "description": f"Click on {element.get('tag', 'element')}",
                    "element": element
                })
        
        return suggestions
    
    async def _broadcast_action_preview(self, action: AgentAction):
        """Broadcast action about to be executed"""
        await self._broadcast({
            "type": "action_preview",
            "data": {
                "action": action.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def _should_wait_for_override(self, action: AgentAction) -> bool:
        """Check if action should wait for human override"""
        # High-risk actions that might need human approval
        high_risk_actions = ["navigate", "submit_form", "click_confirm", "delete"]
        return action.type in high_risk_actions
    
    async def _request_intervention_for_action(self, action: AgentAction):
        """Request intervention for specific action"""
        request_id = str(uuid.uuid4())
        
        request = InterventionRequest(
            request_id=request_id,
            step_index=0,  # Will be updated
            reason=f"High-risk action requires approval: {action.type}",
            suggested_actions=[
                {"type": "approve", "description": "Approve action execution"},
                {"type": "reject", "description": "Reject and suggest alternative"},
                {"type": "modify", "description": "Modify action parameters"}
            ],
            context={"action": action.to_dict()}
        )
        
        with self.state_lock:
            self.intervention_requests[request_id] = request
        
        await self._broadcast({
            "type": MessageType.INTERVENTION_REQUEST,
            "data": asdict(request)
        })
    
    def _track_action_result(self, action: AgentAction, result: Any):
        """Track action execution result"""
        result_data = {
            "action": action.to_dict(),
            "result": str(result),
            "timestamp": datetime.now().isoformat()
        }
        
        with self.state_lock:
            self.metrics_history.append(result_data)
    
    def _update_performance_metrics(self, session_state: CollaborationState, step_data: Dict):
        """Update performance metrics for session"""
        metrics = session_state.performance_metrics
        
        # Calculate success rate
        total_actions = len(session_state.action_history)
        successful_actions = sum(1 for a in session_state.action_history if a.get("success", False))
        metrics["success_rate"] = successful_actions / total_actions if total_actions > 0 else 0
        
        # Calculate average duration
        durations = [a.get("duration", 0) for a in session_state.action_history if "duration" in a]
        metrics["avg_duration"] = sum(durations) / len(durations) if durations else 0
        
        # Track error count
        metrics["error_count"] = sum(1 for a in session_state.action_history if not a.get("success", True))
        
        # Update timestamp
        metrics["last_updated"] = datetime.now().isoformat()
    
    def _serialize_browser_state(self, page: Optional[Page]) -> Dict:
        """Serialize browser state for transmission"""
        if not page:
            return {}
        
        try:
            screenshot = page.screenshot(quality=self.screenshot_quality)
            screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
            
            return {
                "url": page.url,
                "title": page.title,
                "screenshot": screenshot_b64,
                "viewport": page.viewport_size,
                "interactive_elements": page.get_interactive_elements()[:20],  # Limit for performance
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error serializing browser state: {e}")
            return {"error": str(e)}
    
    def _serialize_agent_state(self, state: AgentState) -> Dict:
        """Serialize agent state for transmission"""
        return {
            "current_step": state.current_step,
            "total_steps": state.total_steps,
            "goal": state.goal,
            "context": state.context,
            "memory": state.memory[-50:] if hasattr(state, 'memory') else [],  # Last 50 memories
            "confidence": state.confidence if hasattr(state, 'confidence') else None
        }
    
    async def _broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    async def _stream_browser_view(self):
        """Stream browser view to connected clients"""
        while self.streaming:
            try:
                if not self.clients:
                    await asyncio.sleep(1 / self.stream_fps)
                    continue
                
                # Get current page from agent service
                page = getattr(self.agent_service, 'current_page', None)
                if page:
                    browser_state = self._serialize_browser_state(page)
                    
                    await self._broadcast({
                        "type": MessageType.BROWSER_VIEW,
                        "data": browser_state
                    })
                
                await asyncio.sleep(1 / self.stream_fps)
                
            except Exception as e:
                logger.error(f"Error streaming browser view: {e}")
                await asyncio.sleep(1)
    
    async def _stream_agent_reasoning(self):
        """Stream agent reasoning to connected clients"""
        last_reasoning = None
        
        while self.streaming:
            try:
                # Get current reasoning from agent service
                current_reasoning = getattr(self.agent_service, 'current_reasoning', None)
                
                if current_reasoning and current_reasoning != last_reasoning:
                    await self._broadcast({
                        "type": MessageType.AGENT_REASONING,
                        "data": {
                            "reasoning": current_reasoning,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    last_reasoning = current_reasoning
                
                await asyncio.sleep(0.5)  # Check twice per second
                
            except Exception as e:
                logger.error(f"Error streaming agent reasoning: {e}")
                await asyncio.sleep(1)
    
    async def _stream_performance_metrics(self):
        """Stream performance metrics to connected clients"""
        while self.streaming:
            try:
                with self.state_lock:
                    for session_id, session_state in self.client_sessions.items():
                        metrics_data = {
                            "session_id": session_id,
                            "metrics": session_state.performance_metrics,
                            "agent_status": session_state.agent_status,
                            "current_step": session_state.current_step,
                            "total_steps": session_state.total_steps
                        }
                        
                        await self._broadcast({
                            "type": MessageType.PERFORMANCE_METRICS,
                            "data": metrics_data
                        })
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Error streaming performance metrics: {e}")
                await asyncio.sleep(1)
    
    async def _handle_client_message(self, websocket: WebSocketServerProtocol, message: Dict):
        """Handle incoming client message"""
        try:
            msg_type = message.get("type")
            data = message.get("data", {})
            
            if msg_type == MessageType.PAUSE:
                await self._handle_pause(data)
            
            elif msg_type == MessageType.RESUME:
                await self._handle_resume(data)
            
            elif msg_type == MessageType.REWIND:
                await self._handle_rewind(data)
            
            elif msg_type == MessageType.INJECT_ACTION:
                await self._handle_inject_action(data)
            
            elif msg_type == MessageType.OVERRIDE_ACTION:
                await self._handle_override_action(data)
            
            elif msg_type == MessageType.FEEDBACK:
                await self._handle_feedback(data)
            
            elif msg_type == MessageType.INTERVENTION_RESPONSE:
                await self._handle_intervention_response(data)
            
            elif msg_type == MessageType.SET_SPEED:
                await self._handle_set_speed(data)
            
            elif msg_type == MessageType.REQUEST_STATE:
                await self._handle_request_state(websocket, data)
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "data": {"message": str(e)}
            }))
    
    async def _handle_pause(self, data: Dict):
        """Handle pause request"""
        session_id = data.get("session_id")
        
        with self.state_lock:
            if session_id in self.client_sessions:
                self.client_sessions[session_id].agent_status = AgentStatus.PAUSED
        
        # Pause agent service
        if hasattr(self.agent_service, 'pause'):
            await self.agent_service.pause()
        
        await self._broadcast({
            "type": MessageType.STATE_UPDATE,
            "data": {"status": "paused", "session_id": session_id}
        })
    
    async def _handle_resume(self, data: Dict):
        """Handle resume request"""
        session_id = data.get("session_id")
        
        with self.state_lock:
            if session_id in self.client_sessions:
                self.client_sessions[session_id].agent_status = AgentStatus.RUNNING
        
        # Resume agent service
        if hasattr(self.agent_service, 'resume'):
            await self.agent_service.resume()
        
        await self._broadcast({
            "type": MessageType.STATE_UPDATE,
            "data": {"status": "running", "session_id": session_id}
        })
    
    async def _handle_rewind(self, data: Dict):
        """Handle rewind request"""
        session_id = data.get("session_id")
        steps_back = data.get("steps", 1)
        
        with self.state_lock:
            if session_id not in self.client_sessions:
                return
            
            session_state = self.client_sessions[session_id]
            
            if len(session_state.rewind_stack) < steps_back:
                await self._broadcast({
                    "type": "error",
                    "data": {"message": f"Cannot rewind {steps_back} steps, only {len(session_state.rewind_stack)} available"}
                })
                return
            
            # Get rewind point
            rewind_point = session_state.rewind_stack[-steps_back]
            
            # Restore state
            if hasattr(self.agent_service, 'restore_state'):
                await self.agent_service.restore_state(rewind_point['agent_state'])
            
            # Update session
            session_state.current_step = rewind_point['step_index']
            session_state.agent_status = AgentStatus.RUNNING
            session_state.rewind_stack = session_state.rewind_stack[:-steps_back]
        
        await self._broadcast({
            "type": MessageType.STATE_UPDATE,
            "data": {
                "status": "rewound",
                "step_index": rewind_point['step_index'],
                "session_id": session_id
            }
        })
    
    async def _handle_inject_action(self, data: Dict):
        """Handle action injection from human"""
        action_data = data.get("action")
        session_id = data.get("session_id")
        
        if not action_data:
            return
        
        # Create action from data
        action = AgentAction.from_dict(action_data)
        
        # Inject into agent service
        if hasattr(self.agent_service, 'inject_action'):
            await self.agent_service.inject_action(action)
        
        # Track injection
        if self.on_action_injected:
            await self.on_action_injected(action, session_id)
        
        await self._broadcast({
            "type": "action_injected",
            "data": {
                "action": action_data,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def _handle_override_action(self, data: Dict):
        """Handle action override from human"""
        original_action = data.get("original_action")
        new_action = data.get("new_action")
        session_id = data.get("session_id")
        
        if hasattr(self.agent_service, 'override_action'):
            await self.agent_service.override_action(original_action, new_action)
        
        await self._broadcast({
            "type": "action_overridden",
            "data": {
                "original": original_action,
                "new": new_action,
                "session_id": session_id
            }
        })
    
    async def _handle_feedback(self, data: Dict):
        """Handle training feedback from human"""
        if not self.enable_training:
            return
        
        feedback = TrainingFeedback(
            step_index=data.get("step_index", 0),
            action_index=data.get("action_index", 0),
            feedback_type=data.get("feedback_type", "neutral"),
            score=data.get("score", 0.5),
            comment=data.get("comment"),
            corrected_action=data.get("corrected_action")
        )
        
        with self.state_lock:
            self.training_data.append(feedback)
            
            # Limit training data size
            if len(self.training_data) > self.max_history:
                self.training_data = self.training_data[-self.max_history:]
        
        # Notify callback
        if self.on_feedback_received:
            await self.on_feedback_received(feedback)
        
        # Update agent with feedback
        if hasattr(self.agent_service, 'process_feedback'):
            await self.agent_service.process_feedback(feedback)
        
        await self._broadcast({
            "type": MessageType.TRAINING_DATA,
            "data": {
                "feedback": asdict(feedback),
                "total_feedback": len(self.training_data)
            }
        })
    
    async def _handle_intervention_response(self, data: Dict):
        """Handle response to intervention request"""
        request_id = data.get("request_id")
        response = data.get("response")
        
        with self.state_lock:
            if request_id not in self.intervention_requests:
                return
            
            request = self.intervention_requests[request_id]
            
            # Update session status
            for session_id, session_state in self.client_sessions.items():
                session_state.agent_status = AgentStatus.RUNNING
            
            # Process response
            if response.get("type") == "approve":
                # Continue with original or modified action
                if hasattr(self.agent_service, 'continue_execution'):
                    await self.agent_service.continue_execution(response.get("action"))
            
            elif response.get("type") == "reject":
                # Skip current action
                if hasattr(self.agent_service, 'skip_action'):
                    await self.agent_service.skip_action()
            
            elif response.get("type") == "inject":
                # Inject new action
                await self._handle_inject_action({
                    "action": response.get("action"),
                    "session_id": data.get("session_id")
                })
            
            # Clean up
            del self.intervention_requests[request_id]
        
        await self._broadcast({
            "type": "intervention_resolved",
            "data": {"request_id": request_id, "response": response}
        })
    
    async def _handle_set_speed(self, data: Dict):
        """Handle speed adjustment"""
        speed = data.get("speed", 1.0)
        
        if hasattr(self.agent_service, 'set_execution_speed'):
            await self.agent_service.set_execution_speed(speed)
        
        await self._broadcast({
            "type": "speed_updated",
            "data": {"speed": speed}
        })
    
    async def _handle_request_state(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle state request from client"""
        session_id = data.get("session_id")
        
        with self.state_lock:
            if session_id in self.client_sessions:
                session_state = self.client_sessions[session_id]
                
                state_data = {
                    "session_id": session_id,
                    "agent_status": session_state.agent_status,
                    "current_step": session_state.current_step,
                    "total_steps": session_state.total_steps,
                    "performance_metrics": session_state.performance_metrics,
                    "action_history": list(session_state.action_history)[-10:],  # Last 10 actions
                    "intervention_requests": [
                        asdict(req) for req in self.intervention_requests.values()
                    ]
                }
                
                await websocket.send(json.dumps({
                    "type": MessageType.STATE_UPDATE,
                    "data": state_data
                }))
    
    async def _client_handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection"""
        client_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        logger.info(f"New client connected: {client_id}")
        
        # Add to clients set
        self.clients.add(websocket)
        
        # Create session state
        with self.state_lock:
            self.client_sessions[session_id] = CollaborationState(
                session_id=session_id
            )
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "data": {
                    "client_id": client_id,
                    "session_id": session_id,
                    "server_time": datetime.now().isoformat(),
                    "features": {
                        "training_enabled": self.enable_training,
                        "stream_fps": self.stream_fps,
                        "screenshot_quality": self.screenshot_quality
                    }
                }
            }))
            
            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        
        finally:
            # Clean up
            self.clients.remove(websocket)
            
            with self.state_lock:
                if session_id in self.client_sessions:
                    del self.client_sessions[session_id]
    
    async def start(self):
        """Start the WebSocket server and streaming tasks"""
        logger.info(f"Starting collaboration server on ws://{self.host}:{self.port}")
        
        # Start streaming tasks
        self.streaming = True
        self.stream_task = asyncio.create_task(self._stream_browser_view())
        asyncio.create_task(self._stream_agent_reasoning())
        asyncio.create_task(self._stream_performance_metrics())
        
        # Start WebSocket server
        server = await websockets.serve(
            self._client_handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=40
        )
        
        logger.info("Collaboration server started successfully")
        
        # Keep server running
        await server.wait_closed()
    
    async def stop(self):
        """Stop the collaboration server"""
        logger.info("Stopping collaboration server")
        
        self.streaming = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        for client in self.clients:
            await client.close()
        
        self.clients.clear()
        
        logger.info("Collaboration server stopped")
    
    def get_training_data(self) -> List[Dict]:
        """Get collected training data"""
        with self.state_lock:
            return [asdict(f) for f in self.training_data]
    
    def clear_training_data(self):
        """Clear collected training data"""
        with self.state_lock:
            self.training_data.clear()
    
    def get_session_stats(self) -> Dict:
        """Get statistics about active sessions"""
        with self.state_lock:
            return {
                "active_clients": len(self.clients),
                "active_sessions": len(self.client_sessions),
                "pending_interventions": len(self.intervention_requests),
                "training_data_points": len(self.training_data),
                "metrics_history_size": len(self.metrics_history)
            }


# Convenience function for easy integration
def create_collaboration_server(
    agent_service: AgentService,
    **kwargs
) -> CollaborationWebSocketServer:
    """
    Create and return a collaboration server instance.
    
    Args:
        agent_service: The agent service to collaborate with
        **kwargs: Additional configuration options
    
    Returns:
        Configured CollaborationWebSocketServer instance
    """
    return CollaborationWebSocketServer(agent_service, **kwargs)


# Example usage
if __name__ == "__main__":
    import asyncio
    from nexus.agent.service import AgentService
    
    # This would be your actual agent service
    # agent_service = AgentService(...)
    
    # For demonstration, we'll create a mock
    class MockAgentService:
        def __init__(self):
            self.current_page = None
            self.current_reasoning = "Starting task..."
            self.current_step = 0
            self.total_steps = 10
        
        async def step(self, state):
            # Mock step implementation
            await asyncio.sleep(1)
            return AgentStep(
                action=AgentAction(type="click", target="button"),
                reasoning="Clicking button to proceed",
                result="Success",
                success=True,
                duration=0.5
            )
    
    async def main():
        # Create mock agent service
        mock_agent = MockAgentService()
        
        # Create collaboration server
        server = create_collaboration_server(
            mock_agent,
            host="localhost",
            port=8765,
            stream_fps=5,
            enable_training=True
        )
        
        # Set up callbacks
        async def on_intervention(request):
            print(f"Intervention requested: {request.reason}")
        
        async def on_feedback(feedback):
            print(f"Feedback received: {feedback.feedback_type} - {feedback.score}")
        
        server.on_intervention_requested = on_intervention
        server.on_feedback_received = on_feedback
        
        # Start server
        try:
            await server.start()
        except KeyboardInterrupt:
            await server.stop()
    
    asyncio.run(main())