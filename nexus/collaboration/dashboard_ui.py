"""
Real-time Collaboration Dashboard for nexus.

Enables humans to observe, guide, and correct AI agents in real-time through
a web dashboard with live view, intervention controls, and training feedback loops.
"""

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from datetime import datetime

import websockets
from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from nexus.agent.service import AgentService
from nexus.agent.views import AgentState, AgentStep
from nexus.actor.page import Page


class DashboardMessageType(str, Enum):
    """Types of messages sent between dashboard and clients."""
    # Server -> Client
    BROWSER_VIEW = "browser_view"
    AGENT_REASONING = "agent_reasoning"
    ACTION_QUEUE = "action_queue"
    PERFORMANCE_METRICS = "performance_metrics"
    AGENT_STATE = "agent_state"
    INTERVENTION_REQUEST = "intervention_request"
    FEEDBACK_REQUEST = "feedback_request"
    
    # Client -> Server
    PAUSE = "pause"
    RESUME = "resume"
    REWIND = "rewind"
    INJECT_CORRECTION = "inject_correction"
    PROVIDE_FEEDBACK = "provide_feedback"
    APPROVE_ACTION = "approve_action"
    REJECT_ACTION = "reject_action"
    SET_SPEED = "set_speed"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


@dataclass
class DashboardConfig:
    """Configuration for the collaboration dashboard."""
    host: str = "localhost"
    port: int = 8765
    screenshot_interval: float = 1.0  # seconds
    max_history_steps: int = 100
    enable_feedback_loop: bool = True
    require_approval_for_actions: bool = False
    allowed_correction_types: List[str] = field(default_factory=lambda: [
        "text_input", "click", "scroll", "navigation", "wait", "custom"
    ])


@dataclass
class PerformanceMetrics:
    """Performance metrics for the agent."""
    steps_completed: int = 0
    success_rate: float = 0.0
    average_step_time: float = 0.0
    total_runtime: float = 0.0
    interventions_count: int = 0
    corrections_applied: int = 0
    feedback_positive: int = 0
    feedback_negative: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Intervention:
    """Represents a human intervention."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    type: str  # "pause", "correction", "feedback", "approval"
    content: Optional[str] = None
    applied: bool = False
    applied_at: Optional[datetime] = None


@dataclass
class TrainingFeedback:
    """Training feedback for improving agent performance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    step_index: int
    action_index: int
    is_positive: bool
    comment: Optional[str] = None
    correction: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class CollaborationDashboard:
    """
    Real-time collaboration dashboard for nexus agents.
    
    Provides WebSocket-based streaming of browser view, agent reasoning,
    action queue, and performance metrics. Allows humans to pause, rewind,
    inject corrections, and provide training feedback.
    """
    
    def __init__(
        self,
        agent_service: AgentService,
        config: Optional[DashboardConfig] = None
    ):
        self.agent_service = agent_service
        self.config = config or DashboardConfig()
        
        # WebSocket connections
        self.connections: Set[WebSocketServerProtocol] = set()
        self.connection_metadata: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}
        
        # State management
        self.is_paused = False
        self.current_speed = 1.0
        self.require_approval = self.config.require_approval_for_actions
        self.pending_approvals: Dict[str, asyncio.Future] = {}
        
        # History and tracking
        self.step_history: List[AgentStep] = []
        self.interventions: List[Intervention] = []
        self.training_feedback: List[TrainingFeedback] = []
        self.metrics = PerformanceMetrics()
        
        # Callbacks for agent integration
        self._on_step_callbacks: List[Callable[[AgentStep], Awaitable[None]]] = []
        self._on_action_callbacks: List[Callable[[Dict[str, Any]], Awaitable[bool]]] = []
        
        # Server state
        self.server: Optional[websockets.WebSocketServer] = None
        self.is_running = False
        self._screenshot_task: Optional[asyncio.Task] = None
        
        # Bind to agent service
        self._bind_to_agent()
    
    def _bind_to_agent(self) -> None:
        """Bind dashboard to agent service for real-time updates."""
        # Store original methods to wrap them
        original_execute_step = self.agent_service.execute_step
        original_think = self.agent_service.think
        
        async def wrapped_execute_step(step: AgentStep) -> None:
            """Wrap step execution to capture metrics and stream updates."""
            start_time = time.time()
            
            # Check if we need approval for this action
            if self.require_approval and step.actions:
                for i, action in enumerate(step.actions):
                    approved = await self._request_approval(step, i, action)
                    if not approved:
                        # Action was rejected, skip or modify
                        await self._handle_rejected_action(step, i, action)
                        continue
            
            # Execute the step
            await original_execute_step(step)
            
            # Update metrics
            step_time = time.time() - start_time
            self.metrics.steps_completed += 1
            self.metrics.total_runtime += step_time
            self.metrics.average_step_time = (
                (self.metrics.average_step_time * (self.metrics.steps_completed - 1) + step_time) 
                / self.metrics.steps_completed
            )
            
            # Store in history
            self.step_history.append(step)
            if len(self.step_history) > self.config.max_history_steps:
                self.step_history.pop(0)
            
            # Broadcast step completion
            await self._broadcast_step_update(step)
            
            # Call registered callbacks
            for callback in self._on_step_callbacks:
                try:
                    await callback(step)
                except Exception as e:
                    print(f"Error in step callback: {e}")
        
        async def wrapped_think() -> AgentStep:
            """Wrap thinking to stream reasoning."""
            step = await original_think()
            
            # Broadcast reasoning
            await self._broadcast({
                "type": DashboardMessageType.AGENT_REASONING,
                "data": {
                    "reasoning": step.reasoning,
                    "step_index": len(self.step_history),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            return step
        
        # Replace methods
        self.agent_service.execute_step = wrapped_execute_step
        self.agent_service.think = wrapped_think
    
    async def start(self) -> None:
        """Start the collaboration dashboard server."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start WebSocket server
        self.server = await serve(
            self._handle_connection,
            self.config.host,
            self.config.port
        )
        
        # Start screenshot streaming task
        self._screenshot_task = asyncio.create_task(self._stream_screenshots())
        
        print(f"Collaboration dashboard started on ws://{self.config.host}:{self.config.port}")
    
    async def stop(self) -> None:
        """Stop the collaboration dashboard server."""
        self.is_running = False
        
        if self._screenshot_task:
            self._screenshot_task.cancel()
            try:
                await self._screenshot_task
            except asyncio.CancelledError:
                pass
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all connections
        for connection in self.connections.copy():
            await connection.close()
        
        print("Collaboration dashboard stopped")
    
    async def _handle_connection(
        self, 
        websocket: WebSocketServerProtocol, 
        path: str
    ) -> None:
        """Handle a new WebSocket connection."""
        self.connections.add(websocket)
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now(),
            "subscriptions": set()
        }
        
        try:
            # Send initial state
            await self._send_initial_state(websocket)
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except ConnectionClosed:
            pass
        finally:
            self.connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
    
    async def _send_initial_state(self, websocket: WebSocketServerProtocol) -> None:
        """Send initial state to a newly connected client."""
        # Send current metrics
        await self._send_to_connection(websocket, {
            "type": DashboardMessageType.PERFORMANCE_METRICS,
            "data": asdict(self.metrics)
        })
        
        # Send current agent state
        if hasattr(self.agent_service, 'state'):
            await self._send_to_connection(websocket, {
                "type": DashboardMessageType.AGENT_STATE,
                "data": {
                    "state": self.agent_service.state.value,
                    "is_paused": self.is_paused,
                    "current_speed": self.current_speed,
                    "require_approval": self.require_approval
                }
            })
        
        # Send action queue if available
        if hasattr(self.agent_service, 'current_plan'):
            await self._send_to_connection(websocket, {
                "type": DashboardMessageType.ACTION_QUEUE,
                "data": {
                    "actions": self.agent_service.current_plan,
                    "current_index": getattr(self.agent_service, 'current_action_index', 0)
                }
            })
    
    async def _handle_message(
        self, 
        websocket: WebSocketServerProtocol, 
        message: str
    ) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == DashboardMessageType.PAUSE:
                await self._handle_pause()
                
            elif message_type == DashboardMessageType.RESUME:
                await self._handle_resume()
                
            elif message_type == DashboardMessageType.REWIND:
                steps = data.get("steps", 1)
                await self._handle_rewind(steps)
                
            elif message_type == DashboardMessageType.INJECT_CORRECTION:
                await self._handle_correction(data.get("correction", {}))
                
            elif message_type == DashboardMessageType.PROVIDE_FEEDBACK:
                await self._handle_feedback(data.get("feedback", {}))
                
            elif message_type == DashboardMessageType.APPROVE_ACTION:
                approval_id = data.get("approval_id")
                if approval_id in self.pending_approvals:
                    self.pending_approvals[approval_id].set_result(True)
                    
            elif message_type == DashboardMessageType.REJECT_ACTION:
                approval_id = data.get("approval_id")
                if approval_id in self.pending_approvals:
                    self.pending_approvals[approval_id].set_result(False)
                    
            elif message_type == DashboardMessageType.SET_SPEED:
                speed = data.get("speed", 1.0)
                await self._handle_set_speed(speed)
                
            elif message_type == DashboardMessageType.SUBSCRIBE:
                subscriptions = data.get("subscriptions", [])
                self.connection_metadata[websocket]["subscriptions"].update(subscriptions)
                
            elif message_type == DashboardMessageType.UNSUBSCRIBE:
                subscriptions = data.get("subscriptions", [])
                self.connection_metadata[websocket]["subscriptions"].difference_update(subscriptions)
                
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")
        except Exception as e:
            print(f"Error handling message: {e}")
    
    async def _handle_pause(self) -> None:
        """Handle pause command."""
        self.is_paused = True
        if hasattr(self.agent_service, 'pause'):
            await self.agent_service.pause()
        
        await self._broadcast({
            "type": DashboardMessageType.AGENT_STATE,
            "data": {
                "is_paused": True,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Record intervention
        self.interventions.append(Intervention(
            type="pause",
            content="Agent paused by human"
        ))
        self.metrics.interventions_count += 1
    
    async def _handle_resume(self) -> None:
        """Handle resume command."""
        self.is_paused = False
        if hasattr(self.agent_service, 'resume'):
            await self.agent_service.resume()
        
        await self._broadcast({
            "type": DashboardMessageType.AGENT_STATE,
            "data": {
                "is_paused": False,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Record intervention
        self.interventions.append(Intervention(
            type="resume",
            content="Agent resumed by human"
        ))
    
    async def _handle_rewind(self, steps: int) -> None:
        """Handle rewind command."""
        if len(self.step_history) < steps:
            steps = len(self.step_history)
        
        if steps > 0:
            # Remove last N steps from history
            rewound_steps = self.step_history[-steps:]
            self.step_history = self.step_history[:-steps]
            
            # Notify agent to rewind
            if hasattr(self.agent_service, 'rewind'):
                await self.agent_service.rewind(steps)
            
            await self._broadcast({
                "type": DashboardMessageType.INTERVENTION_REQUEST,
                "data": {
                    "type": "rewind",
                    "steps_rewound": steps,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Record intervention
            self.interventions.append(Intervention(
                type="rewind",
                content=f"Rewound {steps} steps"
            ))
            self.metrics.interventions_count += 1
    
    async def _handle_correction(self, correction: Dict[str, Any]) -> None:
        """Handle correction injection."""
        correction_type = correction.get("type")
        
        if correction_type not in self.config.allowed_correction_types:
            await self._broadcast({
                "type": "error",
                "data": {"message": f"Correction type '{correction_type}' not allowed"}
            })
            return
        
        # Apply correction to agent
        if hasattr(self.agent_service, 'apply_correction'):
            success = await self.agent_service.apply_correction(correction)
            
            if success:
                self.metrics.corrections_applied += 1
                
                await self._broadcast({
                    "type": DashboardMessageType.INTERVENTION_REQUEST,
                    "data": {
                        "type": "correction_applied",
                        "correction": correction,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # Record intervention
                self.interventions.append(Intervention(
                    type="correction",
                    content=json.dumps(correction),
                    applied=True,
                    applied_at=datetime.now()
                ))
            else:
                await self._broadcast({
                    "type": "error",
                    "data": {"message": "Failed to apply correction"}
                })
    
    async def _handle_feedback(self, feedback: Dict[str, Any]) -> None:
        """Handle training feedback."""
        if not self.config.enable_feedback_loop:
            return
        
        # Store feedback
        training_feedback = TrainingFeedback(
            step_index=feedback.get("step_index", len(self.step_history) - 1),
            action_index=feedback.get("action_index", 0),
            is_positive=feedback.get("is_positive", True),
            comment=feedback.get("comment"),
            correction=feedback.get("correction"),
            context={
                "step_history_length": len(self.step_history),
                "current_metrics": asdict(self.metrics)
            }
        )
        
        self.training_feedback.append(training_feedback)
        
        # Update metrics
        if training_feedback.is_positive:
            self.metrics.feedback_positive += 1
        else:
            self.metrics.feedback_negative += 1
        
        # Send feedback to agent for learning
        if hasattr(self.agent_service, 'process_feedback'):
            await self.agent_service.process_feedback(training_feedback)
        
        await self._broadcast({
            "type": DashboardMessageType.FEEDBACK_REQUEST,
            "data": {
                "feedback_id": training_feedback.id,
                "received": True,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def _handle_set_speed(self, speed: float) -> None:
        """Handle speed adjustment."""
        self.current_speed = max(0.1, min(5.0, speed))  # Clamp between 0.1x and 5x
        
        if hasattr(self.agent_service, 'set_speed'):
            await self.agent_service.set_speed(self.current_speed)
        
        await self._broadcast({
            "type": DashboardMessageType.AGENT_STATE,
            "data": {
                "current_speed": self.current_speed,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def _request_approval(
        self, 
        step: AgentStep, 
        action_index: int, 
        action: Dict[str, Any]
    ) -> bool:
        """Request human approval for an action."""
        approval_id = str(uuid.uuid4())
        
        # Create future for approval response
        future = asyncio.Future()
        self.pending_approvals[approval_id] = future
        
        # Broadcast approval request
        await self._broadcast({
            "type": DashboardMessageType.INTERVENTION_REQUEST,
            "data": {
                "type": "approval_required",
                "approval_id": approval_id,
                "step_index": len(self.step_history),
                "action_index": action_index,
                "action": action,
                "reasoning": step.reasoning,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        try:
            # Wait for approval with timeout
            approved = await asyncio.wait_for(future, timeout=30.0)
            return approved
        except asyncio.TimeoutError:
            # Default to approved if no response
            return True
        finally:
            self.pending_approvals.pop(approval_id, None)
    
    async def _handle_rejected_action(
        self, 
        step: AgentStep, 
        action_index: int, 
        action: Dict[str, Any]
    ) -> None:
        """Handle a rejected action."""
        # Record intervention
        self.interventions.append(Intervention(
            type="action_rejected",
            content=json.dumps({
                "step_index": len(self.step_history),
                "action_index": action_index,
                "action": action
            })
        ))
        
        await self._broadcast({
            "type": DashboardMessageType.INTERVENTION_REQUEST,
            "data": {
                "type": "action_rejected",
                "step_index": len(self.step_history),
                "action_index": action_index,
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def _stream_screenshots(self) -> None:
        """Stream browser screenshots to connected clients."""
        while self.is_running:
            try:
                # Check if we have any connections interested in browser view
                has_browser_view_subscribers = any(
                    "browser_view" in self.connection_metadata.get(conn, {}).get("subscriptions", set())
                    for conn in self.connections
                )
                
                if has_browser_view_subscribers and hasattr(self.agent_service, 'page'):
                    page: Page = self.agent_service.page
                    
                    if page and not page.is_closed():
                        # Take screenshot
                        screenshot = await page.screenshot(type="jpeg", quality=70)
                        screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
                        
                        # Broadcast to subscribers
                        await self._broadcast({
                            "type": DashboardMessageType.BROWSER_VIEW,
                            "data": {
                                "screenshot": screenshot_base64,
                                "timestamp": datetime.now().isoformat(),
                                "url": page.url
                            }
                        }, subscriptions={"browser_view"})
                
                await asyncio.sleep(self.config.screenshot_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error streaming screenshot: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _broadcast_step_update(self, step: AgentStep) -> None:
        """Broadcast step completion to all connected clients."""
        # Update metrics
        self.metrics.last_updated = datetime.now()
        
        # Calculate success rate (simple moving average)
        if self.step_history:
            recent_successes = sum(
                1 for s in self.step_history[-10:] 
                if getattr(s, 'success', True)
            )
            self.metrics.success_rate = recent_successes / min(10, len(self.step_history))
        
        # Broadcast step data
        await self._broadcast({
            "type": DashboardMessageType.AGENT_REASONING,
            "data": {
                "step_index": len(self.step_history) - 1,
                "reasoning": step.reasoning,
                "actions": step.actions,
                "success": getattr(step, 'success', True),
                "error": getattr(step, 'error', None),
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Broadcast updated metrics
        await self._broadcast({
            "type": DashboardMessageType.PERFORMANCE_METRICS,
            "data": asdict(self.metrics)
        })
        
        # Broadcast action queue if available
        if hasattr(self.agent_service, 'current_plan'):
            await self._broadcast({
                "type": DashboardMessageType.ACTION_QUEUE,
                "data": {
                    "actions": self.agent_service.current_plan,
                    "current_index": getattr(self.agent_service, 'current_action_index', 0)
                }
            })
    
    async def _broadcast(
        self, 
        message: Dict[str, Any], 
        subscriptions: Optional[Set[str]] = None
    ) -> None:
        """Broadcast message to all connected clients."""
        if not self.connections:
            return
        
        message_str = json.dumps(message)
        
        for connection in self.connections.copy():
            try:
                # Check subscriptions if specified
                if subscriptions:
                    conn_subs = self.connection_metadata.get(connection, {}).get("subscriptions", set())
                    if not subscriptions.intersection(conn_subs):
                        continue
                
                await connection.send(message_str)
            except ConnectionClosed:
                self.connections.discard(connection)
                self.connection_metadata.pop(connection, None)
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
    
    async def _send_to_connection(
        self, 
        connection: WebSocketServerProtocol, 
        message: Dict[str, Any]
    ) -> None:
        """Send message to a specific connection."""
        try:
            await connection.send(json.dumps(message))
        except ConnectionClosed:
            self.connections.discard(connection)
            self.connection_metadata.pop(connection, None)
        except Exception as e:
            print(f"Error sending to connection: {e}")
    
    def register_step_callback(
        self, 
        callback: Callable[[AgentStep], Awaitable[None]]
    ) -> None:
        """Register a callback to be called after each agent step."""
        self._on_step_callbacks.append(callback)
    
    def register_action_callback(
        self, 
        callback: Callable[[Dict[str, Any]], Awaitable[bool]]
    ) -> None:
        """Register a callback to be called before each action execution."""
        self._on_action_callbacks.append(callback)
    
    async def get_training_data(self) -> List[Dict[str, Any]]:
        """Get collected training data for improving the agent."""
        return [
            {
                "feedback": asdict(fb),
                "context": {
                    "step_history": [asdict(s) for s in self.step_history[-10:]],
                    "metrics": asdict(self.metrics),
                    "interventions": [asdict(i) for i in self.interventions[-5:]]
                }
            }
            for fb in self.training_feedback
        ]
    
    async def export_session_data(self) -> Dict[str, Any]:
        """Export complete session data for analysis."""
        return {
            "session_id": str(uuid.uuid4()),
            "start_time": datetime.now().isoformat(),
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "step_history": [asdict(s) for s in self.step_history],
            "interventions": [asdict(i) for i in self.interventions],
            "training_feedback": [asdict(fb) for fb in self.training_feedback],
            "total_connections": len(self.connections)
        }


# Integration helper functions
async def create_collaboration_dashboard(
    agent_service: AgentService,
    host: str = "localhost",
    port: int = 8765,
    **kwargs
) -> CollaborationDashboard:
    """Create and start a collaboration dashboard for an agent service."""
    config = DashboardConfig(host=host, port=port, **kwargs)
    dashboard = CollaborationDashboard(agent_service, config)
    await dashboard.start()
    return dashboard


def integrate_with_agent_service(agent_service: AgentService) -> None:
    """
    Integrate collaboration capabilities into an existing agent service.
    
    Adds collaboration methods to the agent service instance.
    """
    # Add collaboration attributes
    agent_service.collaboration_dashboard = None
    agent_service.is_paused = False
    agent_service.current_speed = 1.0
    
    # Add collaboration methods
    async def start_collaboration(
        host: str = "localhost", 
        port: int = 8765, 
        **kwargs
    ) -> CollaborationDashboard:
        """Start the collaboration dashboard for this agent."""
        dashboard = await create_collaboration_dashboard(
            agent_service, host, port, **kwargs
        )
        agent_service.collaboration_dashboard = dashboard
        return dashboard
    
    async def pause_agent() -> None:
        """Pause the agent execution."""
        agent_service.is_paused = True
        if agent_service.collaboration_dashboard:
            await agent_service.collaboration_dashboard._handle_pause()
    
    async def resume_agent() -> None:
        """Resume the agent execution."""
        agent_service.is_paused = False
        if agent_service.collaboration_dashboard:
            await agent_service.collaboration_dashboard._handle_resume()
    
    async def set_agent_speed(speed: float) -> None:
        """Set the execution speed of the agent."""
        agent_service.current_speed = max(0.1, min(5.0, speed))
        if agent_service.collaboration_dashboard:
            await agent_service.collaboration_dashboard._handle_set_speed(speed)
    
    # Attach methods to agent service
    agent_service.start_collaboration = start_collaboration
    agent_service.pause_agent = pause_agent
    agent_service.resume_agent = resume_agent
    agent_service.set_agent_speed = set_agent_speed


# Example usage
if __name__ == "__main__":
    # This would typically be integrated with an existing agent service
    print("Collaboration Dashboard Module")
    print("Import this module and use with your nexus agent service.")
    print("Example:")
    print("  from nexus.collaboration.dashboard_ui import create_collaboration_dashboard")
    print("  dashboard = await create_collaboration_dashboard(agent_service)")