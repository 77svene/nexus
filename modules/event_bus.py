"""
Cross-Extension Communication Bus for Stable Diffusion WebUI
Implements typed event system with priority queues, capability declarations, and permission controls.
"""

import heapq
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging
from modules import extensions

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event processing priority levels"""
    CRITICAL = 0    # System-level events (model loading, etc.)
    HIGH = 1        # Pre/post processing hooks
    NORMAL = 2      # Standard extension communication
    LOW = 3         # Background tasks, logging
    MONITOR = 4     # Monitoring/analytics only

class EventCategory(Enum):
    """Standard event categories for common operations"""
    MODEL = auto()          # Model loading, sharing, switching
    PROCESSING = auto()     # Pre/post processing hooks
    UI = auto()             # UI updates and interactions
    NETWORK = auto()        # Network/LoRA operations
    SAMPLING = auto()       # Sampling process events
    UPSCALING = auto()      # Upscaling operations
    CUSTOM = auto()         # Extension-specific custom events

@dataclass(order=True)
class Event:
    """Event container with priority ordering"""
    priority: int
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: EventCategory = EventCategory.CUSTOM
    event_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # Extension name that published
    target: Optional[str] = None  # Specific target extension (None for broadcast)
    requires_response: bool = False
    response_event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Subscription:
    """Subscription record with callback and filters"""
    subscription_id: str
    callback: Callable[[Event], Any]
    event_type: str
    category: Optional[EventCategory] = None
    source_filter: Optional[str] = None
    priority_threshold: EventPriority = EventPriority.MONITOR
    once: bool = False  # Auto-unsubscribe after first call
    extension_name: str = ""

class Capability(Enum):
    """Extension capability declarations"""
    PUBLISH_MODEL_EVENTS = auto()
    SUBSCRIBE_MODEL_EVENTS = auto()
    PUBLISH_PROCESSING_EVENTS = auto()
    SUBSCRIBE_PROCESSING_EVENTS = auto()
    PUBLISH_UI_EVENTS = auto()
    SUBSCRIBE_UI_EVENTS = auto()
    SHARE_MODELS = auto()
    ACCESS_SHARED_MODELS = auto()
    MODIFY_GLOBAL_STATE = auto()
    READ_GLOBAL_STATE = auto()

class EventBus:
    """
    Central event bus for cross-extension communication.
    Implements pub/sub with priority queues, capability-based permissions, and typed events.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern to ensure single bus instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        with self._lock:
            # Event queue: heap of (priority, timestamp, event)
            self._event_queue: List[Tuple[int, float, Event]] = []
            
            # Subscriptions: event_type -> list of subscriptions
            self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
            
            # Extension capabilities and permissions
            self._extension_capabilities: Dict[str, Set[Capability]] = {}
            self._extension_permissions: Dict[str, Dict[str, Set[str]]] = {}
            
            # Event history for debugging (limited size)
            self._event_history: List[Event] = []
            self._max_history = 1000
            
            # Processing thread
            self._processing_thread: Optional[threading.Thread] = None
            self._running = False
            self._process_lock = threading.Lock()
            
            # Response tracking
            self._pending_responses: Dict[str, threading.Event] = {}
            self._response_data: Dict[str, Event] = {}
            
            # Standard event type registry
            self._register_standard_events()
            
            self._initialized = True
            logger.info("EventBus initialized")
    
    def _register_standard_events(self):
        """Register standard event types for common operations"""
        self.standard_events = {
            # Model events
            "model.loading": EventCategory.MODEL,
            "model.loaded": EventCategory.MODEL,
            "model.unloading": EventCategory.MODEL,
            "model.sharing.request": EventCategory.MODEL,
            "model.sharing.response": EventCategory.MODEL,
            
            # Processing events
            "processing.pre": EventCategory.PROCESSING,
            "processing.post": EventCategory.PROCESSING,
            "processing.hijack": EventCategory.PROCESSING,
            
            # UI events
            "ui.tab.created": EventCategory.UI,
            "ui.component.updated": EventCategory.UI,
            
            # Sampling events
            "sampling.start": EventCategory.SAMPLING,
            "sampling.step": EventCategory.SAMPLING,
            "sampling.complete": EventCategory.SAMPLING,
            
            # Upscaling events
            "upscaling.start": EventCategory.UPSCALING,
            "upscaling.complete": EventCategory.UPSCALING,
            
            # Network events (LoRA, etc.)
            "network.loaded": EventCategory.NETWORK,
            "network.applied": EventCategory.NETWORK,
        }
    
    def start_processing(self):
        """Start the event processing thread"""
        with self._process_lock:
            if self._running:
                return
                
            self._running = True
            self._processing_thread = threading.Thread(
                target=self._process_events,
                daemon=True,
                name="EventBusProcessor"
            )
            self._processing_thread.start()
            logger.info("EventBus processing thread started")
    
    def stop_processing(self):
        """Stop the event processing thread"""
        with self._process_lock:
            self._running = False
            if self._processing_thread:
                self._processing_thread.join(timeout=5.0)
                self._processing_thread = None
            logger.info("EventBus processing thread stopped")
    
    def register_extension(self, extension_name: str, capabilities: List[Capability]):
        """
        Register an extension with its capabilities.
        Must be called during extension initialization.
        """
        with self._lock:
            self._extension_capabilities[extension_name] = set(capabilities)
            
            # Initialize permission matrix
            if extension_name not in self._extension_permissions:
                self._extension_permissions[extension_name] = {
                    "publish": set(),
                    "subscribe": set()
                }
            
            # Auto-grant permissions based on capabilities
            for cap in capabilities:
                if cap.name.startswith("PUBLISH_"):
                    event_category = cap.name.replace("PUBLISH_", "").lower()
                    self._extension_permissions[extension_name]["publish"].add(event_category)
                elif cap.name.startswith("SUBSCRIBE_"):
                    event_category = cap.name.replace("SUBSCRIBE_", "").lower()
                    self._extension_permissions[extension_name]["subscribe"].add(event_category)
            
            logger.info(f"Registered extension '{extension_name}' with capabilities: {[c.name for c in capabilities]}")
    
    def grant_permission(self, extension_name: str, permission_type: str, event_category: str):
        """Grant specific permission to an extension"""
        with self._lock:
            if extension_name not in self._extension_permissions:
                self._extension_permissions[extension_name] = {"publish": set(), "subscribe": set()}
            
            if permission_type in ["publish", "subscribe"]:
                self._extension_permissions[extension_name][permission_type].add(event_category)
                logger.debug(f"Granted {permission_type} permission for {event_category} to {extension_name}")
    
    def check_permission(self, extension_name: str, action: str, event_category: str) -> bool:
        """Check if extension has permission for action on event category"""
        with self._lock:
            if extension_name not in self._extension_permissions:
                return False
            
            # System extensions have all permissions
            if extension_name in ["core", "builtin"]:
                return True
            
            return event_category in self._extension_permissions[extension_name].get(action, set())
    
    def subscribe(self, 
                 callback: Callable[[Event], Any],
                 event_type: str,
                 category: Optional[EventCategory] = None,
                 source_filter: Optional[str] = None,
                 priority_threshold: EventPriority = EventPriority.MONITOR,
                 once: bool = False,
                 extension_name: Optional[str] = None) -> str:
        """
        Subscribe to events with optional filters.
        Returns subscription ID for later unsubscribe.
        """
        with self._lock:
            # Check permissions if extension_name provided
            if extension_name:
                event_cat = category or self._get_event_category(event_type)
                if not self.check_permission(extension_name, "subscribe", event_cat.name.lower()):
                    raise PermissionError(f"Extension '{extension_name}' not permitted to subscribe to {event_cat.name} events")
            
            subscription_id = str(uuid.uuid4())
            subscription = Subscription(
                subscription_id=subscription_id,
                callback=callback,
                event_type=event_type,
                category=category,
                source_filter=source_filter,
                priority_threshold=priority_threshold,
                once=once,
                extension_name=extension_name or ""
            )
            
            self._subscriptions[event_type].append(subscription)
            
            # Sort by priority (lower number = higher priority)
            self._subscriptions[event_type].sort(key=lambda s: s.priority_threshold.value)
            
            logger.debug(f"New subscription {subscription_id} for event '{event_type}'")
            return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove subscription by ID"""
        with self._lock:
            for event_type, subs in self._subscriptions.items():
                for i, sub in enumerate(subs):
                    if sub.subscription_id == subscription_id:
                        del subs[i]
                        logger.debug(f"Unsubscribed {subscription_id}")
                        return True
            return False
    
    def publish(self,
                event_type: str,
                data: Dict[str, Any],
                category: Optional[EventCategory] = None,
                priority: EventPriority = EventPriority.NORMAL,
                source: Optional[str] = None,
                target: Optional[str] = None,
                requires_response: bool = False,
                timeout: float = 5.0) -> Optional[Event]:
        """
        Publish an event to the bus.
        If requires_response=True, waits for response and returns it.
        """
        with self._lock:
            # Check permissions if source provided
            if source:
                event_cat = category or self._get_event_category(event_type)
                if not self.check_permission(source, "publish", event_cat.name.lower()):
                    raise PermissionError(f"Extension '{source}' not permitted to publish {event_cat.name} events")
            
            event = Event(
                priority=priority.value,
                category=category or self._get_event_category(event_type),
                event_type=event_type,
                data=data,
                source=source or "",
                target=target,
                requires_response=requires_response
            )
            
            # Add to queue
            heapq.heappush(self._event_queue, (event.priority, event.timestamp, event))
            
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            logger.debug(f"Published event '{event_type}' from '{source}' (priority: {priority.name})")
            
            # If response required, wait for it
            if requires_response:
                response_event = threading.Event()
                self._pending_responses[event.event_id] = response_event
                
                # Release lock while waiting
                self._lock.release()
                try:
                    if response_event.wait(timeout):
                        response = self._response_data.get(event.event_id)
                        return response
                    else:
                        logger.warning(f"Response timeout for event {event.event_id}")
                        return None
                finally:
                    self._lock.acquire()
                    # Cleanup
                    self._pending_responses.pop(event.event_id, None)
                    self._response_data.pop(event.event_id, None)
            
            return event
    
    def respond(self, original_event: Event, response_data: Dict[str, Any], source: Optional[str] = None):
        """Respond to an event that requires response"""
        with self._lock:
            if original_event.event_id in self._pending_responses:
                response_event = Event(
                    priority=original_event.priority,
                    category=original_event.category,
                    event_type=f"{original_event.event_type}.response",
                    data=response_data,
                    source=source or "",
                    target=original_event.source,
                    response_event_id=original_event.event_id
                )
                
                self._response_data[original_event.event_id] = response_event
                self._pending_responses[original_event.event_id].set()
                logger.debug(f"Sent response for event {original_event.event_id}")
    
    def _process_events(self):
        """Main event processing loop"""
        while self._running:
            try:
                with self._lock:
                    if not self._event_queue:
                        # No events, release lock and sleep
                        self._lock.release()
                        try:
                            time.sleep(0.01)  # Prevent busy waiting
                        finally:
                            self._lock.acquire()
                        continue
                    
                    # Get next event
                    _, _, event = heapq.heappop(self._event_queue)
                
                # Process outside lock to avoid deadlocks
                self._dispatch_event(event)
                
            except Exception as e:
                logger.error(f"Error in event processing: {e}", exc_info=True)
                time.sleep(0.1)  # Prevent rapid error loops
    
    def _dispatch_event(self, event: Event):
        """Dispatch event to matching subscribers"""
        matching_subs = []
        
        with self._lock:
            # Find all matching subscriptions
            for sub in self._subscriptions.get(event.event_type, []):
                # Check priority threshold
                if event.priority > sub.priority_threshold.value:
                    continue
                
                # Check category filter
                if sub.category and event.category != sub.category:
                    continue
                
                # Check source filter
                if sub.source_filter and event.source != sub.source_filter:
                    continue
                
                # Check target (if specified, only deliver to target)
                if event.target and sub.extension_name != event.target:
                    continue
                
                matching_subs.append(sub)
        
        # Call callbacks outside lock
        for sub in matching_subs:
            try:
                # Check if this is a one-time subscription
                if sub.once:
                    with self._lock:
                        self._subscriptions[event.event_type].remove(sub)
                
                # Call the callback
                sub.callback(event)
                
            except Exception as e:
                logger.error(f"Error in event callback for {event.event_type}: {e}", exc_info=True)
    
    def _get_event_category(self, event_type: str) -> EventCategory:
        """Determine category from event type"""
        if event_type in self.standard_events:
            return self.standard_events[event_type]
        
        # Infer from prefix
        prefix = event_type.split('.')[0].lower()
        category_map = {
            "model": EventCategory.MODEL,
            "processing": EventCategory.PROCESSING,
            "ui": EventCategory.UI,
            "network": EventCategory.NETWORK,
            "sampling": EventCategory.SAMPLING,
            "upscaling": EventCategory.UPSCALING,
        }
        
        return category_map.get(prefix, EventCategory.CUSTOM)
    
    def get_extension_capabilities(self, extension_name: str) -> Set[Capability]:
        """Get capabilities registered for an extension"""
        with self._lock:
            return self._extension_capabilities.get(extension_name, set())
    
    def get_event_history(self, 
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[Event]:
        """Get recent event history with optional filter"""
        with self._lock:
            history = self._event_history[-limit:] if not event_type else [
                e for e in self._event_history[-limit:] if e.event_type == event_type
            ]
            return list(reversed(history))
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
    
    def create_model_sharing_api(self):
        """Create standard API for model sharing between extensions"""
        
        def request_model(model_name: str, model_type: str, requesting_extension: str) -> Optional[Any]:
            """Request a model from another extension"""
            response = self.publish(
                event_type="model.sharing.request",
                data={
                    "model_name": model_name,
                    "model_type": model_type,
                    "requesting_extension": requesting_extension
                },
                category=EventCategory.MODEL,
                priority=EventPriority.HIGH,
                source=requesting_extension,
                requires_response=True,
                timeout=10.0
            )
            
            if response and response.data.get("model"):
                return response.data["model"]
            return None
        
        def provide_model(model_name: str, model: Any, providing_extension: str):
            """Provide a model in response to a request"""
            # This would be called by extensions that can provide models
            # They would subscribe to "model.sharing.request" events
            pass
        
        return request_model, provide_model
    
    def create_processing_hooks(self):
        """Create standard pre/post processing hooks"""
        
        def pre_process(data: Dict[str, Any], extension_name: str) -> Dict[str, Any]:
            """Run pre-processing hooks"""
            event = self.publish(
                event_type="processing.pre",
                data=data,
                category=EventCategory.PROCESSING,
                priority=EventPriority.HIGH,
                source=extension_name,
                requires_response=True,
                timeout=5.0
            )
            
            if event and event.data:
                return event.data
            return data
        
        def post_process(data: Dict[str, Any], extension_name: str) -> Dict[str, Any]:
            """Run post-processing hooks"""
            event = self.publish(
                event_type="processing.post",
                data=data,
                category=EventCategory.PROCESSING,
                priority=EventPriority.HIGH,
                source=extension_name,
                requires_response=True,
                timeout=5.0
            )
            
            if event and event.data:
                return event.data
            return data
        
        return pre_process, post_process

# Global instance
event_bus = EventBus()

# Convenience functions for common operations
def subscribe(event_type: str, callback: Callable[[Event], Any], **kwargs) -> str:
    """Global subscribe function"""
    return event_bus.subscribe(callback, event_type, **kwargs)

def publish(event_type: str, data: Dict[str, Any], **kwargs) -> Optional[Event]:
    """Global publish function"""
    return event_bus.publish(event_type, data, **kwargs)

def register_extension(extension_name: str, capabilities: List[Capability]):
    """Register extension with event bus"""
    return event_bus.register_extension(extension_name, capabilities)

# Auto-initialize when module is loaded
def initialize():
    """Initialize event bus and start processing"""
    # Register built-in extensions
    builtin_extensions = [
        "LDSR",
        "Lora",
        "ScuNET",
        "SwinIR",
        "extra-options-section",
        "prompt-bracket-checker",
    ]
    
    for ext_name in builtin_extensions:
        capabilities = [
            Capability.PUBLISH_PROCESSING_EVENTS,
            Capability.SUBSCRIBE_PROCESSING_EVENTS,
            Capability.PUBLISH_MODEL_EVENTS,
            Capability.SUBSCRIBE_MODEL_EVENTS,
        ]
        
        if ext_name == "Lora":
            capabilities.extend([
                Capability.PUBLISH_NETWORK_EVENTS,
                Capability.SUBSCRIBE_NETWORK_EVENTS,
            ])
        
        register_extension(ext_name, capabilities)
    
    # Start processing thread
    event_bus.start_processing()
    logger.info("EventBus auto-initialized with built-in extensions")

# Hook into extension loading system
original_load_extensions = extensions.load_extensions

def patched_load_extensions():
    """Patched version that registers extensions with event bus"""
    original_load_extensions()
    
    # Register all loaded extensions
    for ext in extensions.extensions:
        if ext.enabled:
            try:
                # Default capabilities for third-party extensions
                default_capabilities = [
                    Capability.PUBLISH_PROCESSING_EVENTS,
                    Capability.SUBSCRIBE_PROCESSING_EVENTS,
                    Capability.PUBLISH_UI_EVENTS,
                    Capability.SUBSCRIBE_UI_EVENTS,
                ]
                
                # Check if extension declares capabilities
                if hasattr(ext, 'event_bus_capabilities'):
                    capabilities = ext.event_bus_capabilities
                else:
                    capabilities = default_capabilities
                
                register_extension(ext.name, capabilities)
                
            except Exception as e:
                logger.error(f"Failed to register extension {ext.name} with event bus: {e}")

# Apply the patch
extensions.load_extensions = patched_load_extensions

# Initialize on module import
initialize()