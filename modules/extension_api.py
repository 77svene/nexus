"""
Cross-Extension Communication Bus for nexus
Pub/sub system enabling modular extension communication with typed events,
priority queues, capability declarations, and permission system.
"""

import threading
import queue
import time
import logging
import inspect
import weakref
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Callable, Type, Union, Tuple
from collections import defaultdict
import functools
import json

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Standard event types
class EventType(Enum):
    """Standard event types for cross-extension communication"""
    # Image processing pipeline
    PRE_PROCESS_IMAGE = auto()
    POST_PROCESS_IMAGE = auto()
    PRE_PROCESS_LATENT = auto()
    POST_PROCESS_LATENT = auto()
    
    # Model operations
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()
    MODEL_LAYER_MODIFIED = auto()
    
    # Sampling events
    SAMPLING_START = auto()
    SAMPLING_STEP = auto()
    SAMPLING_END = auto()
    
    # UI events
    UI_TAB_CREATED = auto()
    UI_ELEMENT_CREATED = auto()
    UI_SETTINGS_CHANGED = auto()
    
    # Extension lifecycle
    EXTENSION_LOADED = auto()
    EXTENSION_UNLOADED = auto()
    EXTENSION_ERROR = auto()
    
    # Custom events (extensions can define their own)
    CUSTOM = auto()

class EventPriority(Enum):
    """Priority levels for event handlers"""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100

@dataclass
class Event:
    """Event object containing type, data, and metadata"""
    event_type: Union[EventType, str]
    data: Dict[str, Any] = field(default_factory=dict)
    source_extension: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    propagation_stopped: bool = False
    
    def stop_propagation(self):
        """Stop event from reaching lower priority handlers"""
        self.propagation_stopped = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_type": self.event_type.name if isinstance(self.event_type, EventType) else self.event_type,
            "data": self.data,
            "source_extension": self.source_extension,
            "timestamp": self.timestamp,
            "priority": self.priority.value
        }

@dataclass
class EventHandler:
    """Registered event handler with metadata"""
    callback: Callable[[Event], Any]
    extension_name: str
    priority: EventPriority = EventPriority.NORMAL
    is_async: bool = False
    description: str = ""
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value < other.priority.value

class Capability(Enum):
    """Standard capabilities extensions can declare"""
    IMAGE_PROCESSING = auto()
    LATENT_PROCESSING = auto()
    MODEL_MODIFICATION = auto()
    SAMPLING_INTERVENTION = auto()
    UI_EXTENSION = auto()
    CUSTOM_MODEL_SUPPORT = auto()
    PREPROCESSING = auto()
    POSTPROCESSING = auto()
    SHARED_STATE = auto()

@dataclass
class ExtensionCapability:
    """Capability declaration with metadata"""
    capability: Capability
    version: str = "1.0"
    description: str = ""
    required_permissions: Set[str] = field(default_factory=set)
    
class Permission(Enum):
    """Permission levels for cross-extension access"""
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    ADMIN = auto()

class ExtensionAPI:
    """Main API class for cross-extension communication"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._event_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._extension_capabilities: Dict[str, Set[Capability]] = defaultdict(set)
        self._extension_permissions: Dict[str, Dict[str, Set[Permission]]] = defaultdict(lambda: defaultdict(set))
        self._extension_registry: Dict[str, weakref.ref] = {}
        self._event_queue = queue.PriorityQueue()
        self._processing_thread = None
        self._running = False
        self._shared_state: Dict[str, Any] = {}
        self._state_lock = threading.RLock()
        
        # Start event processing thread
        self.start_processing()
        
        logger.info("ExtensionAPI initialized")
    
    def start_processing(self):
        """Start the event processing thread"""
        if self._running:
            return
            
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._process_events,
            name="ExtensionAPI-EventProcessor",
            daemon=True
        )
        self._processing_thread.start()
    
    def stop_processing(self):
        """Stop the event processing thread"""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
    
    def _process_events(self):
        """Background thread for processing events"""
        while self._running:
            try:
                # Get event with timeout to allow checking _running flag
                priority, timestamp, event = self._event_queue.get(timeout=0.1)
                self._dispatch_event(event)
                self._event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
    
    def register_extension(self, extension_name: str, extension_module=None) -> bool:
        """Register an extension with the API"""
        with self._lock:
            if extension_name in self._extension_registry:
                logger.warning(f"Extension '{extension_name}' already registered")
                return False
            
            if extension_module:
                self._extension_registry[extension_name] = weakref.ref(extension_module)
            
            # Publish extension loaded event
            self.publish_event(Event(
                event_type=EventType.EXTENSION_LOADED,
                data={"extension_name": extension_name},
                source_extension="extension_api"
            ))
            
            logger.info(f"Extension '{extension_name}' registered with API")
            return True
    
    def unregister_extension(self, extension_name: str) -> bool:
        """Unregister an extension from the API"""
        with self._lock:
            if extension_name not in self._extension_registry:
                return False
            
            # Remove all handlers for this extension
            for event_type in list(self._event_handlers.keys()):
                self._event_handlers[event_type] = [
                    h for h in self._event_handlers[event_type] 
                    if h.extension_name != extension_name
                ]
            
            # Clean up capabilities and permissions
            self._extension_capabilities.pop(extension_name, None)
            self._extension_permissions.pop(extension_name, None)
            self._extension_registry.pop(extension_name, None)
            
            # Publish extension unloaded event
            self.publish_event(Event(
                event_type=EventType.EXTENSION_UNLOADED,
                data={"extension_name": extension_name},
                source_extension="extension_api"
            ))
            
            logger.info(f"Extension '{extension_name}' unregistered from API")
            return True
    
    def declare_capabilities(self, extension_name: str, capabilities: List[ExtensionCapability]):
        """Declare capabilities provided by an extension"""
        with self._lock:
            for cap in capabilities:
                self._extension_capabilities[extension_name].add(cap.capability)
                logger.debug(f"Extension '{extension_name}' declared capability: {cap.capability.name}")
    
    def request_permissions(self, extension_name: str, target_extension: str, 
                          permissions: Set[Permission], capability: Capability) -> bool:
        """Request permissions to access another extension's capabilities"""
        with self._lock:
            # Check if target extension has the capability
            if capability not in self._extension_capabilities.get(target_extension, set()):
                logger.warning(f"Extension '{target_extension}' does not have capability {capability.name}")
                return False
            
            # Grant permissions
            self._extension_permissions[target_extension][extension_name].update(permissions)
            logger.info(f"Granted {len(permissions)} permission(s) to '{extension_name}' for '{target_extension}'")
            return True
    
    def check_permission(self, extension_name: str, target_extension: str, 
                        permission: Permission) -> bool:
        """Check if an extension has permission to access another extension"""
        with self._lock:
            return permission in self._extension_permissions.get(target_extension, {}).get(extension_name, set())
    
    def subscribe(self, event_type: Union[EventType, str], callback: Callable[[Event], Any],
                 extension_name: str, priority: EventPriority = EventPriority.NORMAL,
                 description: str = "") -> bool:
        """Subscribe to an event type"""
        with self._lock:
            event_key = event_type.name if isinstance(event_type, EventType) else event_type
            
            handler = EventHandler(
                callback=callback,
                extension_name=extension_name,
                priority=priority,
                description=description
            )
            
            # Insert in sorted order by priority
            handlers = self._event_handlers[event_key]
            handlers.append(handler)
            handlers.sort()
            
            logger.debug(f"Extension '{extension_name}' subscribed to '{event_key}'")
            return True
    
    def unsubscribe(self, event_type: Union[EventType, str], callback: Callable[[Event], Any],
                   extension_name: str) -> bool:
        """Unsubscribe from an event type"""
        with self._lock:
            event_key = event_type.name if isinstance(event_type, EventType) else event_type
            
            if event_key not in self._event_handlers:
                return False
            
            handlers = self._event_handlers[event_key]
            original_count = len(handlers)
            
            # Remove matching handlers
            self._event_handlers[event_key] = [
                h for h in handlers 
                if not (h.callback == callback and h.extension_name == extension_name)
            ]
            
            removed = original_count - len(self._event_handlers[event_key])
            if removed > 0:
                logger.debug(f"Extension '{extension_name}' unsubscribed from '{event_key}'")
                return True
            return False
    
    def publish_event(self, event: Event, synchronous: bool = False) -> bool:
        """Publish an event to all subscribers"""
        event_key = event.event_type.name if isinstance(event.event_type, EventType) else event.event_type
        
        if synchronous:
            return self._dispatch_event(event)
        else:
            # Add to priority queue (lower priority value = higher priority)
            self._event_queue.put((
                event.priority.value,
                event.timestamp,
                event
            ))
            return True
    
    def _dispatch_event(self, event: Event) -> bool:
        """Dispatch event to all handlers"""
        event_key = event.event_type.name if isinstance(event.event_type, EventType) else event.event_type
        
        if event_key not in self._event_handlers:
            return True
        
        handlers = self._event_handlers[event_key].copy()  # Copy to avoid modification during iteration
        
        for handler in handlers:
            if event.propagation_stopped:
                break
            
            try:
                # Check if handler's extension has permission to handle this event
                if event.source_extension and event.source_extension != handler.extension_name:
                    # For now, allow all handlers. Extensions can implement their own permission checks.
                    pass
                
                # Execute handler
                if handler.is_async:
                    # Run in thread pool for async handlers
                    threading.Thread(
                        target=handler.callback,
                        args=(event,),
                        daemon=True
                    ).start()
                else:
                    handler.callback(event)
                    
            except Exception as e:
                logger.error(f"Error in event handler '{handler.extension_name}': {e}", exc_info=True)
                
                # Publish error event
                error_event = Event(
                    event_type=EventType.EXTENSION_ERROR,
                    data={
                        "error": str(e),
                        "handler_extension": handler.extension_name,
                        "event_type": event_key
                    },
                    source_extension="extension_api"
                )
                # Don't use queue for error events to avoid recursion
                self._dispatch_event(error_event)
        
        return True
    
    def get_extension_capabilities(self, extension_name: str) -> Set[Capability]:
        """Get capabilities declared by an extension"""
        return self._extension_capabilities.get(extension_name, set()).copy()
    
    def get_extensions_with_capability(self, capability: Capability) -> List[str]:
        """Get all extensions that have a specific capability"""
        return [
            ext_name for ext_name, caps in self._extension_capabilities.items()
            if capability in caps
        ]
    
    def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get value from shared state"""
        with self._state_lock:
            return self._shared_state.get(key, default)
    
    def set_shared_state(self, key: str, value: Any, extension_name: str = "") -> bool:
        """Set value in shared state"""
        with self._state_lock:
            self._shared_state[key] = value
            
            # Publish state change event
            self.publish_event(Event(
                event_type="shared_state_changed",
                data={"key": key, "value": value, "extension": extension_name},
                source_extension=extension_name or "extension_api"
            ))
            return True
    
    def call_extension_method(self, extension_name: str, method_name: str, 
                            *args, **kwargs) -> Any:
        """Call a method on another extension (if permitted)"""
        with self._lock:
            ref = self._extension_registry.get(extension_name)
            if not ref:
                raise ValueError(f"Extension '{extension_name}' not registered")
            
            extension_module = ref()
            if not extension_module:
                raise ValueError(f"Extension '{extension_name}' module no longer exists")
            
            if not hasattr(extension_module, method_name):
                raise AttributeError(f"Extension '{extension_name}' has no method '{method_name}'")
            
            method = getattr(extension_module, method_name)
            
            # Check if caller has execute permission
            # For now, we'll allow all calls. Extensions should implement their own permission checks.
            
            return method(*args, **kwargs)
    
    def create_standard_api(self) -> Dict[str, Callable]:
        """Create standard API functions for common operations"""
        return {
            # Image processing
            "pre_process_image": lambda image, extension_name, **kwargs: self.publish_event(
                Event(
                    event_type=EventType.PRE_PROCESS_IMAGE,
                    data={"image": image, "kwargs": kwargs},
                    source_extension=extension_name
                )
            ),
            "post_process_image": lambda image, extension_name, **kwargs: self.publish_event(
                Event(
                    event_type=EventType.POST_PROCESS_IMAGE,
                    data={"image": image, "kwargs": kwargs},
                    source_extension=extension_name
                )
            ),
            
            # Model operations
            "register_model_type": lambda model_type, extension_name: self.set_shared_state(
                f"model_type_{model_type}", extension_name, extension_name
            ),
            "get_model_handler": lambda model_type: self.get_shared_state(f"model_type_{model_type}"),
            
            # Sampling hooks
            "register_sampling_step": lambda callback, extension_name, priority=EventPriority.NORMAL: 
                self.subscribe(EventType.SAMPLING_STEP, callback, extension_name, priority),
            
            # Shared configuration
            "get_config": lambda key, default=None: self.get_shared_state(f"config_{key}", default),
            "set_config": lambda key, value, extension_name: self.set_shared_state(f"config_{key}", value, extension_name),
        }

# Global singleton instance
extension_api = ExtensionAPI()

# Decorator for automatic event subscription
def subscribe_to(event_type: Union[EventType, str], priority: EventPriority = EventPriority.NORMAL):
    """Decorator to automatically subscribe a method to an event"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get extension name from instance
            extension_name = getattr(self, 'extension_name', None) or self.__class__.__name__
            
            # Subscribe to event
            extension_api.subscribe(
                event_type=event_type,
                callback=lambda event: func(self, event, *args, **kwargs),
                extension_name=extension_name,
                priority=priority,
                description=func.__doc__ or ""
            )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# Context manager for temporary event subscription
class EventSubscription:
    """Context manager for temporary event subscriptions"""
    
    def __init__(self, event_type: Union[EventType, str], callback: Callable[[Event], Any],
                 extension_name: str, priority: EventPriority = EventPriority.NORMAL):
        self.event_type = event_type
        self.callback = callback
        self.extension_name = extension_name
        self.priority = priority
    
    def __enter__(self):
        extension_api.subscribe(
            self.event_type, self.callback, self.extension_name, self.priority
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        extension_api.unsubscribe(self.event_type, self.callback, self.extension_name)

# Standard extension interface
class ExtensionBase:
    """Base class for extensions that want to use the API"""
    
    def __init__(self, extension_name: str):
        self.extension_name = extension_name
        self.api = extension_api
        
        # Register with API
        self.api.register_extension(extension_name, self)
        
        # Declare capabilities
        self.declare_capabilities()
        
        # Set up standard API
        self.standard_api = self.api.create_standard_api()
    
    def declare_capabilities(self):
        """Override to declare extension capabilities"""
        pass
    
    def get_api(self) -> Dict[str, Callable]:
        """Get standard API functions"""
        return self.standard_api
    
    def cleanup(self):
        """Clean up when extension is unloaded"""
        self.api.unregister_extension(self.extension_name)

# Integration with existing extensions
def integrate_with_existing_extensions():
    """Integrate API with existing built-in extensions"""
    
    # Example: Hook into Lora extension
    try:
        from extensions_builtins.Lora import lora
        
        # Create wrapper for Lora model loading
        original_load = getattr(lora, 'load_loras', None)
        if original_load:
            @functools.wraps(original_load)
            def wrapped_load(*args, **kwargs):
                # Publish pre-event
                extension_api.publish_event(Event(
                    event_type="lora_loading_start",
                    data={"args": args, "kwargs": kwargs},
                    source_extension="Lora"
                ))
                
                result = original_load(*args, **kwargs)
                
                # Publish post-event
                extension_api.publish_event(Event(
                    event_type="lora_loading_end",
                    data={"result": result, "args": args, "kwargs": kwargs},
                    source_extension="Lora"
                ))
                
                return result
            
            setattr(lora, 'load_loras', wrapped_load)
            logger.info("Integrated with Lora extension")
    except ImportError:
        logger.debug("Lora extension not available for integration")
    
    # Example: Hook into LDSR extension
    try:
        from extensions_builtins.LDSR import ldsr_model
        
        # Register LDSR capabilities
        extension_api.declare_capabilities("LDSR", [
            ExtensionCapability(
                capability=Capability.IMAGE_PROCESSING,
                version="1.0",
                description="LDSR image upscaling"
            )
        ])
        logger.info("Registered LDSR capabilities")
    except ImportError:
        logger.debug("LDSR extension not available for integration")

# Initialize integration when module is loaded
integrate_with_existing_extensions()

# Export public API
__all__ = [
    'ExtensionAPI',
    'extension_api',
    'Event',
    'EventType',
    'EventPriority',
    'EventHandler',
    'Capability',
    'Permission',
    'ExtensionCapability',
    'ExtensionBase',
    'subscribe_to',
    'EventSubscription',
    'integrate_with_existing_extensions'
]