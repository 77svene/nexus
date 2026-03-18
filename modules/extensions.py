from __future__ import annotations

import configparser
import dataclasses
import os
import threading
import re
import time
import json
import copy
import queue
import heapq
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass, field

from modules import shared, errors, cache, scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir, extensions_builtin_dir, script_path  # noqa: F401

extensions: list[Extension] = []
extension_paths: dict[str, Extension] = {}
loaded_extensions: dict[str, Exception] = {}


os.makedirs(extensions_dir, exist_ok=True)

# Cross-Extension Communication Bus
class EventPriority(Enum):
    """Priority levels for event processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass(order=True)
class Event:
    """Represents a typed event in the communication bus."""
    priority: int = field(compare=True)
    timestamp: float = field(compare=False)
    event_type: str = field(compare=False)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)
    source: str = field(compare=False, default="")
    target: Optional[str] = field(compare=False, default=None)
    propagation_stopped: bool = field(compare=False, default=False)
    
    def stop_propagation(self):
        """Stop event from being processed by further handlers."""
        self.propagation_stopped = True

class ExtensionCapabilities(Enum):
    """Capabilities that extensions can declare."""
    PUBLISH_EVENTS = "publish_events"
    SUBSCRIBE_EVENTS = "subscribe_events"
    SHARE_MODELS = "share_models"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    ACCESS_NETWORK = "access_network"
    ACCESS_FILESYSTEM = "access_filesystem"
    MODIFY_UI = "modify_ui"

class Permission(Enum):
    """Permission levels for extension operations."""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4

@dataclass
class ExtensionPermission:
    """Permission configuration for an extension."""
    capabilities: Set[ExtensionCapabilities] = field(default_factory=set)
    event_publish: Set[str] = field(default_factory=set)  # Event types this extension can publish
    event_subscribe: Set[str] = field(default_factory=set)  # Event types this extension can subscribe to
    max_priority: EventPriority = EventPriority.NORMAL
    allowed_targets: Set[str] = field(default_factory=set)  # Extensions this can send events to

class CrossExtensionBus:
    """
    Cross-Extension Communication Bus implementing pub/sub with typed events and priority queues.
    Enables extensions to communicate without tight coupling.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one bus instance exists."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the communication bus."""
        if self._initialized:
            return
            
        self._subscribers: Dict[str, List[Tuple[int, Callable, str]]] = {}  # event_type -> [(priority, callback, extension_name)]
        self._event_queue: List[Event] = []  # Priority queue (heap)
        self._processing_lock = threading.RLock()
        self._extension_permissions: Dict[str, ExtensionPermission] = {}
        self._extension_capabilities: Dict[str, Set[ExtensionCapabilities]] = {}
        self._event_history: List[Dict[str, Any]] = []  # For debugging
        self._max_history = 1000
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_processing = threading.Event()
        self._initialized = True
        
        # Register standard event types
        self._register_standard_events()
        
        # Start event processing thread
        self._start_processing_thread()
    
    def _register_standard_events(self):
        """Register standard event types for common operations."""
        standard_events = [
            # Pre/Post processing events
            "pre_process_image",
            "post_process_image",
            "pre_process_text",
            "post_process_text",
            
            # Model events
            "model_loading",
            "model_loaded",
            "model_unloading",
            "model_sharing_request",
            "model_sharing_response",
            
            # UI events
            "ui_element_created",
            "ui_tab_changed",
            "ui_settings_updated",
            
            # Extension lifecycle events
            "extension_loaded",
            "extension_enabled",
            "extension_disabled",
            "extension_error",
            
            # System events
            "system_startup",
            "system_shutdown",
            "settings_changed",
            
            # Custom data sharing
            "data_request",
            "data_response",
            "state_sync"
        ]
        
        for event_type in standard_events:
            self._subscribers.setdefault(event_type, [])
    
    def _start_processing_thread(self):
        """Start background thread for processing events."""
        def process_events():
            while not self._stop_processing.is_set():
                try:
                    self._process_next_event(timeout=1.0)
                except Exception as e:
                    print(f"Error in event processing thread: {e}")
                    time.sleep(0.1)
        
        self._processing_thread = threading.Thread(
            target=process_events,
            name="ExtensionBusProcessor",
            daemon=True
        )
        self._processing_thread.start()
    
    def register_extension(self, extension_name: str, capabilities: Set[ExtensionCapabilities], 
                          permissions: Optional[ExtensionPermission] = None):
        """
        Register an extension with the communication bus.
        
        Args:
            extension_name: Name of the extension
            capabilities: Set of capabilities the extension provides
            permissions: Optional permission configuration
        """
        with self._lock:
            self._extension_capabilities[extension_name] = capabilities
            
            if permissions is None:
                permissions = ExtensionPermission(capabilities=capabilities)
            
            self._extension_permissions[extension_name] = permissions
            
            # Auto-configure based on capabilities
            if ExtensionCapabilities.PUBLISH_EVENTS in capabilities:
                permissions.event_publish = set(self._subscribers.keys())
            
            if ExtensionCapabilities.SUBSCRIBE_EVENTS in capabilities:
                permissions.event_subscribe = set(self._subscribers.keys())
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None], 
                 extension_name: str, priority: int = EventPriority.NORMAL.value) -> bool:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
            extension_name: Name of the subscribing extension
            priority: Processing priority (higher = processed first)
            
        Returns:
            bool: True if subscription successful
        """
        with self._lock:
            # Check permissions
            if not self._check_permission(extension_name, "subscribe", event_type):
                print(f"Extension {extension_name} not allowed to subscribe to {event_type}")
                return False
            
            # Initialize event type if not exists
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            # Add subscriber with priority
            heapq.heappush(self._subscribers[event_type], (priority, callback, extension_name))
            
            # Sort by priority (heap maintains order)
            self._subscribers[event_type].sort(key=lambda x: x[0], reverse=True)
            
            return True
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None], extension_name: str) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
            extension_name: Name of the extension
            
        Returns:
            bool: True if unsubscription successful
        """
        with self._lock:
            if event_type not in self._subscribers:
                return False
            
            # Find and remove the subscription
            for i, (prio, cb, ext_name) in enumerate(self._subscribers[event_type]):
                if cb == callback and ext_name == extension_name:
                    self._subscribers[event_type].pop(i)
                    heapq.heapify(self._subscribers[event_type])
                    return True
            
            return False
    
    def publish(self, event_type: str, data: Dict[str, Any], 
               source: str, target: Optional[str] = None,
               priority: EventPriority = EventPriority.NORMAL) -> bool:
        """
        Publish an event to the bus.
        
        Args:
            event_type: Type of event to publish
            data: Event data payload
            source: Name of the publishing extension
            target: Optional target extension name (None for broadcast)
            priority: Event priority
            
        Returns:
            bool: True if event was published successfully
        """
        with self._lock:
            # Check permissions
            if not self._check_permission(source, "publish", event_type):
                print(f"Extension {source} not allowed to publish {event_type}")
                return False
            
            # Check target permission if specified
            if target and not self._check_target_permission(source, target):
                print(f"Extension {source} not allowed to send events to {target}")
                return False
            
            # Create event
            event = Event(
                priority=priority.value,
                timestamp=time.time(),
                event_type=event_type,
                data=data,
                source=source,
                target=target
            )
            
            # Add to priority queue
            heapq.heappush(self._event_queue, event)
            
            # Record in history
            self._record_event_history(event)
            
            return True
    
    def _process_next_event(self, timeout: float = 0.1):
        """Process the next event in the queue."""
        try:
            # Get event with highest priority (lowest number in heap)
            if self._event_queue:
                with self._processing_lock:
                    if self._event_queue:
                        event = heapq.heappop(self._event_queue)
                        self._dispatch_event(event)
        except Exception as e:
            print(f"Error processing event: {e}")
    
    def _dispatch_event(self, event: Event):
        """Dispatch an event to all subscribers."""
        event_type = event.event_type
        
        if event_type not in self._subscribers:
            return
        
        # Get subscribers sorted by priority
        subscribers = sorted(self._subscribers[event_type], key=lambda x: x[0], reverse=True)
        
        for priority, callback, ext_name in subscribers:
            if event.propagation_stopped:
                break
            
            # Check if this is a targeted event
            if event.target and ext_name != event.target:
                continue
            
            try:
                # Call the subscriber callback
                callback(event)
            except Exception as e:
                print(f"Error in event handler {ext_name} for {event_type}: {e}")
                # Publish error event
                self.publish(
                    "extension_error",
                    {"error": str(e), "handler": ext_name, "event_type": event_type},
                    "system",
                    priority=EventPriority.HIGH
                )
    
    def _check_permission(self, extension_name: str, operation: str, event_type: str) -> bool:
        """Check if an extension has permission for an operation."""
        if extension_name not in self._extension_permissions:
            return False
        
        perm = self._extension_permissions[extension_name]
        
        if operation == "publish":
            return event_type in perm.event_publish
        elif operation == "subscribe":
            return event_type in perm.event_subscribe
        
        return False
    
    def _check_target_permission(self, source: str, target: str) -> bool:
        """Check if source extension can send events to target extension."""
        if source not in self._extension_permissions:
            return False
        
        perm = self._extension_permissions[source]
        
        # If allowed_targets is empty, allow all
        if not perm.allowed_targets:
            return True
        
        return target in perm.allowed_targets
    
    def _record_event_history(self, event: Event):
        """Record event in history for debugging."""
        history_entry = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "source": event.source,
            "target": event.target,
            "priority": event.priority,
            "data_keys": list(event.data.keys()) if event.data else []
        }
        
        self._event_history.append(history_entry)
        
        # Keep history size limited
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]
    
    def get_event_history(self, event_type: Optional[str] = None, 
                         source: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history with optional filters."""
        with self._lock:
            history = self._event_history.copy()
            
            if event_type:
                history = [e for e in history if e["event_type"] == event_type]
            
            if source:
                history = [e for e in history if e["source"] == source]
            
            return history[-limit:]
    
    def get_subscribers(self, event_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Get current subscribers, optionally filtered by event type."""
        with self._lock:
            if event_type:
                if event_type in self._subscribers:
                    return {event_type: [ext for _, _, ext in self._subscribers[event_type]]}
                return {}
            
            result = {}
            for evt_type, subs in self._subscribers.items():
                result[evt_type] = [ext for _, _, ext in subs]
            
            return result
    
    def clear_extension_subscriptions(self, extension_name: str):
        """Remove all subscriptions for an extension."""
        with self._lock:
            for event_type in list(self._subscribers.keys()):
                self._subscribers[event_type] = [
                    (prio, cb, ext) for prio, cb, ext in self._subscribers[event_type]
                    if ext != extension_name
                ]
                heapq.heapify(self._subscribers[event_type])
    
    def shutdown(self):
        """Shutdown the event processing thread."""
        self._stop_processing.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
    
    def __del__(self):
        """Cleanup when bus is destroyed."""
        self.shutdown()

# Global bus instance
extension_bus = CrossExtensionBus()

# Standard API for common operations
class ExtensionAPI:
    """Standard API for common extension operations."""
    
    @staticmethod
    def pre_process_image(image_data: Any, params: Dict[str, Any], source: str) -> Any:
        """Standard API for image pre-processing."""
        event_data = {
            "image_data": image_data,
            "params": params,
            "processed": False
        }
        
        extension_bus.publish(
            "pre_process_image",
            event_data,
            source,
            priority=EventPriority.HIGH
        )
        
        return event_data.get("image_data", image_data)
    
    @staticmethod
    def post_process_image(image_data: Any, params: Dict[str, Any], source: str) -> Any:
        """Standard API for image post-processing."""
        event_data = {
            "image_data": image_data,
            "params": params,
            "processed": False
        }
        
        extension_bus.publish(
            "post_process_image",
            event_data,
            source,
            priority=EventPriority.HIGH
        )
        
        return event_data.get("image_data", image_data)
    
    @staticmethod
    def request_model_sharing(model_name: str, source: str, callback: Callable) -> bool:
        """Request access to a model from other extensions."""
        request_id = f"{source}_{model_name}_{time.time()}"
        
        def handle_response(event: Event):
            if event.data.get("request_id") == request_id:
                callback(event.data.get("model_data"), event.source)
        
        # Subscribe to response
        extension_bus.subscribe(
            "model_sharing_response",
            handle_response,
            source
        )
        
        # Publish request
        return extension_bus.publish(
            "model_sharing_request",
            {
                "model_name": model_name,
                "request_id": request_id,
                "requester": source
            },
            source,
            priority=EventPriority.HIGH
        )
    
    @staticmethod
    def share_model(model_name: str, model_data: Any, source: str, target: Optional[str] = None):
        """Share a model with other extensions."""
        return extension_bus.publish(
            "model_sharing_response",
            {
                "model_name": model_name,
                "model_data": model_data,
                "request_id": None  # Will be filled by responder
            },
            source,
            target=target,
            priority=EventPriority.HIGH
        )
    
    @staticmethod
    def sync_state(state_key: str, state_data: Any, source: str):
        """Synchronize state across extensions."""
        return extension_bus.publish(
            "state_sync",
            {
                "state_key": state_key,
                "state_data": state_data,
                "timestamp": time.time()
            },
            source,
            priority=EventPriority.NORMAL
        )

# Atomic State Management System
class ExtensionStateTransaction:
    """Transaction-based state management for extensions with rollback capability."""
    
    def __init__(self, extension: 'Extension'):
        self.extension = extension
        self.original_state = self._capture_state()
        self.checkpoints: List[Dict[str, Any]] = []
        self.failed = False
        self.error = None
        
    def _capture_state(self) -> Dict[str, Any]:
        """Capture current extension state for rollback."""
        return {
            'enabled': self.extension.enabled,
            'status': self.extension.status,
            'metadata': copy.deepcopy(self.extension.metadata.config._sections) if hasattr(self.extension.metadata, 'config') else {},
            'custom_state': copy.deepcopy(getattr(self.extension, '_custom_state', {}))
        }
    
    def checkpoint(self):
        """Create a checkpoint of current state."""
        self.checkpoints.append(self._capture_state())
    
    def rollback(self):
        """Rollback to original state."""
        self._restore_state(self.original_state)
        self.failed = True
        
    def rollback_to_checkpoint(self):
        """Rollback to last checkpoint."""
        if self.checkpoints:
            checkpoint_state = self.checkpoints.pop()
            self._restore_state(checkpoint_state)
    
    def _restore_state(self, state: Dict[str, Any]):
        """Restore extension to captured state."""
        self.extension.enabled = state['enabled']
        self.extension.status = state['status']
        if hasattr(self.extension.metadata, 'config') and state['metadata']:
            self.extension.metadata.config._sections = state['metadata']
        if '_custom_state' in state:
            self.extension._custom_state = state['custom_state']
    
    def commit(self):
        """Commit transaction changes."""
        self.checkpoints.clear()
        self.failed = False
        self.error = None
    
    def mark_failed(self, error: Exception):
        """Mark transaction as failed with error."""
        self.failed = True
        self.error = error


class ExtensionStateManager:
    """Manages atomic state operations for all extensions."""
    
    def __init__(self):
        self.active_transactions: Dict[str, ExtensionStateTransaction] = {}
        self.stability_ratings: Dict[str, float] = {}
        self.crash_counts: Dict[str, int] = {}
        self.usage_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self._load_stability_data()
    
    def _load_stability_data(self):
        """Load stability data from cache."""
        try:
            data = cache.cache('extension_stability', 'stability_data', None, None)
            if data:
                self.stability_ratings = data.get('ratings', {})
                self.crash_counts = data.get('crashes', {})
                self.usage_counts = data.get('usage', {})
        except Exception:
            pass
    
    def _save_stability_data(self):
        """Save stability data to cache."""
        try:
            data = {
                'ratings': self.stability_ratings,
                'crashes': self.crash_counts,
                'usage': self.usage_counts
            }
            cache.cache('extension_stability', 'stability_data', data, None)
        except Exception:
            pass
    
    def start_transaction(self, extension: 'Extension') -> ExtensionStateTransaction:
        """Start a new transaction for an extension."""
        with self.lock:
            if extension.name in self.active_transactions:
                raise RuntimeError(f"Extension {extension.name} already has an active transaction")
            
            transaction = ExtensionStateTransaction(extension)
            self.active_transactions[extension.name] = transaction
            
            # Increment usage count
            self.usage_counts[extension.name] = self.usage_counts.get(extension.name, 0) + 1
            
            return transaction
    
    def commit_transaction(self, extension_name: str):
        """Commit an extension's transaction."""
        with self.lock:
            if extension_name in self.active_transactions:
                transaction = self.active_transactions[extension_name]
                transaction.commit()
                del self.active_transactions[extension_name]
                self._update_stability_rating(extension_name, success=True)
                self._save_stability_data()
    
    def rollback_transaction(self, extension_name: str, error: Optional[Exception] = None):
        """Rollback an extension's transaction."""
        with self.lock:
            if extension_name in self.active_transactions:
                transaction = self.active_transactions[extension_name]
                if error:
                    transaction.mark_failed(error)
                transaction.rollback()
                del self.active_transactions[extension_name]
                self._record_crash(extension_name)
                self._update_stability_rating(extension_name, success=False)
                self._save_stability_data()
    
    def _record_crash(self, extension_name: str):
        """Record a crash for an extension."""
        self.crash_counts[extension_name] = self.crash_counts.get(extension_name, 0) + 1
        
        # Auto-disable if crash threshold exceeded
        if self.crash_counts[extension_name] >= 3:
            extension = extension_paths.get(extension_name)
            if extension:
                extension.enabled = False
                self._notify_extension_disabled(extension_name, "excessive crashes")
    
    def _update_stability_rating(self, extension_name: str, success: bool):
        """Update stability rating based on success/failure."""
        crashes = self.crash_counts.get(extension_name, 0)
        usage = self.usage_counts.get(extension_name, 1)
        
        if success:
            # Successful run improves stability
            current_rating = self.stability_ratings.get(extension_name, 1.0)
            new_rating = min(1.0, current_rating + 0.1)
        else:
            # Crash reduces stability
            current_rating = self.stability_ratings.get(extension_name, 1.0)
            new_rating = max(0.0, current_rating - 0.2)
        
        self.stability_ratings[extension_name] = new_rating
    
    def _notify_extension_disabled(self, extension_name: str, reason: str):
        """Notify user that extension was disabled."""
        print(f"Extension '{extension_name}' automatically disabled due to: {reason}")
        errors.report(f"Extension '{extension_name}' was automatically disabled: {reason}")
    
    def get_stability_rating(self, extension_name: str) -> float:
        """Get stability rating for an extension (0.0 to 1.0)."""
        return self.stability_ratings.get(extension_name, 1.0)
    
    def get_marketplace_data(self) -> List[Dict[str, Any]]:
        """Get marketplace data with stability ratings."""
        marketplace_data = []
        for ext in extensions:
            rating = self.get_stability_rating(ext.name)
            crashes = self.crash_counts.get(ext.name, 0)
            usage = self.usage_counts.get(ext.name, 0)
            
            marketplace_data.append({
                'name': ext.name,
                'canonical_name': ext.canon,
                'stability_rating': rating,
                'crash_count': crashes,
                'usage_count': usage,
                'capabilities': list(extension_bus._extension_capabilities.get(ext.name, set()))
            })
        
        return marketplace_data