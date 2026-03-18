"""
core/streaming/priority_queue.py

Streaming Agent Communication Protocol with priority queues and backpressure handling.
Enables real-time streaming communication between nexus with flow control and message prioritization.
"""

import asyncio
import heapq
import struct
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Awaitable
import msgpack
from contextlib import asynccontextmanager

from monitoring.metrics_collector import MetricsCollector
from monitoring.tracing import Tracer
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.distributed.state_manager import StateManager


class MessagePriority(IntEnum):
    """Message priority levels for queue ordering."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


class MessageType(Enum):
    """Types of streaming messages."""
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"
    FLOW_CONTROL = "flow_control"
    PRIORITY_UPDATE = "priority_update"
    CLOSE = "close"


@dataclass
class StreamMessage:
    """A message in the streaming protocol."""
    id: str
    stream_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    priority: MessagePriority
    payload: bytes
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)
    ttl: float = 30.0  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    encrypted: bool = False
    
    def __lt__(self, other: 'StreamMessage') -> bool:
        """Compare messages for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.sequence != other.sequence:
            return self.sequence < other.sequence
        return self.timestamp < other.timestamp
    
    def is_expired(self) -> bool:
        """Check if message has exceeded TTL."""
        return time.time() - self.timestamp > self.ttl
    
    def encode(self) -> bytes:
        """Encode message to binary format."""
        header = struct.pack(
            '!I16s16s16sBBIdd',
            len(self.id.encode()),
            self.id.encode()[:16].ljust(16, b'\0'),
            self.stream_id.encode()[:16].ljust(16, b'\0'),
            self.sender_id.encode()[:16].ljust(16, b'\0'),
            self.message_type.value.encode()[0],
            self.priority,
            self.sequence,
            self.timestamp,
            self.ttl
        )
        
        metadata_packed = msgpack.packb(self.metadata)
        metadata_len = struct.pack('!I', len(metadata_packed))
        
        payload_len = struct.pack('!I', len(self.payload))
        
        flags = 0
        if self.compressed:
            flags |= 0x01
        if self.encrypted:
            flags |= 0x02
        flags_byte = struct.pack('!B', flags)
        
        return header + metadata_len + metadata_packed + payload_len + self.payload + flags_byte
    
    @classmethod
    def decode(cls, data: bytes) -> 'StreamMessage':
        """Decode message from binary format."""
        header_size = struct.calcsize('!I16s16s16sBBIdd')
        header = data[:header_size]
        
        id_len = struct.unpack('!I', header[:4])[0]
        msg_id = header[4:20].rstrip(b'\0').decode()
        stream_id = header[20:36].rstrip(b'\0').decode()
        sender_id = header[36:52].rstrip(b'\0').decode()
        msg_type_char = header[52]
        priority = header[53]
        sequence = struct.unpack('!I', header[54:58])[0]
        timestamp, ttl = struct.unpack('!dd', header[58:74])
        
        offset = header_size
        metadata_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        metadata = msgpack.unpackb(data[offset:offset+metadata_len])
        offset += metadata_len
        
        payload_len = struct.unpack('!I', data[offset:offset+4])[0]
        offset += 4
        payload = data[offset:offset+payload_len]
        offset += payload_len
        
        flags = struct.unpack('!B', data[offset:offset+1])[0]
        compressed = bool(flags & 0x01)
        encrypted = bool(flags & 0x02)
        
        # Convert message type character back to enum
        msg_type_map = {
            ord('d'): MessageType.DATA,
            ord('c'): MessageType.CONTROL,
            ord('h'): MessageType.HEARTBEAT,
            ord('a'): MessageType.ACK,
            ord('n'): MessageType.NACK,
            ord('f'): MessageType.FLOW_CONTROL,
            ord('p'): MessageType.PRIORITY_UPDATE,
            ord('x'): MessageType.CLOSE,
        }
        message_type = msg_type_map.get(msg_type_char, MessageType.DATA)
        
        return cls(
            id=msg_id,
            stream_id=stream_id,
            sender_id=sender_id,
            receiver_id="",  # Will be set by receiver
            message_type=message_type,
            priority=MessagePriority(priority),
            payload=payload,
            sequence=sequence,
            timestamp=timestamp,
            ttl=ttl,
            metadata=metadata,
            compressed=compressed,
            encrypted=encrypted
        )


@dataclass
class FlowControlWindow:
    """Flow control window for backpressure handling."""
    window_size: int = 1000
    current: int = 0
    last_update: float = field(default_factory=time.time)
    min_rate: float = 10.0  # Minimum messages per second
    max_rate: float = 10000.0  # Maximum messages per second
    adaptive: bool = True
    
    def can_send(self) -> bool:
        """Check if we can send more messages."""
        return self.current < self.window_size
    
    def consume(self, count: int = 1) -> bool:
        """Consume window capacity."""
        if self.current + count <= self.window_size:
            self.current += count
            return True
        return False
    
    def release(self, count: int = 1):
        """Release window capacity."""
        self.current = max(0, self.current - count)
    
    def update_window(self, new_size: int):
        """Update window size based on receiver feedback."""
        self.window_size = max(self.min_rate, min(self.max_rate, new_size))
        self.last_update = time.time()
    
    def adjust_based_on_latency(self, latency: float, target_latency: float = 0.1):
        """Adaptively adjust window based on latency."""
        if not self.adaptive:
            return
        
        if latency > target_latency * 2:
            # Reduce window if latency is too high
            self.window_size = max(self.min_rate, int(self.window_size * 0.8))
        elif latency < target_latency * 0.5:
            # Increase window if latency is low
            self.window_size = min(self.max_rate, int(self.window_size * 1.2))


class PriorityQueue:
    """
    Priority queue with backpressure handling for streaming agent communication.
    
    Features:
    - Multiple priority levels
    - Flow control with adaptive window sizing
    - Message expiration (TTL)
    - Dead letter queue for failed messages
    - Metrics integration
    - Distributed state support
    """
    
    def __init__(
        self,
        agent_id: str,
        max_size: int = 10000,
        metrics_collector: Optional[MetricsCollector] = None,
        tracer: Optional[Tracer] = None,
        state_manager: Optional[StateManager] = None
    ):
        self.agent_id = agent_id
        self.max_size = max_size
        
        # Core data structures
        self._queue: List[Tuple[int, float, str, StreamMessage]] = []
        self._stream_queues: Dict[str, List[StreamMessage]] = defaultdict(list)
        self._sequence_counters: Dict[str, int] = defaultdict(int)
        self._pending_acks: Dict[str, StreamMessage] = {}
        self._dead_letter_queue: List[StreamMessage] = []
        
        # Flow control
        self._flow_windows: Dict[str, FlowControlWindow] = {}
        self._global_window = FlowControlWindow(window_size=max_size)
        
        # State
        self._closed = False
        self._processing = False
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._not_full = asyncio.Event()
        self._not_full.set()
        
        # Integration
        self.metrics = metrics_collector or MetricsCollector()
        self.tracer = tracer or Tracer()
        self.state_manager = state_manager
        self.circuit_breaker = CircuitBreaker(name=f"priority_queue_{agent_id}")
        self.retry_policy = RetryPolicy(max_retries=3)
        
        # Callbacks
        self._on_message_callbacks: List[Callable[[StreamMessage], Awaitable[None]]] = []
        self._on_flow_control_callbacks: List[Callable[[str, int], Awaitable[None]]] = []
        
        # Statistics
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "dropped": 0,
            "expired": 0,
            "retried": 0,
            "dead_lettered": 0,
            "flow_control_blocks": 0,
        }
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_messages())
        self._metrics_task = asyncio.create_task(self._report_metrics())
    
    async def enqueue(
        self,
        message: StreamMessage,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Enqueue a message with priority handling and backpressure.
        
        Args:
            message: The message to enqueue
            timeout: Maximum time to wait if queue is full
            
        Returns:
            True if message was enqueued, False otherwise
        """
        if self._closed:
            raise RuntimeError("Queue is closed")
        
        start_time = time.time()
        
        with self.tracer.trace("priority_queue.enqueue") as span:
            span.set_attributes({
                "agent_id": self.agent_id,
                "stream_id": message.stream_id,
                "priority": message.priority.name,
                "message_type": message.message_type.value,
            })
            
            try:
                # Check flow control
                if not await self._check_flow_control(message):
                    self._stats["flow_control_blocks"] += 1
                    self.metrics.increment("queue.flow_control_blocks")
                    return False
                
                async with self._lock:
                    # Check queue capacity
                    if len(self._queue) >= self.max_size:
                        if timeout is None or timeout <= 0:
                            self._stats["dropped"] += 1
                            self.metrics.increment("queue.dropped")
                            await self._send_to_dead_letter(message, "queue_full")
                            return False
                        
                        # Wait for space
                        self._not_full.clear()
                        try:
                            await asyncio.wait_for(self._not_full.wait(), timeout)
                        except asyncio.TimeoutError:
                            self._stats["dropped"] += 1
                            self.metrics.increment("queue.dropped")
                            await self._send_to_dead_letter(message, "timeout")
                            return False
                    
                    # Assign sequence number
                    self._sequence_counters[message.stream_id] += 1
                    message.sequence = self._sequence_counters[message.stream_id]
                    
                    # Add to queue
                    heapq.heappush(
                        self._queue,
                        (message.priority, message.timestamp, message.id, message)
                    )
                    
                    # Also add to stream-specific queue for ordered delivery
                    self._stream_queues[message.stream_id].append(message)
                    
                    # Update statistics
                    self._stats["enqueued"] += 1
                    self.metrics.increment("queue.enqueued")
                    self.metrics.observe("queue.size", len(self._queue))
                    
                    # Notify waiting consumers
                    self._not_empty.set()
                    
                    # Persist state if state manager available
                    if self.state_manager:
                        await self.state_manager.set(
                            f"queue:{self.agent_id}:last_sequence:{message.stream_id}",
                            message.sequence
                        )
                    
                    span.set_attribute("enqueue_latency", time.time() - start_time)
                    return True
                    
            except Exception as e:
                span.record_exception(e)
                self.metrics.increment("queue.enqueue_errors")
                raise
    
    async def dequeue(
        self,
        timeout: Optional[float] = None,
        stream_id: Optional[str] = None
    ) -> Optional[StreamMessage]:
        """
        Dequeue the highest priority message.
        
        Args:
            timeout: Maximum time to wait for a message
            stream_id: Optional filter by stream ID
            
        Returns:
            The dequeued message or None if timeout
        """
        if self._closed and not self._queue:
            return None
        
        start_time = time.time()
        
        with self.tracer.trace("priority_queue.dequeue") as span:
            span.set_attributes({
                "agent_id": self.agent_id,
                "stream_filter": stream_id or "all",
            })
            
            try:
                async with self._lock:
                    # Wait for messages if queue is empty
                    if not self._queue:
                        if timeout is None or timeout <= 0:
                            return None
                        
                        self._not_empty.clear()
                        try:
                            await asyncio.wait_for(self._not_empty.wait(), timeout)
                        except asyncio.TimeoutError:
                            return None
                    
                    # Find message matching criteria
                    message = None
                    if stream_id:
                        # Find highest priority message for specific stream
                        for i, (prio, ts, msg_id, msg) in enumerate(self._queue):
                            if msg.stream_id == stream_id:
                                message = msg
                                # Remove from heap (inefficient but necessary for filtering)
                                self._queue.pop(i)
                                heapq.heapify(self._queue)
                                break
                    else:
                        # Get highest priority message overall
                        _, _, _, message = heapq.heappop(self._queue)
                    
                    if not message:
                        return None
                    
                    # Remove from stream queue
                    if message.stream_id in self._stream_queues:
                        try:
                            self._stream_queues[message.stream_id].remove(message)
                        except ValueError:
                            pass
                    
                    # Check expiration
                    if message.is_expired():
                        self._stats["expired"] += 1
                        self.metrics.increment("queue.expired")
                        await self._send_to_dead_letter(message, "expired")
                        return await self.dequeue(timeout, stream_id)  # Try next
                    
                    # Update statistics
                    self._stats["dequeued"] += 1
                    self.metrics.increment("queue.dequeued")
                    self.metrics.observe("queue.size", len(self._queue))
                    self.metrics.observe(
                        "queue.wait_time",
                        time.time() - message.timestamp
                    )
                    
                    # Notify waiting producers
                    self._not_full.set()
                    
                    # Update flow control
                    await self._update_flow_control(message)
                    
                    # Notify callbacks
                    for callback in self._on_message_callbacks:
                        try:
                            await callback(message)
                        except Exception as e:
                            self.metrics.increment("queue.callback_errors")
                    
                    span.set_attribute("dequeue_latency", time.time() - start_time)
                    return message
                    
            except Exception as e:
                span.record_exception(e)
                self.metrics.increment("queue.dequeue_errors")
                raise
    
    async def peek(self, stream_id: Optional[str] = None) -> Optional[StreamMessage]:
        """Peek at the next message without removing it."""
        async with self._lock:
            if not self._queue:
                return None
            
            if stream_id:
                for _, _, _, msg in self._queue:
                    if msg.stream_id == stream_id:
                        return msg
                return None
            else:
                _, _, _, msg = self._queue[0]
                return msg
    
    async def update_priority(
        self,
        message_id: str,
        new_priority: MessagePriority
    ) -> bool:
        """Update priority of a queued message."""
        async with self._lock:
            for i, (prio, ts, msg_id, msg) in enumerate(self._queue):
                if msg.id == message_id:
                    # Remove old entry
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    
                    # Update priority and re-add
                    msg.priority = new_priority
                    heapq.heappush(
                        self._queue,
                        (new_priority, ts, msg_id, msg)
                    )
                    
                    self.metrics.increment("queue.priority_updates")
                    return True
            return False
    
    async def acknowledge(self, message_id: str) -> bool:
        """Acknowledge successful processing of a message."""
        async with self._lock:
            if message_id in self._pending_acks:
                message = self._pending_acks.pop(message_id)
                # Release flow control window
                if message.stream_id in self._flow_windows:
                    self._flow_windows[message.stream_id].release()
                self.metrics.increment("queue.acks")
                return True
            return False
    
    async def negative_acknowledge(
        self,
        message_id: str,
        requeue: bool = True
    ) -> bool:
        """Negatively acknowledge a message (processing failed)."""
        async with self._lock:
            if message_id in self._pending_acks:
                message = self._pending_acks.pop(message_id)
                
                if requeue:
                    # Requeue with lower priority
                    new_priority = min(
                        MessagePriority.BACKGROUND,
                        MessagePriority(message.priority + 1)
                    )
                    message.priority = new_priority
                    await self.enqueue(message)
                    self._stats["retried"] += 1
                    self.metrics.increment("queue.retries")
                else:
                    await self._send_to_dead_letter(message, "nack")
                
                # Release flow control window
                if message.stream_id in self._flow_windows:
                    self._flow_windows[message.stream_id].release()
                
                self.metrics.increment("queue.nacks")
                return True
            return False
    
    async def update_flow_window(
        self,
        stream_id: str,
        window_size: int
    ):
        """Update flow control window for a stream."""
        async with self._lock:
            if stream_id not in self._flow_windows:
                self._flow_windows[stream_id] = FlowControlWindow()
            
            self._flow_windows[stream_id].update_window(window_size)
            
            # Notify callbacks
            for callback in self._on_flow_control_callbacks:
                try:
                    await callback(stream_id, window_size)
                except Exception as e:
                    self.metrics.increment("queue.flow_control_callback_errors")
            
            self.metrics.gauge(
                f"queue.flow_window.{stream_id}",
                window_size
            )
    
    async def get_stream_messages(
        self,
        stream_id: str,
        limit: int = 100
    ) -> List[StreamMessage]:
        """Get messages for a specific stream in order."""
        async with self._lock:
            messages = self._stream_queues.get(stream_id, [])
            return sorted(messages, key=lambda m: m.sequence)[:limit]
    
    async def clear_stream(self, stream_id: str) -> int:
        """Clear all messages for a stream."""
        async with self._lock:
            count = 0
            # Remove from main queue
            new_queue = []
            for item in self._queue:
                _, _, _, msg = item
                if msg.stream_id != stream_id:
                    new_queue.append(item)
                else:
                    count += 1
            
            self._queue = new_queue
            heapq.heapify(self._queue)
            
            # Clear stream queue
            if stream_id in self._stream_queues:
                count += len(self._stream_queues[stream_id])
                del self._stream_queues[stream_id]
            
            # Clear sequence counter
            if stream_id in self._sequence_counters:
                del self._sequence_counters[stream_id]
            
            self.metrics.increment("queue.stream_clears")
            return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        async with self._lock:
            stats = {
                **self._stats,
                "size": len(self._queue),
                "streams": len(self._stream_queues),
                "pending_acks": len(self._pending_acks),
                "dead_letter_size": len(self._dead_letter_queue),
                "flow_windows": {
                    sid: {
                        "window_size": fw.window_size,
                        "current": fw.current,
                        "utilization": fw.current / fw.window_size if fw.window_size > 0 else 0
                    }
                    for sid, fw in self._flow_windows.items()
                }
            }
            
            # Add priority distribution
            priority_dist = defaultdict(int)
            for _, _, _, msg in self._queue:
                priority_dist[msg.priority.name] += 1
            stats["priority_distribution"] = dict(priority_dist)
            
            return stats
    
    def on_message(self, callback: Callable[[StreamMessage], Awaitable[None]]):
        """Register callback for message events."""
        self._on_message_callbacks.append(callback)
    
    def on_flow_control(self, callback: Callable[[str, int], Awaitable[None]]):
        """Register callback for flow control events."""
        self._on_flow_control_callbacks.append(callback)
    
    async def close(self):
        """Close the queue and cleanup resources."""
        self._closed = True
        
        # Cancel background tasks
        self._cleanup_task.cancel()
        self._metrics_task.cancel()
        
        try:
            await self._cleanup_task
            await self._metrics_task
        except asyncio.CancelledError:
            pass
        
        # Move remaining messages to dead letter queue
        async with self._lock:
            for _, _, _, msg in self._queue:
                await self._send_to_dead_letter(msg, "queue_closed")
            self._queue.clear()
            self._stream_queues.clear()
        
        self.metrics.gauge("queue.size", 0)
    
    async def _check_flow_control(self, message: StreamMessage) -> bool:
        """Check if message can be sent based on flow control."""
        stream_id = message.stream_id
        
        # Check global window
        if not self._global_window.can_send():
            return False
        
        # Check stream-specific window
        if stream_id not in self._flow_windows:
            self._flow_windows[stream_id] = FlowControlWindow()
        
        window = self._flow_windows[stream_id]
        if not window.can_send():
            return False
        
        # Consume window capacity
        window.consume()
        self._global_window.consume()
        
        # Track pending acknowledgment
        self._pending_acks[message.id] = message
        
        return True
    
    async def _update_flow_control(self, message: StreamMessage):
        """Update flow control after message processing."""
        stream_id = message.stream_id
        
        if stream_id in self._flow_windows:
            window = self._flow_windows[stream_id]
            # Don't release here - wait for acknowledgment
            pass
    
    async def _send_to_dead_letter(self, message: StreamMessage, reason: str):
        """Send message to dead letter queue."""
        message.metadata["dead_letter_reason"] = reason
        message.metadata["dead_letter_timestamp"] = time.time()
        self._dead_letter_queue.append(message)
        self._stats["dead_lettered"] += 1
        self.metrics.increment("queue.dead_lettered")
        
        # Limit dead letter queue size
        if len(self._dead_letter_queue) > 1000:
            self._dead_letter_queue = self._dead_letter_queue[-1000:]
    
    async def _cleanup_expired_messages(self):
        """Background task to clean up expired messages."""
        while not self._closed:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                async with self._lock:
                    now = time.time()
                    expired = []
                    
                    # Check main queue
                    for i, (prio, ts, msg_id, msg) in enumerate(self._queue):
                        if msg.is_expired():
                            expired.append((i, msg))
                    
                    # Remove expired messages
                    for i, msg in reversed(expired):
                        self._queue.pop(i)
                        self._stats["expired"] += 1
                        self.metrics.increment("queue.expired")
                        await self._send_to_dead_letter(msg, "expired")
                    
                    if expired:
                        heapq.heapify(self._queue)
                    
                    # Check stream queues
                    for stream_id, messages in list(self._stream_queues.items()):
                        expired_in_stream = [
                            msg for msg in messages if msg.is_expired()
                        ]
                        for msg in expired_in_stream:
                            messages.remove(msg)
                            self._stats["expired"] += 1
                            self.metrics.increment("queue.expired")
                            await self._send_to_dead_letter(msg, "expired")
                        
                        if not messages:
                            del self._stream_queues[stream_id]
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.increment("queue.cleanup_errors")
                await asyncio.sleep(1)  # Backoff on error
    
    async def _report_metrics(self):
        """Background task to report metrics."""
        while not self._closed:
            try:
                await asyncio.sleep(5)  # Report every 5 seconds
                
                stats = await self.get_stats()
                
                self.metrics.gauge("queue.size", stats["size"])
                self.metrics.gauge("queue.streams", stats["streams"])
                self.metrics.gauge("queue.pending_acks", stats["pending_acks"])
                self.metrics.gauge("queue.dead_letter_size", stats["dead_letter_size"])
                
                for priority, count in stats["priority_distribution"].items():
                    self.metrics.gauge(f"queue.priority.{priority}", count)
                
                for stream_id, window_info in stats["flow_windows"].items():
                    self.metrics.gauge(
                        f"queue.flow_utilization.{stream_id}",
                        window_info["utilization"]
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.increment("queue.metrics_errors")
                await asyncio.sleep(1)


class StreamingProtocol:
    """
    High-level streaming protocol for agent-to-agent communication.
    
    Provides:
    - Stream creation and management
    - Message routing and delivery
    - Backpressure coordination
    - Priority-based message scheduling
    """
    
    def __init__(
        self,
        agent_id: str,
        metrics_collector: Optional[MetricsCollector] = None,
        tracer: Optional[Tracer] = None,
        state_manager: Optional[StateManager] = None
    ):
        self.agent_id = agent_id
        self.queues: Dict[str, PriorityQueue] = {}
        self.streams: Dict[str, Dict[str, Any]] = {}
        self.peers: Dict[str, asyncio.StreamWriter] = {}
        
        self.metrics = metrics_collector or MetricsCollector()
        self.tracer = tracer or Tracer()
        self.state_manager = state_manager
        
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._running = False
        self._server: Optional[asyncio.Server] = None
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8888):
        """Start the streaming protocol server."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            host,
            port
        )
        
        self.metrics.increment("protocol.server_started")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def connect_to_peer(self, peer_id: str, host: str, port: int):
        """Connect to another agent."""
        reader, writer = await asyncio.open_connection(host, port)
        self.peers[peer_id] = writer
        
        # Start receiving task
        asyncio.create_task(self._receive_from_peer(peer_id, reader))
        
        self.metrics.increment("protocol.peer_connected")
    
    async def create_stream(
        self,
        stream_id: str,
        receiver_id: str,
        priority: MessagePriority = MessagePriority.MEDIUM,
        window_size: int = 1000
    ) -> str:
        """Create a new streaming connection."""
        if stream_id in self.streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        self.streams[stream_id] = {
            "receiver_id": receiver_id,
            "priority": priority,
            "window_size": window_size,
            "created_at": time.time(),
            "message_count": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
        }
        
        # Create queue for this stream
        self.queues[stream_id] = PriorityQueue(
            agent_id=self.agent_id,
            metrics_collector=self.metrics,
            tracer=self.tracer,
            state_manager=self.state_manager
        )
        
        # Send stream creation control message
        control_msg = StreamMessage(
            id=str(uuid.uuid4()),
            stream_id=stream_id,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.CONTROL,
            priority=MessagePriority.HIGH,
            payload=msgpack.packb({
                "action": "create_stream",
                "window_size": window_size,
                "priority": priority.value
            }),
            metadata={"stream_created": time.time()}
        )
        
        await self.send_message(control_msg)
        
        self.metrics.increment("protocol.stream_created")
        return stream_id
    
    async def send_message(self, message: StreamMessage) -> bool:
        """Send a message through the protocol."""
        if message.receiver_id not in self.peers:
            self.metrics.increment("protocol.send_errors")
            return False
        
        with self.tracer.trace("protocol.send_message") as span:
            span.set_attributes({
                "stream_id": message.stream_id,
                "receiver_id": message.receiver_id,
                "message_type": message.message_type.value,
            })
            
            try:
                # Encode message
                encoded = message.encode()
                
                # Send to peer
                writer = self.peers[message.receiver_id]
                writer.write(encoded)
                await writer.drain()
                
                # Update statistics
                if message.stream_id in self.streams:
                    self.streams[message.stream_id]["message_count"] += 1
                    self.streams[message.stream_id]["bytes_sent"] += len(encoded)
                
                self.metrics.increment("protocol.messages_sent")
                self.metrics.observe("protocol.message_size", len(encoded))
                
                return True
                
            except Exception as e:
                span.record_exception(e)
                self.metrics.increment("protocol.send_errors")
                return False
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle incoming connection from another agent."""
        peer_id = None
        
        try:
            # Read initial handshake
            data = await reader.read(1024)
            if not data:
                return
            
            # Parse handshake (simplified)
            handshake = msgpack.unpackb(data)
            peer_id = handshake.get("agent_id")
            
            if not peer_id:
                return
            
            self.peers[peer_id] = writer
            self.metrics.increment("protocol.connection_accepted")
            
            # Send handshake response
            response = msgpack.packb({
                "agent_id": self.agent_id,
                "status": "connected",
                "timestamp": time.time()
            })
            writer.write(response)
            await writer.drain()
            
            # Start receiving messages
            await self._receive_from_peer(peer_id, reader)
            
        except Exception as e:
            self.metrics.increment("protocol.connection_errors")
        finally:
            if peer_id and peer_id in self.peers:
                del self.peers[peer_id]
            writer.close()
            await writer.wait_closed()
    
    async def _receive_from_peer(
        self,
        peer_id: str,
        reader: asyncio.StreamReader
    ):
        """Receive messages from a peer."""
        while self._running:
            try:
                # Read message header (simplified - in production would parse binary protocol)
                header_data = await reader.read(1024)
                if not header_data:
                    break
                
                # Decode message
                message = StreamMessage.decode(header_data)
                message.receiver_id = self.agent_id
                
                # Update statistics
                if message.stream_id in self.streams:
                    self.streams[message.stream_id]["bytes_received"] += len(header_data)
                
                # Route to appropriate handler
                await self._route_message(message)
                
                self.metrics.increment("protocol.messages_received")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.increment("protocol.receive_errors")
                await asyncio.sleep(0.1)  # Backoff on error
    
    async def _route_message(self, message: StreamMessage):
        """Route message to appropriate handler."""
        # Handle control messages
        if message.message_type == MessageType.CONTROL:
            await self._handle_control_message(message)
            return
        
        # Handle flow control messages
        if message.message_type == MessageType.FLOW_CONTROL:
            await self._handle_flow_control(message)
            return
        
        # Handle priority updates
        if message.message_type == MessageType.PRIORITY_UPDATE:
            await self._handle_priority_update(message)
            return
        
        # Route to registered handlers
        handlers = self._message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                self.metrics.increment("protocol.handler_errors")
    
    async def _handle_control_message(self, message: StreamMessage):
        """Handle control messages."""
        try:
            data = msgpack.unpackb(message.payload)
            action = data.get("action")
            
            if action == "create_stream":
                stream_id = message.stream_id
                window_size = data.get("window_size", 1000)
                priority = MessagePriority(data.get("priority", MessagePriority.MEDIUM))
                
                # Create local queue for stream
                self.queues[stream_id] = PriorityQueue(
                    agent_id=self.agent_id,
                    metrics_collector=self.metrics,
                    tracer=self.tracer,
                    state_manager=self.state_manager
                )
                
                # Update flow window
                await self.queues[stream_id].update_flow_window(stream_id, window_size)
                
                # Store stream info
                self.streams[stream_id] = {
                    "sender_id": message.sender_id,
                    "priority": priority,
                    "window_size": window_size,
                    "created_at": time.time(),
                    "message_count": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                }
                
                self.metrics.increment("protocol.stream_accepted")
                
        except Exception as e:
            self.metrics.increment("protocol.control_message_errors")
    
    async def _handle_flow_control(self, message: StreamMessage):
        """Handle flow control messages."""
        try:
            data = msgpack.unpackb(message.payload)
            window_size = data.get("window_size")
            
            if message.stream_id in self.queues:
                await self.queues[message.stream_id].update_flow_window(
                    message.stream_id,
                    window_size
                )
                
                self.metrics.increment("protocol.flow_control_updates")
                
        except Exception as e:
            self.metrics.increment("protocol.flow_control_errors")
    
    async def _handle_priority_update(self, message: StreamMessage):
        """Handle priority update messages."""
        try:
            data = msgpack.unpackb(message.payload)
            message_id = data.get("message_id")
            new_priority = MessagePriority(data.get("priority"))
            
            if message.stream_id in self.queues:
                await self.queues[message.stream_id].update_priority(
                    message_id,
                    new_priority
                )
                
                self.metrics.increment("protocol.priority_updates")
                
        except Exception as e:
            self.metrics.increment("protocol.priority_update_errors")
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[StreamMessage], Awaitable[None]]
    ):
        """Register a handler for a message type."""
        self._message_handlers[message_type].append(handler)
    
    async def close_stream(self, stream_id: str):
        """Close a stream."""
        if stream_id not in self.streams:
            return
        
        # Send close message
        close_msg = StreamMessage(
            id=str(uuid.uuid4()),
            stream_id=stream_id,
            sender_id=self.agent_id,
            receiver_id=self.streams[stream_id].get("receiver_id", ""),
            message_type=MessageType.CLOSE,
            priority=MessagePriority.HIGH,
            payload=msgpack.packb({"action": "close_stream"}),
            metadata={"stream_closed": time.time()}
        )
        
        await self.send_message(close_msg)
        
        # Cleanup local state
        if stream_id in self.queues:
            await self.queues[stream_id].close()
            del self.queues[stream_id]
        
        if stream_id in self.streams:
            del self.streams[stream_id]
        
        self.metrics.increment("protocol.stream_closed")
    
    async def shutdown(self):
        """Shutdown the protocol."""
        self._running = False
        
        # Close all streams
        for stream_id in list(self.streams.keys()):
            await self.close_stream(stream_id)
        
        # Close all peer connections
        for peer_id, writer in list(self.peers.items()):
            writer.close()
            await writer.wait_closed()
        
        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        self.metrics.increment("protocol.shutdown")


# Export main classes
__all__ = [
    "MessagePriority",
    "MessageType",
    "StreamMessage",
    "FlowControlWindow",
    "PriorityQueue",
    "StreamingProtocol"
]