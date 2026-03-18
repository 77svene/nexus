# core/streaming/protocol.py

import asyncio
import struct
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from monitoring.tracing import TracingManager
from monitoring.metrics_collector import MetricsCollector
from core.distributed.state_manager import StateManager
from core.distributed.executor import DistributedExecutor
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy


class MessageType(IntEnum):
    """Protocol message types for streaming communication."""
    STREAM_OPEN = auto()
    STREAM_DATA = auto()
    STREAM_ACK = auto()
    STREAM_CLOSE = auto()
    STREAM_PAUSE = auto()
    STREAM_RESUME = auto()
    STREAM_ERROR = auto()
    STREAM_HEARTBEAT = auto()
    PRIORITY_UPDATE = auto()


class StreamState(IntEnum):
    """Possible states of a streaming connection."""
    INITIALIZING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    CLOSING = auto()
    CLOSED = auto()
    ERROR = auto()


class Priority(IntEnum):
    """Message priority levels for queue management."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class StreamMessage:
    """Binary protocol message structure."""
    message_type: MessageType
    stream_id: str
    sequence_number: int
    priority: Priority
    payload: bytes
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Binary format: 
    # Header: 1 byte type + 16 bytes stream_id + 4 bytes sequence + 1 byte priority + 8 bytes timestamp
    # Payload: variable length
    HEADER_FORMAT = "!B16sI B d"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    def to_bytes(self) -> bytes:
        """Serialize message to binary format."""
        header = struct.pack(
            self.HEADER_FORMAT,
            self.message_type.value,
            self.stream_id.encode()[:16].ljust(16, b'\0'),
            self.sequence_number,
            self.priority.value,
            self.timestamp
        )
        return header + self.payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'StreamMessage':
        """Deserialize message from binary format."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Message too short: {len(data)} bytes")
        
        header = data[:cls.HEADER_SIZE]
        payload = data[cls.HEADER_SIZE:]
        
        msg_type_val, stream_id_bytes, seq_num, priority_val, timestamp = struct.unpack(
            cls.HEADER_FORMAT, header
        )
        
        stream_id = stream_id_bytes.decode().rstrip('\0')
        
        return cls(
            message_type=MessageType(msg_type_val),
            stream_id=stream_id,
            sequence_number=seq_num,
            priority=Priority(priority_val),
            payload=payload,
            timestamp=timestamp
        )


@dataclass
class StreamChunk:
    """Represents a chunk of data in a stream."""
    data: bytes
    sequence: int
    priority: Priority
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackpressureController:
    """Handles flow control and backpressure for streaming."""
    
    def __init__(self, initial_window_size: int = 1024 * 1024):  # 1MB default
        self.window_size = initial_window_size
        self.bytes_in_flight = 0
        self.acked_bytes = 0
        self.last_window_update = time.time()
        self.congestion_window = 1
        self.slow_start_threshold = 65535
        self.rtt_samples: List[float] = []
        self.min_rtt = float('inf')
        
    def can_send(self, data_size: int) -> bool:
        """Check if data can be sent within current window."""
        return (self.bytes_in_flight + data_size) <= self.window_size
    
    def on_send(self, data_size: int):
        """Record that data has been sent."""
        self.bytes_in_flight += data_size
    
    def on_ack(self, acked_size: int, rtt: float):
        """Handle acknowledgement and adjust window."""
        self.bytes_in_flight = max(0, self.bytes_in_flight - acked_size)
        self.acked_bytes += acked_size
        
        # Update RTT samples
        self.rtt_samples.append(rtt)
        if len(self.rtt_samples) > 100:
            self.rtt_samples.pop(0)
        
        self.min_rtt = min(self.min_rtt, rtt)
        
        # AIMD congestion control
        if self.bytes_in_flight < self.window_size:
            # Slow start
            if self.congestion_window < self.slow_start_threshold:
                self.congestion_window *= 2
            else:
                # Congestion avoidance
                self.congestion_window += 1 / self.congestion_window
        else:
            # Congestion detected
            self.slow_start_threshold = max(self.window_size // 2, 2)
            self.congestion_window = self.slow_start_threshold
        
        # Update window size based on congestion window
        self.window_size = int(self.congestion_window * 1460)  # MSS = 1460
    
    def update_window(self, new_size: int):
        """Update window size from remote endpoint."""
        self.window_size = new_size
        self.last_window_update = time.time()


class PriorityQueueManager:
    """Manages priority queues for streaming messages."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.queues: Dict[Priority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size // len(Priority))
            for priority in Priority
        }
        self.dropped_messages: Dict[Priority, int] = defaultdict(int)
        self.total_enqueued = 0
        self.total_dequeued = 0
        
    async def enqueue(self, message: StreamMessage) -> bool:
        """Add message to appropriate priority queue."""
        queue = self.queues[message.priority]
        
        try:
            if queue.full():
                # Drop lowest priority message if queue is full
                lowest_priority = max(self.queues.keys())
                if message.priority < lowest_priority:
                    try:
                        self.queues[lowest_priority].get_nowait()
                        self.dropped_messages[lowest_priority] += 1
                    except asyncio.QueueEmpty:
                        pass
            
            await queue.put(message)
            self.total_enqueued += 1
            return True
            
        except asyncio.QueueFull:
            self.dropped_messages[message.priority] += 1
            return False
    
    async def dequeue(self) -> Optional[StreamMessage]:
        """Get highest priority message from queues."""
        # Check queues in priority order
        for priority in sorted(Priority):
            queue = self.queues[priority]
            if not queue.empty():
                try:
                    message = queue.get_nowait()
                    self.total_dequeued += 1
                    return message
                except asyncio.QueueEmpty:
                    continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_sizes": {p.name: q.qsize() for p, q in self.queues.items()},
            "dropped_messages": dict(self.dropped_messages),
            "total_enqueued": self.total_enqueued,
            "total_dequeued": self.total_dequeued,
            "backlog": self.total_enqueued - self.total_dequeued
        }


class StreamConnection:
    """Manages a single streaming connection between nexus."""
    
    def __init__(
        self,
        stream_id: str,
        local_agent_id: str,
        remote_agent_id: str,
        backpressure_controller: BackpressureController,
        message_handler: Callable,
        tracing_manager: TracingManager,
        metrics_collector: MetricsCollector
    ):
        self.stream_id = stream_id
        self.local_agent_id = local_agent_id
        self.remote_agent_id = remote_agent_id
        self.state = StreamState.INITIALIZING
        self.sequence_number = 0
        self.acked_sequence = -1
        self.backpressure = backpressure_controller
        self.message_handler = message_handler
        self.tracing = tracing_manager
        self.metrics = metrics_collector
        
        self.incoming_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        self.outgoing_queue = PriorityQueueManager()
        self.pending_acks: Dict[int, Tuple[StreamMessage, float]] = {}
        
        self.last_activity = time.time()
        self.heartbeat_interval = 30.0  # seconds
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.start_time = time.time()
        
    async def start(self):
        """Start the stream connection."""
        self.state = StreamState.ACTIVE
        self.heartbeat_task = asyncio.create_task(self._send_heartbeats())
        
        # Send stream open message
        open_msg = StreamMessage(
            message_type=MessageType.STREAM_OPEN,
            stream_id=self.stream_id,
            sequence_number=0,
            priority=Priority.HIGH,
            payload=b'',
            metadata={
                "local_agent": self.local_agent_id,
                "remote_agent": self.remote_agent_id,
                "protocol_version": "1.0"
            }
        )
        await self.send_message(open_msg)
        
        self.metrics.record_stream_opened(self.stream_id)
        self.tracing.start_span(f"stream:{self.stream_id}")
    
    async def send_message(self, message: StreamMessage) -> bool:
        """Send a message through the stream."""
        if self.state not in (StreamState.ACTIVE, StreamState.PAUSED):
            return False
        
        # Check backpressure
        if not self.backpressure.can_send(len(message.payload)):
            self.metrics.record_backpressure_event(self.stream_id)
            return False
        
        # Enqueue with priority
        success = await self.outgoing_queue.enqueue(message)
        if success:
            self.backpressure.on_send(len(message.payload))
            self.pending_acks[message.sequence_number] = (message, time.time())
            self.messages_sent += 1
            self.bytes_sent += len(message.payload)
            
            self.metrics.record_message_sent(
                self.stream_id, 
                message.message_type.name,
                len(message.payload)
            )
        
        return success
    
    async def receive_message(self, message: StreamMessage):
        """Process incoming message."""
        self.last_activity = time.time()
        self.messages_received += 1
        self.bytes_received += len(message.payload)
        
        self.metrics.record_message_received(
            self.stream_id,
            message.message_type.name,
            len(message.payload)
        )
        
        if message.message_type == MessageType.STREAM_DATA:
            # Add to incoming queue
            chunk = StreamChunk(
                data=message.payload,
                sequence=message.sequence_number,
                priority=message.priority,
                timestamp=message.timestamp,
                metadata=message.metadata
            )
            await self.incoming_queue.put(chunk)
            
            # Send ACK
            ack_msg = StreamMessage(
                message_type=MessageType.STREAM_ACK,
                stream_id=self.stream_id,
                sequence_number=message.sequence_number,
                priority=Priority.HIGH,
                payload=b''
            )
            await self.send_message(ack_msg)
            
        elif message.message_type == MessageType.STREAM_ACK:
            # Handle acknowledgement
            if message.sequence_number in self.pending_acks:
                sent_msg, sent_time = self.pending_acks.pop(message.sequence_number)
                rtt = time.time() - sent_time
                self.backpressure.on_ack(len(sent_msg.payload), rtt)
                self.acked_sequence = max(self.acked_sequence, message.sequence_number)
                
                self.metrics.record_ack_received(self.stream_id, rtt)
        
        elif message.message_type == MessageType.STREAM_PAUSE:
            self.state = StreamState.PAUSED
            self.metrics.record_stream_paused(self.stream_id)
            
        elif message.message_type == MessageType.STREAM_RESUME:
            self.state = StreamState.ACTIVE
            self.metrics.record_stream_resumed(self.stream_id)
            
        elif message.message_type == MessageType.STREAM_CLOSE:
            await self.close()
            
        elif message.message_type == MessageType.STREAM_ERROR:
            self.state = StreamState.ERROR
            self.metrics.record_stream_error(self.stream_id, message.payload.decode())
            
        elif message.message_type == MessageType.STREAM_HEARTBEAT:
            # Update activity timestamp
            pass
        
        # Call message handler for application-level processing
        if self.message_handler:
            await self.message_handler(self.stream_id, message)
    
    async def _send_heartbeats(self):
        """Send periodic heartbeat messages."""
        while self.state == StreamState.ACTIVE:
            await asyncio.sleep(self.heartbeat_interval)
            
            if self.state == StreamState.ACTIVE:
                heartbeat = StreamMessage(
                    message_type=MessageType.STREAM_HEARTBEAT,
                    stream_id=self.stream_id,
                    sequence_number=self.sequence_number,
                    priority=Priority.BACKGROUND,
                    payload=b'',
                    metadata={"timestamp": time.time()}
                )
                await self.send_message(heartbeat)
    
    async def close(self):
        """Close the stream connection."""
        self.state = StreamState.CLOSING
        
        # Send close message
        close_msg = StreamMessage(
            message_type=MessageType.STREAM_CLOSE,
            stream_id=self.stream_id,
            sequence_number=self.sequence_number,
            priority=Priority.HIGH,
            payload=b''
        )
        await self.send_message(close_msg)
        
        # Wait for pending messages
        timeout = 5.0
        start_time = time.time()
        while self.pending_acks and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        # Cancel heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        self.state = StreamState.CLOSED
        duration = time.time() - self.start_time
        
        self.metrics.record_stream_closed(
            self.stream_id,
            duration,
            self.bytes_sent,
            self.bytes_received
        )
        self.tracing.end_span(f"stream:{self.stream_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "stream_id": self.stream_id,
            "state": self.state.name,
            "local_agent": self.local_agent_id,
            "remote_agent": self.remote_agent_id,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "sequence_number": self.sequence_number,
            "acked_sequence": self.acked_sequence,
            "pending_acks": len(self.pending_acks),
            "uptime": time.time() - self.start_time,
            "last_activity": self.last_activity,
            "backpressure_window": self.backpressure.window_size,
            "queue_stats": self.outgoing_queue.get_stats()
        }


class StreamingProtocol:
    """
    Main streaming protocol implementation for agent-to-agent communication.
    
    Features:
    - Real-time streaming with backpressure handling
    - Priority-based message queuing
    - Flow control and congestion management
    - Integration with distributed systems components
    - Monitoring and metrics collection
    """
    
    def __init__(
        self,
        agent_id: str,
        state_manager: StateManager,
        executor: DistributedExecutor,
        tracing_manager: TracingManager,
        metrics_collector: MetricsCollector,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_policy: Optional[RetryPolicy] = None
    ):
        self.agent_id = agent_id
        self.state_manager = state_manager
        self.executor = executor
        self.tracing = tracing_manager
        self.metrics = metrics_collector
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.retry_policy = retry_policy or RetryPolicy()
        
        self.streams: Dict[str, StreamConnection] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.transport_handlers: Dict[str, Callable] = {}
        
        self.running = False
        self.receive_task: Optional[asyncio.Task] = None
        self.processing_task: Optional[asyncio.Task] = None
        
        # Global backpressure controller
        self.global_backpressure = BackpressureController()
        
        # Protocol configuration
        self.max_streams = 1000
        self.stream_timeout = 300.0  # 5 minutes
        self.cleanup_interval = 60.0  # 1 minute
        
        # Register with state manager for coordination
        self._register_with_state_manager()
    
    def _register_with_state_manager(self):
        """Register streaming protocol with distributed state manager."""
        self.state_manager.register_component(
            "streaming_protocol",
            {
                "agent_id": self.agent_id,
                "capabilities": ["streaming", "backpressure", "priority_queues"],
                "max_streams": self.max_streams
            }
        )
    
    async def start(self):
        """Start the streaming protocol."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.receive_task = asyncio.create_task(self._receive_loop())
        self.processing_task = asyncio.create_task(self._processing_loop())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
        self.metrics.record_protocol_started()
        self.tracing.start_span("streaming_protocol")
        
        print(f"Streaming protocol started for agent {self.agent_id}")
    
    async def stop(self):
        """Stop the streaming protocol."""
        self.running = False
        
        # Close all streams
        for stream_id in list(self.streams.keys()):
            await self.close_stream(stream_id)
        
        # Cancel background tasks
        if self.receive_task:
            self.receive_task.cancel()
        if self.processing_task:
            self.processing_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.receive_task, self.processing_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.metrics.record_protocol_stopped()
        self.tracing.end_span("streaming_protocol")
        
        print(f"Streaming protocol stopped for agent {self.agent_id}")
    
    async def create_stream(
        self,
        remote_agent_id: str,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new streaming connection to a remote agent."""
        if len(self.streams) >= self.max_streams:
            raise RuntimeError(f"Maximum streams limit ({self.max_streams}) reached")
        
        # Generate unique stream ID
        stream_id = f"{self.agent_id}:{remote_agent_id}:{uuid.uuid4().hex[:8]}"
        
        # Create stream connection
        stream = StreamConnection(
            stream_id=stream_id,
            local_agent_id=self.agent_id,
            remote_agent_id=remote_agent_id,
            backpressure_controller=BackpressureController(),
            message_handler=self._get_message_handler(remote_agent_id),
            tracing_manager=self.tracing,
            metrics_collector=self.metrics
        )
        
        self.streams[stream_id] = stream
        
        # Start the stream
        await stream.start()
        
        # Register with state manager
        await self.state_manager.set(
            f"stream:{stream_id}",
            {
                "state": stream.state.name,
                "local_agent": self.agent_id,
                "remote_agent": remote_agent_id,
                "created_at": time.time(),
                "priority": priority.name
            }
        )
        
        return stream_id
    
    async def send_data(
        self,
        stream_id: str,
        data: bytes,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send data through an existing stream."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self.streams[stream_id]
        
        if stream.state != StreamState.ACTIVE:
            return False
        
        # Create data message
        message = StreamMessage(
            message_type=MessageType.STREAM_DATA,
            stream_id=stream_id,
            sequence_number=stream.sequence_number,
            priority=priority,
            payload=data,
            metadata=metadata or {}
        )
        
        # Send with retry policy
        success = await self.retry_policy.execute(
            lambda: stream.send_message(message),
            operation_name=f"send_data:{stream_id}"
        )
        
        if success:
            stream.sequence_number += 1
        
        return success
    
    async def send_stream(
        self,
        stream_id: str,
        data_generator,
        priority: Priority = Priority.MEDIUM,
        chunk_size: int = 8192
    ):
        """Send a stream of data using a generator."""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self.streams[stream_id]
        
        try:
            async for chunk in data_generator:
                if stream.state != StreamState.ACTIVE:
                    break
                
                # Split large chunks
                for i in range(0, len(chunk), chunk_size):
                    chunk_part = chunk[i:i + chunk_size]
                    
                    success = await self.send_data(
                        stream_id,
                        chunk_part,
                        priority=priority,
                        metadata={"chunk_index": i // chunk_size}
                    )
                    
                    if not success:
                        # Apply backpressure
                        await asyncio.sleep(0.1)
        
        except Exception as e:
            await self._handle_stream_error(stream_id, str(e))
    
    async def receive_data(self, stream_id: str) -> Optional[StreamChunk]:
        """Receive data from a stream."""
        if stream_id not in self.streams:
            return None
        
        stream = self.streams[stream_id]
        
        try:
            # Wait for data with timeout
            chunk = await asyncio.wait_for(
                stream.incoming_queue.get(),
                timeout=self.stream_timeout
            )
            return chunk
        
        except asyncio.TimeoutError:
            await self.close_stream(stream_id)
            return None
    
    async def close_stream(self, stream_id: str):
        """Close a streaming connection."""
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            await stream.close()
            del self.streams[stream_id]
            
            # Update state manager
            await self.state_manager.delete(f"stream:{stream_id}")
    
    async def pause_stream(self, stream_id: str):
        """Pause a streaming connection."""
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            
            pause_msg = StreamMessage(
                message_type=MessageType.STREAM_PAUSE,
                stream_id=stream_id,
                sequence_number=stream.sequence_number,
                priority=Priority.HIGH,
                payload=b''
            )
            
            await stream.send_message(pause_msg)
            stream.state = StreamState.PAUSED
    
    async def resume_stream(self, stream_id: str):
        """Resume a paused streaming connection."""
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            
            resume_msg = StreamMessage(
                message_type=MessageType.STREAM_RESUME,
                stream_id=stream_id,
                sequence_number=stream.sequence_number,
                priority=Priority.HIGH,
                payload=b''
            )
            
            await stream.send_message(resume_msg)
            stream.state = StreamState.ACTIVE
    
    def register_message_handler(self, remote_agent_id: str, handler: Callable):
        """Register a handler for incoming messages from a specific agent."""
        self.message_handlers[remote_agent_id] = handler
    
    def register_transport_handler(self, transport_type: str, handler: Callable):
        """Register a handler for a specific transport type."""
        self.transport_handlers[transport_type] = handler
    
    def _get_message_handler(self, remote_agent_id: str) -> Callable:
        """Get the appropriate message handler for a remote agent."""
        return self.message_handlers.get(remote_agent_id, self._default_message_handler)
    
    async def _default_message_handler(self, stream_id: str, message: StreamMessage):
        """Default message handler."""
        # Log unhandled messages
        self.metrics.record_unhandled_message(stream_id, message.message_type.name)
    
    async def _handle_stream_error(self, stream_id: str, error: str):
        """Handle stream errors."""
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            
            error_msg = StreamMessage(
                message_type=MessageType.STREAM_ERROR,
                stream_id=stream_id,
                sequence_number=stream.sequence_number,
                priority=Priority.HIGH,
                payload=error.encode()
            )
            
            await stream.send_message(error_msg)
            stream.state = StreamState.ERROR
            
            self.metrics.record_stream_error(stream_id, error)
    
    async def _receive_loop(self):
        """Main loop for receiving messages from transport layer."""
        while self.running:
            try:
                # This would integrate with actual transport layer
                # For now, simulate receiving messages
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.record_protocol_error("receive_loop", str(e))
                await asyncio.sleep(1)
    
    async def _processing_loop(self):
        """Main loop for processing outgoing messages."""
        while self.running:
            try:
                # Process messages from all streams
                for stream_id, stream in list(self.streams.items()):
                    if stream.state == StreamState.ACTIVE:
                        message = await stream.outgoing_queue.dequeue()
                        if message:
                            # Send through transport layer
                            await self._send_through_transport(stream.remote_agent_id, message)
                
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.record_protocol_error("processing_loop", str(e))
                await asyncio.sleep(1)
    
    async def _send_through_transport(self, remote_agent_id: str, message: StreamMessage):
        """Send message through appropriate transport."""
        # This would integrate with actual transport layer
        # For now, simulate sending
        transport_handler = self.transport_handlers.get("default")
        if transport_handler:
            await transport_handler(remote_agent_id, message.to_bytes())
    
    async def _cleanup_loop(self):
        """Periodically clean up inactive streams."""
        while self.running:
            try:
                current_time = time.time()
                streams_to_close = []
                
                for stream_id, stream in self.streams.items():
                    # Check for timeout
                    if (current_time - stream.last_activity) > self.stream_timeout:
                        streams_to_close.append(stream_id)
                    
                    # Check for error state
                    elif stream.state == StreamState.ERROR:
                        streams_to_close.append(stream_id)
                
                # Close timed out or errored streams
                for stream_id in streams_to_close:
                    await self.close_stream(stream_id)
                
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.record_protocol_error("cleanup_loop", str(e))
                await asyncio.sleep(10)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        stream_stats = [stream.get_stats() for stream in self.streams.values()]
        
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "total_streams": len(self.streams),
            "active_streams": sum(1 for s in self.streams.values() if s.state == StreamState.ACTIVE),
            "global_backpressure": {
                "window_size": self.global_backpressure.window_size,
                "bytes_in_flight": self.global_backpressure.bytes_in_flight
            },
            "stream_stats": stream_stats,
            "handlers": {
                "message_handlers": list(self.message_handlers.keys()),
                "transport_handlers": list(self.transport_handlers.keys())
            }
        }
    
    async def update_priority(self, stream_id: str, new_priority: Priority):
        """Update priority for a stream."""
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            
            priority_msg = StreamMessage(
                message_type=MessageType.PRIORITY_UPDATE,
                stream_id=stream_id,
                sequence_number=stream.sequence_number,
                priority=Priority.HIGH,
                payload=struct.pack("!B", new_priority.value)
            )
            
            await stream.send_message(priority_msg)
            
            # Update state manager
            await self.state_manager.update(
                f"stream:{stream_id}",
                {"priority": new_priority.name}
            )


# Factory function for easy instantiation
def create_streaming_protocol(
    agent_id: str,
    state_manager: StateManager,
    executor: DistributedExecutor,
    tracing_manager: TracingManager,
    metrics_collector: MetricsCollector,
    **kwargs
) -> StreamingProtocol:
    """Create and configure a streaming protocol instance."""
    return StreamingProtocol(
        agent_id=agent_id,
        state_manager=state_manager,
        executor=executor,
        tracing_manager=tracing_manager,
        metrics_collector=metrics_collector,
        **kwargs
    )


# Integration with existing distributed system
class StreamingProtocolAdapter:
    """Adapter to integrate streaming protocol with existing distributed components."""
    
    def __init__(self, protocol: StreamingProtocol):
        self.protocol = protocol
        self._setup_integrations()
    
    def _setup_integrations(self):
        """Setup integrations with existing distributed components."""
        # Register with executor for task-based streaming
        self.protocol.executor.register_capability(
            "streaming",
            self._execute_streaming_task
        )
        
        # Register with state manager for coordination
        self.protocol.state_manager.watch_key(
            "streaming:config",
            self._on_config_update
        )
    
    async def _execute_streaming_task(self, task_config: Dict[str, Any]):
        """Execute a streaming task through the executor."""
        stream_id = await self.protocol.create_stream(
            remote_agent_id=task_config["remote_agent"],
            priority=Priority[task_config.get("priority", "MEDIUM")]
        )
        
        # Send task data
        await self.protocol.send_data(
            stream_id,
            task_config["data"].encode(),
            priority=Priority[task_config.get("priority", "MEDIUM")]
        )
        
        # Receive response
        response = await self.protocol.receive_data(stream_id)
        
        # Close stream
        await self.protocol.close_stream(stream_id)
        
        return response.data.decode() if response else None
    
    async def _on_config_update(self, key: str, value: Any):
        """Handle configuration updates from state manager."""
        if key == "streaming:config":
            # Update protocol configuration
            if "max_streams" in value:
                self.protocol.max_streams = value["max_streams"]
            if "stream_timeout" in value:
                self.protocol.stream_timeout = value["stream_timeout"]