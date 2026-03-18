"""
core/streaming/backpressure.py

Streaming Agent Communication Protocol with backpressure handling and priority queues.
Enables real-time streaming communication between nexus with flow control and message prioritization.
"""

import asyncio
import struct
import uuid
import time
import heapq
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict
import msgpack

# Import existing modules for integration
from monitoring.tracing import get_tracer
from monitoring.metrics_collector import MetricsCollector
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy

logger = logging.getLogger(__name__)

class MessagePriority(IntEnum):
    """Priority levels for message processing"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BULK = 4

class MessageType(IntEnum):
    """Types of streaming messages"""
    STREAM_START = 0x01
    STREAM_DATA = 0x02
    STREAM_END = 0x03
    ACK = 0x04
    NACK = 0x05
    FLOW_CONTROL = 0x06
    HEARTBEAT = 0x07
    ERROR = 0x08

@dataclass
class StreamMessage:
    """Represents a message in the streaming protocol"""
    message_id: bytes = field(default_factory=lambda: uuid.uuid4().bytes)
    message_type: MessageType = MessageType.STREAM_DATA
    priority: MessagePriority = MessagePriority.NORMAL
    stream_id: str = ""
    sequence: int = 0
    payload: bytes = b""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: float = 30.0  # Time-to-live in seconds
    
    def is_expired(self) -> bool:
        """Check if message has exceeded TTL"""
        return time.time() - self.timestamp > self.ttl
    
    def encode(self) -> bytes:
        """Encode message to binary format"""
        # Protocol format: [4B length][1B type][1B priority][16B msg_id][4B seq][4B stream_id_len][stream_id][payload]
        header = struct.pack(
            "!IBB16sI",
            len(self.payload),
            self.message_type,
            self.priority,
            self.message_id,
            self.sequence
        )
        
        stream_id_bytes = self.stream_id.encode('utf-8')
        stream_id_header = struct.pack("!I", len(stream_id_bytes))
        
        # Pack metadata with msgpack
        metadata_bytes = msgpack.packb(self.metadata) if self.metadata else b""
        metadata_header = struct.pack("!I", len(metadata_bytes))
        
        return header + stream_id_header + stream_id_bytes + metadata_header + metadata_bytes + self.payload
    
    @classmethod
    def decode(cls, data: bytes) -> 'StreamMessage':
        """Decode message from binary format"""
        if len(data) < 26:  # Minimum header size
            raise ValueError("Message too short")
        
        # Parse header
        payload_len, msg_type, priority, msg_id, sequence = struct.unpack(
            "!IBB16sI", data[:26]
        )
        
        offset = 26
        
        # Parse stream ID
        stream_id_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        stream_id = data[offset:offset+stream_id_len].decode('utf-8')
        offset += stream_id_len
        
        # Parse metadata
        metadata_len = struct.unpack("!I", data[offset:offset+4])[0]
        offset += 4
        metadata = msgpack.unpackb(data[offset:offset+metadata_len]) if metadata_len > 0 else {}
        offset += metadata_len
        
        # Parse payload
        payload = data[offset:offset+payload_len]
        
        return cls(
            message_id=msg_id,
            message_type=MessageType(msg_type),
            priority=MessagePriority(priority),
            stream_id=stream_id,
            sequence=sequence,
            payload=payload,
            metadata=metadata
        )

class BackpressureQueue:
    """Priority queue with backpressure handling"""
    
    def __init__(self, maxsize: int = 1000, high_watermark: float = 0.8, low_watermark: float = 0.3):
        self.maxsize = maxsize
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._queue: List[Tuple[int, float, StreamMessage]] = []  # Heap: (priority, timestamp, message)
        self._queue_map: Dict[bytes, StreamMessage] = {}  # For O(1) lookup by message_id
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._lock = asyncio.Lock()
        self._metrics = MetricsCollector()
        
    @property
    def size(self) -> int:
        return len(self._queue)
    
    @property
    def is_full(self) -> bool:
        return self.size >= self.maxsize
    
    @property
    def utilization(self) -> float:
        return self.size / self.maxsize
    
    async def put(self, message: StreamMessage) -> bool:
        """Add message to queue with backpressure handling"""
        async with self._lock:
            if self.is_full:
                self._metrics.increment("backpressure.queue_full")
                return False
            
            # Check if we need to apply backpressure
            if self.utilization >= self.high_watermark and not self._paused:
                self._apply_backpressure()
            
            # Add to heap (priority, timestamp for FIFO within same priority)
            heapq.heappush(self._queue, (message.priority, message.timestamp, message))
            self._queue_map[message.message_id] = message
            
            self._metrics.gauge("backpressure.queue_size", self.size)
            self._metrics.increment("backpressure.messages_enqueued")
            
            return True
    
    async def get(self) -> Optional[StreamMessage]:
        """Get highest priority message from queue"""
        async with self._lock:
            if not self._queue:
                return None
            
            # Get highest priority message (lowest priority number)
            _, _, message = heapq.heappop(self._queue)
            del self._queue_map[message.message_id]
            
            # Check if we can release backpressure
            if self._paused and self.utilization <= self.low_watermark:
                self._release_backpressure()
            
            self._metrics.gauge("backpressure.queue_size", self.size)
            self._metrics.increment("backpressure.messages_dequeued")
            
            return message
    
    def _apply_backpressure(self):
        """Apply backpressure by pausing producers"""
        self._paused = True
        self._pause_event.clear()
        logger.warning(f"Backpressure applied: queue at {self.utilization:.1%} capacity")
        self._metrics.increment("backpressure.applied")
    
    def _release_backpressure(self):
        """Release backpressure and resume producers"""
        self._paused = False
        self._pause_event.set()
        logger.info("Backpressure released")
        self._metrics.increment("backpressure.released")
    
    async def wait_for_capacity(self):
        """Wait until queue has capacity (for producers)"""
        if self._paused:
            await self._pause_event.wait()
    
    def remove_by_stream(self, stream_id: str) -> int:
        """Remove all messages for a specific stream (for cleanup)"""
        removed = 0
        with self._lock:
            # Rebuild queue without messages from this stream
            new_queue = []
            for item in self._queue:
                if item[2].stream_id != stream_id:
                    new_queue.append(item)
                else:
                    removed += 1
                    del self._queue_map[item[2].message_id]
            
            self._queue = new_queue
            heapq.heapify(self._queue)
        
        if removed > 0:
            self._metrics.increment("backpressure.messages_removed", removed)
        
        return removed

class FlowController:
    """Manages flow control between nexus"""
    
    def __init__(self, window_size: int = 100, ack_timeout: float = 5.0):
        self.window_size = window_size
        self.ack_timeout = ack_timeout
        self._pending_acks: Dict[bytes, asyncio.Future] = {}  # message_id -> future
        self._send_window: Set[bytes] = set()  # Messages in flight
        self._sequence_numbers: Dict[str, int] = defaultdict(int)  # stream_id -> next sequence
        self._metrics = MetricsCollector()
        
    def get_next_sequence(self, stream_id: str) -> int:
        """Get next sequence number for a stream"""
        seq = self._sequence_numbers[stream_id]
        self._sequence_numbers[stream_id] += 1
        return seq
    
    async def send_with_flow_control(
        self, 
        message: StreamMessage, 
        transport: asyncio.Transport,
        retry_policy: Optional[RetryPolicy] = None
    ) -> bool:
        """Send message with flow control and optional retry"""
        if len(self._send_window) >= self.window_size:
            self._metrics.increment("flow_control.window_full")
            return False
        
        message_id = message.message_id
        self._send_window.add(message_id)
        
        # Create future for ACK
        future = asyncio.Future()
        self._pending_acks[message_id] = future
        
        try:
            # Send message
            transport.write(message.encode())
            self._metrics.increment("flow_control.messages_sent")
            
            # Wait for ACK with timeout
            try:
                await asyncio.wait_for(future, self.ack_timeout)
                self._metrics.increment("flow_control.acks_received")
                return True
            except asyncio.TimeoutError:
                self._metrics.increment("flow_control.acks_timeout")
                if retry_policy:
                    return await retry_policy.execute(
                        lambda: self._retry_send(message, transport)
                    )
                return False
        finally:
            self._send_window.discard(message_id)
            self._pending_acks.pop(message_id, None)
    
    async def _retry_send(self, message: StreamMessage, transport: asyncio.Transport) -> bool:
        """Retry sending a message"""
        transport.write(message.encode())
        future = asyncio.Future()
        self._pending_acks[message.message_id] = future
        
        try:
            await asyncio.wait_for(future, self.ack_timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def handle_ack(self, message_id: bytes):
        """Handle incoming ACK"""
        if message_id in self._pending_acks:
            self._pending_acks[message_id].set_result(True)
    
    def handle_nack(self, message_id: bytes, reason: str = ""):
        """Handle incoming NACK"""
        if message_id in self._pending_acks:
            self._pending_acks[message_id].set_exception(
                Exception(f"NACK received: {reason}")
            )

class StreamManager:
    """Manages active streams between nexus"""
    
    def __init__(self):
        self._streams: Dict[str, 'AgentStream'] = {}
        self._stream_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._metrics = MetricsCollector()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            name="stream_manager"
        )
    
    async def create_stream(
        self, 
        stream_id: str, 
        source_agent: str,
        target_agent: str,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> 'AgentStream':
        """Create a new stream between nexus"""
        if stream_id in self._streams:
            raise ValueError(f"Stream {stream_id} already exists")
        
        stream = AgentStream(
            stream_id=stream_id,
            source_agent=source_agent,
            target_agent=target_agent,
            priority=priority,
            manager=self
        )
        
        self._streams[stream_id] = stream
        self._metrics.increment("streams.created")
        
        # Notify callbacks
        for callback in self._stream_callbacks.get("stream_created", []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stream)
                else:
                    callback(stream)
            except Exception as e:
                logger.error(f"Error in stream_created callback: {e}")
        
        return stream
    
    def get_stream(self, stream_id: str) -> Optional['AgentStream']:
        """Get existing stream by ID"""
        return self._streams.get(stream_id)
    
    async def close_stream(self, stream_id: str, reason: str = "completed"):
        """Close and cleanup a stream"""
        stream = self._streams.pop(stream_id, None)
        if stream:
            await stream.close(reason)
            self._metrics.increment("streams.closed")
            
            # Notify callbacks
            for callback in self._stream_callbacks.get("stream_closed", []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(stream, reason)
                    else:
                        callback(stream, reason)
                except Exception as e:
                    logger.error(f"Error in stream_closed callback: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for stream events"""
        self._stream_callbacks[event].append(callback)
    
    async def broadcast_to_stream(self, stream_id: str, message: StreamMessage):
        """Broadcast message to all participants in a stream"""
        stream = self.get_stream(stream_id)
        if stream:
            await stream.broadcast(message)

class AgentStream:
    """Represents a streaming connection between nexus"""
    
    def __init__(
        self,
        stream_id: str,
        source_agent: str,
        target_agent: str,
        priority: MessagePriority,
        manager: StreamManager
    ):
        self.stream_id = stream_id
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.priority = priority
        self.manager = manager
        self.created_at = time.time()
        self.last_activity = self.created_at
        
        # Stream state
        self._is_active = True
        self._sequence = 0
        self._buffer = bytearray()
        self._subscribers: Set[Callable] = set()
        
        # Flow control
        self._flow_controller = FlowController()
        self._incoming_queue = BackpressureQueue(maxsize=500)
        
        # Metrics
        self._metrics = MetricsCollector()
        self._tracer = get_tracer()
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start stream processing"""
        if not self._processing_task:
            self._processing_task = asyncio.create_task(self._process_incoming())
    
    async def send(self, payload: bytes, metadata: Optional[Dict] = None) -> bool:
        """Send data through the stream"""
        if not self._is_active:
            return False
        
        message = StreamMessage(
            message_type=MessageType.STREAM_DATA,
            priority=self.priority,
            stream_id=self.stream_id,
            sequence=self._sequence,
            payload=payload,
            metadata=metadata or {}
        )
        
        self._sequence += 1
        self.last_activity = time.time()
        
        # Add to incoming queue for processing
        success = await self._incoming_queue.put(message)
        if success:
            self._metrics.increment(f"stream.{self.stream_id}.messages_sent")
        
        return success
    
    async def _process_incoming(self):
        """Process incoming messages from the queue"""
        while self._is_active:
            try:
                # Wait for backpressure if needed
                await self._incoming_queue.wait_for_capacity()
                
                message = await self._incoming_queue.get()
                if not message:
                    await asyncio.sleep(0.01)  # Prevent busy waiting
                    continue
                
                # Process message
                await self._handle_message(message)
                
            except Exception as e:
                logger.error(f"Error processing stream {self.stream_id}: {e}")
                self._metrics.increment(f"stream.{self.stream_id}.processing_errors")
    
    async def _handle_message(self, message: StreamMessage):
        """Handle incoming message"""
        with self._tracer.start_span("stream.message_handle") as span:
            span.set_attribute("stream.id", self.stream_id)
            span.set_attribute("message.type", message.message_type.name)
            span.set_attribute("message.priority", message.priority.name)
            
            try:
                # Notify subscribers
                for callback in self._subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in stream subscriber: {e}")
                
                self._metrics.increment(f"stream.{self.stream_id}.messages_processed")
                
            except Exception as e:
                span.record_exception(e)
                raise
    
    def subscribe(self, callback: Callable[[StreamMessage], Awaitable[None]]):
        """Subscribe to stream messages"""
        self._subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from stream messages"""
        self._subscribers.discard(callback)
    
    async def broadcast(self, message: StreamMessage):
        """Broadcast message to all subscribers"""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Error broadcasting to subscriber: {e}")
    
    async def close(self, reason: str = "completed"):
        """Close the stream"""
        self._is_active = False
        
        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Send stream end message
        end_message = StreamMessage(
            message_type=MessageType.STREAM_END,
            priority=self.priority,
            stream_id=self.stream_id,
            sequence=self._sequence,
            payload=b"",
            metadata={"reason": reason}
        )
        
        await self.broadcast(end_message)
        
        # Cleanup
        self._subscribers.clear()
        self._metrics.increment(f"stream.{self.stream_id}.closed")

class StreamingAgentCommunicator:
    """Main class for agent-to-agent streaming communication"""
    
    def __init__(self, agent_id: str, max_concurrent_streams: int = 100):
        self.agent_id = agent_id
        self.max_concurrent_streams = max_concurrent_streams
        
        # Core components
        self.stream_manager = StreamManager()
        self.outgoing_queue = BackpressureQueue(maxsize=10000)
        self.flow_controllers: Dict[str, FlowController] = {}  # target_agent -> controller
        
        # Connection management
        self._connections: Dict[str, asyncio.Transport] = {}
        self._connection_futures: Dict[str, asyncio.Future] = {}
        
        # Background tasks
        self._sending_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Metrics and monitoring
        self._metrics = MetricsCollector()
        self._tracer = get_tracer()
        
        # Resilience
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=60.0,
            name=f"communicator_{agent_id}"
        )
        self._retry_policy = RetryPolicy(
            max_retries=3,
            backoff_factor=2.0,
            max_backoff=30.0
        )
        
        logger.info(f"Streaming communicator initialized for agent {agent_id}")
    
    async def start(self):
        """Start the communicator"""
        self._sending_task = asyncio.create_task(self._process_outgoing())
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
        
        # Register metrics
        self._metrics.register_gauge("communicator.active_streams", 
                                    lambda: len(self.stream_manager._streams))
        self._metrics.register_gauge("communicator.outgoing_queue_size",
                                    lambda: self.outgoing_queue.size)
    
    async def stop(self):
        """Stop the communicator gracefully"""
        # Cancel background tasks
        if self._sending_task:
            self._sending_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._sending_task, self._heartbeat_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Close all streams
        for stream_id in list(self.stream_manager._streams.keys()):
            await self.stream_manager.close_stream(stream_id, "communicator_shutdown")
        
        # Close connections
        for transport in self._connections.values():
            transport.close()
        
        logger.info(f"Streaming communicator stopped for agent {self.agent_id}")
    
    async def connect_to_agent(self, target_agent: str, host: str, port: int):
        """Establish connection to another agent"""
        if target_agent in self._connections:
            return
        
        try:
            # Create connection with backpressure
            transport, protocol = await asyncio.get_event_loop().create_connection(
                lambda: AgentProtocol(self, target_agent),
                host, port
            )
            
            self._connections[target_agent] = transport
            self._connection_futures[target_agent] = asyncio.Future()
            self.flow_controllers[target_agent] = FlowController()
            
            self._metrics.increment("communicator.connections_established")
            logger.info(f"Connected to agent {target_agent} at {host}:{port}")
            
        except Exception as e:
            self._metrics.increment("communicator.connection_errors")
            logger.error(f"Failed to connect to agent {target_agent}: {e}")
            raise
    
    async def create_stream(
        self,
        target_agent: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        stream_id: Optional[str] = None
    ) -> AgentStream:
        """Create a new stream to target agent"""
        if target_agent not in self._connections:
            raise ValueError(f"No connection to agent {target_agent}")
        
        stream_id = stream_id or f"{self.agent_id}-{target_agent}-{uuid.uuid4().hex[:8]}"
        
        stream = await self.stream_manager.create_stream(
            stream_id=stream_id,
            source_agent=self.agent_id,
            target_agent=target_agent,
            priority=priority
        )
        
        # Send stream start message
        start_message = StreamMessage(
            message_type=MessageType.STREAM_START,
            priority=priority,
            stream_id=stream_id,
            payload=b"",
            metadata={
                "source_agent": self.agent_id,
                "target_agent": target_agent,
                "created_at": time.time()
            }
        )
        
        await self._queue_message(target_agent, start_message)
        await stream.start()
        
        return stream
    
    async def send_to_stream(self, stream_id: str, payload: bytes, metadata: Optional[Dict] = None):
        """Send data to an existing stream"""
        stream = self.stream_manager.get_stream(stream_id)
        if not stream:
            raise ValueError(f"Stream {stream_id} not found")
        
        return await stream.send(payload, metadata)
    
    async def _queue_message(self, target_agent: str, message: StreamMessage):
        """Queue message for sending with backpressure handling"""
        # Wait for backpressure if needed
        await self.outgoing_queue.wait_for_capacity()
        
        # Add target agent to metadata for routing
        message.metadata["target_agent"] = target_agent
        
        success = await self.outgoing_queue.put(message)
        if not success:
            self._metrics.increment("communicator.messages_dropped")
            logger.warning(f"Message dropped due to full queue for agent {target_agent}")
        
        return success
    
    async def _process_outgoing(self):
        """Process outgoing messages from queue"""
        while True:
            try:
                message = await self.outgoing_queue.get()
                if not message:
                    await asyncio.sleep(0.01)
                    continue
                
                target_agent = message.metadata.get("target_agent")
                if not target_agent or target_agent not in self._connections:
                    logger.warning(f"No connection for target agent: {target_agent}")
                    continue
                
                # Get flow controller for target
                flow_controller = self.flow_controllers.get(target_agent)
                if not flow_controller:
                    continue
                
                # Send with flow control and retry
                transport = self._connections[target_agent]
                
                success = await self._circuit_breaker.execute(
                    lambda: flow_controller.send_with_flow_control(
                        message, transport, self._retry_policy
                    )
                )
                
                if success:
                    self._metrics.increment("communicator.messages_sent")
                else:
                    self._metrics.increment("communicator.messages_failed")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing outgoing messages: {e}")
                self._metrics.increment("communicator.processing_errors")
    
    async def _send_heartbeats(self):
        """Send periodic heartbeats to maintain connections"""
        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                for target_agent in list(self._connections.keys()):
                    heartbeat = StreamMessage(
                        message_type=MessageType.HEARTBEAT,
                        priority=MessagePriority.LOW,
                        stream_id="heartbeat",
                        payload=b"",
                        metadata={
                            "source_agent": self.agent_id,
                            "timestamp": time.time()
                        }
                    )
                    
                    await self._queue_message(target_agent, heartbeat)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending heartbeats: {e}")
    
    def handle_incoming_message(self, source_agent: str, data: bytes):
        """Handle incoming message from another agent"""
        try:
            message = StreamMessage.decode(data)
            
            # Update metrics
            self._metrics.increment("communicator.messages_received")
            self._metrics.increment(f"communicator.messages_received.{message.message_type.name}")
            
            # Handle based on message type
            if message.message_type == MessageType.STREAM_START:
                asyncio.create_task(self._handle_stream_start(source_agent, message))
            elif message.message_type == MessageType.STREAM_DATA:
                asyncio.create_task(self._handle_stream_data(source_agent, message))
            elif message.message_type == MessageType.STREAM_END:
                asyncio.create_task(self._handle_stream_end(source_agent, message))
            elif message.message_type == MessageType.ACK:
                self._handle_ack(source_agent, message)
            elif message.message_type == MessageType.NACK:
                self._handle_nack(source_agent, message)
            elif message.message_type == MessageType.HEARTBEAT:
                self._handle_heartbeat(source_agent, message)
                
        except Exception as e:
            logger.error(f"Error handling incoming message from {source_agent}: {e}")
            self._metrics.increment("communicator.message_decode_errors")
    
    async def _handle_stream_start(self, source_agent: str, message: StreamMessage):
        """Handle incoming stream start"""
        stream_id = message.stream_id
        
        # Create stream if it doesn't exist
        if not self.stream_manager.get_stream(stream_id):
            stream = await self.stream_manager.create_stream(
                stream_id=stream_id,
                source_agent=source_agent,
                target_agent=self.agent_id,
                priority=message.priority
            )
            await stream.start()
            
            # Send ACK
            ack = StreamMessage(
                message_type=MessageType.ACK,
                priority=message.priority,
                stream_id=stream_id,
                sequence=message.sequence,
                message_id=message.message_id
            )
            await self._queue_message(source_agent, ack)
    
    async def _handle_stream_data(self, source_agent: str, message: StreamMessage):
        """Handle incoming stream data"""
        stream = self.stream_manager.get_stream(message.stream_id)
        if stream:
            # Add to stream's incoming queue
            await stream._incoming_queue.put(message)
            
            # Send ACK
            ack = StreamMessage(
                message_type=MessageType.ACK,
                priority=message.priority,
                stream_id=message.stream_id,
                sequence=message.sequence,
                message_id=message.message_id
            )
            await self._queue_message(source_agent, ack)
    
    async def _handle_stream_end(self, source_agent: str, message: StreamMessage):
        """Handle stream end"""
        await self.stream_manager.close_stream(
            message.stream_id,
            message.metadata.get("reason", "remote_close")
        )
    
    def _handle_ack(self, source_agent: str, message: StreamMessage):
        """Handle ACK message"""
        flow_controller = self.flow_controllers.get(source_agent)
        if flow_controller:
            flow_controller.handle_ack(message.message_id)
    
    def _handle_nack(self, source_agent: str, message: StreamMessage):
        """Handle NACK message"""
        flow_controller = self.flow_controllers.get(source_agent)
        if flow_controller:
            flow_controller.handle_nack(
                message.message_id,
                message.metadata.get("reason", "")
            )
    
    def _handle_heartbeat(self, source_agent: str, message: StreamMessage):
        """Handle heartbeat message"""
        # Update connection health
        self._metrics.gauge(
            f"communicator.heartbeat_latency.{source_agent}",
            time.time() - message.timestamp
        )

class AgentProtocol(asyncio.Protocol):
    """Asyncio protocol for agent communication"""
    
    def __init__(self, communicator: StreamingAgentCommunicator, agent_id: str):
        self.communicator = communicator
        self.agent_id = agent_id
        self.transport: Optional[asyncio.Transport] = None
        self._buffer = bytearray()
    
    def connection_made(self, transport: asyncio.Transport):
        """Called when connection is established"""
        self.transport = transport
        logger.info(f"Connection established with agent {self.agent_id}")
    
    def data_received(self, data: bytes):
        """Called when data is received"""
        self._buffer.extend(data)
        
        # Process complete messages
        while len(self._buffer) >= 4:
            # Read message length
            msg_len = struct.unpack("!I", self._buffer[:4])[0]
            
            if len(self._buffer) < 4 + msg_len:
                break  # Wait for more data
            
            # Extract message
            message_data = self._buffer[4:4+msg_len]
            self._buffer = self._buffer[4+msg_len:]
            
            # Handle message
            self.communicator.handle_incoming_message(self.agent_id, message_data)
    
    def connection_lost(self, exc: Optional[Exception]):
        """Called when connection is lost"""
        if exc:
            logger.error(f"Connection lost with agent {self.agent_id}: {exc}")
        else:
            logger.info(f"Connection closed with agent {self.agent_id}")
        
        # Cleanup
        if self.agent_id in self.communicator._connections:
            del self.communicator._connections[self.agent_id]
        if self.agent_id in self.communicator.flow_controllers:
            del self.communicator.flow_controllers[self.agent_id]

# Integration with existing modules
class StreamingIntegration:
    """Integration layer with existing SOVEREIGN modules"""
    
    @staticmethod
    def integrate_with_executor(executor_module):
        """Integrate streaming with distributed executor"""
        # Add streaming capabilities to executor
        executor_module.streaming_communicator = None
        
        original_execute = executor_module.execute_task
        
        async def streaming_execute(task, *args, **kwargs):
            # Check if task supports streaming
            if hasattr(task, 'streaming_enabled') and task.streaming_enabled:
                # Use streaming execution
                communicator = executor_module.streaming_communicator
                if not communicator:
                    communicator = StreamingAgentCommunicator(f"executor-{uuid.uuid4().hex[:8]}")
                    await communicator.start()
                    executor_module.streaming_communicator = communicator
                
                # Create stream for task
                stream = await communicator.create_stream(
                    target_agent=task.target_agent,
                    priority=MessagePriority.NORMAL
                )
                
                # Stream task execution
                async for result in task.stream_execute(*args, **kwargs):
                    await stream.send(result)
                
                await stream.close()
                return {"status": "streamed", "stream_id": stream.stream_id}
            else:
                # Use original execution
                return await original_execute(task, *args, **kwargs)
        
        executor_module.execute_task = streaming_execute
    
    @staticmethod
    def integrate_with_tracing(tracing_module):
        """Integrate streaming with tracing"""
        original_start_span = tracing_module.start_span
        
        def streaming_start_span(name, *args, **kwargs):
            span = original_start_span(name, *args, **kwargs)
            
            # Add streaming context to span
            if hasattr(tracing_module, 'current_stream_id'):
                span.set_attribute("stream.id", tracing_module.current_stream_id)
            
            return span
        
        tracing_module.start_span = streaming_start_span
    
    @staticmethod
    def integrate_with_metrics(metrics_module):
        """Integrate streaming metrics with existing metrics collector"""
        # Register streaming metrics
        metrics_module.register_counter("streaming.messages.sent")
        metrics_module.register_counter("streaming.messages.received")
        metrics_module.register_counter("streaming.backpressure.applied")
        metrics_module.register_gauge("streaming.active_streams")
        metrics_module.register_histogram("streaming.message.size")

# Factory function for easy instantiation
def create_streaming_communicator(
    agent_id: str,
    max_concurrent_streams: int = 100,
    enable_monitoring: bool = True
) -> StreamingAgentCommunicator:
    """Create and configure a streaming communicator"""
    communicator = StreamingAgentCommunicator(agent_id, max_concurrent_streams)
    
    if enable_monitoring:
        # Integrate with monitoring
        StreamingIntegration.integrate_with_metrics(MetricsCollector())
    
    return communicator

# Export main classes
__all__ = [
    'StreamMessage',
    'MessagePriority',
    'MessageType',
    'BackpressureQueue',
    'FlowController',
    'StreamManager',
    'AgentStream',
    'StreamingAgentCommunicator',
    'StreamingIntegration',
    'create_streaming_communicator'
]