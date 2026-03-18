"""memory/retrieval.py - Persistent Agent Memory System with semantic search and intelligent forgetting"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, AsyncIterator
import numpy as np
from collections import defaultdict
import pickle

# Integration with existing modules
from monitoring.metrics_collector import MetricsCollector, MetricType
from monitoring.tracing import TracingManager, Span
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from core.distributed.state_manager import DistributedStateManager

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories stored in the system"""
    EPISODIC = "episodic"  # Event-based memories
    SEMANTIC = "semantic"  # Fact-based knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"  # Short-term working memory
    CONSOLIDATED = "consolidated"  # Merged/summarized memories


class MemoryImportance(Enum):
    """Importance levels for memory retention"""
    CRITICAL = 1.0  # Never forget
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    EPHEMERAL = 0.1  # Quick forgetting


@dataclass
class Memory:
    """Individual memory unit with metadata"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: MemoryImportance = MemoryImportance.MEDIUM
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source_agent: Optional[str] = None
    decay_rate: float = 0.1  # How quickly this memory fades
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['importance'] = self.importance.value
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Memory':
        """Create Memory from dictionary"""
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        data['memory_type'] = MemoryType(data['memory_type'])
        data['importance'] = MemoryImportance(data['importance'])
        return cls(**data)
    
    def calculate_retention(self, current_time: float) -> float:
        """Calculate memory retention based on forgetting curve"""
        time_since_access = current_time - self.last_accessed
        time_since_creation = current_time - self.created_at
        
        # Ebbinghaus forgetting curve with importance weighting
        base_retention = np.exp(-self.decay_rate * time_since_access / 3600)
        
        # Boost retention based on importance and access frequency
        importance_boost = self.importance.value * 0.5
        frequency_boost = min(0.3, self.access_count * 0.05)
        
        # Consolidated memories decay slower
        consolidation_bonus = 0.2 if self.memory_type == MemoryType.CONSOLIDATED else 0
        
        retention = min(1.0, base_retention + importance_boost + frequency_boost + consolidation_bonus)
        return max(0.0, retention)


class VectorStore(ABC):
    """Abstract base class for vector storage backends"""
    
    @abstractmethod
    async def add(self, memory_id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Add embedding to vector store"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete embedding from store"""
        pass
    
    @abstractmethod
    async def update(self, memory_id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Update existing embedding"""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, dimension: int = 768, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Maps FAISS index to memory_id
        self.metadata_store = {}
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            elif self.index_type == "IVF":
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                self.index.nprobe = 10
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
                
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            
        except ImportError:
            logger.warning("FAISS not installed, falling back to numpy-based search")
            self.index = None
    
    async def add(self, memory_id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Add embedding to FAISS index"""
        try:
            if self.index is None:
                return False
            
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            if isinstance(self.index, type(self.index)) and not self.index.is_trained:
                # Train index if needed (for IVF)
                self.index.train(embedding.reshape(1, -1))
            
            faiss_idx = self.index.ntotal
            self.index.add(embedding.reshape(1, -1))
            self.id_map[faiss_idx] = memory_id
            self.metadata_store[memory_id] = metadata
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to FAISS: {e}")
            return False
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar embeddings"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.index.ntotal)
            )
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:  # Valid index
                    memory_id = self.id_map.get(idx)
                    if memory_id:
                        results.append((memory_id, float(dist)))
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def delete(self, memory_id: str) -> bool:
        """Delete from FAISS (rebuild index)"""
        # FAISS doesn't support deletion, so we mark for rebuild
        # In production, use a more sophisticated approach
        return True
    
    async def update(self, memory_id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Update embedding (requires delete + add)"""
        await self.delete(memory_id)
        return await self.add(memory_id, embedding, metadata)


class MemoryConsolidator:
    """Consolidates similar memories and extracts patterns"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.consolidation_rules = []
        
    def add_consolidation_rule(self, rule: callable):
        """Add custom consolidation rule"""
        self.consolidation_rules.append(rule)
    
    async def consolidate_memories(
        self, 
        memories: List[Memory],
        vector_store: VectorStore
    ) -> List[Memory]:
        """Consolidate similar memories into higher-level memories"""
        if len(memories) < 2:
            return memories
        
        # Group memories by type and tags
        grouped = defaultdict(list)
        for memory in memories:
            key = (memory.memory_type, tuple(sorted(memory.tags)))
            grouped[key].append(memory)
        
        consolidated = []
        
        for (mem_type, tags), group in grouped.items():
            if len(group) < 2:
                consolidated.extend(group)
                continue
            
            # Find clusters of similar memories
            clusters = await self._cluster_memories(group, vector_store)
            
            for cluster in clusters:
                if len(cluster) > 1:
                    # Create consolidated memory
                    consolidated_memory = await self._create_consolidated_memory(cluster)
                    consolidated.append(consolidated_memory)
                else:
                    consolidated.extend(cluster)
        
        return consolidated
    
    async def _cluster_memories(
        self, 
        memories: List[Memory], 
        vector_store: VectorStore
    ) -> List[List[Memory]]:
        """Cluster similar memories together"""
        if not memories:
            return []
        
        # Simple clustering based on embedding similarity
        clusters = []
        used = set()
        
        for i, mem1 in enumerate(memories):
            if i in used:
                continue
                
            cluster = [mem1]
            used.add(i)
            
            if mem1.embedding is not None:
                for j, mem2 in enumerate(memories[i+1:], i+1):
                    if j in used or mem2.embedding is None:
                        continue
                    
                    similarity = np.dot(mem1.embedding, mem2.embedding)
                    if similarity >= self.similarity_threshold:
                        cluster.append(mem2)
                        used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    async def _create_consolidated_memory(self, memories: List[Memory]) -> Memory:
        """Create a consolidated memory from a cluster"""
        # Combine content
        combined_content = " | ".join([m.content for m in memories[:5]])  # Limit to 5
        if len(memories) > 5:
            combined_content += f" ... and {len(memories) - 5} more"
        
        # Average embeddings
        embeddings = [m.embedding for m in memories if m.embedding is not None]
        avg_embedding = np.mean(embeddings, axis=0) if embeddings else None
        
        # Calculate new importance (max of cluster)
        max_importance = max(m.importance for m in memories)
        
        # Combine tags
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)
        
        # Create new memory ID
        content_hash = hashlib.md5(combined_content.encode()).hexdigest()[:12]
        memory_id = f"consolidated_{content_hash}_{int(time.time())}"
        
        return Memory(
            id=memory_id,
            content=combined_content,
            embedding=avg_embedding,
            memory_type=MemoryType.CONSOLIDATED,
            importance=max_importance,
            tags=list(all_tags),
            metadata={
                "source_memories": [m.id for m in memories],
                "consolidation_count": len(memories),
                "original_types": [m.memory_type.value for m in memories]
            }
        )


class ForgettingCurve:
    """Implements configurable forgetting curves"""
    
    def __init__(self, base_decay: float = 0.1, importance_weight: float = 0.5):
        self.base_decay = base_decay
        self.importance_weight = importance_weight
    
    def calculate_forgetting_probability(
        self, 
        memory: Memory, 
        current_time: float
    ) -> float:
        """Calculate probability of forgetting a memory"""
        retention = memory.calculate_retention(current_time)
        forgetting_prob = 1.0 - retention
        
        # Adjust based on memory type
        if memory.memory_type == MemoryType.CONSOLIDATED:
            forgetting_prob *= 0.5  # Consolidated memories are harder to forget
        elif memory.memory_type == MemoryType.WORKING:
            forgetting_prob *= 2.0  # Working memory forgets faster
        
        return min(1.0, max(0.0, forgetting_prob))
    
    def should_forget(
        self, 
        memory: Memory, 
        current_time: float,
        threshold: float = 0.7
    ) -> bool:
        """Determine if a memory should be forgotten"""
        forgetting_prob = self.calculate_forgetting_probability(memory, current_time)
        return forgetting_prob >= threshold


class AgentMemorySystem:
    """
    Main memory system for nexus with semantic search,
    consolidation, and intelligent forgetting
    """
    
    def __init__(
        self,
        agent_id: str,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[Any] = None,
        consolidator: Optional[MemoryConsolidator] = None,
        forgetting_curve: Optional[ForgettingCurve] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        state_manager: Optional[DistributedStateManager] = None,
        max_memories: int = 10000,
        consolidation_interval: int = 3600,  # 1 hour
        forgetting_interval: int = 300,  # 5 minutes
        persistence_path: str = "./memory_store"
    ):
        self.agent_id = agent_id
        self.vector_store = vector_store or FAISSVectorStore()
        self.embedding_model = embedding_model
        self.consolidator = consolidator or MemoryConsolidator()
        self.forgetting_curve = forgetting_curve or ForgettingCurve()
        self.metrics = metrics_collector or MetricsCollector()
        self.state_manager = state_manager
        self.max_memories = max_memories
        self.persistence_path = persistence_path
        
        # Memory storage
        self.memories: Dict[str, Memory] = {}
        self.memory_index: Dict[str, List[str]] = defaultdict(list)  # tag -> memory_ids
        
        # Background tasks
        self._consolidation_task = None
        self._forgetting_task = None
        self._persistence_task = None
        self.consolidation_interval = consolidation_interval
        self.forgetting_interval = forgetting_interval
        
        # Circuit breaker for vector store operations
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            name=f"memory_vector_store_{agent_id}"
        )
        
        # Retry policy for failed operations
        self.retry_policy = RetryPolicy(
            max_retries=3,
            backoff_factor=2.0,
            retry_on=(ConnectionError, TimeoutError)
        )
        
        # Tracing
        self.tracer = TracingManager()
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize memory system"""
        logger.info(f"Initializing memory system for agent {self.agent_id}")
        
        # Load existing memories from persistence
        asyncio.create_task(self._load_from_disk())
        
        # Start background tasks
        self._start_background_tasks()
        
        # Register metrics
        self._register_metrics()
    
    def _register_metrics(self):
        """Register metrics for monitoring"""
        self.metrics.register_metric(
            name="memory_count",
            metric_type=MetricType.GAUGE,
            description="Total number of memories stored"
        )
        self.metrics.register_metric(
            name="memory_search_latency",
            metric_type=MetricType.HISTOGRAM,
            description="Latency of memory search operations"
        )
        self.metrics.register_metric(
            name="memory_consolidation_count",
            metric_type=MetricType.COUNTER,
            description="Number of memory consolidations performed"
        )
        self.metrics.register_metric(
            name="memory_forgetting_count",
            metric_type=MetricType.COUNTER,
            description="Number of memories forgotten"
        )
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._consolidation_task = asyncio.create_task(
            self._periodic_consolidation()
        )
        self._forgetting_task = asyncio.create_task(
            self._periodic_forgetting()
        )
        self._persistence_task = asyncio.create_task(
            self._periodic_persistence()
        )
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        source_agent: Optional[str] = None
    ) -> str:
        """
        Store a new memory
        
        Returns:
            Memory ID
        """
        with self.tracer.start_span("store_memory") as span:
            span.set_attribute("agent_id", self.agent_id)
            span.set_attribute("memory_type", memory_type.value)
            
            try:
                # Generate memory ID
                content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
                memory_id = f"{self.agent_id}_{content_hash}_{int(time.time())}"
                
                # Generate embedding if model available
                embedding = None
                if self.embedding_model:
                    embedding = await self._generate_embedding(content)
                
                # Create memory object
                memory = Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    memory_type=memory_type,
                    importance=importance,
                    tags=tags or [],
                    metadata=metadata or {},
                    source_agent=source_agent or self.agent_id
                )
                
                # Store in memory
                self.memories[memory_id] = memory
                
                # Update index
                for tag in memory.tags:
                    self.memory_index[tag].append(memory_id)
                
                # Store in vector store if embedding exists
                if embedding is not None:
                    await self._store_in_vector_store(memory)
                
                # Check memory limit
                await self._enforce_memory_limit()
                
                # Update metrics
                self.metrics.set_gauge("memory_count", len(self.memories))
                
                logger.debug(f"Stored memory {memory_id} for agent {self.agent_id}")
                return memory_id
                
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                span.set_status("error", str(e))
                raise
    
    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_importance: Optional[MemoryImportance] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories based on semantic search and filters
        
        Returns:
            List of (Memory, relevance_score) tuples
        """
        with self.tracer.start_span("retrieve_memories") as span:
            span.set_attribute("agent_id", self.agent_id)
            span.set_attribute("query_length", len(query))
            
            start_time = time.time()
            
            try:
                results = []
                
                # Generate query embedding
                query_embedding = None
                if self.embedding_model:
                    query_embedding = await self._generate_embedding(query)
                
                # Semantic search if embedding available
                if query_embedding is not None:
                    semantic_results = await self._semantic_search(
                        query_embedding, 
                        top_k=top_k * 2  # Get more for filtering
                    )
                    
                    for memory_id, similarity in semantic_results:
                        if memory_id in self.memories:
                            memory = self.memories[memory_id]
                            
                            # Apply filters
                            if not self._apply_filters(memory, memory_types, tags, min_importance, time_range):
                                continue
                            
                            # Update access metadata
                            memory.last_accessed = time.time()
                            memory.access_count += 1
                            
                            results.append((memory, similarity))
                
                # Fallback to keyword search if no semantic results
                if not results:
                    results = await self._keyword_search(
                        query, 
                        top_k, 
                        memory_types, 
                        tags, 
                        min_importance, 
                        time_range
                    )
                
                # Sort by relevance and limit
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:top_k]
                
                # Update metrics
                latency = time.time() - start_time
                self.metrics.observe_histogram("memory_search_latency", latency)
                
                logger.debug(f"Retrieved {len(results)} memories for query")
                return results
                
            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")
                span.set_status("error", str(e))
                return []
    
    async def _semantic_search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Perform semantic search using vector store"""
        async with self.circuit_breaker:
            return await self.retry_policy.execute(
                lambda: self.vector_store.search(query_embedding, top_k)
            )
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        memory_types: Optional[List[MemoryType]],
        tags: Optional[List[str]],
        min_importance: Optional[MemoryImportance],
        time_range: Optional[Tuple[float, float]]
    ) -> List[Tuple[Memory, float]]:
        """Fallback keyword-based search"""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories.values():
            # Apply filters
            if not self._apply_filters(memory, memory_types, tags, min_importance, time_range):
                continue
            
            # Simple keyword matching
            content_lower = memory.content.lower()
            if query_lower in content_lower:
                # Calculate simple relevance score
                relevance = content_lower.count(query_lower) / len(content_lower.split())
                results.append((memory, relevance))
        
        return results
    
    def _apply_filters(
        self,
        memory: Memory,
        memory_types: Optional[List[MemoryType]],
        tags: Optional[List[str]],
        min_importance: Optional[MemoryImportance],
        time_range: Optional[Tuple[float, float]]
    ) -> bool:
        """Apply filters to memory"""
        if memory_types and memory.memory_type not in memory_types:
            return False
        
        if tags and not any(tag in memory.tags for tag in tags):
            return False
        
        if min_importance and memory.importance.value < min_importance.value:
            return False
        
        if time_range:
            start_time, end_time = time_range
            if not (start_time <= memory.created_at <= end_time):
                return False
        
        return True
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        try:
            if hasattr(self.embedding_model, 'encode'):
                # Sentence-transformers style
                return self.embedding_model.encode(text)
            elif hasattr(self.embedding_model, 'embed'):
                # Custom embedding model
                return await self.embedding_model.embed(text)
            else:
                logger.warning("Embedding model does not have encode or embed method")
                return None
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def _store_in_vector_store(self, memory: Memory) -> bool:
        """Store memory embedding in vector store"""
        if memory.embedding is None:
            return False
        
        try:
            return await self.vector_store.add(
                memory.id,
                memory.embedding,
                {
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance.value,
                    "tags": memory.tags,
                    "created_at": memory.created_at
                }
            )
        except Exception as e:
            logger.error(f"Failed to store in vector store: {e}")
            return False
    
    async def _periodic_consolidation(self):
        """Periodically consolidate memories"""
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval)
                
                with self.tracer.start_span("memory_consolidation"):
                    logger.info(f"Starting memory consolidation for agent {self.agent_id}")
                    
                    # Get all memories
                    all_memories = list(self.memories.values())
                    
                    # Consolidate
                    consolidated = await self.consolidator.consolidate_memories(
                        all_memories,
                        self.vector_store
                    )
                    
                    # Replace memories with consolidated versions
                    self.memories.clear()
                    self.memory_index.clear()
                    
                    for memory in consolidated:
                        self.memories[memory.id] = memory
                        for tag in memory.tags:
                            self.memory_index[tag].append(memory.id)
                    
                    # Update metrics
                    self.metrics.increment_counter("memory_consolidation_count")
                    self.metrics.set_gauge("memory_count", len(self.memories))
                    
                    logger.info(f"Consolidation complete. Memories: {len(all_memories)} -> {len(consolidated)}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation failed: {e}")
    
    async def _periodic_forgetting(self):
        """Periodically forget low-retention memories"""
        while True:
            try:
                await asyncio.sleep(self.forgetting_interval)
                
                with self.tracer.start_span("memory_forgetting"):
                    current_time = time.time()
                    memories_to_forget = []
                    
                    for memory_id, memory in self.memories.items():
                        if self.forgetting_curve.should_forget(memory, current_time):
                            memories_to_forget.append(memory_id)
                    
                    # Forget memories
                    for memory_id in memories_to_forget:
                        await self.forget_memory(memory_id)
                    
                    if memories_to_forget:
                        logger.info(f"Forgot {len(memories_to_forget)} memories")
                        self.metrics.increment_counter(
                            "memory_forgetting_count", 
                            len(memories_to_forget)
                        )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Forgetting failed: {e}")
    
    async def forget_memory(self, memory_id: str) -> bool:
        """Forget a specific memory"""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        # Remove from vector store
        if memory.embedding is not None:
            await self.vector_store.delete(memory_id)
        
        # Remove from index
        for tag in memory.tags:
            if memory_id in self.memory_index[tag]:
                self.memory_index[tag].remove(memory_id)
        
        # Remove from memory store
        del self.memories[memory_id]
        
        # Update metrics
        self.metrics.set_gauge("memory_count", len(self.memories))
        
        return True
    
    async def _enforce_memory_limit(self):
        """Enforce maximum memory limit by forgetting least important memories"""
        if len(self.memories) <= self.max_memories:
            return
        
        # Sort memories by retention (lowest first)
        current_time = time.time()
        memories_by_retention = sorted(
            self.memories.values(),
            key=lambda m: m.calculate_retention(current_time)
        )
        
        # Forget lowest retention memories until under limit
        forget_count = len(self.memories) - self.max_memories
        for memory in memories_by_retention[:forget_count]:
            await self.forget_memory(memory.id)
        
        logger.info(f"Enforced memory limit, forgot {forget_count} memories")
    
    async def _periodic_persistence(self):
        """Periodically persist memories to disk"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._save_to_disk()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Persistence failed: {e}")
    
    async def _save_to_disk(self):
        """Save memories to disk"""
        try:
            import os
            os.makedirs(self.persistence_path, exist_ok=True)
            
            filepath = os.path.join(self.persistence_path, f"{self.agent_id}_memories.pkl")
            
            # Prepare data for serialization
            data = {
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "memories": [memory.to_dict() for memory in self.memories.values()],
                "memory_index": dict(self.memory_index)
            }
            
            # Save with pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved {len(self.memories)} memories to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save memories to disk: {e}")
    
    async def _load_from_disk(self):
        """Load memories from disk"""
        try:
            import os
            filepath = os.path.join(self.persistence_path, f"{self.agent_id}_memories.pkl")
            
            if not os.path.exists(filepath):
                logger.info(f"No existing memories found at {filepath}")
                return
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Restore memories
            for memory_data in data.get("memories", []):
                memory = Memory.from_dict(memory_data)
                self.memories[memory.id] = memory
            
            # Restore index
            self.memory_index = defaultdict(list, data.get("memory_index", {}))
            
            logger.info(f"Loaded {len(self.memories)} memories from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load memories from disk: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory system"""
        current_time = time.time()
        
        # Calculate statistics
        total_memories = len(self.memories)
        avg_retention = np.mean([
            m.calculate_retention(current_time) 
            for m in self.memories.values()
        ]) if self.memories else 0
        
        type_counts = defaultdict(int)
        importance_counts = defaultdict(int)
        
        for memory in self.memories.values():
            type_counts[memory.memory_type.value] += 1
            importance_counts[memory.importance.value] += 1
        
        return {
            "agent_id": self.agent_id,
            "total_memories": total_memories,
            "average_retention": float(avg_retention),
            "memory_types": dict(type_counts),
            "importance_distribution": dict(importance_counts),
            "vector_store_size": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') and self.vector_store.index else 0,
            "index_size": sum(len(ids) for ids in self.memory_index.values())
        }
    
    async def clear_memories(self, memory_types: Optional[List[MemoryType]] = None):
        """Clear memories, optionally filtered by type"""
        if memory_types:
            to_remove = [
                mem_id for mem_id, mem in self.memories.items()
                if mem.memory_type in memory_types
            ]
            for mem_id in to_remove:
                await self.forget_memory(mem_id)
        else:
            # Clear all
            self.memories.clear()
            self.memory_index.clear()
            # Note: vector store would need separate clearing
        
        logger.info(f"Cleared memories for agent {self.agent_id}")
    
    async def shutdown(self):
        """Shutdown memory system gracefully"""
        logger.info(f"Shutting down memory system for agent {self.agent_id}")
        
        # Cancel background tasks
        if self._consolidation_task:
            self._consolidation_task.cancel()
        if self._forgetting_task:
            self._forgetting_task.cancel()
        if self._persistence_task:
            self._persistence_task.cancel()
        
        # Final persistence
        await self._save_to_disk()
        
        logger.info(f"Memory system shutdown complete for agent {self.agent_id}")


# Factory function for easy instantiation
def create_memory_system(
    agent_id: str,
    embedding_model: Optional[Any] = None,
    vector_store_type: str = "faiss",
    **kwargs
) -> AgentMemorySystem:
    """Factory function to create memory system with common configurations"""
    
    # Create vector store based on type
    if vector_store_type == "faiss":
        vector_store = FAISSVectorStore()
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")
    
    # Create memory system
    return AgentMemorySystem(
        agent_id=agent_id,
        vector_store=vector_store,
        embedding_model=embedding_model,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    from sentence_transformers import SentenceTransformer
    
    async def example_usage():
        # Initialize embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create memory system
        memory_system = create_memory_system(
            agent_id="agent_001",
            embedding_model=model,
            max_memories=1000
        )
        
        # Store some memories
        await memory_system.store_memory(
            "The user prefers dark mode interfaces",
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH,
            tags=["preferences", "ui"]
        )
        
        await memory_system.store_memory(
            "User asked about Python asyncio patterns",
            memory_type=MemoryType.EPISODIC,
            importance=MemoryImportance.MEDIUM,
            tags=["questions", "python"]
        )
        
        # Retrieve memories
        results = await memory_system.retrieve_memories(
            "What are the user's preferences?",
            top_k=5
        )
        
        for memory, score in results:
            print(f"Score: {score:.3f} - {memory.content}")
        
        # Get statistics
        stats = await memory_system.get_memory_stats()
        print(f"Memory stats: {stats}")
        
        # Shutdown
        await memory_system.shutdown()
    
    # Run example
    asyncio.run(example_usage())