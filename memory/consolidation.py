"""
memory/consolidation.py - Persistent Agent Memory System
Long-term memory with semantic search, consolidation, and forgetting mechanisms.
"""

import asyncio
import time
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib
from collections import defaultdict

# Integration with existing modules
from core.distributed.state_manager import StateManager
from core.distributed.executor import DistributedExecutor
from monitoring.metrics_collector import MetricsCollector
from monitoring.tracing import TracingManager
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy

# Vector database abstraction (pluggable backends)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class MemoryType(Enum):
    """Types of memories with different retention policies."""
    EPISODIC = "episodic"      # Specific events/experiences
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"        # Temporary working memory

class MemoryImportance(Enum):
    """Importance levels affecting retention."""
    CRITICAL = 1.0      # Never forget (system configs, core knowledge)
    HIGH = 0.8          # Long retention
    MEDIUM = 0.5        # Standard retention
    LOW = 0.3           # Short retention
    EPHEMERAL = 0.1     # Very short retention

@dataclass
class Memory:
    """Individual memory unit with metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    consolidation_level: int = 0  # How many times consolidated
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_agent: Optional[str] = None
    related_memories: List[str] = field(default_factory=list)  # IDs of related memories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        if 'memory_type' in data:
            data['memory_type'] = MemoryType(data['memory_type'])
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_accessed' in data:
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        if 'tags' in data:
            data['tags'] = set(data['tags'])
        return cls(**data)

class VectorStore:
    """Abstract vector store interface."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    async def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        raise NotImplementedError
    
    async def search(self, query_embedding: List[float], top_k: int = 10, 
                    filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        raise NotImplementedError
    
    async def delete(self, id: str):
        raise NotImplementedError
    
    async def update(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        raise NotImplementedError

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, dimension: int = 768, collection_name: str = "agent_memories"):
        super().__init__(dimension)
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./memory/vector_db"
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        self.collection.add(
            embeddings=[embedding],
            ids=[id],
            metadatas=[metadata or {}]
        )
    
    async def search(self, query_embedding: List[float], top_k: int = 10,
                    filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        if not results['ids'][0]:
            return []
        
        return list(zip(results['ids'][0], results['distances'][0]))
    
    async def delete(self, id: str):
        self.collection.delete(ids=[id])
    
    async def update(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        self.collection.update(
            ids=[id],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )

class FaissVectorStore(VectorStore):
    """FAISS implementation of vector store (for high-performance scenarios)."""
    
    def __init__(self, dimension: int = 768):
        super().__init__(dimension)
        if not FAISS_AVAILABLE:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu")
        
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = {}  # Maps FAISS index to memory ID
        self.metadata_map = {}  # Maps memory ID to metadata
        self.next_index = 0
    
    async def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        embedding_array = np.array([embedding], dtype=np.float32)
        self.index.add(embedding_array)
        self.id_map[self.next_index] = id
        self.metadata_map[id] = metadata or {}
        self.next_index += 1
    
    async def search(self, query_embedding: List[float], top_k: int = 10,
                    filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        query_array = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_array, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            memory_id = self.id_map.get(idx)
            if memory_id and self._matches_filter(memory_id, filter_metadata):
                results.append((memory_id, float(distance)))
        
        return results
    
    def _matches_filter(self, memory_id: str, filter_metadata: Dict[str, Any]) -> bool:
        if not filter_metadata:
            return True
        metadata = self.metadata_map.get(memory_id, {})
        for key, value in filter_metadata.items():
            if metadata.get(key) != value:
                return False
        return True
    
    async def delete(self, id: str):
        # FAISS doesn't support deletion, mark as deleted
        for idx, mem_id in list(self.id_map.items()):
            if mem_id == id:
                del self.id_map[idx]
                break
        if id in self.metadata_map:
            del self.metadata_map[id]
    
    async def update(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        # FAISS doesn't support update, delete and re-add
        await self.delete(id)
        await self.add(id, embedding, metadata)

class ForgettingCurve:
    """Ebbinghaus-inspired forgetting curve with importance weighting."""
    
    def __init__(self, 
                 base_retention: float = 0.5,
                 importance_weight: float = 2.0,
                 access_boost: float = 0.1):
        self.base_retention = base_retention
        self.importance_weight = importance_weight
        self.access_boost = access_boost
    
    def calculate_retention(self, 
                           memory: Memory, 
                           current_time: Optional[datetime] = None) -> float:
        """Calculate current retention probability for a memory."""
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Time since last access (in days)
        time_delta = (current_time - memory.last_accessed).total_seconds() / 86400
        
        # Base forgetting curve: R = e^(-t/S)
        # Where S is stability (modified by importance and access)
        stability = self.base_retention * (1 + memory.importance * self.importance_weight)
        stability *= (1 + memory.access_count * self.access_boost)
        
        retention = np.exp(-time_delta / stability)
        
        # Apply consolidation bonus
        consolidation_bonus = 1 + (memory.consolidation_level * 0.1)
        retention = min(1.0, retention * consolidation_bonus)
        
        return float(retention)
    
    def should_forget(self, memory: Memory, threshold: float = 0.1) -> bool:
        """Determine if a memory should be forgotten."""
        retention = self.calculate_retention(memory)
        return retention < threshold

class MemoryConsolidation:
    """Memory consolidation strategies."""
    
    @staticmethod
    def cluster_memories(memories: List[Memory], 
                        similarity_threshold: float = 0.85) -> List[List[Memory]]:
        """Cluster similar memories for consolidation."""
        if not memories:
            return []
        
        # Simple clustering based on tags and metadata
        clusters = []
        used = set()
        
        for i, mem1 in enumerate(memories):
            if i in used:
                continue
            
            cluster = [mem1]
            used.add(i)
            
            for j, mem2 in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                # Calculate similarity (simplified)
                tag_overlap = len(mem1.tags & mem2.tags) / max(len(mem1.tags | mem2.tags), 1)
                type_match = 1.0 if mem1.memory_type == mem2.memory_type else 0.0
                
                similarity = (tag_overlap * 0.7 + type_match * 0.3)
                
                if similarity >= similarity_threshold:
                    cluster.append(mem2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    def consolidate_cluster(cluster: List[Memory]) -> Memory:
        """Consolidate a cluster of similar memories into one."""
        if not cluster:
            raise ValueError("Empty cluster")
        
        if len(cluster) == 1:
            return cluster[0]
        
        # Sort by importance and recency
        cluster.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        # Take the most important/recent as base
        base = cluster[0]
        
        # Combine content (summarize in real implementation)
        contents = [m.content for m in cluster]
        combined_content = f"Consolidated from {len(cluster)} memories:\n" + "\n".join(contents[:5])
        
        # Combine tags
        all_tags = set()
        for mem in cluster:
            all_tags.update(mem.tags)
        
        # Calculate new importance (weighted average)
        total_weight = sum(m.importance for m in cluster)
        weighted_importance = sum(m.importance * m.importance for m in cluster) / total_weight
        
        # Create consolidated memory
        consolidated = Memory(
            id=base.id,
            content=combined_content,
            memory_type=base.memory_type,
            importance=weighted_importance,
            created_at=min(m.created_at for m in cluster),
            last_accessed=max(m.last_accessed for m in cluster),
            access_count=sum(m.access_count for m in cluster),
            consolidation_level=max(m.consolidation_level for m in cluster) + 1,
            tags=all_tags,
            metadata={
                "consolidated_from": [m.id for m in cluster],
                "consolidation_count": len(cluster)
            },
            source_agent=base.source_agent,
            related_memories=list(set(mid for m in cluster for mid in m.related_memories))
        )
        
        return consolidated

class PersistentMemorySystem:
    """
    Main persistent memory system for nexus.
    Integrates with SOVEREIGN's distributed infrastructure.
    """
    
    def __init__(self,
                 agent_id: str,
                 state_manager: Optional[StateManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 tracing: Optional[TracingManager] = None,
                 vector_store_type: str = "chroma",
                 embedding_dimension: int = 768,
                 config: Optional[Dict[str, Any]] = None):
        
        self.agent_id = agent_id
        self.state_manager = state_manager or StateManager()
        self.metrics = metrics_collector or MetricsCollector()
        self.tracing = tracing or TracingManager()
        
        # Configuration
        self.config = config or {}
        self.consolidation_interval = self.config.get("consolidation_interval", 3600)  # 1 hour
        self.forgetting_threshold = self.config.get("forgetting_threshold", 0.1)
        self.max_memories = self.config.get("max_memories", 10000)
        self.batch_size = self.config.get("batch_size", 100)
        
        # Initialize vector store
        self.vector_store = self._create_vector_store(vector_store_type, embedding_dimension)
        
        # Forgetting curve
        self.forgetting_curve = ForgettingCurve(
            base_retention=self.config.get("base_retention", 0.5),
            importance_weight=self.config.get("importance_weight", 2.0),
            access_boost=self.config.get("access_boost", 0.1)
        )
        
        # Memory consolidation
        self.consolidation = MemoryConsolidation()
        
        # In-memory cache for frequently accessed memories
        self.memory_cache: Dict[str, Memory] = {}
        self.cache_size = self.config.get("cache_size", 1000)
        
        # Background tasks
        self._consolidation_task = None
        self._cleanup_task = None
        self._running = False
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            name=f"memory_system_{agent_id}"
        )
        
        # Retry policy for transient failures
        self.retry_policy = RetryPolicy(
            max_retries=3,
            backoff_factor=2,
            retry_on=(ConnectionError, TimeoutError)
        )
        
        # Register metrics
        self._register_metrics()
    
    def _create_vector_store(self, store_type: str, dimension: int) -> VectorStore:
        """Factory method for vector store creation."""
        if store_type == "chroma" and CHROMA_AVAILABLE:
            return ChromaVectorStore(dimension=dimension, 
                                   collection_name=f"agent_{self.agent_id}_memories")
        elif store_type == "faiss" and FAISS_AVAILABLE:
            return FaissVectorStore(dimension=dimension)
        else:
            # Fallback to in-memory simple store
            return InMemoryVectorStore(dimension=dimension)
    
    def _register_metrics(self):
        """Register metrics for monitoring."""
        self.metrics.register_counter("memory_operations_total", 
                                     labels=["operation", "status"])
        self.metrics.register_histogram("memory_operation_duration_seconds",
                                       labels=["operation"])
        self.metrics.register_gauge("memory_count", 
                                   labels=["agent", "type"])
        self.metrics.register_gauge("memory_cache_size")
        self.metrics.register_counter("memory_consolidations_total")
        self.metrics.register_counter("memory_forgets_total")
    
    async def start(self):
        """Start background tasks."""
        if self._running:
            return
        
        self._running = True
        
        # Start consolidation task
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Load existing memories from state manager
        await self._load_existing_memories()
        
        self.tracing.log_event("memory_system_started", 
                              {"agent_id": self.agent_id})
    
    async def stop(self):
        """Stop background tasks and persist state."""
        self._running = False
        
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final consolidation
        await self.consolidate_memories()
        
        # Persist cache
        await self._persist_cache()
        
        self.tracing.log_event("memory_system_stopped", 
                              {"agent_id": self.agent_id})
    
    async def add_memory(self,
                        content: str,
                        memory_type: MemoryType = MemoryType.EPISODIC,
                        importance: float = 0.5,
                        tags: Optional[Set[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        source_agent: Optional[str] = None,
                        generate_embedding: bool = True) -> str:
        """Add a new memory to the system."""
        start_time = time.time()
        
        try:
            # Create memory object
            memory = Memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or set(),
                metadata=metadata or {},
                source_agent=source_agent or self.agent_id
            )
            
            # Generate embedding if requested
            if generate_embedding:
                memory.embedding = await self._generate_embedding(content)
            
            # Store in vector store
            if memory.embedding:
                await self.circuit_breaker.call(
                    self.vector_store.add,
                    memory.id,
                    memory.embedding,
                    {
                        "memory_type": memory_type.value,
                        "importance": importance,
                        "agent_id": self.agent_id
                    }
                )
            
            # Store in state manager
            await self._store_memory(memory)
            
            # Add to cache
            self._add_to_cache(memory)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.observe_histogram("memory_operation_duration_seconds", 
                                         duration, {"operation": "add"})
            self.metrics.inc_counter("memory_operations_total", 
                                   {"operation": "add", "status": "success"})
            
            self.tracing.log_event("memory_added", {
                "memory_id": memory.id,
                "type": memory_type.value,
                "importance": importance
            })
            
            return memory.id
            
        except Exception as e:
            self.metrics.inc_counter("memory_operations_total", 
                                   {"operation": "add", "status": "failure"})
            self.tracing.log_error("memory_add_failed", {"error": str(e)})
            raise
    
    async def retrieve_memory(self,
                             query: str,
                             top_k: int = 5,
                             memory_type: Optional[MemoryType] = None,
                             min_importance: float = 0.0,
                             tags: Optional[Set[str]] = None) -> List[Memory]:
        """Retrieve memories based on semantic similarity."""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Prepare filters
            filter_metadata = {"agent_id": self.agent_id}
            if memory_type:
                filter_metadata["memory_type"] = memory_type.value
            
            # Search vector store
            results = await self.circuit_breaker.call(
                self.vector_store.search,
                query_embedding,
                top_k=top_k * 2,  # Get more to filter
                filter_metadata=filter_metadata
            )
            
            # Retrieve full memories
            memories = []
            for memory_id, distance in results:
                memory = await self._get_memory(memory_id)
                if not memory:
                    continue
                
                # Apply filters
                if memory.importance < min_importance:
                    continue
                
                if tags and not tags.intersection(memory.tags):
                    continue
                
                # Update access statistics
                memory.last_accessed = datetime.utcnow()
                memory.access_count += 1
                await self._update_memory(memory)
                
                memories.append(memory)
                
                if len(memories) >= top_k:
                    break
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.observe_histogram("memory_operation_duration_seconds", 
                                         duration, {"operation": "retrieve"})
            self.metrics.inc_counter("memory_operations_total", 
                                   {"operation": "retrieve", "status": "success"})
            
            return memories
            
        except Exception as e:
            self.metrics.inc_counter("memory_operations_total", 
                                   {"operation": "retrieve", "status": "failure"})
            self.tracing.log_error("memory_retrieve_failed", {"error": str(e)})
            raise
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        # Check cache first
        if memory_id in self.memory_cache:
            memory = self.memory_cache[memory_id]
            memory.last_accessed = datetime.utcnow()
            memory.access_count += 1
            return memory
        
        # Load from storage
        memory = await self._get_memory(memory_id)
        if memory:
            # Update access
            memory.last_accessed = datetime.utcnow()
            memory.access_count += 1
            await self._update_memory(memory)
            
            # Add to cache
            self._add_to_cache(memory)
        
        return memory
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the system."""
        try:
            # Remove from vector store
            await self.circuit_breaker.call(
                self.vector_store.delete,
                memory_id
            )
            
            # Remove from state manager
            await self.state_manager.delete(f"memory:{self.agent_id}:{memory_id}")
            
            # Remove from cache
            if memory_id in self.memory_cache:
                del self.memory_cache[memory_id]
            
            self.metrics.inc_counter("memory_operations_total", 
                                   {"operation": "delete", "status": "success"})
            
            return True
            
        except Exception as e:
            self.metrics.inc_counter("memory_operations_total", 
                                   {"operation": "delete", "status": "failure"})
            self.tracing.log_error("memory_delete_failed", 
                                  {"memory_id": memory_id, "error": str(e)})
            return False
    
    async def consolidate_memories(self) -> int:
        """Consolidate similar memories to reduce redundancy."""
        start_time = time.time()
        
        try:
            # Get all memories for this agent
            memory_keys = await self.state_manager.list_keys(f"memory:{self.agent_id}:*")
            
            if not memory_keys:
                return 0
            
            # Load memories in batches
            all_memories = []
            for i in range(0, len(memory_keys), self.batch_size):
                batch_keys = memory_keys[i:i + self.batch_size]
                batch_memories = await asyncio.gather(
                    *[self._get_memory_from_key(key) for key in batch_keys]
                )
                all_memories.extend([m for m in batch_memories if m])
            
            # Cluster similar memories
            clusters = self.consolidation.cluster_memories(all_memories)
            
            # Consolidate each cluster
            consolidated_count = 0
            for cluster in clusters:
                if len(cluster) > 1:
                    consolidated = self.consolidation.consolidate_cluster(cluster)
                    
                    # Delete old memories
                    for memory in cluster:
                        if memory.id != consolidated.id:
                            await self.delete_memory(memory.id)
                    
                    # Store consolidated memory
                    await self._store_memory(consolidated)
                    
                    # Update vector store
                    if consolidated.embedding:
                        await self.circuit_breaker.call(
                            self.vector_store.update,
                            consolidated.id,
                            consolidated.embedding,
                            {
                                "memory_type": consolidated.memory_type.value,
                                "importance": consolidated.importance,
                                "agent_id": self.agent_id
                            }
                        )
                    
                    consolidated_count += 1
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.observe_histogram("memory_operation_duration_seconds", 
                                         duration, {"operation": "consolidate"})
            self.metrics.inc_counter("memory_consolidations_total", consolidated_count)
            
            self.tracing.log_event("memory_consolidation_completed", {
                "clusters_processed": len(clusters),
                "memories_consolidated": consolidated_count
            })
            
            return consolidated_count
            
        except Exception as e:
            self.tracing.log_error("memory_consolidation_failed", {"error": str(e)})
            return 0
    
    async def apply_forgetting(self) -> int:
        """Apply forgetting curve to remove low-retention memories."""
        start_time = time.time()
        
        try:
            # Get all memories
            memory_keys = await self.state_manager.list_keys(f"memory:{self.agent_id}:*")
            
            if not memory_keys:
                return 0
            
            # Load and evaluate memories
            forgotten_count = 0
            for key in memory_keys:
                memory = await self._get_memory_from_key(key)
                if not memory:
                    continue
                
                # Check if should be forgotten
                if self.forgetting_curve.should_forget(memory, self.forgetting_threshold):
                    await self.delete_memory(memory.id)
                    forgotten_count += 1
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.observe_histogram("memory_operation_duration_seconds", 
                                         duration, {"operation": "forget"})
            self.metrics.inc_counter("memory_forgets_total", forgotten_count)
            
            self.tracing.log_event("memory_forgetting_completed", {
                "memories_forgotten": forgotten_count
            })
            
            return forgotten_count
            
        except Exception as e:
            self.tracing.log_error("memory_forgetting_failed", {"error": str(e)})
            return 0
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        memory_keys = await self.state_manager.list_keys(f"memory:{self.agent_id}:*")
        
        stats = {
            "total_memories": len(memory_keys),
            "cache_size": len(self.memory_cache),
            "memory_types": defaultdict(int),
            "importance_distribution": defaultdict(int),
            "consolidation_levels": defaultdict(int),
            "oldest_memory": None,
            "newest_memory": None,
            "average_access_count": 0
        }
        
        if not memory_keys:
            return stats
        
        # Sample some memories for statistics
        sample_size = min(100, len(memory_keys))
        sample_keys = np.random.choice(memory_keys, sample_size, replace=False)
        
        total_access = 0
        for key in sample_keys:
            memory = await self._get_memory_from_key(key)
            if not memory:
                continue
            
            stats["memory_types"][memory.memory_type.value] += 1
            
            importance_bucket = f"{memory.importance:.1f}"
            stats["importance_distribution"][importance_bucket] += 1
            
            stats["consolidation_levels"][memory.consolidation_level] += 1
            
            total_access += memory.access_count
            
            if not stats["oldest_memory"] or memory.created_at < stats["oldest_memory"]:
                stats["oldest_memory"] = memory.created_at
            
            if not stats["newest_memory"] or memory.created_at > stats["newest_memory"]:
                stats["newest_memory"] = memory.created_at
        
        stats["average_access_count"] = total_access / sample_size if sample_size > 0 else 0
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats["memory_types"] = dict(stats["memory_types"])
        stats["importance_distribution"] = dict(stats["importance_distribution"])
        stats["consolidation_levels"] = dict(stats["consolidation_levels"])
        
        return stats
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text. Override with actual embedding model."""
        # Placeholder implementation - in production, use a real embedding model
        # This could integrate with LLM services or local models
        hash_obj = hashlib.md5(text.encode())
        hash_digest = hash_obj.digest()
        
        # Convert hash to pseudo-embedding (for demonstration only)
        embedding = []
        for i in range(0, len(hash_digest), 4):
            chunk = hash_digest[i:i+4]
            value = int.from_bytes(chunk, byteorder='big') / (2**32)
            embedding.append(value)
        
        # Pad or truncate to required dimension
        target_dim = 768  # Standard embedding dimension
        if len(embedding) < target_dim:
            embedding.extend([0.0] * (target_dim - len(embedding)))
        else:
            embedding = embedding[:target_dim]
        
        return embedding
    
    async def _store_memory(self, memory: Memory):
        """Store memory in state manager."""
        key = f"memory:{self.agent_id}:{memory.id}"
        await self.state_manager.set(key, memory.to_dict())
    
    async def _get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory from state manager."""
        key = f"memory:{self.agent_id}:{memory_id}"
        data = await self.state_manager.get(key)
        if data:
            return Memory.from_dict(data)
        return None
    
    async def _get_memory_from_key(self, key: str) -> Optional[Memory]:
        """Retrieve memory from state manager using full key."""
        data = await self.state_manager.get(key)
        if data:
            return Memory.from_dict(data)
        return None
    
    async def _update_memory(self, memory: Memory):
        """Update memory in state manager and cache."""
        await self._store_memory(memory)
        self._add_to_cache(memory)
    
    def _add_to_cache(self, memory: Memory):
        """Add memory to LRU cache."""
        if memory.id in self.memory_cache:
            # Move to end (most recently used)
            self.memory_cache.pop(memory.id)
        
        self.memory_cache[memory.id] = memory
        
        # Evict if cache is full
        if len(self.memory_cache) > self.cache_size:
            # Remove oldest (first item in dict)
            oldest_id = next(iter(self.memory_cache))
            del self.memory_cache[oldest_id]
        
        # Update cache size metric
        self.metrics.set_gauge("memory_cache_size", len(self.memory_cache))
    
    async def _load_existing_memories(self):
        """Load existing memories from storage into cache."""
        try:
            memory_keys = await self.state_manager.list_keys(f"memory:{self.agent_id}:*")
            
            # Load most recent/accessed memories into cache
            memories = []
            for key in memory_keys[:self.cache_size]:
                memory = await self._get_memory_from_key(key)
                if memory:
                    memories.append(memory)
            
            # Sort by last accessed (most recent first)
            memories.sort(key=lambda m: m.last_accessed, reverse=True)
            
            # Add to cache
            for memory in memories[:self.cache_size]:
                self._add_to_cache(memory)
            
            self.tracing.log_event("memories_loaded", {
                "count": len(memories),
                "cache_size": len(self.memory_cache)
            })
            
        except Exception as e:
            self.tracing.log_error("memory_load_failed", {"error": str(e)})
    
    async def _persist_cache(self):
        """Persist cached memories to storage."""
        for memory in self.memory_cache.values():
            await self._store_memory(memory)
    
    async def _consolidation_loop(self):
        """Background task for periodic memory consolidation."""
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval)
                if self._running:
                    await self.consolidate_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.tracing.log_error("consolidation_loop_error", {"error": str(e)})
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_loop(self):
        """Background task for periodic cleanup of forgotten memories."""
        while self._running:
            try:
                await asyncio.sleep(self.consolidation_interval * 2)  # Less frequent
                if self._running:
                    await self.apply_forgetting()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.tracing.log_error("cleanup_loop_error", {"error": str(e)})
                await asyncio.sleep(60)  # Wait before retrying

class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing/fallback."""
    
    def __init__(self, dimension: int = 768):
        super().__init__(dimension)
        self.embeddings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    async def add(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        self.embeddings[id] = embedding
        self.metadata[id] = metadata or {}
    
    async def search(self, query_embedding: List[float], top_k: int = 10,
                    filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        if not self.embeddings:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for id, embedding in self.embeddings.items():
            if filter_metadata and not self._matches_filter(id, filter_metadata):
                continue
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
            norm_a = sum(a * a for a in query_embedding) ** 0.5
            norm_b = sum(b * b for b in embedding) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_a * norm_b)
            
            # Convert to distance (lower is better)
            distance = 1 - similarity
            similarities.append((id, distance))
        
        # Sort by distance (ascending)
        similarities.sort(key=lambda x: x[1])
        
        return similarities[:top_k]
    
    def _matches_filter(self, id: str, filter_metadata: Dict[str, Any]) -> bool:
        metadata = self.metadata.get(id, {})
        for key, value in filter_metadata.items():
            if metadata.get(key) != value:
                return False
        return True
    
    async def delete(self, id: str):
        if id in self.embeddings:
            del self.embeddings[id]
        if id in self.metadata:
            del self.metadata[id]
    
    async def update(self, id: str, embedding: List[float], metadata: Dict[str, Any] = None):
        self.embeddings[id] = embedding
        if metadata:
            self.metadata[id] = metadata

# Factory function for easy instantiation
def create_memory_system(agent_id: str, **kwargs) -> PersistentMemorySystem:
    """Create a memory system with sensible defaults."""
    defaults = {
        "vector_store_type": "chroma" if CHROMA_AVAILABLE else "memory",
        "config": {
            "consolidation_interval": 3600,  # 1 hour
            "forgetting_threshold": 0.1,
            "max_memories": 10000,
            "cache_size": 1000,
            "base_retention": 0.5,
            "importance_weight": 2.0,
            "access_boost": 0.1
        }
    }
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        if key == "config" and isinstance(value, dict):
            defaults["config"].update(value)
        else:
            defaults[key] = value
    
    return PersistentMemorySystem(agent_id=agent_id, **defaults)

# Integration with SOVEREIGN's distributed executor
class DistributedMemorySystem:
    """
    Distributed memory system that coordinates across multiple nexus.
    Uses the existing distributed infrastructure.
    """
    
    def __init__(self,
                 executor: DistributedExecutor,
                 consensus_protocol: str = "raft"):
        self.executor = executor
        self.consensus_protocol = consensus_protocol
        self.agent_memories: Dict[str, PersistentMemorySystem] = {}
        
        # Shared vector store for cross-agent memory sharing
        self.shared_vector_store = None
        if CHROMA_AVAILABLE:
            self.shared_vector_store = ChromaVectorStore(
                collection_name="shared_agent_memories"
            )
    
    async def get_agent_memory(self, agent_id: str) -> PersistentMemorySystem:
        """Get or create memory system for an agent."""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = create_memory_system(agent_id)
            await self.agent_memories[agent_id].start()
        
        return self.agent_memories[agent_id]
    
    async def share_memory(self,
                          from_agent: str,
                          to_agent: str,
                          memory_id: str,
                          importance_boost: float = 0.1) -> bool:
        """Share a memory between nexus."""
        try:
            # Get source memory
            source_memory_system = await self.get_agent_memory(from_agent)
            memory = await source_memory_system.get_memory(memory_id)
            
            if not memory:
                return False
            
            # Create shared version
            shared_memory = Memory(
                content=memory.content,
                memory_type=memory.memory_type,
                importance=min(1.0, memory.importance + importance_boost),
                tags=memory.tags | {"shared", f"from_{from_agent}"},
                metadata={**memory.metadata, "shared_from": from_agent},
                source_agent=from_agent
            )
            
            # Add to target agent's memory
            target_memory_system = await self.get_agent_memory(to_agent)
            await target_memory_system.add_memory(
                content=shared_memory.content,
                memory_type=shared_memory.memory_type,
                importance=shared_memory.importance,
                tags=shared_memory.tags,
                metadata=shared_memory.metadata,
                source_agent=from_agent
            )
            
            # Also add to shared vector store if available
            if self.shared_vector_store and memory.embedding:
                await self.shared_vector_store.add(
                    f"{from_agent}_{to_agent}_{memory_id}",
                    memory.embedding,
                    {
                        "source_agent": from_agent,
                        "target_agent": to_agent,
                        "original_memory_id": memory_id
                    }
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to share memory: {e}")
            return False
    
    async def search_across_nexus(self,
                                  query: str,
                                  agent_ids: Optional[List[str]] = None,
                                  top_k: int = 10) -> List[Tuple[str, Memory]]:
        """Search for memories across multiple nexus."""
        if not agent_ids:
            agent_ids = list(self.agent_memories.keys())
        
        # Search in parallel
        tasks = []
        for agent_id in agent_ids:
            if agent_id in self.agent_memories:
                memory_system = self.agent_memories[agent_id]
                tasks.append(memory_system.retrieve_memory(query, top_k=top_k))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and rank results
        all_memories = []
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                continue
            
            for memory in result:
                all_memories.append((agent_id, memory))
        
        # Sort by importance and recency
        all_memories.sort(
            key=lambda x: (x[1].importance, x[1].last_accessed),
            reverse=True
        )
        
        return all_memories[:top_k]
    
    async def consolidate_across_nexus(self) -> Dict[str, int]:
        """Consolidate memories across all nexus."""
        consolidation_results = {}
        
        for agent_id, memory_system in self.agent_memories.items():
            try:
                count = await memory_system.consolidate_memories()
                consolidation_results[agent_id] = count
            except Exception as e:
                logging.error(f"Consolidation failed for agent {agent_id}: {e}")
                consolidation_results[agent_id] = 0
        
        return consolidation_results

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    async def test_memory_system():
        # Create a memory system
        memory_system = create_memory_system("test_agent_1")
        
        # Start the system
        await memory_system.start()
        
        # Add some memories
        memory_ids = []
        
        # Episodic memory
        memory_id = await memory_system.add_memory(
            content="The agent learned to optimize API calls by implementing caching",
            memory_type=MemoryType.EPISODIC,
            importance=0.7,
            tags={"learning", "api", "optimization"},
            metadata={"context": "performance_tuning"}
        )
        memory_ids.append(memory_id)
        
        # Semantic memory
        memory_id = await memory_system.add_memory(
            content="REST APIs should use proper HTTP methods: GET for retrieval, POST for creation, PUT for updates, DELETE for removal",
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            tags={"api", "rest", "best_practices"},
            metadata={"category": "knowledge"}
        )
        memory_ids.append(memory_id)
        
        # Procedural memory
        memory_id = await memory_system.add_memory(
            content="To implement circuit breaker pattern: 1) Track failure counts, 2) Open circuit after threshold, 3) Allow test requests after timeout",
            memory_type=MemoryType.PROCEDURAL,
            importance=0.8,
            tags={"resilience", "pattern", "implementation"},
            metadata={"skill": "circuit_breaker"}
        )
        memory_ids.append(memory_id)
        
        print(f"Added {len(memory_ids)} memories")
        
        # Retrieve memories
        print("\nSearching for 'API optimization'...")
        results = await memory_system.retrieve_memory("API optimization", top_k=2)
        for memory in results:
            print(f"  - {memory.content[:100]}... (importance: {memory.importance})")
        
        # Get memory stats
        stats = await memory_system.get_memory_stats()
        print(f"\nMemory stats: {json.dumps(stats, indent=2, default=str)}")
        
        # Stop the system
        await memory_system.stop()
        print("\nMemory system stopped")
    
    # Run the test
    asyncio.run(test_memory_system())