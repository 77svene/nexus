"""
Persistent Agent Memory System - Vector Store Implementation
Long-term memory for nexus with semantic search, memory consolidation, and forgetting mechanisms.
"""

import asyncio
import uuid
import json
import time
import math
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from pathlib import Path
import sqlite3
import pickle

# Integration with existing modules
from core.distributed.state_manager import StateManager, StateKey
from core.resilience.circuit_breaker import CircuitBreaker
from core.resilience.retry_policy import RetryPolicy
from monitoring.metrics_collector import MetricsCollector
from monitoring.tracing import trace_operation
from core.composition.capability_graph import CapabilityGraph

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of agent memories"""
    EPISODIC = "episodic"  # Specific events/experiences
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    WORKING = "working"  # Temporary context
    CONSOLIDATED = "consolidated"  # Summarized memories

class ImportanceLevel(Enum):
    """Importance levels for memories"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    MINIMAL = 0.1

@dataclass
class Memory:
    """Represents a single memory unit"""
    id: str
    content: str
    embedding: np.ndarray
    memory_type: MemoryType
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    decay_rate: float = 0.1  # Forgetting curve parameter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        data['memory_type'] = self.memory_type.value
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data['tags'] = list(self.tags)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary"""
        data['embedding'] = np.array(data['embedding'])
        data['memory_type'] = MemoryType(data['memory_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        data['tags'] = set(data.get('tags', []))
        return cls(**data)

class VectorStoreBackend(Enum):
    """Supported vector store backends"""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    SQLITE = "sqlite"  # Fallback for development

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    backend: VectorStoreBackend = VectorStoreBackend.SQLITE
    embedding_dim: int = 1536  # Default for OpenAI embeddings
    index_path: str = "./data/vector_store"
    similarity_threshold: float = 0.7
    max_memories: int = 10000
    consolidation_threshold: int = 1000
    forgetting_enabled: bool = True
    forgetting_check_interval: int = 3600  # seconds
    decay_function: str = "exponential"  # exponential, linear, or logarithmic
    importance_boost_factor: float = 1.2
    access_boost_factor: float = 1.1

class VectorStore:
    """
    Persistent vector store for agent memory system.
    Implements semantic search, memory consolidation, and forgetting mechanisms.
    """
    
    def __init__(self, agent_id: str, config: Optional[VectorStoreConfig] = None):
        self.agent_id = agent_id
        self.config = config or VectorStoreConfig()
        self.metrics = MetricsCollector(namespace=f"memory.vector_store.{agent_id}")
        self.circuit_breaker = CircuitBreaker(
            name=f"vector_store_{agent_id}",
            failure_threshold=5,
            reset_timeout=60
        )
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            backoff_factor=2.0
        )
        
        # Initialize storage
        self._init_storage()
        
        # Memory cache for fast access
        self.memory_cache: Dict[str, Memory] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Background tasks
        self._forgetting_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Integration with state manager
        self.state_manager = StateManager()
        self.capability_graph = CapabilityGraph()
        
        logger.info(f"VectorStore initialized for agent {agent_id} with backend {self.config.backend.value}")
    
    def _init_storage(self):
        """Initialize storage backend based on configuration"""
        try:
            if self.config.backend == VectorStoreBackend.SQLITE:
                self._init_sqlite()
            elif self.config.backend == VectorStoreBackend.FAISS:
                self._init_faiss()
            elif self.config.backend == VectorStoreBackend.CHROMA:
                self._init_chroma()
            else:
                raise ValueError(f"Unsupported backend: {self.config.backend}")
                
            # Create index directory if it doesn't exist
            Path(self.config.index_path).mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            self.metrics.increment("initialization_errors")
            raise
    
    def _init_sqlite(self):
        """Initialize SQLite backend for development/fallback"""
        self.db_path = Path(self.config.index_path) / f"{self.agent_id}_memories.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create memories table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,
                tags TEXT,
                decay_rate REAL DEFAULT 0.1,
                agent_id TEXT NOT NULL
            )
        ''')
        
        # Create indexes for performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)')
        
        self.conn.commit()
        logger.debug("SQLite backend initialized")
    
    def _init_faiss(self):
        """Initialize FAISS backend"""
        try:
            import faiss
            
            self.index_path = Path(self.config.index_path) / f"{self.agent_id}_faiss.index"
            self.metadata_path = Path(self.config.index_path) / f"{self.agent_id}_metadata.json"
            
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
                
            # Load metadata if exists
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)
            else:
                self.metadata_store = {}
                
        except ImportError:
            logger.warning("FAISS not installed, falling back to SQLite")
            self.config.backend = VectorStoreBackend.SQLITE
            self._init_sqlite()
    
    def _init_chroma(self):
        """Initialize ChromaDB backend"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.config.index_path
            ))
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=f"agent_{self.agent_id}_memories",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB backend initialized")
            
        except ImportError:
            logger.warning("ChromaDB not installed, falling back to SQLite")
            self.config.backend = VectorStoreBackend.SQLITE
            self._init_sqlite()
    
    @trace_operation("vector_store.add_memory")
    async def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = ImportanceLevel.MEDIUM.value,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None
    ) -> str:
        """
        Add a new memory to the vector store.
        
        Args:
            content: Text content of the memory
            embedding: Vector embedding of the content
            memory_type: Type of memory
            importance: Importance score (0-1)
            metadata: Additional metadata
            tags: Set of tags for categorization
            
        Returns:
            Memory ID
        """
        try:
            memory_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                importance=importance,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {},
                tags=tags or set()
            )
            
            # Store based on backend
            if self.config.backend == VectorStoreBackend.SQLITE:
                await self._add_to_sqlite(memory)
            elif self.config.backend == VectorStoreBackend.FAISS:
                await self._add_to_faiss(memory)
            elif self.config.backend == VectorStoreBackend.CHROMA:
                await self._add_to_chroma(memory)
            
            # Update cache
            self.memory_cache[memory_id] = memory
            self.embedding_cache[memory_id] = embedding
            
            # Update metrics
            self.metrics.increment("memories_added")
            self.metrics.gauge("total_memories", len(self.memory_cache))
            
            # Update state manager
            await self.state_manager.set(
                StateKey(f"memory:{self.agent_id}:{memory_id}"),
                memory.to_dict()
            )
            
            logger.debug(f"Added memory {memory_id} of type {memory_type.value}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            self.metrics.increment("add_memory_errors")
            raise
    
    async def _add_to_sqlite(self, memory: Memory):
        """Add memory to SQLite backend"""
        self.cursor.execute('''
            INSERT INTO memories 
            (id, content, embedding, memory_type, importance, created_at, 
             last_accessed, access_count, metadata, tags, decay_rate, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id,
            memory.content,
            pickle.dumps(memory.embedding),
            memory.memory_type.value,
            memory.importance,
            memory.created_at.isoformat(),
            memory.last_accessed.isoformat(),
            memory.access_count,
            json.dumps(memory.metadata),
            json.dumps(list(memory.tags)),
            memory.decay_rate,
            self.agent_id
        ))
        self.conn.commit()
    
    async def _add_to_faiss(self, memory: Memory):
        """Add memory to FAISS backend"""
        import faiss
        
        # Normalize embedding for cosine similarity
        embedding = memory.embedding / np.linalg.norm(memory.embedding)
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        # Store metadata
        self.metadata_store[memory.id] = {
            'content': memory.content,
            'memory_type': memory.memory_type.value,
            'importance': memory.importance,
            'created_at': memory.created_at.isoformat(),
            'last_accessed': memory.last_accessed.isoformat(),
            'access_count': memory.access_count,
            'metadata': memory.metadata,
            'tags': list(memory.tags),
            'decay_rate': memory.decay_rate
        }
        
        # Persist
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata_store, f)
    
    async def _add_to_chroma(self, memory: Memory):
        """Add memory to ChromaDB backend"""
        self.collection.add(
            embeddings=[memory.embedding.tolist()],
            documents=[memory.content],
            metadatas=[{
                'memory_type': memory.memory_type.value,
                'importance': memory.importance,
                'created_at': memory.created_at.isoformat(),
                'last_accessed': memory.last_accessed.isoformat(),
                'access_count': memory.access_count,
                'decay_rate': memory.decay_rate,
                **memory.metadata
            }],
            ids=[memory.id]
        )
    
    @trace_operation("vector_store.search")
    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        tags: Optional[Set[str]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Search for similar memories using semantic search.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
            time_range: Filter by time range (start, end)
            tags: Filter by tags
            
        Returns:
            List of (memory, similarity_score) tuples
        """
        try:
            start_time = time.time()
            
            if self.config.backend == VectorStoreBackend.SQLITE:
                results = await self._search_sqlite(
                    query_embedding, k, memory_types, min_importance, time_range, tags
                )
            elif self.config.backend == VectorStoreBackend.FAISS:
                results = await self._search_faiss(
                    query_embedding, k, memory_types, min_importance, time_range, tags
                )
            elif self.config.backend == VectorStoreBackend.CHROMA:
                results = await self._search_chroma(
                    query_embedding, k, memory_types, min_importance, time_range, tags
                )
            
            # Update access patterns for retrieved memories
            for memory, score in results:
                await self._update_memory_access(memory.id)
            
            # Record metrics
            search_time = time.time() - start_time
            self.metrics.histogram("search_latency", search_time)
            self.metrics.increment("searches_performed")
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.metrics.increment("search_errors")
            raise
    
    async def _search_sqlite(
        self,
        query_embedding: np.ndarray,
        k: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float,
        time_range: Optional[Tuple[datetime, datetime]],
        tags: Optional[Set[str]]
    ) -> List[Tuple[Memory, float]]:
        """Search using SQLite with brute-force cosine similarity"""
        # Build query
        query = """
            SELECT id, content, embedding, memory_type, importance, created_at,
                   last_accessed, access_count, metadata, tags, decay_rate
            FROM memories 
            WHERE agent_id = ?
        """
        params = [self.agent_id]
        
        if memory_types:
            type_placeholders = ','.join(['?'] * len(memory_types))
            query += f" AND memory_type IN ({type_placeholders})"
            params.extend([mt.value for mt in memory_types])
        
        if min_importance > 0:
            query += " AND importance >= ?"
            params.append(min_importance)
        
        if time_range:
            query += " AND created_at BETWEEN ? AND ?"
            params.extend([time_range[0].isoformat(), time_range[1].isoformat()])
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        # Calculate similarities
        results = []
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for row in rows:
            memory_id, content, embedding_blob, memory_type, importance, created_at, \
            last_accessed, access_count, metadata, tags_json, decay_rate = row
            
            # Deserialize
            embedding = pickle.loads(embedding_blob)
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(query_norm, embedding_norm)
            
            if similarity >= self.config.similarity_threshold:
                # Check tags filter
                memory_tags = set(json.loads(tags_json))
                if tags and not tags.intersection(memory_tags):
                    continue
                
                # Create memory object
                memory = Memory(
                    id=memory_id,
                    content=content,
                    embedding=embedding,
                    memory_type=MemoryType(memory_type),
                    importance=importance,
                    created_at=datetime.fromisoformat(created_at),
                    last_accessed=datetime.fromisoformat(last_accessed),
                    access_count=access_count,
                    metadata=json.loads(metadata),
                    tags=memory_tags,
                    decay_rate=decay_rate
                )
                
                results.append((memory, float(similarity)))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    async def _search_faiss(
        self,
        query_embedding: np.ndarray,
        k: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float,
        time_range: Optional[Tuple[datetime, datetime]],
        tags: Optional[Set[str]]
    ) -> List[Tuple[Memory, float]]:
        """Search using FAISS index"""
        import faiss
        
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_norm.reshape(1, -1).astype('float32'),
            min(k * 2, self.index.ntotal)  # Get more results for filtering
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Get memory ID from metadata (FAISS doesn't store IDs)
            # This is a simplified approach - in production, maintain ID mapping
            memory_id = list(self.metadata_store.keys())[idx]
            metadata = self.metadata_store[memory_id]
            
            # Apply filters
            if memory_types and metadata['memory_type'] not in [mt.value for mt in memory_types]:
                continue
            
            if metadata['importance'] < min_importance:
                continue
            
            if time_range:
                created_at = datetime.fromisoformat(metadata['created_at'])
                if not (time_range[0] <= created_at <= time_range[1]):
                    continue
            
            if tags:
                memory_tags = set(metadata.get('tags', []))
                if not tags.intersection(memory_tags):
                    continue
            
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=metadata['content'],
                embedding=self.embedding_cache.get(memory_id, np.zeros(self.config.embedding_dim)),
                memory_type=MemoryType(metadata['memory_type']),
                importance=metadata['importance'],
                created_at=datetime.fromisoformat(metadata['created_at']),
                last_accessed=datetime.fromisoformat(metadata['last_accessed']),
                access_count=metadata['access_count'],
                metadata=metadata.get('metadata', {}),
                tags=set(metadata.get('tags', [])),
                decay_rate=metadata.get('decay_rate', 0.1)
            )
            
            # Convert FAISS distance to similarity (1 - distance for cosine)
            similarity = 1.0 - distance
            results.append((memory, similarity))
        
        # Sort and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    async def _search_chroma(
        self,
        query_embedding: np.ndarray,
        k: int,
        memory_types: Optional[List[MemoryType]],
        min_importance: float,
        time_range: Optional[Tuple[datetime, datetime]],
        tags: Optional[Set[str]]
    ) -> List[Tuple[Memory, float]]:
        """Search using ChromaDB"""
        # Build where clause for filtering
        where_clause = {}
        
        if memory_types:
            where_clause["memory_type"] = {"$in": [mt.value for mt in memory_types]}
        
        if min_importance > 0:
            where_clause["importance"] = {"$gte": min_importance}
        
        if time_range:
            where_clause["created_at"] = {
                "$gte": time_range[0].isoformat(),
                "$lte": time_range[1].isoformat()
            }
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        # Convert to Memory objects
        memories = []
        for i, (doc, metadata, distance, memory_id) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
            results['ids'][0]
        )):
            # Check tags filter
            if tags:
                memory_tags = set(metadata.get('tags', []))
                if not tags.intersection(memory_tags):
                    continue
            
            # Create memory object
            memory = Memory(
                id=memory_id,
                content=doc,
                embedding=np.array(results['embeddings'][0][i]),
                memory_type=MemoryType(metadata['memory_type']),
                importance=metadata['importance'],
                created_at=datetime.fromisoformat(metadata['created_at']),
                last_accessed=datetime.fromisoformat(metadata['last_accessed']),
                access_count=metadata['access_count'],
                metadata={k: v for k, v in metadata.items() 
                         if k not in ['memory_type', 'importance', 'created_at', 
                                     'last_accessed', 'access_count', 'decay_rate']},
                tags=set(metadata.get('tags', [])),
                decay_rate=metadata.get('decay_rate', 0.1)
            )
            
            # Convert distance to similarity
            similarity = 1.0 - distance
            memories.append((memory, similarity))
        
        return memories
    
    async def _update_memory_access(self, memory_id: str):
        """Update access statistics for a memory"""
        try:
            now = datetime.utcnow()
            
            if memory_id in self.memory_cache:
                memory = self.memory_cache[memory_id]
                memory.last_accessed = now
                memory.access_count += 1
                
                # Boost importance based on access
                memory.importance = min(
                    1.0,
                    memory.importance * self.config.access_boost_factor
                )
            
            # Update in backend
            if self.config.backend == VectorStoreBackend.SQLITE:
                self.cursor.execute('''
                    UPDATE memories 
                    SET last_accessed = ?, access_count = access_count + 1,
                        importance = MIN(1.0, importance * ?)
                    WHERE id = ?
                ''', (now.isoformat(), self.config.access_boost_factor, memory_id))
                self.conn.commit()
            
            elif self.config.backend == VectorStoreBackend.CHROMA:
                # ChromaDB doesn't support updates, so we'd need to delete and re-add
                pass
            
            # Update state manager
            await self.state_manager.set(
                StateKey(f"memory_access:{self.agent_id}:{memory_id}"),
                {"last_accessed": now.isoformat(), "access_count": 1}
            )
            
        except Exception as e:
            logger.warning(f"Failed to update memory access for {memory_id}: {e}")
    
    @trace_operation("vector_store.consolidate")
    async def consolidate_memories(
        self,
        cluster_count: int = 10,
        similarity_threshold: float = 0.8
    ) -> List[Memory]:
        """
        Consolidate memories by clustering similar memories and creating summaries.
        
        Args:
            cluster_count: Number of clusters to create
            similarity_threshold: Threshold for considering memories similar
            
        Returns:
            List of new consolidated memories
        """
        try:
            logger.info(f"Starting memory consolidation for agent {self.agent_id}")
            
            # Get all memories
            all_memories = list(self.memory_cache.values())
            
            if len(all_memories) < self.config.consolidation_threshold:
                logger.debug("Not enough memories for consolidation")
                return []
            
            # Extract embeddings
            embeddings = np.array([m.embedding for m in all_memories])
            
            # Simple k-means clustering
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(
                n_clusters=min(cluster_count, len(all_memories)),
                random_state=42
            )
            clusters = kmeans.fit_predict(embeddings)
            
            # Group memories by cluster
            clustered_memories = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_memories:
                    clustered_memories[cluster_id] = []
                clustered_memories[cluster_id].append(all_memories[idx])
            
            # Create consolidated memories for each cluster
            consolidated_memories = []
            for cluster_id, memories in clustered_memories.items():
                if len(memories) < 2:
                    continue
                
                # Calculate cluster centroid
                cluster_embeddings = np.array([m.embedding for m in memories])
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Create summary content
                summary_content = self._create_memory_summary(memories)
                
                # Calculate consolidated importance (max of cluster)
                max_importance = max(m.importance for m in memories)
                
                # Create consolidated memory
                consolidated_memory = Memory(
                    id=str(uuid.uuid4()),
                    content=summary_content,
                    embedding=centroid,
                    memory_type=MemoryType.CONSOLIDATED,
                    importance=max_importance * self.config.importance_boost_factor,
                    created_at=datetime.utcnow(),
                    last_accessed=datetime.utcnow(),
                    metadata={
                        "source_memory_ids": [m.id for m in memories],
                        "cluster_size": len(memories),
                        "consolidation_date": datetime.utcnow().isoformat()
                    },
                    tags=set().union(*[m.tags for m in memories]),
                    decay_rate=0.05  # Slower decay for consolidated memories
                )
                
                # Add to store
                await self.add_memory(
                    content=consolidated_memory.content,
                    embedding=consolidated_memory.embedding,
                    memory_type=MemoryType.CONSOLIDATED,
                    importance=consolidated_memory.importance,
                    metadata=consolidated_memory.metadata,
                    tags=consolidated_memory.tags
                )
                
                consolidated_memories.append(consolidated_memory)
                
                # Mark original memories for deletion (optional)
                for memory in memories:
                    memory.importance *= 0.5  # Reduce importance
                
                logger.debug(f"Consolidated {len(memories)} memories into cluster {cluster_id}")
            
            # Update metrics
            self.metrics.increment("consolidation_operations")
            self.metrics.gauge("consolidated_memories_created", len(consolidated_memories))
            
            logger.info(f"Consolidation complete: created {len(consolidated_memories)} consolidated memories")
            return consolidated_memories
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            self.metrics.increment("consolidation_errors")
            return []
    
    def _create_memory_summary(self, memories: List[Memory]) -> str:
        """Create a summary of multiple memories"""
        # Simple extractive summary - in production, use LLM for better summaries
        contents = [m.content for m in memories]
        
        # Take first 100 chars of each memory and combine
        summary_parts = []
        for i, content in enumerate(contents[:5]):  # Limit to first 5
            truncated = content[:100] + "..." if len(content) > 100 else content
            summary_parts.append(f"Memory {i+1}: {truncated}")
        
        summary = " | ".join(summary_parts)
        
        if len(memories) > 5:
            summary += f" ... and {len(memories) - 5} more related memories"
        
        return summary
    
    @trace_operation("vector_store.forget")
    async def forget_memories(
        self,
        importance_threshold: float = 0.1,
        age_threshold_days: int = 30,
        max_to_forget: int = 100
    ) -> int:
        """
        Forget memories based on importance decay and age.
        
        Args:
            importance_threshold: Minimum importance to keep
            age_threshold_days: Maximum age in days to keep low-importance memories
            max_to_forget: Maximum number of memories to forget in one operation
            
        Returns:
            Number of memories forgotten
        """
        try:
            if not self.config.forgetting_enabled:
                return 0
            
            logger.info(f"Starting forgetting process for agent {self.agent_id}")
            
            now = datetime.utcnow()
            forgotten_count = 0
            
            # Get all memories
            all_memories = list(self.memory_cache.values())
            
            for memory in all_memories:
                if forgotten_count >= max_to_forget:
                    break
                
                # Calculate current importance with decay
                time_since_access = (now - memory.last_accessed).total_seconds()
                decay_factor = self._calculate_decay_factor(
                    time_since_access,
                    memory.decay_rate,
                    memory.importance
                )
                
                current_importance = memory.importance * decay_factor
                
                # Check if memory should be forgotten
                age_days = (now - memory.created_at).days
                
                should_forget = (
                    current_importance < importance_threshold or
                    (age_days > age_threshold_days and current_importance < 0.3)
                )
                
                if should_forget:
                    await self._delete_memory(memory.id)
                    forgotten_count += 1
                    
                    logger.debug(f"Forgot memory {memory.id} (importance: {current_importance:.3f}, age: {age_days} days)")
            
            # Update metrics
            self.metrics.increment("forgetting_operations")
            self.metrics.gauge("memories_forgotten", forgotten_count)
            
            logger.info(f"Forgetting complete: removed {forgotten_count} memories")
            return forgotten_count
            
        except Exception as e:
            logger.error(f"Forgetting process failed: {e}")
            self.metrics.increment("forgetting_errors")
            return 0
    
    def _calculate_decay_factor(
        self,
        time_since_access: float,
        decay_rate: float,
        initial_importance: float
    ) -> float:
        """Calculate decay factor based on configured decay function"""
        if self.config.decay_function == "exponential":
            return math.exp(-decay_rate * time_since_access / 86400)  # Convert seconds to days
        elif self.config.decay_function == "linear":
            return max(0, 1 - (decay_rate * time_since_access / 86400))
        elif self.config.decay_function == "logarithmic":
            return 1 / (1 + decay_rate * math.log(1 + time_since_access / 86400))
        else:
            return math.exp(-decay_rate * time_since_access / 86400)  # Default to exponential
    
    async def _delete_memory(self, memory_id: str):
        """Delete a memory from all storage locations"""
        try:
            # Remove from cache
            if memory_id in self.memory_cache:
                del self.memory_cache[memory_id]
            if memory_id in self.embedding_cache:
                del self.embedding_cache[memory_id]
            
            # Remove from backend
            if self.config.backend == VectorStoreBackend.SQLITE:
                self.cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
                self.conn.commit()
            
            elif self.config.backend == VectorStoreBackend.FAISS:
                # FAISS doesn't support deletion, so we'd need to rebuild index
                # For now, just remove from metadata
                if memory_id in self.metadata_store:
                    del self.metadata_store[memory_id]
                    with open(self.metadata_path, 'w') as f:
                        json.dump(self.metadata_store, f)
            
            elif self.config.backend == VectorStoreBackend.CHROMA:
                self.collection.delete(ids=[memory_id])
            
            # Remove from state manager
            await self.state_manager.delete(StateKey(f"memory:{self.agent_id}:{memory_id}"))
            
            self.metrics.increment("memories_deleted")
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            self.metrics.increment("delete_errors")
    
    async def start_background_tasks(self):
        """Start background tasks for forgetting and consolidation"""
        if self._running:
            return
        
        self._running = True
        
        # Start forgetting task
        if self.config.forgetting_enabled:
            self._forgetting_task = asyncio.create_task(self._forgetting_loop())
        
        # Start consolidation task
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
        
        logger.info("Background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        self._running = False
        
        if self._forgetting_task:
            self._forgetting_task.cancel()
            try:
                await self._forgetting_task
            except asyncio.CancelledError:
                pass
        
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background tasks stopped")
    
    async def _forgetting_loop(self):
        """Background loop for periodic forgetting"""
        while self._running:
            try:
                await asyncio.sleep(self.config.forgetting_check_interval)
                await self.forget_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Forgetting loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _consolidation_loop(self):
        """Background loop for periodic consolidation"""
        while self._running:
            try:
                # Run consolidation every 6 hours
                await asyncio.sleep(6 * 3600)
                await self.consolidate_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation loop error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory store"""
        try:
            stats = {
                "total_memories": len(self.memory_cache),
                "memory_types": {},
                "importance_distribution": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                    "minimal": 0
                },
                "average_age_days": 0,
                "average_access_count": 0,
                "backend": self.config.backend.value,
                "embedding_dimension": self.config.embedding_dim
            }
            
            if not self.memory_cache:
                return stats
            
            # Calculate statistics
            total_age = 0
            total_access = 0
            
            for memory in self.memory_cache.values():
                # Count by type
                type_name = memory.memory_type.value
                stats["memory_types"][type_name] = stats["memory_types"].get(type_name, 0) + 1
                
                # Count by importance
                if memory.importance >= 0.8:
                    stats["importance_distribution"]["critical"] += 1
                elif memory.importance >= 0.6:
                    stats["importance_distribution"]["high"] += 1
                elif memory.importance >= 0.4:
                    stats["importance_distribution"]["medium"] += 1
                elif memory.importance >= 0.2:
                    stats["importance_distribution"]["low"] += 1
                else:
                    stats["importance_distribution"]["minimal"] += 1
                
                # Calculate age
                age_days = (datetime.utcnow() - memory.created_at).days
                total_age += age_days
                
                # Count access
                total_access += memory.access_count
            
            stats["average_age_days"] = total_age / len(self.memory_cache)
            stats["average_access_count"] = total_access / len(self.memory_cache)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close connections and clean up"""
        await self.stop_background_tasks()
        
        if hasattr(self, 'conn'):
            self.conn.close()
        
        logger.info(f"VectorStore closed for agent {self.agent_id}")

# Factory function for creating vector stores
def create_vector_store(
    agent_id: str,
    backend: str = "sqlite",
    **kwargs
) -> VectorStore:
    """Factory function to create a vector store with specified backend"""
    config = VectorStoreConfig(
        backend=VectorStoreBackend(backend),
        **kwargs
    )
    return VectorStore(agent_id, config)

# Integration with capability graph
class MemoryCapability:
    """Capability for memory operations that can be registered in capability graph"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    async def remember(
        self,
        content: str,
        embedding: np.ndarray,
        memory_type: str = "episodic",
        importance: float = 0.5
    ) -> str:
        """Remember something"""
        return await self.vector_store.add_memory(
            content=content,
            embedding=embedding,
            memory_type=MemoryType(memory_type),
            importance=importance
        )
    
    async def recall(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Recall similar memories"""
        results = await self.vector_store.search(query, k)
        return [
            {
                "content": memory.content,
                "similarity": score,
                "importance": memory.importance,
                "type": memory.memory_type.value
            }
            for memory, score in results
        ]
    
    async def forget_old(self) -> int:
        """Forget old, unimportant memories"""
        return await self.vector_store.forget_memories()
    
    async def consolidate(self) -> List[Dict[str, Any]]:
        """Consolidate memories"""
        consolidated = await self.vector_store.consolidate_memories()
        return [
            {
                "id": m.id,
                "content": m.content,
                "source_count": len(m.metadata.get("source_memory_ids", []))
            }
            for m in consolidated
        ]

# Register memory capability in capability graph
def register_memory_capability(agent_id: str, vector_store: VectorStore):
    """Register memory capability in the capability graph"""
    graph = CapabilityGraph()
    memory_capability = MemoryCapability(vector_store)
    
    graph.add_capability(
        name="memory",
        description="Long-term memory with semantic search",
        instance=memory_capability,
        methods=["remember", "recall", "forget_old", "consolidate"]
    )
    
    logger.info(f"Memory capability registered for agent {agent_id}")