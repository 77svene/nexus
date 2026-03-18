# LLM Context Optimization Engine

from typing import Any, Dict, List, Optional, Tuple, Set
import time
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum

from nexus.dom.serializer.clickable_elements import ClickableElementDetector
from nexus.dom.serializer.paint_order import PaintOrderRemover
from nexus.dom.utils import cap_text_length
from nexus.dom.views import (
    DOMRect,
    DOMSelectorMap,
    EnhancedDOMTreeNode,
    NodeType,
    PropagatingBounds,
    SerializedDOMState,
    SimplifiedNode,
)

DISABLED_ELEMENTS = {'style', 'script', 'head', 'meta', 'link', 'title', 'noscript', 'template'}

# SVG child elements to skip (decorative only, no interaction value)
SVG_ELEMENTS = {
    'path', 'rect', 'g', 'circle', 'ellipse', 'line', 'polyline', 'polygon',
    'use', 'defs', 'clipPath', 'mask', 'pattern', 'image', 'text', 'tspan',
    'linearGradient', 'radialGradient', 'stop', 'filter', 'feGaussianBlur',
    'feOffset', 'feBlend', 'feColorMatrix', 'animate', 'animateTransform',
}

# Common page patterns for caching
COMMON_PATTERNS = {
    'navigation': ['nav', 'header', 'menu', 'navbar'],
    'main_content': ['main', 'article', 'content', 'section'],
    'sidebar': ['aside', 'sidebar', 'widget'],
    'footer': ['footer', 'copyright', 'legal'],
    'form': ['form', 'input', 'button', 'select', 'textarea'],
}

class CompressionLevel(Enum):
    """Different compression levels for different use cases"""
    MINIMAL = "minimal"      # Keep only interactive elements
    BALANCED = "balanced"    # Keep structure + interactive + important text
    DETAILED = "detailed"    # Keep more context for complex tasks
    FULL = "full"           # Keep everything (no compression)

@dataclass
class ElementFingerprint:
    """Unique fingerprint for element pattern caching"""
    tag_name: str
    attributes_hash: str
    text_hash: str
    structure_hash: str
    relevance_score: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElementFingerprint':
        return cls(**data)

class SemanticCompressor:
    """
    Semantic compression engine for DOM trees.
    Reduces token usage while maintaining semantic meaning.
    """
    
    def __init__(self, task_context: Optional[str] = None):
        self.task_context = task_context
        self.compression_stats = {
            'original_tokens': 0,
            'compressed_tokens': 0,
            'compression_ratio': 0.0,
            'nodes_removed': 0,
            'nodes_compressed': 0,
        }
        
        # Common word patterns for semantic compression
        self.semantic_patterns = {
            'navigation': ['home', 'about', 'contact', 'login', 'sign', 'menu', 'nav'],
            'action': ['submit', 'buy', 'add', 'delete', 'edit', 'save', 'cancel'],
            'content': ['title', 'heading', 'description', 'text', 'content'],
            'interactive': ['button', 'input', 'select', 'checkbox', 'radio'],
        }
        
        # Relevance scoring weights
        self.relevance_weights = {
            'task_match': 0.4,
            'interactive': 0.25,
            'visibility': 0.15,
            'semantic_importance': 0.1,
            'text_density': 0.1,
        }
    
    def compress_dom_tree(self, root: EnhancedDOMTreeNode, 
                         compression_level: CompressionLevel = CompressionLevel.BALANCED,
                         max_depth: int = 15) -> EnhancedDOMTreeNode:
        """
        Compress DOM tree with semantic understanding.
        Returns optimized tree for LLM consumption.
        """
        # Reset stats
        self.compression_stats = {
            'original_tokens': 0,
            'compressed_tokens': 0,
            'compression_ratio': 0.0,
            'nodes_removed': 0,
            'nodes_compressed': 0,
            'start_time': time.time(),
        }
        
        # Create compressed copy
        compressed_root = self._compress_node(root, depth=0, max_depth=max_depth, 
                                             compression_level=compression_level)
        
        # Calculate final stats
        self.compression_stats['end_time'] = time.time()
        self.compression_stats['processing_time'] = (
            self.compression_stats['end_time'] - self.compression_stats['start_time']
        )
        
        if self.compression_stats['original_tokens'] > 0:
            self.compression_stats['compression_ratio'] = (
                1 - (self.compression_stats['compressed_tokens'] / 
                     self.compression_stats['original_tokens'])
            )
        
        return compressed_root
    
    def _compress_node(self, node: EnhancedDOMTreeNode, depth: int, 
                      max_depth: int, compression_level: CompressionLevel) -> Optional[EnhancedDOMTreeNode]:
        """Recursively compress a node and its children"""
        
        # Skip if max depth exceeded
        if depth > max_depth:
            self.compression_stats['nodes_removed'] += 1
            return None
        
        # Skip disabled elements entirely
        if node.tag_name.lower() in DISABLED_ELEMENTS:
            self.compression_stats['nodes_removed'] += 1
            return None
        
        # Skip decorative SVG elements
        if node.tag_name.lower() in SVG_ELEMENTS and not self._is_interactive_svg(node):
            self.compression_stats['nodes_removed'] += 1
            return None
        
        # Calculate relevance score for this node
        relevance_score = self._calculate_node_relevance(node)
        
        # Apply compression based on level and relevance
        should_compress = self._should_compress_node(node, relevance_score, compression_level)
        
        if should_compress == 'remove':
            self.compression_stats['nodes_removed'] += 1
            return None
        
        # Create compressed node
        compressed_node = self._create_compressed_node(node, relevance_score, compression_level)
        
        # Compress children
        compressed_children = []
        for child in node.children:
            compressed_child = self._compress_node(child, depth + 1, max_depth, compression_level)
            if compressed_child:
                compressed_children.append(compressed_child)
        
        compressed_node.children = compressed_children
        
        # Update stats
        self.compression_stats['nodes_compressed'] += 1
        
        return compressed_node
    
    def _calculate_node_relevance(self, node: EnhancedDOMTreeNode) -> float:
        """Calculate relevance score for a node based on multiple factors"""
        score = 0.0
        
        # 1. Task context matching
        if self.task_context:
            task_score = self._calculate_task_relevance(node, self.task_context)
            score += task_score * self.relevance_weights['task_match']
        
        # 2. Interactive elements are highly relevant
        if self._is_interactive_element(node):
            score += 1.0 * self.relevance_weights['interactive']
        
        # 3. Visibility and size
        visibility_score = self._calculate_visibility_score(node)
        score += visibility_score * self.relevance_weights['visibility']
        
        # 4. Semantic importance (headings, main content, etc.)
        semantic_score = self._calculate_semantic_importance(node)
        score += semantic_score * self.relevance_weights['semantic_importance']
        
        # 5. Text density (more text = more context)
        text_score = self._calculate_text_density(node)
        score += text_score * self.relevance_weights['text_density']
        
        return min(score, 1.0)  # Normalize to 0-1
    
    def _calculate_task_relevance(self, node: EnhancedDOMTreeNode, task_context: str) -> float:
        """Calculate how relevant a node is to the given task context"""
        if not task_context:
            return 0.0
        
        task_words = set(task_context.lower().split())
        node_text = self._extract_node_text(node).lower()
        
        # Check for direct word matches
        text_words = set(node_text.split())
        matches = task_words.intersection(text_words)
        
        if not matches:
            return 0.0
        
        # Calculate relevance based on match quality
        match_ratio = len(matches) / len(task_words) if task_words else 0
        
        # Boost for exact phrase matches
        if task_context.lower() in node_text:
            match_ratio = min(match_ratio * 1.5, 1.0)
        
        return match_ratio
    
    def _is_interactive_element(self, node: EnhancedDOMTreeNode) -> bool:
        """Check if node is an interactive element"""
        interactive_tags = {'button', 'input', 'select', 'textarea', 'a', 'area', 'details'}
        if node.tag_name.lower() in interactive_tags:
            return True
        
        # Check for interactive attributes
        interactive_attrs = ['onclick', 'onchange', 'onsubmit', 'role', 'tabindex']
        for attr in interactive_attrs:
            if node.attributes.get(attr):
                return True
        
        # Check for ARIA roles
        aria_role = node.attributes.get('role', '').lower()
        interactive_roles = {'button', 'link', 'checkbox', 'radio', 'textbox', 'combobox'}
        if aria_role in interactive_roles:
            return True
        
        return False
    
    def _is_interactive_svg(self, node: EnhancedDOMTreeNode) -> bool:
        """Check if SVG element is interactive"""
        # SVG elements with click handlers or ARIA attributes
        if node.attributes.get('onclick') or node.attributes.get('role'):
            return True
        
        # Check for SVG links
        if node.tag_name.lower() == 'a' and node.attributes.get('href'):
            return True
        
        return False
    
    def _calculate_visibility_score(self, node: EnhancedDOMTreeNode) -> float:
        """Calculate visibility score based on DOM properties"""
        score = 0.0
        
        # Check if element is hidden
        style = node.attributes.get('style', '')
        if 'display:none' in style or 'visibility:hidden' in style:
            return 0.0
        
        # Check bounding box size
        if node.bounding_box:
            area = node.bounding_box.width * node.bounding_box.height
            if area > 100:  # Reasonable minimum visible size
                score += 0.5
            if area > 1000:  # Larger visible area
                score += 0.3
        
        # Check opacity
        if 'opacity:0' in style or 'opacity: 0' in style:
            score *= 0.2
        
        return min(score, 1.0)
    
    def _calculate_semantic_importance(self, node: EnhancedDOMTreeNode) -> float:
        """Calculate semantic importance based on HTML structure"""
        score = 0.0
        
        # Headings are semantically important
        if node.tag_name.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            score += 0.7
        
        # Main content areas
        semantic_roles = ['main', 'article', 'section', 'nav', 'aside', 'header', 'footer']
        role = node.attributes.get('role', '').lower()
        if role in semantic_roles or node.tag_name.lower() in semantic_roles:
            score += 0.6
        
        # ARIA landmarks
        aria_role = node.attributes.get('role', '').lower()
        landmark_roles = ['banner', 'navigation', 'main', 'complementary', 'contentinfo']
        if aria_role in landmark_roles:
            score += 0.8
        
        return min(score, 1.0)
    
    def _calculate_text_density(self, node: EnhancedDOMTreeNode) -> float:
        """Calculate text density score"""
        text = self._extract_node_text(node)
        if not text:
            return 0.0
        
        # Normalize by length (longer text = more context)
        length_score = min(len(text) / 200, 1.0)  # Cap at 200 chars
        
        # Check for meaningful text (not just whitespace or single chars)
        meaningful_chars = sum(1 for c in text if c.isalnum())
        if meaningful_chars < 3:
            return 0.1
        
        return length_score
    
    def _extract_node_text(self, node: EnhancedDOMTreeNode) -> str:
        """Extract text content from node and its children"""
        texts = []
        
        # Get direct text
        if node.text:
            texts.append(node.text.strip())
        
        # Get child text (limited depth to avoid recursion)
        for child in node.children[:5]:  # Limit to first 5 children
            child_text = self._extract_node_text(child)
            if child_text:
                texts.append(child_text)
        
        return ' '.join(texts)
    
    def _should_compress_node(self, node: EnhancedDOMTreeNode, 
                            relevance_score: float, 
                            compression_level: CompressionLevel) -> str:
        """Determine if and how to compress a node"""
        
        # Always keep highly relevant nodes
        if relevance_score > 0.7:
            return 'keep'
        
        # Apply compression based on level
        if compression_level == CompressionLevel.MINIMAL:
            # Only keep interactive elements and high relevance
            if relevance_score < 0.3:
                return 'remove'
            elif relevance_score < 0.5:
                return 'compress_text'
        
        elif compression_level == CompressionLevel.BALANCED:
            # Keep structure but compress low relevance
            if relevance_score < 0.2:
                return 'remove'
            elif relevance_score < 0.4:
                return 'compress_text'
        
        elif compression_level == CompressionLevel.DETAILED:
            # Keep more nodes, only remove very low relevance
            if relevance_score < 0.1:
                return 'remove'
        
        # FULL compression keeps everything
        return 'keep'
    
    def _create_compressed_node(self, node: EnhancedDOMTreeNode, 
                              relevance_score: float,
                              compression_level: CompressionLevel) -> EnhancedDOMTreeNode:
        """Create a compressed version of the node"""
        
        # Create copy of node
        compressed = EnhancedDOMTreeNode(
            node_id=node.node_id,
            node_type=node.node_type,
            tag_name=node.tag_name,
            attributes=node.attributes.copy() if node.attributes else {},
            text=node.text,
            children=[],
            parent=node.parent,
            bounding_box=node.bounding_box,
            ax_node=node.ax_node,
        )
        
        # Apply text compression if needed
        if compression_level != CompressionLevel.FULL and relevance_score < 0.4:
            compressed.text = self._compress_text(compressed.text)
        
        # Add relevance metadata
        compressed.attributes['_relevance_score'] = str(relevance_score)
        compressed.attributes['_compressed'] = 'true'
        
        return compressed
    
    def _compress_text(self, text: Optional[str]) -> Optional[str]:
        """Compress text content"""
        if not text:
            return text
        
        # Remove extra whitespace
        compressed = ' '.join(text.split())
        
        # Truncate long text
        if len(compressed) > 100:
            compressed = compressed[:97] + '...'
        
        return compressed
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.compression_stats.copy()

class ElementRelevanceScorer:
    """
    Scores elements based on relevance to current task.
    Uses lightweight models for fast scoring.
    """
    
    def __init__(self, task_context: Optional[str] = None):
        self.task_context = task_context
        self.scoring_cache = {}
        
        # Lightweight feature extractors
        self.feature_extractors = [
            self._extract_text_features,
            self._extract_structural_features,
            self._extract_visual_features,
            self._extract_interactive_features,
        ]
    
    def score_element(self, element: EnhancedDOMTreeNode) -> float:
        """Score element relevance (0-1)"""
        # Create cache key
        cache_key = self._create_element_fingerprint(element)
        
        # Check cache
        if cache_key in self.scoring_cache:
            return self.scoring_cache[cache_key]
        
        # Extract features
        features = {}
        for extractor in self.feature_extractors:
            features.update(extractor(element))
        
        # Calculate score using simple weighted average
        score = (
            features.get('text_relevance', 0) * 0.3 +
            features.get('interactive_score', 0) * 0.3 +
            features.get('visibility_score', 0) * 0.2 +
            features.get('semantic_score', 0) * 0.2
        )
        
        # Cache result
        self.scoring_cache[cache_key] = score
        
        return score
    
    def _create_element_fingerprint(self, element: EnhancedDOMTreeNode) -> str:
        """Create unique fingerprint for caching"""
        fingerprint_data = {
            'tag': element.tag_name,
            'attributes': sorted(element.attributes.items()) if element.attributes else [],
            'text': element.text[:100] if element.text else '',
        }
        return hashlib.md5(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    
    def _extract_text_features(self, element: EnhancedDOMTreeNode) -> Dict[str, float]:
        """Extract text-based features"""
        features = {}
        
        text = element.text or ''
        if self.task_context:
            # Simple keyword matching
            task_words = set(self.task_context.lower().split())
            text_words = set(text.lower().split())
            
            if task_words and text_words:
                overlap = len(task_words.intersection(text_words))
                features['text_relevance'] = overlap / len(task_words)
            else:
                features['text_relevance'] = 0.0
        else:
            # No task context, score based on text length
            features['text_relevance'] = min(len(text) / 100, 1.0)
        
        return features
    
    def _extract_structural_features(self, element: EnhancedDOMTreeNode) -> Dict[str, float]:
        """Extract structural features"""
        features = {}
        
        # Depth penalty (deeper = less important)
        depth = self._calculate_depth(element)
        features['depth_score'] = max(0, 1 - (depth / 10))
        
        # Child count (too many = noise)
        child_count = len(element.children) if element.children else 0
        features['child_score'] = 1.0 if child_count < 5 else 0.5
        
        return features
    
    def _extract_visual_features(self, element: EnhancedDOMTreeNode) -> Dict[str, float]:
        """Extract visual features"""
        features = {}
        
        if element.bounding_box:
            area = element.bounding_box.width * element.bounding_box.height
            features['visibility_score'] = min(area / 10000, 1.0)  # Normalize by typical size
        else:
            features['visibility_score'] = 0.0
        
        return features
    
    def _extract_interactive_features(self, element: EnhancedDOMTreeNode) -> Dict[str, float]:
        """Extract interactive features"""
        features = {}
        
        # Check for interactive attributes
        interactive_score = 0.0
        
        if element.tag_name.lower() in ['button', 'input', 'select', 'textarea', 'a']:
            interactive_score += 0.5
        
        if element.attributes.get('onclick') or element.attributes.get('href'):
            interactive_score += 0.3
        
        if element.attributes.get('role'):
            interactive_score += 0.2
        
        features['interactive_score'] = min(interactive_score, 1.0)
        
        return features
    
    def _calculate_depth(self, node: EnhancedDOMTreeNode, current_depth: int = 0) -> int:
        """Calculate node depth in tree"""
        if not node.parent:
            return current_depth
        return self._calculate_depth(node.parent, current_depth + 1)

class PagePatternCache:
    """
    Caches common page patterns and element fingerprints.
    Dramatically speeds up repeated visits to similar pages.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self.pattern_cache = {}
        self.element_fingerprints = {}
        self.max_cache_size = max_cache_size
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'pattern_matches': 0,
            'cache_size': 0,
        }
    
    def get_cached_pattern(self, page_signature: str) -> Optional[Dict[str, Any]]:
        """Get cached pattern for similar page"""
        if page_signature in self.pattern_cache:
            self.stats['hits'] += 1
            return self.pattern_cache[page_signature]
        
        self.stats['misses'] += 1
        return None
    
    def cache_pattern(self, page_signature: str, pattern_data: Dict[str, Any]):
        """Cache page pattern"""
        # Implement LRU-like eviction if cache is full
        if len(self.pattern_cache) >= self.max_cache_size:
            self._evict_old_patterns()
        
        self.pattern_cache[page_signature] = pattern_data
        self.stats['cache_size'] = len(self.pattern_cache)
    
    def get_element_relevance(self, fingerprint: str) -> Optional[float]:
        """Get cached element relevance score"""
        if fingerprint in self.element_fingerprints:
            return self.element_fingerprints[fingerprint].relevance_score
        return None
    
    def cache_element_relevance(self, fingerprint: str, relevance_score: float):
        """Cache element relevance score"""
        self.element_fingerprints[fingerprint] = ElementFingerprint(
            tag_name="",  # Simplified for cache
            attributes_hash=fingerprint,
            text_hash="",
            structure_hash="",
            relevance_score=relevance_score,
            timestamp=time.time()
        )
    
    def _evict_old_patterns(self):
        """Evict oldest patterns when cache is full"""
        if not self.pattern_cache:
            return
        
        # Simple FIFO eviction
        oldest_key = next(iter(self.pattern_cache))
        del self.pattern_cache[oldest_key]
    
    def generate_page_signature(self, root: EnhancedDOMTreeNode) -> str:
        """Generate signature for page pattern matching"""
        # Extract key structural features
        features = {
            'tag_distribution': self._get_tag_distribution(root),
            'depth_profile': self._get_depth_profile(root),
            'interactive_count': self._count_interactive_elements(root),
            'text_density': self._calculate_text_density(root),
        }
        
        return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
    
    def _get_tag_distribution(self, node: EnhancedDOMTreeNode, 
                            distribution: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        """Get distribution of tag types in tree"""
        if distribution is None:
            distribution = defaultdict(int)
        
        distribution[node.tag_name.lower()] += 1
        
        for child in node.children:
            self._get_tag_distribution(child, distribution)
        
        return dict(distribution)
    
    def _get_depth_profile(self, node: EnhancedDOMTreeNode, 
                          profile: Optional[List[int]] = None, 
                          current_depth: int = 0) -> List[int]:
        """Get depth profile of tree"""
        if profile is None:
            profile = []
        
        if current_depth >= len(profile):
            profile.extend([0] * (current_depth - len(profile) + 1))
        
        profile[current_depth] += 1
        
        for child in node.children:
            self._get_depth_profile(child, profile, current_depth + 1)
        
        return profile
    
    def _count_interactive_elements(self, node: EnhancedDOMTreeNode) -> int:
        """Count interactive elements in tree"""
        count = 0
        
        # Check if current node is interactive
        interactive_tags = {'button', 'input', 'select', 'textarea', 'a'}
        if node.tag_name.lower() in interactive_tags:
            count += 1
        
        # Count in children
        for child in node.children:
            count += self._count_interactive_elements(child)
        
        return count
    
    def _calculate_text_density(self, node: EnhancedDOMTreeNode) -> float:
        """Calculate text density in tree"""
        total_text = 0
        total_nodes = 0
        
        def traverse(n: EnhancedDOMTreeNode):
            nonlocal total_text, total_nodes
            total_nodes += 1
            if n.text:
                total_text += len(n.text)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        
        return total_text / total_nodes if total_nodes > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.copy()

class AdaptiveElementLocator:
    """
    Adaptive Element Location Engine with multi-strategy fallback system.
    Implements strategy pattern with pluggable locators for robust element discovery.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.strategies = [
            self._css_strategy,
            self._xpath_strategy,
            self._accessibility_strategy,
            self._visual_strategy,
            self._llm_guided_strategy
        ]
        self.strategy_weights = {
            'css': 0.3,
            'xpath': 0.25,
            'accessibility': 0.25,
            'visual': 0.15,
            'llm': 0.05
        }
        
    def generate_adaptive_selector(self, node: EnhancedDOMTreeNode) -> Dict[str, Any]:
        """Generate adaptive selector with multiple strategies and fallback chains."""
        strategies_results = []
        
        for strategy_func in self.strategies:
            try:
                result = strategy_func(node)
                if result:
                    strategies_results.append(result)
            except Exception:
                continue
        
        # Sort by robustness score
        strategies_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Build fallback chain
        fallback_chain = []
        for result in strategies_results[:3]:  # Top 3 strategies
            fallback_chain.append({
                'strategy': result['strategy'],
                'selector': result['selector'],
                'score': result['score'],
                'type': result['type']
            })
        
        # Calculate overall robustness
        overall_score = self._calculate_robustness_score(strategies_results)
        
        return {
            'primary': strategies_results[0] if strategies_results else None,
            'fallback_chain': fallback_chain,
            'overall_score': overall_score,
            'strategies_count': len(strategies_results),
            'adaptive': True
        }
    
    def _css_strategy(self, node: EnhancedDOMTreeNode) -> Optional[Dict[str, Any]]:
        """Generate CSS selector with robustness scoring."""
        if not node.attributes:
            return None
            
        # Build selector parts
        selector_parts = []
        
        # Tag name
        selector_parts.append(node.tag_name.lower())
        
        # ID (most specific)
        if node.attributes.get('id'):
            selector_parts.append(f"#{node.attributes['id']}")
            score = 0.95
        # Classes
        elif node.attributes.get('class'):
            classes = node.attributes['class'].split()
            selector_parts.append('.' + '.'.join(classes[:2]))  # Limit to first 2 classes
            score = 0.8
        # Other attributes
        else:
            # Use data attributes or other stable attributes
            for attr in ['data-testid', 'data-id', 'name', 'type']:
                if node.attributes.get(attr):
                    selector_parts.append(f'[{attr}="{node.attributes[attr]}"]')
                    score = 0.7
                    break
            else:
                score = 0.5
        
        selector = ''.join(selector_parts)
        
        return {
            'strategy': 'css',
            'selector': selector,
            'score': score,
            'type': 'css',
            'specificity': len(selector_parts)
        }
    
    def _xpath_strategy(self, node: EnhancedDOMTreeNode) -> Optional[Dict[str, Any]]:
        """Generate XPath selector with position awareness."""
        if not node.parent:
            return None
        
        # Build XPath based on position and attributes
        xpath_parts = []
        
        # Start from current node
        current = node
        while current and current.parent:
            # Get position among siblings
            siblings = [child for child in current.parent.children 
                       if child.tag_name == current.tag_name]
            position = siblings.index(current) + 1 if len(siblings) > 1 else None
            
            # Build segment
            segment = current.tag_name.lower()
            if position:
                segment += f'[{position}]'
            
            xpath_parts.insert(0, segment)
            current = current.parent
        
        # Add attributes for specificity
        if node.attributes.get('id'):
            xpath_parts[-1] += f'[@id="{node.attributes["id"]}"]'
            score = 0.9
        elif node.attributes.get('class'):
            xpath_parts[-1] += f'[contains(@class, "{node.attributes["class"].split()[0]}")]'
            score = 0.75
        else:
            score = 0.6
        
        xpath = '//' + '/'.join(xpath_parts)
        
        return {
            'strategy': 'xpath',
            'selector': xpath,
            'score': score,
            'type': 'xpath',
            'depth': len(xpath_parts)
        }
    
    def _accessibility_strategy(self, node: EnhancedDOMTreeNode) -> Optional[Dict[str, Any]]:
        """Generate selector using accessibility tree analysis."""
        if not node.ax_node:
            return None
        
        ax = node.ax_node
        selector_parts = []
        
        # Role-based selector
        if ax.role:
            selector_parts.append(f'[role="{ax.role}"]')
        
        # Name-based selector
        if ax.name:
            # Use aria-label or text content
            if node.attributes.get('aria-label'):
                selector_parts.append(f'[aria-label="{node.attributes["aria-label"]}"]')
            else:
                # Use text content as selector (for buttons, links, etc.)
                selector_parts.append(f':has-text("{ax.name[:50]}")')  # Limit length
        
        # Properties-based selector
        if ax.properties:
            for prop in ax.properties[:2]:  # Limit to first 2 properties
                if prop.get('name') and prop.get('value'):
                    selector_parts.append(f'[{prop["name"]}="{prop["value"]}"]')
        
        if not selector_parts:
            return None
        
        selector = ''.join(selector_parts)
        score = 0.85 if ax.role and ax.name else 0.7
        
        return {
            'strategy': 'accessibility',
            'selector': selector,
            'score': score,
            'type': 'accessibility',
            'role': ax.role,
            'name': ax.name
        }
    
    def _visual_strategy(self, node: EnhancedDOMTreeNode) -> Optional[Dict[str, Any]]:
        """Generate selector using visual recognition hints."""
        if not node.bounding_box:
            return None
        
        bbox = node.bounding_box
        
        # Visual position-based selector
        selector_parts = []
        
        # Size-based filtering
        if bbox.width > 0 and bbox.height > 0:
            # Relative position in viewport
            if hasattr(node, 'viewport_info'):
                vp = node.viewport_info
                rel_x = bbox.x / vp.width if vp.width > 0 else 0
                rel_y = bbox.y / vp.height if vp.height > 0 else 0
                
                # Create selector based on position
                if rel_x < 0.3 and rel_y < 0.3:
                    selector_parts.append('[data-visual-position="top-left"]')
                elif rel_x > 0.7 and rel_y < 0.3:
                    selector_parts.append('[data-visual-position="top-right"]')
                elif rel_x < 0.3 and rel_y > 0.7:
                    selector_parts.append('[data-visual-position="bottom-left"]')
                elif rel_x > 0.7 and rel_y > 0.7:
                    selector_parts.append('[data-visual-position="bottom-right"]')
                else:
                    selector_parts.append('[data-visual-position="center"]')
                
                score = 0.6
            else:
                # No viewport info, use size only
                if bbox.width * bbox.height > 10000:  # Large element
                    selector_parts.append('[data-visual-size="large"]')
                    score = 0.5
                else:
                    return None
        else:
            return None
        
        selector = ''.join(selector_parts)
        
        return {
            'strategy': 'visual',
            'selector': selector,
            'score': score,
            'type': 'visual',
            'position': (bbox.x, bbox.y),
            'size': (bbox.width, bbox.height)
        }
    
    def _llm_guided_strategy(self, node: EnhancedDOMTreeNode) -> Optional[Dict[str, Any]]:
        """Generate selector using LLM guidance hints."""
        # This would integrate with an LLM for intelligent selector generation
        # For now, return a placeholder
        
        # Check for semantic hints
        semantic_hints = []
        
        # Check for common patterns
        if node.tag_name.lower() == 'button':
            semantic_hints.append('button')
        elif node.tag_name.lower() == 'a':
            semantic_hints.append('link')
        elif node.tag_name.lower() == 'input':
            input_type = node.attributes.get('type', 'text')
            semantic_hints.append(f'input-{input_type}')
        
        # Check for ARIA attributes
        if node.attributes.get('aria-label'):
            semantic_hints.append(f'aria-{node.attributes["aria-label"][:20]}')
        
        if not semantic_hints:
            return None
        
        selector = f'[data-llm-hint="{"-".join(semantic_hints)}"]'
        
        return {
            'strategy': 'llm',
            'selector': selector,
            'score': 0.4,  # Lower confidence for LLM hints
            'type': 'llm',
            'hints': semantic_hints
        }
    
    def _calculate_robustness_score(self, strategies_results: List[Dict[str, Any]]) -> float:
        """Calculate overall robustness score from multiple strategies."""
        if not strategies_results:
            return 0.0
        
        # Weighted average of top strategies
        total_score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(strategies_results[:3]):  # Top 3 strategies
            weight = 1.0 / (i + 1)  # Higher weight for better strategies
            total_score += result['score'] * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

class LLMContextOptimizer:
    """
    Main LLM Context Optimization Engine.
    Orchestrates compression, relevance scoring, and caching.
    """
    
    def __init__(self, task_context: Optional[str] = None, 
                 compression_level: CompressionLevel = CompressionLevel.BALANCED,
                 enable_caching: bool = True):
        self.task_context = task_context
        self.compression_level = compression_level
        self.enable_caching = enable_caching
        
        # Initialize components
        self.compressor = SemanticCompressor(task_context)
        self.relevance_scorer = ElementRelevanceScorer(task_context)
        self.pattern_cache = PagePatternCache()
        
        # Statistics
        self.optimization_stats = {
            'pages_processed': 0,
            'total_compression_ratio': 0.0,
            'cache_hits': 0,
            'processing_time': 0.0,
        }
    
    def optimize_for_llm(self, root: EnhancedDOMTreeNode) -> Dict[str, Any]:
        """
        Optimize DOM tree for LLM consumption.
        Returns optimized tree with metadata.
        """
        start_time = time.time()
        
        # Check cache for page pattern
        page_signature = self.pattern_cache.generate_page_signature(root)
        cached_pattern = None
        
        if self.enable_caching:
            cached_pattern = self.pattern_cache.get_cached_pattern(page_signature)
            if cached_pattern:
                self.optimization_stats['cache_hits'] += 1
                return cached_pattern
        
        # Apply semantic compression
        compressed_root = self.compressor.compress_dom_tree(
            root, 
            compression_level=self.compression_level
        )
        
        # Score elements for relevance
        self._score_elements_recursive(compressed_root)
        
        # Generate optimized representation
        optimized_representation = self._create_optimized_representation(compressed_root)
        
        # Cache the pattern
        if self.enable_caching:
            self.pattern_cache.cache_pattern(page_signature, optimized_representation)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.optimization_stats['pages_processed'] += 1
        self.optimization_stats['processing_time'] += processing_time
        
        compression_stats = self.compressor.get_compression_stats()
        if compression_stats['compression_ratio'] > 0:
            self.optimization_stats['total_compression_ratio'] = (
                (self.optimization_stats['total_compression_ratio'] * 
                 (self.optimization_stats['pages_processed'] - 1) + 
                 compression_stats['compression_ratio']) / 
                self.optimization_stats['pages_processed']
            )
        
        # Add metadata to result
        optimized_representation['_metadata'] = {
            'compression_stats': compression_stats,
            'optimization_time': processing_time,
            'task_context': self.task_context,
            'compression_level': self.compression_level.value,
            'page_signature': page_signature,
            'cache_used': cached_pattern is not None,
        }
        
        return optimized_representation
    
    def _score_elements_recursive(self, node: EnhancedDOMTreeNode):
        """Recursively score all elements in tree"""
        # Score current node
        relevance_score = self.relevance_scorer.score_element(node)
        node.attributes['_relevance_score'] = str(relevance_score)
        
        # Cache the score
        if self.enable_caching:
            fingerprint = self.relevance_scorer._create_element_fingerprint(node)
            self.pattern_cache.cache_element_relevance(fingerprint, relevance_score)
        
        # Recursively score children
        for child in node.children:
            self._score_elements_recursive(child)
    
    def _create_optimized_representation(self, root: EnhancedDOMTreeNode) -> Dict[str, Any]:
        """Create optimized representation for LLM"""
        representation = {
            'dom_tree': self._serialize_tree_for_llm(root),
            'interactive_elements': self._extract_interactive_elements(root),
            'semantic_structure': self._extract_semantic_structure(root),
            'text_content': self._extract_optimized_text(root),
            'relevance_scores': self._collect_relevance_scores(root),
        }
        
        return representation
    
    def _serialize_tree_for_llm(self, node: EnhancedDOMTreeNode, 
                               depth: int = 0, 
                               max_depth: int = 10) -> Dict[str, Any]:
        """Serialize tree in LLM-friendly format"""
        if depth > max_depth:
            return {'truncated': True}
        
        serialized = {
            'tag': node.tag_name.lower(),
            'id': node.attributes.get('id'),
            'classes': node.attributes.get('class', '').split() if node.attributes.get('class') else [],
            'text': self._compress_text_for_llm(node.text),
            'relevance': float(node.attributes.get('_relevance_score', 0)),
            'children': [],
        }
        
        # Add important attributes
        important_attrs = ['role', 'aria-label', 'type', 'name', 'placeholder', 'value']
        for attr in important_attrs:
            if node.attributes.get(attr):
                serialized[attr] = node.attributes[attr]
        
        # Add bounding box if available
        if node.bounding_box:
            serialized['position'] = {
                'x': node.bounding_box.x,
                'y': node.bounding_box.y,
                'width': node.bounding_box.width,
                'height': node.bounding_box.height,
            }
        
        # Serialize children (only if relevant)
        for child in node.children:
            child_relevance = float(child.attributes.get('_relevance_score', 0))
            if child_relevance > 0.1:  # Only include somewhat relevant children
                child_serialized = self._serialize_tree_for_llm(child, depth + 1, max_depth)
                if child_serialized:
                    serialized['children'].append(child_serialized)
        
        # Simplify if no children and low relevance
        if not serialized['children'] and serialized['relevance'] < 0.3:
            serialized.pop('children', None)
        
        return serialized
    
    def _extract_interactive_elements(self, node: EnhancedDOMTreeNode) -> List[Dict[str, Any]]:
        """Extract all interactive elements with their selectors"""
        interactive_elements = []
        
        def traverse(n: EnhancedDOMTreeNode):
            # Check if interactive
            interactive_tags = {'button', 'input', 'select', 'textarea', 'a', 'area'}
            is_interactive = (
                n.tag_name.lower() in interactive_tags or
                n.attributes.get('onclick') or
                n.attributes.get('role') in ['button', 'link', 'checkbox', 'radio']
            )
            
            if is_interactive:
                # Generate selector
                locator = AdaptiveElementLocator()
                selector_info = locator.generate_adaptive_selector(n)
                
                element_info = {
                    'tag': n.tag_name.lower(),
                    'text': self._compress_text_for_llm(n.text),
                    'relevance': float(n.attributes.get('_relevance_score', 0)),
                    'selector': selector_info['primary']['selector'] if selector_info['primary'] else None,
                    'selector_strategy': selector_info['primary']['strategy'] if selector_info['primary'] else None,
                    'attributes': {
                        k: v for k, v in n.attributes.items() 
                        if k in ['id', 'class', 'name', 'type', 'role', 'aria-label']
                    },
                }
                
                if n.bounding_box:
                    element_info['position'] = {
                        'x': n.bounding_box.x,
                        'y': n.bounding_box.y,
                    }
                
                interactive_elements.append(element_info)
            
            # Traverse children
            for child in n.children:
                traverse(child)
        
        traverse(node)
        
        # Sort by relevance
        interactive_elements.sort(key=lambda x: x['relevance'], reverse=True)
        
        return interactive_elements
    
    def _extract_semantic_structure(self, node: EnhancedDOMTreeNode) -> Dict[str, Any]:
        """Extract semantic structure of page"""
        structure = {
            'headings': [],
            'navigation': [],
            'main_content': [],
            'forms': [],
            'landmarks': [],
        }
        
        def traverse(n: EnhancedDOMTreeNode, path: str = ''):
            current_path = f"{path}/{n.tag_name.lower()}"
            
            # Check for headings
            if n.tag_name.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                structure['headings'].append({
                    'level': int(n.tag_name[1]),
                    'text': self._compress_text_for_llm(n.text),
                    'path': current_path,
                })
            
            # Check for navigation
            if (n.tag_name.lower() == 'nav' or 
                n.attributes.get('role') == 'navigation' or
                any(keyword in (n.attributes.get('class', '') or '').lower() 
                    for keyword in ['nav', 'menu', 'navbar'])):
                structure['navigation'].append({
                    'text': self._compress_text_for_llm(n.text)[:100],
                    'path': current_path,
                })
            
            # Check for main content
            if (n.tag_name.lower() == 'main' or 
                n.attributes.get('role') == 'main'):
                structure['main_content'].append({
                    'text': self._compress_text_for_llm(n.text)[:200],
                    'path': current_path,
                })
            
            # Check for forms
            if n.tag_name.lower() == 'form':
                form_fields = []
                for child in n.children:
                    if child.tag_name.lower() in ['input', 'select', 'textarea']:
                        form_fields.append({
                            'type': child.attributes.get('type', child.tag_name.lower()),
                            'name': child.attributes.get('name'),
                            'label': self._compress_text_for_llm(child.text),
                        })
                
                structure['forms'].append({
                    'fields': form_fields[:10],  # Limit fields
                    'path': current_path,
                })
            
            # Check for ARIA landmarks
            role = n.attributes.get('role', '')
            if role in ['banner', 'navigation', 'main', 'complementary', 'contentinfo']:
                structure['landmarks'].append({
                    'role': role,
                    'text': self._compress_text_for_llm(n.text)[:100],
                    'path': current_path,
                })
            
            # Traverse children
            for child in n.children:
                traverse(child, current_path)
        
        traverse(node)
        
        # Limit arrays
        for key in structure:
            if isinstance(structure[key], list):
                structure[key] = structure[key][:5]  # Keep top 5 of each
        
        return structure
    
    def _extract_optimized_text(self, node: EnhancedDOMTreeNode) -> str:
        """Extract optimized text content"""
        text_parts = []
        
        def traverse(n: EnhancedDOMTreeNode, depth: int = 0):
            if depth > 5:  # Limit depth for text extraction
                return
            
            # Add node text if relevant
            relevance = float(n.attributes.get('_relevance_score', 0))
            if n.text and relevance > 0.2:
                compressed_text = self._compress_text_for_llm(n.text)
                if compressed_text:
                    text_parts.append(compressed_text)
            
            # Traverse children
            for child in n.children:
                traverse(child, depth + 1)
        
        traverse(node)
        
        # Join and compress final text
        full_text = ' '.join(text_parts)
        return self._compress_text_for_llm(full_text, max_length=1000)
    
    def _collect_relevance_scores(self, node: EnhancedDOMTreeNode) -> Dict[str, float]:
        """Collect relevance scores for analysis"""
        scores = {}
        
        def traverse(n: EnhancedDOMTreeNode, path: str = ''):
            current_path = f"{path}/{n.tag_name.lower()}"
            relevance = float(n.attributes.get('_relevance_score', 0))
            
            if relevance > 0.3:  # Only collect meaningful scores
                scores[current_path] = relevance
            
            for child in n.children:
                traverse(child, current_path)
        
        traverse(node)
        
        # Sort by relevance
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20])
    
    def _compress_text_for_llm(self, text: Optional[str], max_length: int = 200) -> Optional[str]:
        """Compress text specifically for LLM consumption"""
        if not text:
            return text
        
        # Remove extra whitespace
        compressed = ' '.join(text.split())
        
        # Truncate if too long
        if len(compressed) > max_length:
            compressed = compressed[:max_length-3] + '...'
        
        return compressed if compressed else None
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = self.optimization_stats.copy()
        stats.update({
            'cache_stats': self.pattern_cache.get_stats(),
            'compression_stats': self.compressor.get_compression_stats(),
        })
        return stats

# Integration with existing serializer
class EnhancedDOMSerializer:
    """
    Enhanced DOM Serializer with LLM Context Optimization.
    Integrates optimization engine into serialization pipeline.
    """
    
    def __init__(self, task_context: Optional[str] = None,
                 compression_level: CompressionLevel = CompressionLevel.BALANCED):
        self.task_context = task_context
        self.compression_level = compression_level
        
        # Initialize components
        self.clickable_detector = ClickableElementDetector()
        self.paint_order_remover = PaintOrderRemover()
        self.llm_optimizer = LLMContextOptimizer(task_context, compression_level)
        
        # Statistics
        self.serialization_stats = {
            'total_serializations': 0,
            'avg_compression_ratio': 0.0,
            'avg_optimization_time': 0.0,
        }
    
    def serialize_dom_tree(self, root: EnhancedDOMTreeNode, 
                          optimize_for_llm: bool = True) -> SerializedDOMState:
        """
        Serialize DOM tree with optional LLM optimization.
        Returns serialized state ready for LLM consumption.
        """
        start_time = time.time()
        
        # Apply paint order removal
        cleaned_root = self.paint_order_remover.remove_paint_order(root)
        
        # Detect clickable elements
        clickable_elements = self.clickable_detector.detect_clickable_elements(cleaned_root)
        
        # Optimize for LLM if requested
        if optimize_for_llm:
            optimized_data = self.llm_optimizer.optimize_for_llm(cleaned_root)
            
            # Create serialized state with optimized data
            serialized_state = SerializedDOMState(
                root_element=cleaned_root,
                clickable_elements=clickable_elements,
                selector_map=self._build_selector_map(cleaned_root),
                element_tree=self._build_element_tree(cleaned_root),
                # Add optimized LLM data
                llm_optimized_data=optimized_data,
                optimization_enabled=True,
            )
        else:
            # Standard serialization without optimization
            serialized_state = SerializedDOMState(
                root_element=cleaned_root,
                clickable_elements=clickable_elements,
                selector_map=self._build_selector_map(cleaned_root),
                element_tree=self._build_element_tree(cleaned_root),
                optimization_enabled=False,
            )
        
        # Update statistics
        serialization_time = time.time() - start_time
        self.serialization_stats['total_serializations'] += 1
        
        if optimize_for_llm:
            optimization_stats = self.llm_optimizer.get_optimization_stats()
            compression_ratio = optimization_stats.get('total_compression_ratio', 0)
            
            # Update running averages
            n = self.serialization_stats['total_serializations']
            self.serialization_stats['avg_compression_ratio'] = (
                (self.serialization_stats['avg_compression_ratio'] * (n - 1) + compression_ratio) / n
            )
            self.serialization_stats['avg_optimization_time'] = (
                (self.serialization_stats['avg_optimization_time'] * (n - 1) + serialization_time) / n
            )
        
        return serialized_state
    
    def _build_selector_map(self, root: EnhancedDOMTreeNode) -> DOMSelectorMap:
        """Build selector map for all elements"""
        selector_map = {}
        locator = AdaptiveElementLocator()
        
        def traverse(node: EnhancedDOMTreeNode):
            selector_info = locator.generate_adaptive_selector(node)
            if selector_info['primary']:
                selector_map[node.node_id] = {
                    'primary': selector_info['primary']['selector'],
                    'fallbacks': [fb['selector'] for fb in selector_info['fallback_chain']],
                    'strategy': selector_info['primary']['strategy'],
                    'score': selector_info['overall_score'],
                }
            
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return selector_map
    
    def _build_element_tree(self, root: EnhancedDOMTreeNode) -> Dict[str, Any]:
        """Build simplified element tree for LLM"""
        def build_node(node: EnhancedDOMTreeNode) -> Dict[str, Any]:
            return {
                'tag': node.tag_name.lower(),
                'id': node.attributes.get('id'),
                'classes': node.attributes.get('class', '').split() if node.attributes.get('class') else [],
                'text': cap_text_length(node.text, 100) if node.text else None,
                'children': [build_node(child) for child in node.children[:10]],  # Limit children
                'relevance': float(node.attributes.get('_relevance_score', 0)),
            }
        
        return build_node(root)
    
    def get_serialization_stats(self) -> Dict[str, Any]:
        """Get serialization statistics"""
        stats = self.serialization_stats.copy()
        stats['llm_optimizer_stats'] = self.llm_optimizer.get_optimization_stats()
        return stats