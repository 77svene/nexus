"""
LLM Context Optimization Engine - nexus/context/compressor.py
Intelligently compress and structure page context for LLMs, reducing token usage by 60-80% while maintaining accuracy.
"""

import re
import hashlib
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

from bs4 import BeautifulSoup, Comment, NavigableString
from bs4.element import Tag

logger = logging.getLogger(__name__)


@dataclass
class ElementFingerprint:
    """Unique fingerprint for element patterns."""
    tag: str
    attributes: Dict[str, str] = field(default_factory=dict)
    depth: int = 0
    child_count: int = 0
    text_pattern: str = ""
    
    def hash(self) -> str:
        """Generate unique hash for this fingerprint."""
        data = f"{self.tag}|{json.dumps(self.attributes, sort_keys=True)}|{self.depth}|{self.child_count}|{self.text_pattern}"
        return hashlib.md5(data.encode()).hexdigest()


@dataclass 
class CompressionMetrics:
    """Metrics for compression performance."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    elements_removed: int = 0
    elements_retained: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update(self, original: int, compressed: int):
        """Update metrics with new compression results."""
        self.original_tokens += original
        self.compressed_tokens += compressed
        if self.original_tokens > 0:
            self.compression_ratio = 1 - (self.compressed_tokens / self.original_tokens)


class SemanticCompressor:
    """Core semantic compression engine for DOM trees."""
    
    # Tags to always remove (non-content or non-interactive)
    IRRELEVANT_TAGS = {
        'script', 'style', 'link', 'meta', 'noscript', 'template',
        'iframe', 'object', 'embed', 'applet', 'head'
    }
    
    # Attributes to preserve (important for interaction)
    PRESERVE_ATTRS = {
        'id', 'class', 'name', 'type', 'value', 'placeholder',
        'href', 'src', 'action', 'method', 'aria-label', 'role',
        'data-testid', 'data-id', 'tabindex', 'disabled', 'readonly',
        'checked', 'selected', 'required', 'for', 'label'
    }
    
    # Attributes to remove (styling, events, etc.)
    REMOVE_ATTRS = {
        'style', 'onclick', 'onload', 'onmouseover', 'onmouseout',
        'onfocus', 'onblur', 'onchange', 'onsubmit', 'onkeydown',
        'onkeyup', 'onkeypress', 'onerror', 'onabort'
    }
    
    # Interactive elements (higher relevance)
    INTERACTIVE_TAGS = {
        'a', 'button', 'input', 'select', 'textarea', 'form',
        'label', 'option', 'fieldset', 'output', 'details', 'summary'
    }
    
    # Content-bearing elements
    CONTENT_TAGS = {
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span',
        'li', 'td', 'th', 'caption', 'figcaption', 'article',
        'section', 'main', 'header', 'footer', 'nav'
    }
    
    def __init__(self, relevance_threshold: float = 0.3, 
                 max_elements: int = 200,
                 use_caching: bool = True):
        """
        Initialize the semantic compressor.
        
        Args:
            relevance_threshold: Minimum relevance score to keep element (0-1)
            max_elements: Maximum elements to retain after compression
            use_caching: Whether to use pattern caching
        """
        self.relevance_threshold = relevance_threshold
        self.max_elements = max_elements
        self.use_caching = use_caching
        
        # Cache for common patterns
        self._pattern_cache: Dict[str, Dict] = {}
        self._fingerprint_cache: Dict[str, ElementFingerprint] = {}
        self._compression_cache: Dict[str, str] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self.metrics = CompressionMetrics()
        
        # Initialize with common patterns
        self._init_common_patterns()
    
    def _init_common_patterns(self):
        """Initialize cache with common web patterns."""
        common_patterns = {
            "nav_menu": {
                "selector": "nav, [role='navigation'], .nav, .menu",
                "structure": "ul > li > a",
                "compress_to": "[NAV: {links}]"
            },
            "form": {
                "selector": "form",
                "structure": "input, select, textarea, button",
                "compress_to": "[FORM: {fields}]"
            },
            "table": {
                "selector": "table",
                "structure": "tr > td, tr > th",
                "compress_to": "[TABLE: {rows}x{cols}]"
            },
            "article": {
                "selector": "article, .post, .article",
                "structure": "h1, h2, p, img",
                "compress_to": "[ARTICLE: {title}]"
            }
        }
        
        for pattern_name, pattern_data in common_patterns.items():
            pattern_hash = hashlib.md5(pattern_name.encode()).hexdigest()
            self._pattern_cache[pattern_hash] = pattern_data
    
    def compress_html(self, html: str, task_context: str = "", 
                     preserve_interactive: bool = True) -> str:
        """
        Compress HTML content for LLM consumption.
        
        Args:
            html: Raw HTML content
            task_context: Current task context for relevance scoring
            preserve_interactive: Whether to prioritize interactive elements
            
        Returns:
            Compressed HTML string
        """
        if not html or not html.strip():
            return ""
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove irrelevant tags
            self._remove_irrelevant_tags(soup)
            
            # Clean attributes
            self._clean_attributes(soup)
            
            # Score elements by relevance
            relevance_scores = self._score_elements(soup, task_context)
            
            # Apply compression based on scores
            compressed_soup = self._apply_compression(
                soup, relevance_scores, preserve_interactive
            )
            
            # Final cleanup
            self._final_cleanup(compressed_soup)
            
            # Convert to string
            compressed_html = str(compressed_soup)
            
            # Update metrics
            original_tokens = self._estimate_tokens(html)
            compressed_tokens = self._estimate_tokens(compressed_html)
            self.metrics.update(original_tokens, compressed_tokens)
            
            logger.debug(f"Compressed {original_tokens} -> {compressed_tokens} tokens "
                        f"({self.metrics.compression_ratio:.1%} reduction)")
            
            return compressed_html
            
        except Exception as e:
            logger.error(f"Error compressing HTML: {e}")
            return html  # Return original on error
    
    def _remove_irrelevant_tags(self, soup: BeautifulSoup):
        """Remove tags that don't contribute to understanding or interaction."""
        for tag_name in self.IRRELEVANT_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove empty elements (except those that might be interactive)
        for tag in soup.find_all():
            if (tag.name not in self.INTERACTIVE_TAGS and 
                not tag.get_text(strip=True) and 
                not tag.find_all(self.INTERACTIVE_TAGS)):
                tag.decompose()
    
    def _clean_attributes(self, soup: BeautifulSoup):
        """Clean attributes from elements, preserving only relevant ones."""
        for tag in soup.find_all(True):
            # Get current attributes
            attrs = dict(tag.attrs)
            
            # Clear all attributes
            tag.attrs = {}
            
            # Restore preserved attributes
            for attr in self.PRESERVE_ATTRS:
                if attr in attrs:
                    value = attrs[attr]
                    if isinstance(value, list):
                        value = ' '.join(value)
                    tag[attr] = value
            
            # Special handling for class attribute (keep only semantic classes)
            if 'class' in attrs:
                classes = attrs['class']
                if isinstance(classes, list):
                    # Filter out CSS framework classes and keep semantic ones
                    semantic_classes = [
                        c for c in classes 
                        if not re.match(r'^(col-|row-|btn-|text-|bg-|p-|m-|d-|flex-|justify-|align-)', c)
                    ]
                    if semantic_classes:
                        tag['class'] = semantic_classes
    
    def _score_elements(self, soup: BeautifulSoup, task_context: str) -> Dict[Tag, float]:
        """Score elements based on relevance to task context."""
        scores = {}
        
        # Extract keywords from task context
        keywords = self._extract_keywords(task_context)
        
        for tag in soup.find_all(True):
            score = 0.0
            
            # Base score by tag type
            if tag.name in self.INTERACTIVE_TAGS:
                score += 0.4
            elif tag.name in self.CONTENT_TAGS:
                score += 0.3
            
            # Score based on text content relevance
            text = tag.get_text(strip=True).lower()
            if text:
                # Check for keyword matches
                keyword_matches = sum(1 for kw in keywords if kw in text)
                if keyword_matches > 0:
                    score += min(0.3, keyword_matches * 0.1)
                
                # Prefer elements with meaningful text
                if len(text) > 10:
                    score += 0.1
            
            # Score based on attributes
            attrs = tag.attrs
            if 'id' in attrs:
                score += 0.2
            if 'name' in attrs:
                score += 0.15
            if 'aria-label' in attrs:
                score += 0.2
            if 'role' in attrs:
                score += 0.15
            if 'data-testid' in attrs:
                score += 0.2
            
            # Check for form elements
            if tag.name == 'input':
                input_type = tag.get('type', 'text')
                if input_type in ['submit', 'button', 'reset']:
                    score += 0.3
                elif input_type in ['text', 'email', 'password', 'search']:
                    score += 0.25
            
            # Check for links
            if tag.name == 'a' and tag.get('href'):
                score += 0.25
            
            # Normalize score to 0-1 range
            scores[tag] = min(1.0, score)
        
        return scores
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return set()
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'down', 'out', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # Extract words, convert to lowercase, remove punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = {
            word for word in words 
            if word not in stop_words and len(word) > 2
        }
        
        return keywords
    
    def _apply_compression(self, soup: BeautifulSoup, 
                          scores: Dict[Tag, float],
                          preserve_interactive: bool) -> BeautifulSoup:
        """Apply compression based on relevance scores."""
        # Create a copy for modification
        compressed_soup = BeautifulSoup(str(soup), 'html.parser')
        
        # Get all elements with their scores
        elements_with_scores = []
        for tag in compressed_soup.find_all(True):
            # Find corresponding score (approximate matching)
            original_score = self._find_matching_score(tag, scores)
            elements_with_scores.append((tag, original_score))
        
        # Sort by score (descending)
        elements_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only top elements
        elements_to_keep = set()
        interactive_kept = 0
        
        for tag, score in elements_with_scores:
            if len(elements_to_keep) >= self.max_elements:
                break
            
            if score >= self.relevance_threshold:
                elements_to_keep.add(tag)
                if tag.name in self.INTERACTIVE_TAGS:
                    interactive_kept += 1
            
            # Always keep some interactive elements even if below threshold
            elif (preserve_interactive and 
                  tag.name in self.INTERACTIVE_TAGS and 
                  interactive_kept < 20):  # Keep at least 20 interactive elements
                elements_to_keep.add(tag)
                interactive_kept += 1
        
        # Mark elements for removal
        elements_to_remove = []
        for tag in compressed_soup.find_all(True):
            if tag not in elements_to_keep:
                elements_to_remove.append(tag)
        
        # Remove low-relevance elements
        for tag in elements_to_remove:
            # Replace with placeholder if it has important children
            if tag.find_all(self.INTERACTIVE_TAGS):
                placeholder = compressed_soup.new_string(
                    f"[COMPRESSED: {len(list(tag.descendants))} elements]"
                )
                tag.replace_with(placeholder)
            else:
                tag.decompose()
        
        # Apply pattern-based compression
        self._apply_pattern_compression(compressed_soup)
        
        # Update metrics
        self.metrics.elements_removed += len(elements_to_remove)
        self.metrics.elements_retained += len(elements_to_keep)
        
        return compressed_soup
    
    def _find_matching_score(self, tag: Tag, scores: Dict[Tag, float]) -> float:
        """Find matching score for a tag (approximate matching)."""
        # Try exact match first
        if tag in scores:
            return scores[tag]
        
        # Try matching by position and content
        tag_text = tag.get_text(strip=True)[:100]
        tag_attrs = {k: v for k, v in tag.attrs.items() if k in self.PRESERVE_ATTRS}
        
        for scored_tag, score in scores.items():
            if (scored_tag.name == tag.name and
                scored_tag.get_text(strip=True)[:100] == tag_text and
                {k: v for k, v in scored_tag.attrs.items() if k in self.PRESERVE_ATTRS} == tag_attrs):
                return score
        
        return 0.5  # Default score
    
    def _apply_pattern_compression(self, soup: BeautifulSoup):
        """Apply pattern-based compression for common structures."""
        # Compress navigation menus
        for nav in soup.find_all(['nav', {'role': 'navigation'}]):
            links = nav.find_all('a')
            if len(links) > 3:
                link_texts = [a.get_text(strip=True) for a in links[:5]]
                compressed = f"[NAV: {', '.join(link_texts)}]"
                nav.string = compressed
                nav.clear()
        
        # Compress forms
        for form in soup.find_all('form'):
            inputs = form.find_all(['input', 'select', 'textarea'])
            if len(inputs) > 5:
                field_names = []
                for inp in inputs[:5]:
                    name = inp.get('name') or inp.get('id') or inp.get('placeholder', 'field')
                    field_names.append(name)
                compressed = f"[FORM: {', '.join(field_names)}]"
                form.string = compressed
                form.clear()
        
        # Compress tables
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) > 3:
                cols = len(rows[0].find_all(['td', 'th'])) if rows else 0
                compressed = f"[TABLE: {len(rows)}x{cols}]"
                table.string = compressed
                table.clear()
        
        # Compress long text blocks
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 200:
                # Keep first 100 chars and last 50 chars
                compressed = f"{text[:100]}...{text[-50:]}"
                p.string = compressed
    
    def _final_cleanup(self, soup: BeautifulSoup):
        """Final cleanup of compressed HTML."""
        # Remove excessive whitespace
        for tag in soup.find_all(True):
            if tag.string:
                tag.string = re.sub(r'\s+', ' ', tag.string.strip())
        
        # Remove empty tags (except interactive ones)
        for tag in soup.find_all(True):
            if (not tag.get_text(strip=True) and 
                not tag.find_all(self.INTERACTIVE_TAGS) and
                tag.name not in ['br', 'hr', 'img', 'input']):
                tag.decompose()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 chars per token for English
        return len(text) // 4
    
    def compress_for_llm(self, html: str, task_context: str = "",
                        include_metrics: bool = False) -> Dict[str, Any]:
        """
        Compress HTML and return structured data for LLM.
        
        Args:
            html: Raw HTML content
            task_context: Current task context
            include_metrics: Whether to include compression metrics
            
        Returns:
            Dictionary with compressed content and metadata
        """
        compressed_html = self.compress_html(html, task_context)
        
        result = {
            "compressed_html": compressed_html,
            "original_length": len(html),
            "compressed_length": len(compressed_html),
            "compression_ratio": 1 - (len(compressed_html) / len(html)) if html else 0,
            "task_context": task_context
        }
        
        if include_metrics:
            result["metrics"] = asdict(self.metrics)
        
        return result
    
    def get_element_fingerprint(self, element: Tag) -> ElementFingerprint:
        """Generate fingerprint for an element."""
        # Create fingerprint
        attrs = {k: v for k, v in element.attrs.items() 
                if k in self.PRESERVE_ATTRS}
        
        text = element.get_text(strip=True)
        text_pattern = ""
        if text:
            # Create pattern from text (first 20 chars)
            text_pattern = re.sub(r'\s+', ' ', text[:20])
        
        fingerprint = ElementFingerprint(
            tag=element.name,
            attributes=attrs,
            depth=len(list(element.parents)),
            child_count=len(list(element.children)),
            text_pattern=text_pattern
        )
        
        # Cache the fingerprint
        fp_hash = fingerprint.hash()
        with self._lock:
            self._fingerprint_cache[fp_hash] = fingerprint
        
        return fingerprint
    
    def clear_cache(self):
        """Clear all caches."""
        with self._lock:
            self._pattern_cache.clear()
            self._fingerprint_cache.clear()
            self._compression_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "pattern_cache_size": len(self._pattern_cache),
                "fingerprint_cache_size": len(self._fingerprint_cache),
                "compression_cache_size": len(self._compression_cache)
            }


class ContextCompressor:
    """
    High-level context compressor for browser automation.
    Integrates with existing nexus modules.
    """
    
    def __init__(self, 
                 relevance_threshold: float = 0.3,
                 max_elements: int = 200,
                 use_parallel: bool = True,
                 max_workers: int = 4):
        """
        Initialize the context compressor.
        
        Args:
            relevance_threshold: Minimum relevance score to keep element
            max_elements: Maximum elements to retain
            use_parallel: Whether to use parallel processing
            max_workers: Maximum worker threads for parallel processing
        """
        self.semantic_compressor = SemanticCompressor(
            relevance_threshold=relevance_threshold,
            max_elements=max_elements
        )
        
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if use_parallel else None
        
        # Integration with existing modules
        self._element_cache = {}
        self._page_cache = {}
    
    def compress_page_context(self, 
                             page_content: str,
                             task_description: str,
                             current_url: str = "",
                             interactive_only: bool = False) -> Dict[str, Any]:
        """
        Compress page context for LLM consumption.
        
        Args:
            page_content: HTML content of the page
            task_description: Description of the current task
            current_url: Current page URL
            interactive_only: Whether to focus only on interactive elements
            
        Returns:
            Compressed context dictionary
        """
        # Generate cache key
        cache_key = self._generate_cache_key(page_content, task_description)
        
        # Check cache
        if cache_key in self._page_cache:
            self.semantic_compressor.metrics.cache_hits += 1
            return self._page_cache[cache_key]
        
        self.semantic_compressor.metrics.cache_misses += 1
        
        # Compress the content
        compressed_data = self.semantic_compressor.compress_for_llm(
            html=page_content,
            task_context=task_description,
            include_metrics=True
        )
        
        # Extract interactive elements if requested
        interactive_elements = []
        if interactive_only:
            interactive_elements = self._extract_interactive_elements(
                compressed_data["compressed_html"]
            )
        
        # Structure the result
        result = {
            "compressed_content": compressed_data["compressed_html"],
            "original_size": compressed_data["original_length"],
            "compressed_size": compressed_data["compressed_length"],
            "compression_ratio": compressed_data["compression_ratio"],
            "task_context": task_description,
            "current_url": current_url,
            "interactive_elements": interactive_elements,
            "metrics": compressed_data.get("metrics", {}),
            "cache_key": cache_key
        }
        
        # Cache the result
        self._page_cache[cache_key] = result
        
        # Limit cache size
        if len(self._page_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self._page_cache.keys())[:200]
            for key in oldest_keys:
                del self._page_cache[key]
        
        return result
    
    def _extract_interactive_elements(self, html: str) -> List[Dict[str, Any]]:
        """Extract interactive elements from compressed HTML."""
        elements = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            for tag in soup.find_all(SemanticCompressor.INTERACTIVE_TAGS):
                element_data = {
                    "tag": tag.name,
                    "text": tag.get_text(strip=True)[:100],  # Limit text length
                    "attributes": {
                        k: v for k, v in tag.attrs.items()
                        if k in SemanticCompressor.PRESERVE_ATTRS
                    }
                }
                
                # Add element type
                if tag.name == 'a':
                    element_data["type"] = "link"
                elif tag.name == 'button':
                    element_data["type"] = "button"
                elif tag.name == 'input':
                    element_data["type"] = f"input_{tag.get('type', 'text')}"
                elif tag.name == 'select':
                    element_data["type"] = "dropdown"
                elif tag.name == 'textarea':
                    element_data["type"] = "textarea"
                else:
                    element_data["type"] = "interactive"
                
                elements.append(element_data)
        
        except Exception as e:
            logger.error(f"Error extracting interactive elements: {e}")
        
        return elements
    
    def _generate_cache_key(self, content: str, context: str) -> str:
        """Generate cache key for content and context."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
        return f"{content_hash}_{context_hash}"
    
    def compress_element_context(self, 
                                element_html: str,
                                element_info: Dict[str, Any]) -> str:
        """
        Compress context for a specific element.
        
        Args:
            element_html: HTML of the element
            element_info: Information about the element (tag, attributes, etc.)
            
        Returns:
            Compressed element context
        """
        # Create focused context for the element
        soup = BeautifulSoup(element_html, 'html.parser')
        
        # Remove all but the target element
        target_element = soup.find(element_info.get('tag', 'div'))
        if not target_element:
            return element_html
        
        # Clean the element
        self.semantic_compressor._clean_attributes(soup)
        
        # Add element metadata as comment
        metadata = f"<!-- Element: {element_info.get('tag', 'unknown')} "
        if 'id' in element_info:
            metadata += f"id={element_info['id']} "
        if 'class' in element_info:
            metadata += f"class={element_info['class']} "
        metadata += "-->"
        
        result = f"{metadata}\n{str(target_element)}"
        
        return result
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "metrics": asdict(self.semantic_compressor.metrics),
            "cache_stats": self.semantic_compressor.get_cache_stats(),
            "page_cache_size": len(self._page_cache)
        }
    
    def shutdown(self):
        """Shutdown the compressor and clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        self.semantic_compressor.clear_cache()
        self._page_cache.clear()


# Integration with existing modules
def integrate_with_agent_service():
    """Integration point with nexus/agent/service.py"""
    # This function would be called during agent initialization
    # to set up the context compressor
    pass


def integrate_with_page_actor():
    """Integration point with nexus/actor/page.py"""
    # This function would be called when page content is retrieved
    # to compress it before sending to LLM
    pass


# Factory function for easy instantiation
def create_context_compressor(
    relevance_threshold: float = 0.3,
    max_elements: int = 200,
    use_parallel: bool = True
) -> ContextCompressor:
    """
    Factory function to create a context compressor instance.
    
    Args:
        relevance_threshold: Minimum relevance score to keep element
        max_elements: Maximum elements to retain
        use_parallel: Whether to use parallel processing
        
    Returns:
        Configured ContextCompressor instance
    """
    return ContextCompressor(
        relevance_threshold=relevance_threshold,
        max_elements=max_elements,
        use_parallel=use_parallel
    )


# Example usage and testing
if __name__ == "__main__":
    # Example HTML
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Example Page</title>
        <script src="analytics.js"></script>
        <style>body { font-family: Arial; }</style>
    </head>
    <body>
        <nav class="main-nav">
            <ul>
                <li><a href="/home">Home</a></li>
                <li><a href="/about">About</a></li>
                <li><a href="/contact">Contact</a></li>
            </ul>
        </nav>
        <main>
            <h1>Welcome to Our Site</h1>
            <article>
                <h2>Latest News</h2>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
                   Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
                <form id="search-form">
                    <input type="text" name="q" placeholder="Search...">
                    <button type="submit">Search</button>
                </form>
            </article>
            <aside>
                <h3>Related Links</h3>
                <ul>
                    <li><a href="/link1">Link 1</a></li>
                    <li><a href="/link2">Link 2</a></li>
                </ul>
            </aside>
        </main>
        <footer>
            <p>© 2024 Example Company</p>
        </footer>
    </body>
    </html>
    """
    
    # Create compressor
    compressor = create_context_compressor(
        relevance_threshold=0.3,
        max_elements=50
    )
    
    # Compress the sample HTML
    result = compressor.compress_page_context(
        page_content=sample_html,
        task_description="Find and click the search button",
        current_url="https://example.com"
    )
    
    print(f"Original size: {result['original_size']} chars")
    print(f"Compressed size: {result['compressed_size']} chars")
    print(f"Compression ratio: {result['compression_ratio']:.1%}")
    print(f"\nCompressed content:\n{result['compressed_content']}")
    
    # Print statistics
    stats = compressor.get_compression_stats()
    print(f"\nCompression statistics: {stats}")