"""
ML-based Element Identification & Self-Healing Selectors
Intelligent element discovery that adapts to UI changes using machine learning.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

import numpy as np
from playwright.async_api import Page, ElementHandle, Locator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nexus.actor.element import Element
from nexus.agent.views import ActionResult

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features extracted from elements."""
    DOM_ATTRIBUTES = "dom_attributes"
    VISUAL_FEATURES = "visual_features"
    TEXT_CONTENT = "text_content"
    STRUCTURAL_CONTEXT = "structural_context"
    INTERACTION_HISTORY = "interaction_history"


@dataclass
class ElementFeatures:
    """Comprehensive feature set for an element."""
    element_id: str
    dom_attributes: Dict[str, str] = field(default_factory=dict)
    visual_features: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    structural_context: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML processing."""
        vectors = []
        
        # DOM attributes (one-hot encoded categorical features)
        dom_vector = self._encode_dom_attributes()
        vectors.append(dom_vector)
        
        # Visual features (normalized numerical values)
        visual_vector = self._encode_visual_features()
        vectors.append(visual_vector)
        
        # Text content (TF-IDF vector)
        text_vector = self._encode_text_content()
        vectors.append(text_vector)
        
        # Structural context
        structural_vector = self._encode_structural_context()
        vectors.append(structural_vector)
        
        return np.concatenate(vectors)
    
    def _encode_dom_attributes(self) -> np.ndarray:
        """Encode DOM attributes as numerical features."""
        important_attrs = ['id', 'class', 'name', 'type', 'role', 'aria-label', 
                          'data-testid', 'placeholder', 'title', 'href']
        
        features = []
        for attr in important_attrs:
            value = self.dom_attributes.get(attr, "")
            # Binary feature: whether attribute exists and is non-empty
            features.append(1.0 if value else 0.0)
            
            # Hash-based feature for attribute value similarity
            if value:
                hash_val = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
                features.append(hash_val / (2**32))
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def _encode_visual_features(self) -> np.ndarray:
        """Encode visual features (position, size, colors, etc.)."""
        visual = self.visual_features
        features = [
            visual.get('x', 0) / 1920,  # Normalize by typical screen width
            visual.get('y', 0) / 1080,  # Normalize by typical screen height
            visual.get('width', 0) / 1920,
            visual.get('height', 0) / 1080,
            visual.get('opacity', 1.0),
            visual.get('z_index', 0) / 1000,
            visual.get('background_color_r', 0) / 255,
            visual.get('background_color_g', 0) / 255,
            visual.get('background_color_b', 0) / 255,
            visual.get('text_color_r', 0) / 255,
            visual.get('text_color_g', 0) / 255,
            visual.get('text_color_b', 0) / 255,
        ]
        return np.array(features)
    
    def _encode_text_content(self) -> np.ndarray:
        """Encode text content using simple character-level features."""
        if not self.text_content:
            return np.zeros(50)
        
        # Simple character frequency encoding
        char_freq = {}
        for char in self.text_content.lower():
            if char.isalnum():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Create fixed-size vector
        features = []
        for i in range(50):
            char = chr(ord('a') + i) if i < 26 else str(i - 26)
            features.append(char_freq.get(char, 0) / max(len(self.text_content), 1))
        
        return np.array(features)
    
    def _encode_structural_context(self) -> np.ndarray:
        """Encode structural context (parent, siblings, etc.)."""
        context = self.structural_context
        features = [
            context.get('depth', 0) / 20,  # Normalize depth
            context.get('child_count', 0) / 50,
            context.get('sibling_index', 0) / 20,
            context.get('has_form_parent', 0),
            context.get('has_table_parent', 0),
            context.get('has_list_parent', 0),
            context.get('is_clickable', 0),
            context.get('is_visible', 1),
            context.get('is_focusable', 0),
        ]
        return np.array(features)


@dataclass
class SelectorCandidate:
    """Candidate selector with confidence score."""
    selector: str
    selector_type: str  # 'css', 'xpath', 'text', 'aria'
    confidence: float
    element_handle: Optional[ElementHandle] = None
    features: Optional[ElementFeatures] = None
    fallback_level: int = 0


class MLIdentifier:
    """
    ML-based element identifier that creates robust selectors
    and provides self-healing capabilities when UI changes occur.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        similarity_threshold: float = 0.75,
        cache_size: int = 1000,
        enable_visual_similarity: bool = True
    ):
        self.model_path = model_path or Path.home() / ".nexus" / "ml_models"
        self.similarity_threshold = similarity_threshold
        self.cache_size = cache_size
        self.enable_visual_similarity = enable_visual_similarity
        
        # ML models
        self.classifier: Optional[RandomForestClassifier] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.feature_scaler = None
        
        # Element cache for quick lookups
        self.element_cache: Dict[str, ElementFeatures] = {}
        self.selector_cache: Dict[str, List[SelectorCandidate]] = {}
        
        # Training data
        self.training_data: List[Tuple[ElementFeatures, str]] = []  # (features, correct_selector)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load ML models."""
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing models
            classifier_path = self.model_path / "element_classifier.pkl"
            vectorizer_path = self.model_path / "text_vectorizer.pkl"
            
            if classifier_path.exists():
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info("Loaded existing element classifier")
            
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Loaded existing text vectorizer")
            
            # Initialize new models if not found
            if self.classifier is None:
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
        except Exception as e:
            logger.warning(f"Failed to initialize ML models: {e}")
            # Fallback to simple models
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=500)
    
    async def extract_element_features(
        self, 
        element: Union[ElementHandle, Element, Locator],
        page: Page
    ) -> ElementFeatures:
        """Extract comprehensive features from an element."""
        try:
            # Get element handle if needed
            if isinstance(element, Element):
                handle = element._element_handle
            elif isinstance(element, Locator):
                handle = await element.element_handle()
            else:
                handle = element
            
            if not handle:
                raise ValueError("Could not get element handle")
            
            # Extract DOM attributes
            dom_attrs = await self._extract_dom_attributes(handle)
            
            # Extract visual features
            visual_features = await self._extract_visual_features(handle, page)
            
            # Extract text content
            text_content = await self._extract_text_content(handle)
            
            # Extract structural context
            structural_context = await self._extract_structural_context(handle)
            
            # Generate unique ID
            element_id = self._generate_element_id(dom_attrs, visual_features, text_content)
            
            return ElementFeatures(
                element_id=element_id,
                dom_attributes=dom_attrs,
                visual_features=visual_features,
                text_content=text_content,
                structural_context=structural_context
            )
            
        except Exception as e:
            logger.error(f"Failed to extract element features: {e}")
            raise
    
    async def _extract_dom_attributes(self, handle: ElementHandle) -> Dict[str, str]:
        """Extract DOM attributes from element."""
        attributes = {}
        
        try:
            # Get all attributes via JavaScript
            attrs_js = """
            (element) => {
                const attrs = {};
                for (const attr of element.attributes) {
                    attrs[attr.name] = attr.value;
                }
                return attrs;
            }
            """
            attributes = await handle.evaluate(attrs_js)
            
            # Add computed properties
            tag_name = await handle.evaluate("el => el.tagName.toLowerCase()")
            attributes['tag'] = tag_name
            
            # Get ARIA attributes
            aria_attrs = await handle.evaluate("""
            (element) => {
                const aria = {};
                for (const attr of element.attributes) {
                    if (attr.name.startsWith('aria-')) {
                        aria[attr.name] = attr.value;
                    }
                }
                return aria;
            }
            """)
            attributes.update(aria_attrs)
            
        except Exception as e:
            logger.debug(f"Error extracting DOM attributes: {e}")
        
        return attributes
    
    async def _extract_visual_features(
        self, 
        handle: ElementHandle, 
        page: Page
    ) -> Dict[str, Any]:
        """Extract visual features from element."""
        features = {}
        
        try:
            # Get bounding box
            bbox = await handle.bounding_box()
            if bbox:
                features.update({
                    'x': bbox['x'],
                    'y': bbox['y'],
                    'width': bbox['width'],
                    'height': bbox['height']
                })
            
            # Get computed styles
            styles_js = """
            (element) => {
                const style = window.getComputedStyle(element);
                return {
                    opacity: parseFloat(style.opacity) || 1,
                    zIndex: parseInt(style.zIndex) || 0,
                    backgroundColor: style.backgroundColor,
                    color: style.color,
                    fontSize: style.fontSize,
                    fontFamily: style.fontFamily,
                    display: style.display,
                    visibility: style.visibility,
                    position: style.position
                };
            }
            """
            styles = await handle.evaluate(styles_js)
            
            # Parse colors
            bg_color = self._parse_color(styles.get('backgroundColor', ''))
            text_color = self._parse_color(styles.get('color', ''))
            
            features.update({
                'opacity': styles.get('opacity', 1),
                'z_index': styles.get('zIndex', 0),
                'background_color_r': bg_color[0],
                'background_color_g': bg_color[1],
                'background_color_b': bg_color[2],
                'text_color_r': text_color[0],
                'text_color_g': text_color[1],
                'text_color_b': text_color[2],
                'font_size': self._parse_font_size(styles.get('fontSize', '16px')),
                'display': styles.get('display', ''),
                'visibility': styles.get('visibility', 'visible'),
                'position': styles.get('position', '')
            })
            
            # Check if element is visible in viewport
            is_visible = await handle.is_visible()
            features['is_visible_in_viewport'] = 1 if is_visible else 0
            
        except Exception as e:
            logger.debug(f"Error extracting visual features: {e}")
        
        return features
    
    async def _extract_text_content(self, handle: ElementHandle) -> str:
        """Extract text content from element."""
        try:
            text = await handle.evaluate("""
            (element) => {
                // Get direct text content
                let text = element.innerText || element.textContent || '';
                
                // Clean up whitespace
                text = text.replace(/\\s+/g, ' ').trim();
                
                // Limit length for ML processing
                return text.substring(0, 500);
            }
            """)
            return text
        except Exception:
            return ""
    
    async def _extract_structural_context(self, handle: ElementHandle) -> Dict[str, Any]:
        """Extract structural context (parent, siblings, etc.)."""
        try:
            context = await handle.evaluate("""
            (element) => {
                const context = {};
                
                // Depth in DOM tree
                let depth = 0;
                let parent = element.parentElement;
                while (parent) {
                    depth++;
                    parent = parent.parentElement;
                }
                context.depth = depth;
                
                // Child count
                context.child_count = element.children.length;
                
                // Sibling index
                const siblings = Array.from(element.parentElement?.children || []);
                context.sibling_index = siblings.indexOf(element);
                
                // Parent context
                let current = element.parentElement;
                context.has_form_parent = false;
                context.has_table_parent = false;
                context.has_list_parent = false;
                
                while (current && current !== document.body) {
                    const tag = current.tagName.toLowerCase();
                    if (tag === 'form') context.has_form_parent = true;
                    if (tag === 'table') context.has_table_parent = true;
                    if (tag === 'ul' || tag === 'ol') context.has_list_parent = true;
                    current = current.parentElement;
                }
                
                // Interaction properties
                const style = window.getComputedStyle(element);
                context.is_clickable = (
                    element.tagName === 'BUTTON' ||
                    element.tagName === 'A' ||
                    element.tagName === 'INPUT' ||
                    style.cursor === 'pointer' ||
                    element.onclick !== null
                ) ? 1 : 0;
                
                context.is_focusable = element.tabIndex >= 0 ? 1 : 0;
                
                return context;
            }
            """)
            return context
        except Exception as e:
            logger.debug(f"Error extracting structural context: {e}")
            return {}
    
    def _generate_element_id(
        self, 
        dom_attrs: Dict[str, str], 
        visual_features: Dict[str, Any],
        text_content: str
    ) -> str:
        """Generate a unique ID for the element based on its features."""
        # Create a stable identifier
        id_parts = []
        
        # Add tag and important attributes
        id_parts.append(dom_attrs.get('tag', 'unknown'))
        
        if 'id' in dom_attrs and dom_attrs['id']:
            id_parts.append(f"id:{dom_attrs['id']}")
        
        if 'data-testid' in dom_attrs and dom_attrs['data-testid']:
            id_parts.append(f"testid:{dom_attrs['data-testid']}")
        
        # Add position-based component for uniqueness
        if 'x' in visual_features and 'y' in visual_features:
            pos_hash = hashlib.md5(
                f"{visual_features['x']:.0f}:{visual_features['y']:.0f}".encode()
            ).hexdigest()[:8]
            id_parts.append(f"pos:{pos_hash}")
        
        # Add text hash
        if text_content:
            text_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
            id_parts.append(f"text:{text_hash}")
        
        return "|".join(id_parts)
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse CSS color string to RGB tuple."""
        if not color_str or color_str == 'transparent':
            return (0, 0, 0)
        
        try:
            # Handle rgb/rgba
            if color_str.startswith('rgb'):
                parts = color_str.replace('rgba(', '').replace('rgb(', '').replace(')', '').split(',')
                r, g, b = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                return (r, g, b)
            
            # Handle hex colors
            elif color_str.startswith('#'):
                hex_str = color_str.lstrip('#')
                if len(hex_str) == 3:
                    hex_str = ''.join([c*2 for c in hex_str])
                r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                return (r, g, b)
            
        except Exception:
            pass
        
        return (0, 0, 0)
    
    def _parse_font_size(self, font_size_str: str) -> float:
        """Parse font size string to pixels."""
        try:
            if 'px' in font_size_str:
                return float(font_size_str.replace('px', ''))
            elif 'pt' in font_size_str:
                return float(font_size_str.replace('pt', '')) * 1.333  # Approx conversion
            elif 'em' in font_size_str:
                return float(font_size_str.replace('em', '')) * 16  # Assuming base 16px
            elif 'rem' in font_size_str:
                return float(font_size_str.replace('rem', '')) * 16
            else:
                return float(font_size_str) if font_size_str else 16.0
        except Exception:
            return 16.0
    
    async def generate_robust_selectors(
        self,
        element: Union[ElementHandle, Element, Locator],
        page: Page,
        max_selectors: int = 5
    ) -> List[SelectorCandidate]:
        """Generate multiple robust selectors for an element."""
        candidates = []
        
        try:
            # Get element handle
            if isinstance(element, Element):
                handle = element._element_handle
            elif isinstance(element, Locator):
                handle = await element.element_handle()
            else:
                handle = element
            
            if not handle:
                return candidates
            
            # Extract features for ML-based selector generation
            features = await self.extract_element_features(handle, page)
            
            # Generate CSS selectors
            css_selectors = await self._generate_css_selectors(handle)
            for selector in css_selectors[:3]:  # Top 3 CSS selectors
                candidates.append(SelectorCandidate(
                    selector=selector,
                    selector_type='css',
                    confidence=0.9,
                    features=features,
                    fallback_level=0
                ))
            
            # Generate XPath selectors
            xpath_selectors = await self._generate_xpath_selectors(handle)
            for selector in xpath_selectors[:2]:  # Top 2 XPath selectors
                candidates.append(SelectorCandidate(
                    selector=selector,
                    selector_type='xpath',
                    confidence=0.85,
                    features=features,
                    fallback_level=1
                ))
            
            # Generate text-based selectors if text exists
            text_content = await self._extract_text_content(handle)
            if text_content and len(text_content) > 2:
                text_selector = f'text="{text_content[:50]}"'
                candidates.append(SelectorCandidate(
                    selector=text_selector,
                    selector_type='text',
                    confidence=0.8,
                    features=features,
                    fallback_level=2
                ))
            
            # Generate ARIA-based selectors
            aria_selectors = await self._generate_aria_selectors(handle)
            for selector in aria_selectors[:2]:
                candidates.append(SelectorCandidate(
                    selector=selector,
                    selector_type='aria',
                    confidence=0.85,
                    features=features,
                    fallback_level=1
                ))
            
            # Sort by confidence and fallback level
            candidates.sort(key=lambda x: (x.fallback_level, -x.confidence))
            
            # Cache the selectors
            element_id = features.element_id
            self.selector_cache[element_id] = candidates[:max_selectors]
            
            # Cache features
            self.element_cache[element_id] = features
            
            # Trim cache if needed
            if len(self.element_cache) > self.cache_size:
                self._trim_cache()
            
            return candidates[:max_selectors]
            
        except Exception as e:
            logger.error(f"Failed to generate selectors: {e}")
            return []
    
    async def _generate_css_selectors(self, handle: ElementHandle) -> List[str]:
        """Generate CSS selectors for an element."""
        selectors = []
        
        try:
            css_js = """
            (element) => {
                const selectors = [];
                
                // ID selector (highest priority)
                if (element.id) {
                    selectors.push(`#${element.id}`);
                }
                
                // Data-testid selector
                if (element.dataset.testid) {
                    selectors.push(`[data-testid="${element.dataset.testid}"]`);
                }
                
                // Class-based selectors
                if (element.className && typeof element.className === 'string') {
                    const classes = element.className.split(' ').filter(c => c.trim());
                    if (classes.length > 0) {
                        // Single class
                        selectors.push(`.${classes[0]}`);
                        
                        // Multiple classes (if not too many)
                        if (classes.length <= 3) {
                            selectors.push('.' + classes.join('.'));
                        }
                    }
                }
                
                // Tag with attributes
                const tag = element.tagName.toLowerCase();
                const attrs = ['name', 'type', 'placeholder', 'title', 'role'];
                
                for (const attr of attrs) {
                    if (element[attr]) {
                        selectors.push(`${tag}[${attr}="${element[attr]}"]`);
                    }
                }
                
                // Position-based selector (nth-child)
                const parent = element.parentElement;
                if (parent) {
                    const siblings = Array.from(parent.children);
                    const index = siblings.indexOf(element) + 1;
                    selectors.push(`${tag}:nth-child(${index})`);
                }
                
                return selectors;
            }
            """
            
            selectors = await handle.evaluate(css_js)
            
        except Exception as e:
            logger.debug(f"Error generating CSS selectors: {e}")
        
        return selectors
    
    async def _generate_xpath_selectors(self, handle: ElementHandle) -> List[str]:
        """Generate XPath selectors for an element."""
        selectors = []
        
        try:
            xpath_js = """
            (element) => {
                const selectors = [];
                
                // Absolute XPath
                function getAbsoluteXPath(el) {
                    if (el.id) {
                        return `//*[@id="${el.id}"]`;
                    }
                    
                    if (el === document.body) {
                        return '/html/body';
                    }
                    
                    let ix = 0;
                    const siblings = el.parentNode ? el.parentNode.childNodes : [];
                    
                    for (let i = 0; i < siblings.length; i++) {
                        const sibling = siblings[i];
                        
                        if (sibling === el) {
                            return getAbsoluteXPath(el.parentNode) + '/' + 
                                   el.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        }
                        
                        if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {
                            ix++;
                        }
                    }
                }
                
                // Relative XPath with text
                function getTextXPath(el) {
                    const text = el.textContent?.trim();
                    if (text && text.length < 50) {
                        return `//${el.tagName.toLowerCase()}[contains(text(), "${text.substring(0, 30)}")]`;
                    }
                    return null;
                }
                
                // Add absolute XPath
                const absolute = getAbsoluteXPath(element);
                if (absolute) selectors.push(absolute);
                
                // Add text-based XPath
                const textXPath = getTextXPath(element);
                if (textXPath) selectors.push(textXPath);
                
                // Add attribute-based XPath
                const attrs = ['id', 'name', 'class', 'type', 'role'];
                for (const attr of attrs) {
                    if (element[attr]) {
                        selectors.push(`//${element.tagName.toLowerCase()}[@${attr}="${element[attr]}"]`);
                    }
                }
                
                return selectors;
            }
            """
            
            selectors = await handle.evaluate(xpath_js)
            
        except Exception as e:
            logger.debug(f"Error generating XPath selectors: {e}")
        
        return selectors
    
    async def _generate_aria_selectors(self, handle: ElementHandle) -> List[str]:
        """Generate ARIA-based selectors for accessibility."""
        selectors = []
        
        try:
            aria_js = """
            (element) => {
                const selectors = [];
                
                // ARIA label
                if (element.getAttribute('aria-label')) {
                    selectors.push(`[aria-label="${element.getAttribute('aria-label')}"]`);
                }
                
                // ARIA role
                if (element.getAttribute('role')) {
                    selectors.push(`[role="${element.getAttribute('role')}"]`);
                }
                
                // ARIA describedby
                if (element.getAttribute('aria-describedby')) {
                    selectors.push(`[aria-describedby="${element.getAttribute('aria-describedby')}"]`);
                }
                
                // ARIA labelledby
                if (element.getAttribute('aria-labelledby')) {
                    selectors.push(`[aria-labelledby="${element.getAttribute('aria-labelledby')}"]`);
                }
                
                // Button/input with accessible name
                const tag = element.tagName.toLowerCase();
                if (tag === 'button' || tag === 'input') {
                    const name = element.getAttribute('name') || 
                                element.getAttribute('aria-label') ||
                                element.textContent?.trim();
                    if (name) {
                        selectors.push(`${tag}[name="${name}"]`);
                    }
                }
                
                return selectors;
            }
            """
            
            selectors = await handle.evaluate(aria_js)
            
        except Exception as e:
            logger.debug(f"Error generating ARIA selectors: {e}")
        
        return selectors
    
    async def find_element_with_healing(
        self,
        page: Page,
        primary_selector: str,
        selector_type: str = 'css',
        context_element: Optional[ElementHandle] = None,
        timeout: int = 5000
    ) -> Optional[ElementHandle]:
        """
        Find an element using primary selector with automatic healing fallbacks.
        """
        start_time = time.time()
        
        # Try primary selector first
        try:
            if selector_type == 'css':
                element = await page.wait_for_selector(primary_selector, timeout=timeout)
            elif selector_type == 'xpath':
                element = await page.wait_for_selector(f'xpath={primary_selector}', timeout=timeout)
            elif selector_type == 'text':
                element = await page.get_by_text(primary_selector.replace('text=', '').strip('"')).first
                element = await element.element_handle()
            elif selector_type == 'aria':
                element = await page.wait_for_selector(primary_selector, timeout=timeout)
            else:
                element = await page.wait_for_selector(primary_selector, timeout=timeout)
            
            if element:
                logger.info(f"Found element with primary selector: {primary_selector}")
                return element
                
        except Exception as e:
            logger.debug(f"Primary selector failed: {primary_selector} - {e}")
        
        # If primary selector fails, try healing strategies
        elapsed = time.time() - start_time
        remaining_timeout = max(1000, timeout - int(elapsed * 1000))
        
        return await self._heal_element_search(
            page, 
            primary_selector, 
            selector_type, 
            context_element, 
            remaining_timeout
        )
    
    async def _heal_element_search(
        self,
        page: Page,
        original_selector: str,
        selector_type: str,
        context_element: Optional[ElementHandle],
        timeout: int
    ) -> Optional[ElementHandle]:
        """
        Attempt to heal broken selector using ML-based strategies.
        """
        logger.info(f"Attempting to heal selector: {original_selector}")
        
        # Strategy 1: Try similar selectors from cache
        healed_element = await self._try_cached_similar_selectors(
            page, original_selector, selector_type, timeout
        )
        if healed_element:
            return healed_element
        
        # Strategy 2: Visual similarity search
        if self.enable_visual_similarity:
            healed_element = await self._find_by_visual_similarity(
                page, original_selector, context_element, timeout
            )
            if healed_element:
                return healed_element
        
        # Strategy 3: Text content matching
        healed_element = await self._find_by_text_similarity(
            page, original_selector, timeout
        )
        if healed_element:
            return healed_element
        
        # Strategy 4: Structural context matching
        healed_element = await self._find_by_structural_context(
            page, original_selector, context_element, timeout
        )
        if healed_element:
            return healed_element
        
        # Strategy 5: ML-based prediction
        if self.classifier and len(self.training_data) > 10:
            healed_element = await self._find_by_ml_prediction(
                page, original_selector, timeout
            )
            if healed_element:
                return healed_element
        
        logger.warning(f"All healing strategies failed for selector: {original_selector}")
        return None
    
    async def _try_cached_similar_selectors(
        self,
        page: Page,
        original_selector: str,
        selector_type: str,
        timeout: int
    ) -> Optional[ElementHandle]:
        """Try similar selectors from the cache."""
        # Look for cached selectors with similar patterns
        for element_id, candidates in self.selector_cache.items():
            for candidate in candidates:
                if candidate.selector_type == selector_type:
                    # Check if selector is similar
                    similarity = self._selector_similarity(original_selector, candidate.selector)
                    if similarity > 0.7:
                        try:
                            if selector_type == 'css':
                                element = await page.wait_for_selector(candidate.selector, timeout=1000)
                            elif selector_type == 'xpath':
                                element = await page.wait_for_selector(f'xpath={candidate.selector}', timeout=1000)
                            else:
                                continue
                            
                            if element:
                                logger.info(f"Found element using cached similar selector: {candidate.selector}")
                                return element
                        except Exception:
                            continue
        
        return None
    
    def _selector_similarity(self, selector1: str, selector2: str) -> float:
        """Calculate similarity between two selectors."""
        # Simple similarity based on common parts
        parts1 = set(selector1.replace('.', ' ').replace('#', ' ').replace('[', ' ').replace(']', ' ').split())
        parts2 = set(selector2.replace('.', ' ').replace('#', ' ').replace('[', ' ').replace(']', ' ').split())
        
        if not parts1 or not parts2:
            return 0.0
        
        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _find_by_visual_similarity(
        self,
        page: Page,
        original_selector: str,
        context_element: Optional[ElementHandle],
        timeout: int
    ) -> Optional[ElementHandle]:
        """Find element by visual similarity to cached elements."""
        try:
            # Get all visible elements
            elements_js = """
            () => {
                const elements = Array.from(document.querySelectorAll('*'));
                return elements
                    .filter(el => {
                        const style = window.getComputedStyle(el);
                        return style.display !== 'none' && 
                               style.visibility !== 'hidden' &&
                               el.offsetWidth > 0 &&
                               el.offsetHeight > 0;
                    })
                    .slice(0, 100); // Limit for performance
            }
            """
            
            elements = await page.evaluate(elements_js)
            
            # For each element, check if it matches any cached element visually
            for element_info in elements:
                try:
                    # Get element handle
                    element = await page.query_selector(element_info['selector'])
                    if not element:
                        continue
                    
                    # Extract features
                    features = await self.extract_element_features(element, page)
                    
                    # Compare with cached features
                    for cached_id, cached_features in self.element_cache.items():
                        similarity = self._feature_similarity(features, cached_features)
                        
                        if similarity > self.similarity_threshold:
                            logger.info(f"Found element by visual similarity: {similarity:.2f}")
                            return element
                            
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Visual similarity search failed: {e}")
        
        return None
    
    def _feature_similarity(self, features1: ElementFeatures, features2: ElementFeatures) -> float:
        """Calculate similarity between two element feature sets."""
        try:
            # Convert to vectors
            vec1 = features1.to_vector()
            vec2 = features2.to_vector()
            
            # Calculate cosine similarity
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            
            # Apply weights to different feature types
            dom_weight = 0.3
            visual_weight = 0.4
            text_weight = 0.2
            structural_weight = 0.1
            
            # Calculate component-wise similarities
            dom_sim = self._dom_similarity(features1.dom_attributes, features2.dom_attributes)
            visual_sim = self._visual_similarity(features1.visual_features, features2.visual_features)
            text_sim = self._text_similarity(features1.text_content, features2.text_content)
            structural_sim = self._structural_similarity(
                features1.structural_context, 
                features2.structural_context
            )
            
            weighted_similarity = (
                dom_weight * dom_sim +
                visual_weight * visual_sim +
                text_weight * text_sim +
                structural_weight * structural_sim
            )
            
            return weighted_similarity
            
        except Exception as e:
            logger.debug(f"Feature similarity calculation failed: {e}")
            return 0.0
    
    def _dom_similarity(self, attrs1: Dict[str, str], attrs2: Dict[str, str]) -> float:
        """Calculate DOM attribute similarity."""
        important_attrs = ['tag', 'id', 'class', 'name', 'type', 'role']
        
        matches = 0
        total = 0
        
        for attr in important_attrs:
            if attr in attrs1 or attr in attrs2:
                total += 1
                if attrs1.get(attr) == attrs2.get(attr):
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def _visual_similarity(self, visual1: Dict[str, Any], visual2: Dict[str, Any]) -> float:
        """Calculate visual feature similarity."""
        # Position similarity (weighted heavily)
        pos_sim = 0.0
        if 'x' in visual1 and 'x' in visual2 and 'y' in visual1 and 'y' in visual2:
            dx = abs(visual1['x'] - visual2['x']) / 1920
            dy = abs(visual1['y'] - visual2['y']) / 1080
            pos_sim = 1.0 - min(1.0, (dx + dy) / 2)
        
        # Size similarity
        size_sim = 0.0
        if 'width' in visual1 and 'width' in visual2:
            dw = abs(visual1['width'] - visual2['width']) / 1920
            dh = abs(visual1['height'] - visual2['height']) / 1080
            size_sim = 1.0 - min(1.0, (dw + dh) / 2)
        
        # Color similarity
        color_sim = 0.0
        if all(k in visual1 for k in ['background_color_r', 'background_color_g', 'background_color_b']) and \
           all(k in visual2 for k in ['background_color_r', 'background_color_g', 'background_color_b']):
            dr = abs(visual1['background_color_r'] - visual2['background_color_r']) / 255
            dg = abs(visual1['background_color_g'] - visual2['background_color_g']) / 255
            db = abs(visual1['background_color_b'] - visual2['background_color_b']) / 255
            color_sim = 1.0 - (dr + dg + db) / 3
        
        return (pos_sim * 0.4 + size_sim * 0.3 + color_sim * 0.3)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text content similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _structural_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate structural context similarity."""
        matches = 0
        total = 0
        
        for key in ['depth', 'has_form_parent', 'has_table_parent', 'has_list_parent']:
            if key in context1 or key in context2:
                total += 1
                if context1.get(key) == context2.get(key):
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    async def _find_by_text_similarity(
        self,
        page: Page,
        original_selector: str,
        timeout: int
    ) -> Optional[ElementHandle]:
        """Find element by text content similarity."""
        try:
            # Extract text from original selector if it's text-based
            text_content = ""
            if original_selector.startswith('text='):
                text_content = original_selector.replace('text=', '').strip('"\'')
            elif '"' in original_selector or "'" in original_selector:
                # Try to extract quoted text
                import re
                match = re.search(r'["\']([^"\']+)["\']', original_selector)
                if match:
                    text_content = match.group(1)
            
            if not text_content:
                return None
            
            # Find elements with similar text
            text_js = f"""
            () => {{
                const elements = Array.from(document.querySelectorAll('*'));
                const targetText = "{text_content}";
                
                return elements
                    .filter(el => {{
                        const text = el.textContent?.trim().toLowerCase() || '';
                        return text.includes(targetText.toLowerCase()) ||
                               targetText.toLowerCase().includes(text);
                    }})
                    .map(el => {{
                        const rect = el.getBoundingClientRect();
                        return {{
                            selector: el.tagName.toLowerCase() + 
                                     (el.id ? '#' + el.id : '') +
                                     (el.className ? '.' + el.className.split(' ')[0] : ''),
                            text: el.textContent?.trim().substring(0, 100),
                            visible: rect.width > 0 && rect.height > 0
                        }};
                    }})
                    .filter(info => info.visible)
                    .slice(0, 5);
            }}
            """
            
            candidates = await page.evaluate(text_js)
            
            for candidate in candidates:
                try:
                    element = await page.wait_for_selector(candidate['selector'], timeout=1000)
                    if element:
                        logger.info(f"Found element by text similarity: {candidate['text'][:50]}...")
                        return element
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Text similarity search failed: {e}")
        
        return None
    
    async def _find_by_structural_context(
        self,
        page: Page,
        original_selector: str,
        context_element: Optional[ElementHandle],
        timeout: int
    ) -> Optional[ElementHandle]:
        """Find element by structural context matching."""
        try:
            # If we have context element, look for similar elements nearby
            if context_element:
                context_js = """
                (contextElement) => {
                    const contextTag = contextElement.tagName.toLowerCase();
                    const contextParent = contextElement.parentElement;
                    
                    if (!contextParent) return [];
                    
                    // Find similar elements in the same parent
                    const siblings = Array.from(contextParent.children);
                    const similar = siblings.filter(el => {
                        return el.tagName.toLowerCase() === contextTag;
                    });
                    
                    return similar.map(el => {
                        const rect = el.getBoundingClientRect();
                        return {
                            selector: el.tagName.toLowerCase() + 
                                     (el.id ? '#' + el.id : '') +
                                     (el.className ? '.' + el.className.split(' ')[0] : ''),
                            visible: rect.width > 0 && rect.height > 0
                        };
                    }).filter(info => info.visible);
                }
                """
                
                candidates = await context_element.evaluate(context_js)
                
                for candidate in candidates:
                    try:
                        element = await page.wait_for_selector(candidate['selector'], timeout=1000)
                        if element:
                            logger.info(f"Found element by structural context")
                            return element
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Structural context search failed: {e}")
        
        return None
    
    async def _find_by_ml_prediction(
        self,
        page: Page,
        original_selector: str,
        timeout: int
    ) -> Optional[ElementHandle]:
        """Find element using ML model prediction."""
        try:
            # This would require a trained model to predict element location
            # For now, return None - this would be implemented with actual ML model
            pass
        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
        
        return None
    
    def record_successful_interaction(
        self,
        element: ElementFeatures,
        selector: str,
        action: str,
        success: bool = True
    ):
        """Record successful interaction for training."""
        interaction = {
            'element_id': element.element_id,
            'selector': selector,
            'action': action,
            'success': success,
            'timestamp': time.time(),
            'features': element.to_vector().tolist()
        }
        
        element.interaction_history.append(interaction)
        
        # Add to training data
        if success:
            self.training_data.append((element, selector))
            
            # Keep training data size manageable
            if len(self.training_data) > 10000:
                self.training_data = self.training_data[-5000:]
    
    async def train_model(self, force: bool = False):
        """Train the ML model on collected data."""
        if len(self.training_data) < 100 and not force:
            logger.info(f"Not enough training data: {len(self.training_data)} samples")
            return
        
        try:
            logger.info(f"Training ML model with {len(self.training_data)} samples")
            
            # Prepare training data
            X = []
            y = []
            
            for features, selector in self.training_data:
                X.append(features.to_vector())
                y.append(selector)
            
            X = np.array(X)
            
            # Train classifier (simplified - in reality would need proper selector encoding)
            # For now, we'll just fit the vectorizer on text content
            text_contents = [f.text_content for f, _ in self.training_data if f.text_content]
            if text_contents:
                self.vectorizer.fit(text_contents)
            
            # Save models
            self._save_models()
            
            logger.info("ML model training completed")
            
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
    
    def _save_models(self):
        """Save trained models to disk."""
        try:
            if self.classifier:
                with open(self.model_path / "element_classifier.pkl", 'wb') as f:
                    pickle.dump(self.classifier, f)
            
            if self.vectorizer:
                with open(self.model_path / "text_vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                    
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _trim_cache(self):
        """Trim cache to stay within size limits."""
        if len(self.element_cache) > self.cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.element_cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            keep_count = int(self.cache_size * 0.8)
            self.element_cache = dict(sorted_items[-keep_count:])
            
            # Also trim selector cache
            element_ids = set(self.element_cache.keys())
            self.selector_cache = {
                k: v for k, v in self.selector_cache.items()
                if k in element_ids
            }
    
    async def get_element_stability_score(
        self,
        element: Union[ElementHandle, Element, Locator],
        page: Page
    ) -> float:
        """
        Calculate stability score for an element (0-1, higher is more stable).
        Stable elements are less likely to change between UI updates.
        """
        try:
            features = await self.extract_element_features(element, page)
            
            stability_score = 0.0
            
            # ID presence increases stability
            if features.dom_attributes.get('id'):
                stability_score += 0.3
            
            # data-testid is very stable
            if features.dom_attributes.get('data-testid'):
                stability_score += 0.4
            
            # ARIA attributes are moderately stable
            aria_attrs = [k for k in features.dom_attributes.keys() if k.startswith('aria-')]
            if aria_attrs:
                stability_score += 0.2
            
            # Text content provides some stability
            if features.text_content and len(features.text_content) > 3:
                stability_score += 0.1
            
            # Structural context (form elements, etc.) are somewhat stable
            if features.structural_context.get('has_form_parent'):
                stability_score += 0.1
            
            # Cap at 1.0
            return min(1.0, stability_score)
            
        except Exception:
            return 0.5  # Default moderate stability


# Integration with existing Element class
def enhance_element_with_ml(element_class):
    """Enhance existing Element class with ML capabilities."""
    
    original_init = element_class.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.ml_identifier = None
        self.element_features = None
        self.selector_candidates = []
    
    element_class.__init__ = new_init
    
    async def generate_robust_selectors(self, page: Page, ml_identifier: MLIdentifier = None):
        """Generate robust selectors for this element."""
        if ml_identifier is None:
            ml_identifier = MLIdentifier()
        
        self.ml_identifier = ml_identifier
        self.selector_candidates = await ml_identifier.generate_robust_selectors(
            self._element_handle, page
        )
        
        return self.selector_candidates
    
    async def find_with_healing(self, page: Page, timeout: int = 5000):
        """Find this element with healing capabilities."""
        if not self.selector_candidates:
            await self.generate_robust_selectors(page)
        
        if not self.selector_candidates:
            raise ValueError("No selector candidates available")
        
        # Try each candidate in order
        for candidate in self.selector_candidates:
            try:
                element = await self.ml_identifier.find_element_with_healing(
                    page,
                    candidate.selector,
                    candidate.selector_type,
                    timeout=timeout
                )
                
                if element:
                    # Update element handle
                    self._element_handle = element
                    return self
                    
            except Exception as e:
                logger.debug(f"Selector candidate failed: {candidate.selector} - {e}")
                continue
        
        raise Exception("All selector candidates failed")
    
    element_class.generate_robust_selectors = generate_robust_selectors
    element_class.find_with_healing = find_with_healing
    
    return element_class


# Apply enhancement to Element class
try:
    from nexus.actor.element import Element
    Element = enhance_element_with_ml(Element)
except ImportError:
    logger.warning("Could not enhance Element class - module not found")


# Export public API
__all__ = [
    'MLIdentifier',
    'ElementFeatures',
    'SelectorCandidate',
    'FeatureType',
    'enhance_element_with_ml'
]