"""
Visual Matcher - Intelligent Element Discovery & Healing
Self-healing selectors that adapt to UI changes using ML-based element identification.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image
import io
import base64

# ML imports - using lightweight models for browser environment
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from playwright.async_api import Page, ElementHandle, Locator

from nexus.actor.element import Element
from nexus.actor.utils import retry


logger = logging.getLogger(__name__)


class MatchingStrategy(Enum):
    """Element matching strategies in order of preference"""
    PRIMARY_SELECTOR = "primary_selector"
    VISUAL_SIMILARITY = "visual_similarity"
    DOM_STRUCTURE = "dom_structure"
    TEXT_CONTENT = "text_content"
    POSITIONAL = "positional"
    FALLBACK_CHAIN = "fallback_chain"


@dataclass
class ElementSignature:
    """Comprehensive element signature for robust identification"""
    element_id: str
    primary_selector: str
    fallback_selectors: List[str] = field(default_factory=list)
    visual_features: Optional[np.ndarray] = None
    dom_features: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    bounding_box: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 1.0
    last_seen: float = field(default_factory=time.time)
    match_count: int = 0
    failure_count: int = 0


@dataclass 
class HealingResult:
    """Result of element healing attempt"""
    success: bool
    element: Optional[Element] = None
    strategy_used: MatchingStrategy = MatchingStrategy.PRIMARY_SELECTOR
    confidence: float = 0.0
    alternative_selectors: List[str] = field(default_factory=list)
    healing_time_ms: float = 0.0


class VisualFeatureExtractor:
    """Lightweight visual feature extractor for element matching"""
    
    def __init__(self, model_type: str = "mobilenet_v2"):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize lightweight CNN model for feature extraction"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Visual matching will use basic features.")
            return
        
        try:
            if self.model_type == "mobilenet_v2":
                # Use MobileNetV2 - lightweight and fast
                self.model = models.mobilenet_v2(pretrained=True)
                # Remove classification head
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                self.model.eval()
                
                # Standard ImageNet transforms
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
            else:
                logger.warning(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize visual model: {e}")
            self.model = None
    
    async def extract_features(self, element: Element) -> Optional[np.ndarray]:
        """Extract visual features from element screenshot"""
        if not self.model or not TORCH_AVAILABLE:
            return None
        
        try:
            # Take screenshot of element
            screenshot = await element.screenshot()
            if not screenshot:
                return None
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot))
            
            # Apply transforms
            tensor = self.transform(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor)
                features = features.squeeze().numpy()
            
            # Normalize features
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    @staticmethod
    def cosine_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


class DOMFeatureExtractor:
    """Extract features from DOM structure for element matching"""
    
    @staticmethod
    async def extract_dom_features(element: Element) -> Dict[str, Any]:
        """Extract comprehensive DOM features"""
        features = {}
        
        try:
            # Basic element properties
            features['tag_name'] = await element.get_tag_name()
            features['id'] = await element.get_attribute('id') or ''
            features['class_names'] = await element.get_attribute('class') or ''
            features['name'] = await element.get_attribute('name') or ''
            features['type'] = await element.get_attribute('type') or ''
            features['role'] = await element.get_attribute('role') or ''
            features['aria_label'] = await element.get_attribute('aria-label') or ''
            features['data_testid'] = await element.get_attribute('data-testid') or ''
            
            # Position and size
            box = await element.bounding_box()
            if box:
                features['bounding_box'] = box
                features['position_hash'] = hashlib.md5(
                    json.dumps(box, sort_keys=True).encode()
                ).hexdigest()[:8]
            
            # Text content
            features['text_content'] = await element.text_content() or ''
            features['inner_text'] = await element.inner_text() or ''
            
            # Parent context
            parent = await element.evaluate('el => el.parentElement ? el.parentElement.tagName : null')
            features['parent_tag'] = parent or ''
            
            # Sibling count
            sibling_count = await element.evaluate('''el => {
                if (!el.parentElement) return 0;
                return el.parentElement.children.length;
            }''')
            features['sibling_count'] = sibling_count or 0
            
            # Index among siblings
            sibling_index = await element.evaluate('''el => {
                if (!el.parentElement) return 0;
                return Array.from(el.parentElement.children).indexOf(el);
            }''')
            features['sibling_index'] = sibling_index or 0
            
        except Exception as e:
            logger.error(f"DOM feature extraction failed: {e}")
        
        return features
    
    @staticmethod
    def calculate_dom_similarity(features1: Dict, features2: Dict) -> float:
        """Calculate similarity between DOM feature sets"""
        if not features1 or not features2:
            return 0.0
        
        score = 0.0
        weights = {
            'tag_name': 0.2,
            'id': 0.15,
            'class_names': 0.1,
            'text_content': 0.15,
            'position_hash': 0.2,
            'parent_tag': 0.1,
            'sibling_index': 0.1
        }
        
        for key, weight in weights.items():
            if key in features1 and key in features2:
                if key == 'class_names':
                    # Calculate Jaccard similarity for class names
                    classes1 = set(features1[key].split())
                    classes2 = set(features2[key].split())
                    if classes1 or classes2:
                        intersection = len(classes1 & classes2)
                        union = len(classes1 | classes2)
                        score += weight * (intersection / union if union > 0 else 0)
                elif key == 'text_content':
                    # Simple text similarity
                    text1 = features1[key].strip().lower()
                    text2 = features2[key].strip().lower()
                    if text1 and text2:
                        # Use sequence matcher for better text comparison
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, text1, text2).ratio()
                        score += weight * similarity
                else:
                    # Exact match for other features
                    if features1[key] == features2[key]:
                        score += weight
        
        return score


class SelectorGenerator:
    """Generate robust selectors for elements"""
    
    @staticmethod
    async def generate_selectors(element: Element) -> List[str]:
        """Generate multiple selector strategies for an element"""
        selectors = []
        
        try:
            # 1. ID-based selector (most stable)
            element_id = await element.get_attribute('id')
            if element_id:
                selectors.append(f'#{element_id}')
            
            # 2. Data attribute selectors (common in modern apps)
            for attr in ['data-testid', 'data-test', 'data-cy', 'data-qa']:
                value = await element.get_attribute(attr)
                if value:
                    selectors.append(f'[{attr}="{value}"]')
            
            # 3. ARIA attributes
            aria_label = await element.get_attribute('aria-label')
            if aria_label:
                selectors.append(f'[aria-label="{aria_label}"]')
            
            role = await element.get_attribute('role')
            if role:
                selectors.append(f'[role="{role}"]')
            
            # 4. Class-based selectors
            class_name = await element.get_attribute('class')
            if class_name:
                classes = class_name.split()
                if classes:
                    # Use most specific class combination
                    selectors.append('.' + '.'.join(classes[:3]))
            
            # 5. Text-based selectors
            text = await element.text_content()
            if text and len(text) < 50:  # Avoid long text selectors
                selectors.append(f'text="{text.strip()}"')
            
            # 6. XPath as fallback
            xpath = await element.evaluate('''el => {
                function getXPath(el) {
                    if (el.id !== '') return '//*[@id="' + el.id + '"]';
                    if (el === document.body) return '/html/body';
                    
                    let ix = 0;
                    let siblings = el.parentNode.childNodes;
                    for (let i = 0; i < siblings.length; i++) {
                        let sibling = siblings[i];
                        if (sibling === el) {
                            return getXPath(el.parentNode) + '/' + el.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        }
                        if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {
                            ix++;
                        }
                    }
                }
                return getXPath(el);
            }''')
            if xpath:
                selectors.append(xpath)
            
        except Exception as e:
            logger.error(f"Selector generation failed: {e}")
        
        return selectors


class VisualMatcher:
    """
    Intelligent element discovery and healing system.
    Uses ML-based element identification to create self-healing selectors.
    """
    
    def __init__(self, page: Page, cache_dir: Optional[str] = None):
        self.page = page
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.nexus' / 'visual_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractors
        self.visual_extractor = VisualFeatureExtractor()
        self.dom_extractor = DOMFeatureExtractor()
        self.selector_generator = SelectorGenerator()
        
        # Element signatures cache
        self.signatures: Dict[str, ElementSignature] = {}
        self._load_cache()
        
        # Configuration
        self.visual_threshold = 0.75
        self.dom_threshold = 0.6
        self.max_candidates = 10
        self.healing_enabled = True
        
    def _load_cache(self):
        """Load element signatures from cache"""
        cache_file = self.cache_dir / 'element_signatures.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for elem_id, sig_data in data.items():
                        # Convert visual features back to numpy array
                        if 'visual_features' in sig_data and sig_data['visual_features']:
                            sig_data['visual_features'] = np.array(sig_data['visual_features'])
                        self.signatures[elem_id] = ElementSignature(**sig_data)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save element signatures to cache"""
        cache_file = self.cache_dir / 'element_signatures.json'
        try:
            data = {}
            for elem_id, signature in self.signatures.items():
                sig_dict = signature.__dict__.copy()
                # Convert numpy array to list for JSON serialization
                if sig_dict['visual_features'] is not None:
                    sig_dict['visual_features'] = sig_dict['visual_features'].tolist()
                data[elem_id] = sig_dict
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    async def find_element_with_healing(
        self,
        selector: str,
        element_id: Optional[str] = None,
        timeout: int = 5000
    ) -> HealingResult:
        """
        Find element with self-healing capabilities.
        
        Args:
            selector: Primary CSS/XPath selector
            element_id: Optional element ID for caching
            timeout: Maximum time to wait for element
            
        Returns:
            HealingResult with element and healing metadata
        """
        start_time = time.time()
        
        # Try primary selector first
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                elem = Element(element, self.page)
                
                # Update signature if we have element_id
                if element_id:
                    await self._update_signature(element_id, elem, selector)
                
                return HealingResult(
                    success=True,
                    element=elem,
                    strategy_used=MatchingStrategy.PRIMARY_SELECTOR,
                    confidence=1.0,
                    healing_time_ms=(time.time() - start_time) * 1000
                )
        except Exception:
            logger.debug(f"Primary selector failed: {selector}")
        
        # Primary selector failed - attempt healing
        if not self.healing_enabled:
            return HealingResult(
                success=False,
                strategy_used=MatchingStrategy.PRIMARY_SELECTOR,
                healing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Try healing strategies
        healing_result = await self._heal_element(selector, element_id, timeout)
        healing_result.healing_time_ms = (time.time() - start_time) * 1000
        
        return healing_result
    
    async def _heal_element(
        self,
        original_selector: str,
        element_id: Optional[str],
        timeout: int
    ) -> HealingResult:
        """Attempt to heal broken selector using multiple strategies"""
        
        strategies = [
            (MatchingStrategy.VISUAL_SIMILARITY, self._try_visual_matching),
            (MatchingStrategy.DOM_STRUCTURE, self._try_dom_matching),
            (MatchingStrategy.TEXT_CONTENT, self._try_text_matching),
            (MatchingStrategy.POSITIONAL, self._try_positional_matching),
            (MatchingStrategy.FALLBACK_CHAIN, self._try_fallback_selectors)
        ]
        
        best_result = None
        best_confidence = 0.0
        
        for strategy, strategy_func in strategies:
            try:
                result = await strategy_func(original_selector, element_id, timeout)
                if result.success and result.confidence > best_confidence:
                    best_result = result
                    best_confidence = result.confidence
                    
                    # If we have high confidence, use this result
                    if best_confidence > 0.9:
                        break
                        
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {e}")
                continue
        
        if best_result:
            # Generate alternative selectors for future use
            if best_result.element:
                alt_selectors = await self.selector_generator.generate_selectors(best_result.element)
                best_result.alternative_selectors = alt_selectors
                
                # Update signature with new selectors
                if element_id:
                    await self._update_signature(
                        element_id,
                        best_result.element,
                        best_result.alternative_selectors[0] if alt_selectors else original_selector
                    )
            
            return best_result
        
        return HealingResult(
            success=False,
            strategy_used=MatchingStrategy.FALLBACK_CHAIN,
            confidence=0.0
        )
    
    async def _try_visual_matching(
        self,
        original_selector: str,
        element_id: Optional[str],
        timeout: int
    ) -> HealingResult:
        """Try to find element using visual similarity"""
        
        if not element_id or element_id not in self.signatures:
            return HealingResult(success=False, confidence=0.0)
        
        signature = self.signatures[element_id]
        if signature.visual_features is None:
            return HealingResult(success=False, confidence=0.0)
        
        # Get all visible elements of same type
        tag_name = signature.dom_features.get('tag_name', 'div')
        candidates = await self.page.query_selector_all(tag_name)
        
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidates[:self.max_candidates]:
            try:
                elem = Element(candidate, self.page)
                
                # Check if element is visible
                is_visible = await elem.is_visible()
                if not is_visible:
                    continue
                
                # Extract visual features
                features = await self.visual_extractor.extract_features(elem)
                if features is None:
                    continue
                
                # Calculate similarity
                similarity = VisualFeatureExtractor.cosine_similarity(
                    signature.visual_features,
                    features
                )
                
                if similarity > best_similarity and similarity > self.visual_threshold:
                    best_similarity = similarity
                    best_match = elem
                    
            except Exception:
                continue
        
        if best_match:
            return HealingResult(
                success=True,
                element=best_match,
                strategy_used=MatchingStrategy.VISUAL_SIMILARITY,
                confidence=best_similarity
            )
        
        return HealingResult(success=False, confidence=0.0)
    
    async def _try_dom_matching(
        self,
        original_selector: str,
        element_id: Optional[str],
        timeout: int
    ) -> HealingResult:
        """Try to find element using DOM structure similarity"""
        
        if not element_id or element_id not in self.signatures:
            return HealingResult(success=False, confidence=0.0)
        
        signature = self.signatures[element_id]
        
        # Get all elements
        candidates = await self.page.query_selector_all('*')
        
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidates[:self.max_candidates * 2]:  # Check more candidates for DOM
            try:
                elem = Element(candidate, self.page)
                
                # Extract DOM features
                features = await self.dom_extractor.extract_dom_features(elem)
                
                # Calculate similarity
                similarity = DOMFeatureExtractor.calculate_dom_similarity(
                    signature.dom_features,
                    features
                )
                
                if similarity > best_similarity and similarity > self.dom_threshold:
                    best_similarity = similarity
                    best_match = elem
                    
            except Exception:
                continue
        
        if best_match:
            return HealingResult(
                success=True,
                element=best_match,
                strategy_used=MatchingStrategy.DOM_STRUCTURE,
                confidence=best_similarity
            )
        
        return HealingResult(success=False, confidence=0.0)
    
    async def _try_text_matching(
        self,
        original_selector: str,
        element_id: Optional[str],
        timeout: int
    ) -> HealingResult:
        """Try to find element by text content"""
        
        if not element_id or element_id not in self.signatures:
            return HealingResult(success=False, confidence=0.0)
        
        signature = self.signatures[element_id]
        text_content = signature.text_content.strip()
        
        if not text_content or len(text_content) < 2:
            return HealingResult(success=False, confidence=0.0)
        
        try:
            # Use text selector
            text_selector = f'text="{text_content}"'
            element = await self.page.wait_for_selector(text_selector, timeout=timeout // 2)
            
            if element:
                elem = Element(element, self.page)
                return HealingResult(
                    success=True,
                    element=elem,
                    strategy_used=MatchingStrategy.TEXT_CONTENT,
                    confidence=0.8
                )
        except Exception:
            pass
        
        return HealingResult(success=False, confidence=0.0)
    
    async def _try_positional_matching(
        self,
        original_selector: str,
        element_id: Optional[str],
        timeout: int
    ) -> HealingResult:
        """Try to find element by position relative to stable elements"""
        
        if not element_id or element_id not in self.signatures:
            return HealingResult(success=False, confidence=0.0)
        
        signature = self.signatures[element_id]
        target_box = signature.bounding_box
        
        if not target_box:
            return HealingResult(success=False, confidence=0.0)
        
        # Find elements at similar position
        try:
            # Use JavaScript to find elements at specific coordinates
            elements_at_pos = await self.page.evaluate('''(x, y) => {
                const elements = document.elementsFromPoint(x, y);
                return elements.map(el => ({
                    tag: el.tagName,
                    id: el.id,
                    classes: el.className
                }));
            }''', target_box['x'] + target_box['width'] / 2, 
                 target_box['y'] + target_box['height'] / 2)
            
            if elements_at_pos:
                # Try to find matching element
                for elem_info in elements_at_pos[:3]:
                    selector_parts = []
                    if elem_info.get('id'):
                        selector_parts.append(f'#{elem_info["id"]}')
                    if elem_info.get('classes'):
                        classes = elem_info['classes'].split()[:2]
                        selector_parts.append('.' + '.'.join(classes))
                    
                    if selector_parts:
                        selector = ''.join(selector_parts)
                        try:
                            element = await self.page.wait_for_selector(selector, timeout=1000)
                            if element:
                                elem = Element(element, self.page)
                                return HealingResult(
                                    success=True,
                                    element=elem,
                                    strategy_used=MatchingStrategy.POSITIONAL,
                                    confidence=0.7
                                )
                        except Exception:
                            continue
                            
        except Exception as e:
            logger.debug(f"Positional matching failed: {e}")
        
        return HealingResult(success=False, confidence=0.0)
    
    async def _try_fallback_selectors(
        self,
        original_selector: str,
        element_id: Optional[str],
        timeout: int
    ) -> HealingResult:
        """Try fallback selectors from signature"""
        
        if not element_id or element_id not in self.signatures:
            return HealingResult(success=False, confidence=0.0)
        
        signature = self.signatures[element_id]
        
        # Try each fallback selector
        for fallback_selector in signature.fallback_selectors:
            if fallback_selector == original_selector:
                continue
                
            try:
                element = await self.page.wait_for_selector(fallback_selector, timeout=1000)
                if element:
                    elem = Element(element, self.page)
                    return HealingResult(
                        success=True,
                        element=elem,
                        strategy_used=MatchingStrategy.FALLBACK_CHAIN,
                        confidence=0.9
                    )
            except Exception:
                continue
        
        return HealingResult(success=False, confidence=0.0)
    
    async def _update_signature(
        self,
        element_id: str,
        element: Element,
        selector: str
    ):
        """Update or create element signature"""
        
        # Extract features
        visual_features = await self.visual_extractor.extract_features(element)
        dom_features = await self.dom_extractor.extract_dom_features(element)
        text_content = await element.text_content() or ""
        bounding_box = await element.bounding_box() or {}
        
        # Generate fallback selectors
        fallback_selectors = await self.selector_generator.generate_selectors(element)
        
        # Create or update signature
        if element_id in self.signatures:
            signature = self.signatures[element_id]
            signature.primary_selector = selector
            signature.fallback_selectors = fallback_selectors
            signature.visual_features = visual_features
            signature.dom_features = dom_features
            signature.text_content = text_content
            signature.bounding_box = bounding_box
            signature.last_seen = time.time()
            signature.match_count += 1
            
            # Update confidence based on match history
            total_attempts = signature.match_count + signature.failure_count
            if total_attempts > 0:
                signature.confidence_score = signature.match_count / total_attempts
        else:
            self.signatures[element_id] = ElementSignature(
                element_id=element_id,
                primary_selector=selector,
                fallback_selectors=fallback_selectors,
                visual_features=visual_features,
                dom_features=dom_features,
                text_content=text_content,
                bounding_box=bounding_box,
                confidence_score=1.0,
                match_count=1,
                failure_count=0
            )
        
        # Save to cache
        self._save_cache()
    
    async def register_element(
        self,
        element: Element,
        element_id: Optional[str] = None
    ) -> str:
        """
        Register an element for tracking and healing.
        
        Args:
            element: Element to register
            element_id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            Element ID for future reference
        """
        if not element_id:
            # Generate unique ID based on element properties
            props = await self.dom_extractor.extract_dom_features(element)
            element_id = hashlib.md5(
                json.dumps(props, sort_keys=True).encode()
            ).hexdigest()[:12]
        
        # Generate primary selector
        selectors = await self.selector_generator.generate_selectors(element)
        primary_selector = selectors[0] if selectors else ""
        
        # Create signature
        await self._update_signature(element_id, element, primary_selector)
        
        return element_id
    
    async def heal_selector(
        self,
        selector: str,
        element_id: Optional[str] = None,
        timeout: int = 5000
    ) -> Tuple[Optional[Element], List[str]]:
        """
        Convenience method to heal a broken selector.
        
        Returns:
            Tuple of (found_element, alternative_selectors)
        """
        result = await self.find_element_with_healing(selector, element_id, timeout)
        
        if result.success:
            return result.element, result.alternative_selectors
        else:
            return None, []
    
    def get_signature(self, element_id: str) -> Optional[ElementSignature]:
        """Get element signature by ID"""
        return self.signatures.get(element_id)
    
    def list_tracked_elements(self) -> List[Dict[str, Any]]:
        """List all tracked elements with their metadata"""
        return [
            {
                'id': sig.element_id,
                'selector': sig.primary_selector,
                'confidence': sig.confidence_score,
                'last_seen': sig.last_seen,
                'match_count': sig.match_count,
                'text_preview': sig.text_content[:50] if sig.text_content else ''
            }
            for sig in self.signatures.values()
        ]
    
    async def cleanup_old_signatures(self, max_age_days: int = 30):
        """Remove old signatures that haven't been seen recently"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        to_remove = []
        for elem_id, signature in self.signatures.items():
            if current_time - signature.last_seen > max_age_seconds:
                to_remove.append(elem_id)
        
        for elem_id in to_remove:
            del self.signatures[elem_id]
        
        if to_remove:
            self._save_cache()
            logger.info(f"Cleaned up {len(to_remove)} old signatures")


# Integration with existing Element class
def patch_element_class():
    """Add healing capabilities to existing Element class"""
    from nexus.actor.element import Element as OriginalElement
    
    class HealingElement(OriginalElement):
        """Element with self-healing capabilities"""
        
        def __init__(self, handle, page, visual_matcher=None):
            super().__init__(handle, page)
            self.visual_matcher = visual_matcher
            self.element_id = None
        
        async def click_with_healing(self, selector: str, element_id: Optional[str] = None):
            """Click element with healing if selector breaks"""
            if self.visual_matcher:
                result = await self.visual_matcher.find_element_with_healing(
                    selector, element_id
                )
                if result.success and result.element:
                    await result.element.click()
                    return True
            return False
        
        async def register_for_healing(self, element_id: Optional[str] = None) -> str:
            """Register element for self-healing"""
            if self.visual_matcher:
                self.element_id = await self.visual_matcher.register_element(
                    self, element_id
                )
                return self.element_id
            return ""
    
    # Monkey patch the original Element class
    OriginalElement.__bases__ = (HealingElement,)


# Auto-patch when module is imported
patch_element_class()


# Example usage in existing code
async def example_usage():
    """Example of how to use VisualMatcher with existing nexus code"""
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Initialize visual matcher
        matcher = VisualMatcher(page)
        
        # Navigate to page
        await page.goto('https://example.com')
        
        # Find element with healing
        result = await matcher.find_element_with_healing(
            selector='#submit-button',
            element_id='submit_btn',
            timeout=5000
        )
        
        if result.success:
            print(f"Found element using {result.strategy_used}")
            print(f"Confidence: {result.confidence}")
            await result.element.click()
            
            # Get alternative selectors for future use
            print(f"Alternative selectors: {result.alternative_selectors}")
        else:
            print("Element not found even with healing")
        
        await browser.close()


if __name__ == "__main__":
    asyncio.run(example_usage())