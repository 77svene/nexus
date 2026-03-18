"""
nexus/element/visual_recognition.py

Adaptive Element Location Engine - Multi-strategy adaptive element location system
combining CSS/XPath, visual recognition, accessibility tree analysis, and LLM-guided
exploration with automatic fallback chains and self-healing selectors.
"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

import numpy as np
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

if TYPE_CHECKING:
    from nexus.actor.page import Page
    from nexus.agent.service import Agent

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of element location strategies."""
    CSS = "css"
    XPATH = "xpath"
    VISUAL = "visual"
    ACCESSIBILITY = "accessibility"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass
class ElementSignature:
    """Signature representing an element's identifying characteristics."""
    css_selector: Optional[str] = None
    xpath: Optional[str] = None
    visual_hash: Optional[str] = None
    accessibility_id: Optional[str] = None
    aria_label: Optional[str] = None
    text_content: Optional[str] = None
    element_type: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0
    strategy_used: StrategyType = StrategyType.CSS
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "css_selector": self.css_selector,
            "xpath": self.xpath,
            "visual_hash": self.visual_hash,
            "accessibility_id": self.accessibility_id,
            "aria_label": self.aria_label,
            "text_content": self.text_content,
            "element_type": self.element_type,
            "attributes": self.attributes,
            "bounding_box": self.bounding_box,
            "confidence": self.confidence,
            "strategy_used": self.strategy_used.value,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElementSignature":
        """Create from dictionary."""
        data["strategy_used"] = StrategyType(data["strategy_used"])
        return cls(**data)


@dataclass
class LocationResult:
    """Result of element location attempt."""
    element: Optional[WebElement]
    signature: ElementSignature
    strategy_used: StrategyType
    confidence: float
    attempts: int = 1
    fallback_used: bool = False
    time_taken: float = 0.0
    error: Optional[str] = None


class ElementLocatorStrategy(ABC):
    """Abstract base class for element location strategies."""
    
    def __init__(self, driver: WebDriver, timeout: float = 10.0):
        self.driver = driver
        self.timeout = timeout
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def locate(self, 
                    description: str, 
                    context: Optional[Dict[str, Any]] = None) -> LocationResult:
        """Locate element using this strategy."""
        pass
    
    @abstractmethod
    def calculate_confidence(self, element: WebElement, description: str) -> float:
        """Calculate confidence score for found element."""
        pass
    
    def get_element_signature(self, element: WebElement) -> ElementSignature:
        """Extract signature from located element."""
        try:
            # Try to generate CSS selector
            css_selector = self._generate_css_selector(element)
            
            # Try to generate XPath
            xpath = self._generate_xpath(element)
            
            # Get accessibility information
            aria_label = element.get_attribute("aria-label")
            accessibility_id = element.get_attribute("data-accessibility-id")
            
            # Get text content
            text_content = element.text
            
            # Get element type
            tag_name = element.tag_name
            element_type = element.get_attribute("type") or tag_name
            
            # Get bounding box
            location = element.location
            size = element.size
            bounding_box = (
                location["x"],
                location["y"],
                location["x"] + size["width"],
                location["y"] + size["height"]
            ) if location and size else None
            
            # Get important attributes
            important_attrs = ["id", "name", "class", "role", "data-testid", "placeholder"]
            attributes = {}
            for attr in important_attrs:
                value = element.get_attribute(attr)
                if value:
                    attributes[attr] = value
            
            return ElementSignature(
                css_selector=css_selector,
                xpath=xpath,
                aria_label=aria_label,
                accessibility_id=accessibility_id,
                text_content=text_content,
                element_type=element_type,
                attributes=attributes,
                bounding_box=bounding_box,
                strategy_used=StrategyType.CSS,
            )
        except Exception as e:
            logger.warning(f"Failed to extract element signature: {e}")
            return ElementSignature()
    
    def _generate_css_selector(self, element: WebElement) -> str:
        """Generate CSS selector for element."""
        try:
            # Try ID first
            element_id = element.get_attribute("id")
            if element_id:
                return f"#{element_id}"
            
            # Try data-testid
            test_id = element.get_attribute("data-testid")
            if test_id:
                return f"[data-testid='{test_id}']"
            
            # Try name
            name = element.get_attribute("name")
            if name:
                tag = element.tag_name
                return f"{tag}[name='{name}']"
            
            # Build path-based selector
            return self._build_css_path(element)
        except:
            return ""
    
    def _build_css_path(self, element: WebElement) -> str:
        """Build CSS path for element."""
        path = []
        current = element
        
        while current:
            tag = current.tag_name.lower()
            
            # Check for unique identifiers
            element_id = current.get_attribute("id")
            if element_id:
                path.insert(0, f"#{element_id}")
                break
            
            # Check siblings
            parent = current.find_element(By.XPATH, "..") if current.tag_name.lower() != "html" else None
            if parent:
                siblings = parent.find_elements(By.XPATH, f"./{tag}")
                if len(siblings) > 1:
                    index = siblings.index(current) + 1
                    path.insert(0, f"{tag}:nth-of-type({index})")
                else:
                    path.insert(0, tag)
            else:
                path.insert(0, tag)
            
            try:
                current = parent if parent and parent.tag_name.lower() != "html" else None
            except:
                break
        
        return " > ".join(path)
    
    def _generate_xpath(self, element: WebElement) -> str:
        """Generate XPath for element."""
        try:
            components = []
            current = element
            
            while current and current.tag_name.lower() != "html":
                tag = current.tag_name.lower()
                
                # Check for attributes that make unique
                element_id = current.get_attribute("id")
                if element_id:
                    components.insert(0, f"//*[@id='{element_id}']")
                    break
                
                # Check siblings
                parent = current.find_element(By.XPATH, "..")
                siblings = parent.find_elements(By.XPATH, f"./{tag}")
                
                if len(siblings) > 1:
                    index = siblings.index(current) + 1
                    components.insert(0, f"{tag}[{index}]")
                else:
                    components.insert(0, tag)
                
                current = parent
            
            return "//" + "/".join(components) if components else ""
        except:
            return ""


class CSSLocatorStrategy(ElementLocatorStrategy):
    """CSS selector-based location strategy."""
    
    async def locate(self, 
                    description: str, 
                    context: Optional[Dict[str, Any]] = None) -> LocationResult:
        start_time = time.time()
        
        try:
            # Extract potential CSS selectors from description
            selectors = self._extract_selectors_from_description(description)
            
            for selector in selectors:
                try:
                    wait = WebDriverWait(self.driver, self.timeout)
                    element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    
                    if element and element.is_displayed():
                        confidence = self.calculate_confidence(element, description)
                        signature = self.get_element_signature(element)
                        signature.css_selector = selector
                        signature.confidence = confidence
                        signature.strategy_used = StrategyType.CSS
                        
                        return LocationResult(
                            element=element,
                            signature=signature,
                            strategy_used=StrategyType.CSS,
                            confidence=confidence,
                            time_taken=time.time() - start_time,
                        )
                except:
                    continue
            
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.CSS,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error="No matching CSS selector found",
            )
            
        except Exception as e:
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.CSS,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e),
            )
    
    def _extract_selectors_from_description(self, description: str) -> List[str]:
        """Extract potential CSS selectors from natural language description."""
        selectors = []
        desc_lower = description.lower()
        
        # Common patterns
        patterns = {
            r"button|btn": ["button", "[role='button']", "input[type='button']", "input[type='submit']"],
            r"input|text field|textbox": ["input[type='text']", "input:not([type='button'])", "textarea"],
            r"link|anchor": ["a", "[role='link']"],
            r"checkbox": ["input[type='checkbox']", "[role='checkbox']"],
            r"radio": ["input[type='radio']", "[role='radio']"],
            r"select|dropdown": ["select", "[role='listbox']"],
            r"image|img": ["img", "[role='img']"],
            r"heading|h[1-6]": ["h1", "h2", "h3", "h4", "h5", "h6", "[role='heading']"],
        }
        
        for pattern, css_patterns in patterns.items():
            if re.search(pattern, desc_lower):
                selectors.extend(css_patterns)
        
        # Try to extract IDs, classes, names from description
        id_match = re.search(r"#([\w-]+)", description)
        if id_match:
            selectors.append(f"#{id_match.group(1)}")
        
        class_match = re.search(r"\.([\w-]+)", description)
        if class_match:
            selectors.append(f".{class_match.group(1)}")
        
        name_match = re.search(r"name=['\"]([^'\"]+)['\"]", description)
        if name_match:
            selectors.append(f"[name='{name_match.group(1)}']")
        
        # Add generic fallbacks
        selectors.extend([
            "[data-testid]",
            "[aria-label]",
            "[placeholder]",
            "[title]",
        ])
        
        return list(set(selectors))  # Remove duplicates
    
    def calculate_confidence(self, element: WebElement, description: str) -> float:
        """Calculate confidence based on attribute matches."""
        confidence = 0.5  # Base confidence
        
        desc_lower = description.lower()
        
        # Check text content
        if element.text and element.text.lower() in desc_lower:
            confidence += 0.3
        
        # Check aria-label
        aria_label = element.get_attribute("aria-label")
        if aria_label and aria_label.lower() in desc_lower:
            confidence += 0.2
        
        # Check placeholder
        placeholder = element.get_attribute("placeholder")
        if placeholder and placeholder.lower() in desc_lower:
            confidence += 0.1
        
        # Check title
        title = element.get_attribute("title")
        if title and title.lower() in desc_lower:
            confidence += 0.1
        
        return min(confidence, 1.0)


class XPathLocatorStrategy(ElementLocatorStrategy):
    """XPath-based location strategy."""
    
    async def locate(self, 
                    description: str, 
                    context: Optional[Dict[str, Any]] = None) -> LocationResult:
        start_time = time.time()
        
        try:
            # Generate XPath expressions from description
            xpaths = self._generate_xpaths_from_description(description)
            
            for xpath in xpaths:
                try:
                    wait = WebDriverWait(self.driver, self.timeout)
                    element = wait.until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    
                    if element and element.is_displayed():
                        confidence = self.calculate_confidence(element, description)
                        signature = self.get_element_signature(element)
                        signature.xpath = xpath
                        signature.confidence = confidence
                        signature.strategy_used = StrategyType.XPATH
                        
                        return LocationResult(
                            element=element,
                            signature=signature,
                            strategy_used=StrategyType.XPATH,
                            confidence=confidence,
                            time_taken=time.time() - start_time,
                        )
                except:
                    continue
            
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.XPATH,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error="No matching XPath found",
            )
            
        except Exception as e:
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.XPATH,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e),
            )
    
    def _generate_xpaths_from_description(self, description: str) -> List[str]:
        """Generate XPath expressions from natural language description."""
        xpaths = []
        desc_lower = description.lower()
        
        # Text-based XPaths
        if any(word in desc_lower for word in ["button", "click", "submit"]):
            xpaths.extend([
                "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
                "//input[@type='button' or @type='submit'][contains(translate(@value, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
                "//*[@role='button'][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
            ])
        
        if any(word in desc_lower for word in ["input", "field", "text", "enter"]):
            xpaths.extend([
                "//input[not(@type='button' or @type='submit' or @type='hidden')]",
                "//textarea",
                "//input[contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
            ])
        
        if any(word in desc_lower for word in ["link", "href", "navigate"]):
            xpaths.extend([
                "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
                "//*[@role='link'][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
            ])
        
        # Generic XPaths
        xpaths.extend([
            "//*[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
            "//*[contains(translate(@title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
            "//*[contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
            "//*[contains(translate(@data-testid, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{}')]".format(desc_lower),
        ])
        
        return xpaths
    
    def calculate_confidence(self, element: WebElement, description: str) -> float:
        """Calculate confidence based on text and attribute matching."""
        confidence = 0.5
        desc_lower = description.lower()
        
        # Check direct text match
        element_text = element.text.lower()
        if element_text and element_text in desc_lower:
            confidence += 0.3
        
        # Check attribute matches
        for attr in ["aria-label", "title", "placeholder", "value"]:
            attr_value = element.get_attribute(attr)
            if attr_value and attr_value.lower() in desc_lower:
                confidence += 0.1
                break
        
        return min(confidence, 1.0)


class VisualLocatorStrategy(ElementLocatorStrategy):
    """Visual recognition-based element location strategy."""
    
    def __init__(self, driver: WebDriver, timeout: float = 10.0):
        super().__init__(driver, timeout)
        self.visual_cache: Dict[str, Any] = {}
    
    async def locate(self, 
                    description: str, 
                    context: Optional[Dict[str, Any]] = None) -> LocationResult:
        start_time = time.time()
        
        try:
            # Take screenshot for visual analysis
            screenshot = self.driver.get_screenshot_as_png()
            
            # Analyze page visually
            visual_elements = await self._analyze_page_visually(screenshot, description)
            
            if visual_elements:
                # Find best matching element
                best_match = self._find_best_visual_match(visual_elements, description)
                
                if best_match:
                    # Try to find the actual WebElement
                    element = await self._locate_element_from_visual(best_match)
                    
                    if element:
                        confidence = self.calculate_confidence(element, description)
                        signature = self.get_element_signature(element)
                        signature.visual_hash = best_match.get("visual_hash")
                        signature.bounding_box = best_match.get("bounding_box")
                        signature.confidence = confidence
                        signature.strategy_used = StrategyType.VISUAL
                        
                        return LocationResult(
                            element=element,
                            signature=signature,
                            strategy_used=StrategyType.VISUAL,
                            confidence=confidence,
                            time_taken=time.time() - start_time,
                        )
            
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.VISUAL,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error="No visual match found",
            )
            
        except Exception as e:
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.VISUAL,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e),
            )
    
    async def _analyze_page_visually(self, 
                                   screenshot: bytes, 
                                   description: str) -> List[Dict[str, Any]]:
        """Analyze page screenshot to identify interactive elements."""
        # This would integrate with computer vision models
        # For now, return mock data - in production, this would use
        # object detection models like YOLO or specialized UI detection models
        
        elements = []
        
        # Mock implementation - in reality, this would use ML models
        try:
            # Convert to PIL Image for processing
            from io import BytesIO
            image = Image.open(BytesIO(screenshot))
            
            # Simple heuristic: look for rectangular regions that might be buttons/inputs
            # This is a placeholder - real implementation would use trained models
            width, height = image.size
            
            # Mock detection of common UI elements
            mock_elements = [
                {
                    "type": "button",
                    "bounding_box": (width * 0.1, height * 0.1, width * 0.3, height * 0.15),
                    "visual_hash": "button_hash_1",
                    "confidence": 0.8,
                },
                {
                    "type": "input",
                    "bounding_box": (width * 0.1, height * 0.2, width * 0.9, height * 0.25),
                    "visual_hash": "input_hash_1",
                    "confidence": 0.7,
                },
            ]
            
            elements.extend(mock_elements)
            
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
        
        return elements
    
    def _find_best_visual_match(self, 
                               elements: List[Dict[str, Any]], 
                               description: str) -> Optional[Dict[str, Any]]:
        """Find the best visual match for the description."""
        if not elements:
            return None
        
        desc_lower = description.lower()
        
        # Score each element based on description
        scored_elements = []
        for element in elements:
            score = 0.0
            
            # Type matching
            element_type = element.get("type", "").lower()
            if element_type in desc_lower:
                score += 0.5
            
            # Position heuristics (buttons often at top, inputs in middle)
            bbox = element.get("bounding_box", (0, 0, 0, 0))
            y_center = (bbox[1] + bbox[3]) / 2
            
            if "submit" in desc_lower or "button" in desc_lower:
                # Buttons often at bottom
                if y_center > 0.7:  # Assuming normalized coordinates
                    score += 0.3
            
            if "search" in desc_lower or "input" in desc_lower:
                # Inputs often at top
                if y_center < 0.3:
                    score += 0.3
            
            # Confidence from visual detection
            score += element.get("confidence", 0.0) * 0.2
            
            scored_elements.append((score, element))
        
        # Return highest scoring element
        if scored_elements:
            scored_elements.sort(key=lambda x: x[0], reverse=True)
            return scored_elements[0][1]
        
        return None
    
    async def _locate_element_from_visual(self, 
                                        visual_element: Dict[str, Any]) -> Optional[WebElement]:
        """Try to locate actual WebElement from visual detection."""
        try:
            # Try to find element at the detected coordinates
            bbox = visual_element.get("bounding_box")
            if not bbox:
                return None
            
            # Calculate center point
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            
            # Use JavaScript to find element at coordinates
            script = """
            var element = document.elementFromPoint(arguments[0], arguments[1]);
            return element;
            """
            
            element = self.driver.execute_script(script, x_center, y_center)
            return element
            
        except Exception as e:
            logger.warning(f"Failed to locate element from visual: {e}")
            return None
    
    def calculate_confidence(self, element: WebElement, description: str) -> float:
        """Calculate confidence based on visual match quality."""
        # Visual strategy gets base confidence from detection model
        # Additional confidence from element properties
        confidence = 0.6  # Base visual confidence
        
        # Check if element is visible and interactable
        if element.is_displayed() and element.is_enabled():
            confidence += 0.2
        
        # Check size (too small might be icon, too large might be container)
        size = element.size
        if size:
            area = size["width"] * size["height"]
            if 1000 < area < 100000:  # Reasonable interactive element size
                confidence += 0.1
        
        return min(confidence, 1.0)


class AccessibilityLocatorStrategy(ElementLocatorStrategy):
    """Accessibility tree-based element location strategy."""
    
    async def locate(self, 
                    description: str, 
                    context: Optional[Dict[str, Any]] = None) -> LocationResult:
        start_time = time.time()
        
        try:
            # Get accessibility tree
            accessibility_tree = await self._get_accessibility_tree()
            
            # Find elements matching description
            matching_elements = self._find_in_accessibility_tree(
                accessibility_tree, description
            )
            
            if matching_elements:
                # Try to locate the first matching element
                for element_info in matching_elements:
                    element = await self._locate_element_from_accessibility(element_info)
                    
                    if element:
                        confidence = self.calculate_confidence(element, description)
                        signature = self.get_element_signature(element)
                        signature.accessibility_id = element_info.get("id")
                        signature.aria_label = element_info.get("name")
                        signature.element_type = element_info.get("role")
                        signature.confidence = confidence
                        signature.strategy_used = StrategyType.ACCESSIBILITY
                        
                        return LocationResult(
                            element=element,
                            signature=signature,
                            strategy_used=StrategyType.ACCESSIBILITY,
                            confidence=confidence,
                            time_taken=time.time() - start_time,
                        )
            
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.ACCESSIBILITY,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error="No matching element in accessibility tree",
            )
            
        except Exception as e:
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.ACCESSIBILITY,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e),
            )
    
    async def _get_accessibility_tree(self) -> Dict[str, Any]:
        """Get the page's accessibility tree."""
        try:
            # Use Chrome DevTools Protocol to get accessibility tree
            # This requires CDP access - simplified implementation
            script = """
            var tree = {children: []};
            
            function getAccessibilityInfo(element, depth) {
                if (depth > 5) return null;  // Limit depth
                
                var info = {
                    role: element.getAttribute('role') || element.tagName.toLowerCase(),
                    name: element.getAttribute('aria-label') || 
                          element.getAttribute('title') || 
                          element.innerText || '',
                    id: element.id,
                    tagName: element.tagName,
                    className: element.className,
                    children: []
                };
                
                // Get computed role
                try {
                    var computedRole = window.getComputedStyle(element).getPropertyValue('role');
                    if (computedRole) info.computedRole = computedRole;
                } catch(e) {}
                
                // Recursively process children
                for (var child of element.children) {
                    var childInfo = getAccessibilityInfo(child, depth + 1);
                    if (childInfo) info.children.push(childInfo);
                }
                
                return info;
            }
            
            tree = getAccessibilityInfo(document.body, 0);
            return tree;
            """
            
            tree = self.driver.execute_script(script)
            return tree or {"children": []}
            
        except Exception as e:
            logger.warning(f"Failed to get accessibility tree: {e}")
            return {"children": []}
    
    def _find_in_accessibility_tree(self, 
                                   tree: Dict[str, Any], 
                                   description: str) -> List[Dict[str, Any]]:
        """Find elements in accessibility tree matching description."""
        matches = []
        desc_lower = description.lower()
        
        def search_tree(node, depth=0):
            if depth > 10:  # Prevent infinite recursion
                return
            
            # Check if this node matches
            node_text = (node.get("name", "") + " " + 
                        node.get("role", "") + " " + 
                        node.get("id", "")).lower()
            
            if any(word in node_text for word in desc_lower.split()):
                matches.append(node)
            
            # Search children
            for child in node.get("children", []):
                search_tree(child, depth + 1)
        
        search_tree(tree)
        return matches
    
    async def _locate_element_from_accessibility(self, 
                                                element_info: Dict[str, Any]) -> Optional[WebElement]:
        """Locate WebElement from accessibility tree information."""
        try:
            # Try multiple strategies to find the element
            
            # 1. Try by ID
            element_id = element_info.get("id")
            if element_id:
                try:
                    element = self.driver.find_element(By.ID, element_id)
                    if element:
                        return element
                except:
                    pass
            
            # 2. Try by aria-label
            aria_label = element_info.get("name")
            if aria_label:
                try:
                    element = self.driver.find_element(
                        By.XPATH, 
                        f"//*[@aria-label='{aria_label}']"
                    )
                    if element:
                        return element
                except:
                    pass
            
            # 3. Try by role and text
            role = element_info.get("role")
            text = element_info.get("name", "")
            if role and text:
                try:
                    # Build XPath based on role and text
                    if role == "button":
                        xpath = f"//button[contains(text(), '{text}')]"
                    elif role == "link":
                        xpath = f"//a[contains(text(), '{text}')]"
                    elif role == "textbox":
                        xpath = f"//input[contains(@placeholder, '{text}')]"
                    else:
                        xpath = f"//*[@role='{role}' and contains(text(), '{text}')]"
                    
                    element = self.driver.find_element(By.XPATH, xpath)
                    if element:
                        return element
                except:
                    pass
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to locate element from accessibility: {e}")
            return None
    
    def calculate_confidence(self, element: WebElement, description: str) -> float:
        """Calculate confidence based on accessibility attributes."""
        confidence = 0.7  # High base confidence for accessibility matches
        
        desc_lower = description.lower()
        
        # Check ARIA attributes
        aria_label = element.get_attribute("aria-label")
        if aria_label and aria_label.lower() in desc_lower:
            confidence += 0.2
        
        # Check role
        role = element.get_attribute("role")
        if role and role.lower() in desc_lower:
            confidence += 0.1
        
        return min(confidence, 1.0)


class LLMGuidedLocatorStrategy(ElementLocatorStrategy):
    """LLM-guided element location strategy."""
    
    def __init__(self, 
                 driver: WebDriver, 
                 agent: Optional["Agent"] = None,
                 timeout: float = 10.0):
        super().__init__(driver, timeout)
        self.agent = agent
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create prompt template for LLM element location."""
        return """
        You are an expert web automation assistant. I need to find a web element on the page.
        
        Description of element to find: {description}
        
        Current page URL: {url}
        Page title: {title}
        
        Here are some visible elements on the page:
        {visible_elements}
        
        Please analyze the page and provide:
        1. The most likely CSS selector or XPath for the element
        2. Alternative selectors if the primary one might fail
        3. A confidence score (0-100) for your recommendation
        4. Any potential challenges or warnings
        
        Format your response as JSON with the following structure:
        {{
            "primary_selector": "CSS selector or XPath",
            "alternative_selectors": ["selector1", "selector2"],
            "confidence": 85,
            "selector_type": "css" or "xpath",
            "reasoning": "Why you chose this selector",
            "warnings": ["warning1", "warning2"]
        }}
        """
    
    async def locate(self, 
                    description: str, 
                    context: Optional[Dict[str, Any]] = None) -> LocationResult:
        start_time = time.time()
        
        if not self.agent:
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.LLM,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error="LLM agent not available",
            )
        
        try:
            # Get page information
            page_info = await self._get_page_info()
            
            # Get visible elements for context
            visible_elements = await self._get_visible_elements_info()
            
            # Create prompt
            prompt = self.prompt_template.format(
                description=description,
                url=page_info.get("url", ""),
                title=page_info.get("title", ""),
                visible_elements=json.dumps(visible_elements[:10], indent=2),  # Limit to 10 elements
            )
            
            # Query LLM
            response = await self.agent.query_llm(prompt)
            
            # Parse response
            selector_info = self._parse_llm_response(response)
            
            if selector_info:
                # Try to locate element with LLM's suggestion
                element = await self._locate_with_llm_suggestion(selector_info)
                
                if element:
                    confidence = selector_info.get("confidence", 50) / 100.0
                    signature = self.get_element_signature(element)
                    signature.confidence = confidence
                    signature.strategy_used = StrategyType.LLM
                    
                    # Store LLM reasoning in attributes
                    signature.attributes["llm_reasoning"] = selector_info.get("reasoning", "")
                    signature.attributes["llm_warnings"] = selector_info.get("warnings", [])
                    
                    return LocationResult(
                        element=element,
                        signature=signature,
                        strategy_used=StrategyType.LLM,
                        confidence=confidence,
                        time_taken=time.time() - start_time,
                    )
            
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.LLM,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error="LLM could not identify element",
            )
            
        except Exception as e:
            return LocationResult(
                element=None,
                signature=ElementSignature(),
                strategy_used=StrategyType.LLM,
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e),
            )
    
    async def _get_page_info(self) -> Dict[str, str]:
        """Get current page information."""
        try:
            return {
                "url": self.driver.current_url,
                "title": self.driver.title,
            }
        except:
            return {"url": "", "title": ""}
    
    async def _get_visible_elements_info(self) -> List[Dict[str, Any]]:
        """Get information about visible elements on the page."""
        try:
            script = """
            var elements = [];
            var allElements = document.querySelectorAll('*');
            
            for (var i = 0; i < Math.min(allElements.length, 50); i++) {
                var el = allElements[i];
                var rect = el.getBoundingClientRect();
                
                if (rect.width > 0 && rect.height > 0) {
                    elements.push({
                        tag: el.tagName.toLowerCase(),
                        id: el.id,
                        className: el.className,
                        text: el.innerText ? el.innerText.substring(0, 100) : '',
                        ariaLabel: el.getAttribute('aria-label'),
                        placeholder: el.getAttribute('placeholder'),
                        type: el.getAttribute('type'),
                        role: el.getAttribute('role'),
                        visible: true
                    });
                }
            }
            
            return elements;
            """
            
            elements = self.driver.execute_script(script)
            return elements or []
            
        except Exception as e:
            logger.warning(f"Failed to get visible elements: {e}")
            return []
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into selector information."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Fallback: try to extract selector directly
            selector_match = re.search(r'(?:selector|xpath|css)[:\s]*["\']?([^"\'}\s]+)', response, re.IGNORECASE)
            if selector_match:
                return {
                    "primary_selector": selector_match.group(1),
                    "alternative_selectors": [],
                    "confidence": 70,
                    "selector_type": "css" if not selector_match.group(1).startswith("//") else "xpath",
                    "reasoning": "Extracted from LLM response",
                    "warnings": [],
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None
    
    async def _locate_with_llm_suggestion(self, 
                                         selector_info: Dict[str, Any]) -> Optional[WebElement]:
        """Locate element using LLM's suggestion."""
        try:
            primary_selector = selector_info.get("primary_selector")
            selector_type = selector_info.get("selector_type", "css")
            
            if not primary_selector:
                return None
            
            # Try primary selector
            try:
                if selector_type == "xpath":
                    element = self.driver.find_element(By.XPATH, primary_selector)
                else:
                    element = self.driver.find_element(By.CSS_SELECTOR, primary_selector)
                
                if element and element.is_displayed():
                    return element
            except:
                pass
            
            # Try alternative selectors
            for alt_selector in selector_info.get("alternative_selectors", []):
                try:
                    if selector_type == "xpath":
                        element = self.driver.find_element(By.XPATH, alt_selector)
                    else:
                        element = self.driver.find_element(By.CSS_SELECTOR, alt_selector)
                    
                    if element and element.is_displayed():
                        return element
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to locate with LLM suggestion: {e}")
            return None
    
    def calculate_confidence(self, element: WebElement, description: str) -> float:
        """Confidence is provided by LLM, so use that."""
        # LLM confidence is already normalized in the response
        return 0.5  # Default, will be overridden by LLM confidence


class SelectorRobustnessScorer:
    """Scores the robustness of element selectors."""
    
    @staticmethod
    def score_selector(selector: str, 
                      selector_type: str = "css",
                      element: Optional[WebElement] = None) -> float:
        """Calculate robustness score for a selector."""
        score = 0.0
        
        if selector_type == "css":
            score = SelectorRobustnessScorer._score_css_selector(selector, element)
        elif selector_type == "xpath":
            score = SelectorRobustnessScorer._score_xpath_selector(selector, element)
        
        return min(max(score, 0.0), 1.0)
    
    @staticmethod
    def _score_css_selector(selector: str, element: Optional[WebElement] = None) -> float:
        """Score CSS selector robustness."""
        score = 0.5  # Base score
        
        # ID selectors are very robust
        if selector.startswith("#") and not " " in selector:
            score += 0.4
        
        # Data attributes are usually stable
        elif "[data-" in selector:
            score += 0.3
        
        # ARIA attributes are good for accessibility
        elif "[aria-" in selector:
            score += 0.2
        
        # Class selectors with specific names
        elif "." in selector and not selector.startswith("."):
            score += 0.1
        
        # Avoid position-based selectors
        if ":nth-child" in selector or ":nth-of-type" in selector:
            score -= 0.2
        
        # Avoid very long selectors (likely brittle)
        if len(selector) > 100:
            score -= 0.1
        
        # Check element stability if provided
        if element:
            # Elements with stable attributes are better
            stable_attrs = ["id", "name", "data-testid", "aria-label"]
            for attr in stable_attrs:
                if element.get_attribute(attr):
                    score += 0.05
        
        return score
    
    @staticmethod
    def _score_xpath_selector(selector: str, element: Optional[WebElement] = None) -> float:
        """Score XPath selector robustness."""
        score = 0.5  # Base score
        
        # ID-based XPaths are robust
        if "@id=" in selector:
            score += 0.4
        
        # Attribute-based XPaths
        elif "@" in selector and "contains" not in selector:
            score += 0.3
        
        # Text-based XPaths can be fragile
        if "text()" in selector:
            score -= 0.1
        
        // Position-based XPaths are fragile
        if re.search(r'\[\d+\]', selector):
            score -= 0.2
        
        // Very long XPaths are likely brittle
        if len(selector) > 150:
            score -= 0.1
        
        return score
    
    @staticmethod
    def generate_robust_alternatives(signature: ElementSignature) -> List[Tuple[str, str, float]]:
        """Generate robust alternative selectors for an element."""
        alternatives = []
        
        if signature.css_selector:
            # Try to improve CSS selector
            improved_css = SelectorRobustnessScorer._improve_css_selector(
                signature.css_selector, signature
            )
            if improved_css:
                score = SelectorRobustnessScorer.score_selector(improved_css, "css")
                alternatives.append((improved_css, "css", score))
        
        if signature.xpath:
            # Try to improve XPath
            improved_xpath = SelectorRobustnessScorer._improve_xpath_selector(
                signature.xpath, signature
            )
            if improved_xpath:
                score = SelectorRobustnessScorer.score_selector(improved_xpath, "xpath")
                alternatives.append((improved_xpath, "xpath", score))
        
        # Generate from attributes
        if signature.attributes:
            for attr, value in signature.attributes.items():
                if attr in ["id", "name", "data-testid", "aria-label"]:
                    css_selector = f"[{attr}='{value}']"
                    score = SelectorRobustnessScorer.score_selector(css_selector, "css")
                    alternatives.append((css_selector, "css", score))
        
        # Sort by score descending
        alternatives.sort(key=lambda x: x[2], reverse=True)
        
        return alternatives
    
    @staticmethod
    def _improve_css_selector(selector: str, signature: ElementSignature) -> Optional[str]:
        """Try to improve CSS selector robustness."""
        # If selector contains position-based parts, try to replace with attribute selectors
        if ":nth-child" in selector or ":nth-of-type" in selector:
            # Try to use ID or data attributes instead
            if signature.attributes.get("id"):
                return f"#{signature.attributes['id']}"
            elif signature.attributes.get("data-testid"):
                return f"[data-testid='{signature.attributes['data-testid']}']"
        
        return selector
    
    @staticmethod
    def _improve_xpath_selector(selector: str, signature: ElementSignature) -> Optional[str]:
        """Try to improve XPath selector robustness."""
        # Replace position-based predicates with attribute predicates
        if re.search(r'\[\d+\]', selector):
            if signature.attributes.get("id"):
                return f"//*[@id='{signature.attributes['id']}']"
            elif signature.attributes.get("name"):
                return f"//*[@name='{signature.attributes['name']}']"
        
        return selector


class AdaptiveElementLocator:
    """
    Adaptive Element Location Engine - Multi-strategy adaptive element location
    with automatic fallback chains and self-healing selectors.
    """
    
    def __init__(self, 
                 driver: WebDriver,
                 agent: Optional["Agent"] = None,
                 timeout: float = 10.0,
                 strategies_order: Optional[List[StrategyType]] = None):
        """
        Initialize the adaptive element locator.
        
        Args:
            driver: Selenium WebDriver instance
            agent: Optional LLM agent for guided exploration
            timeout: Default timeout for element location
            strategies_order: Order of strategies to try (default: CSS -> XPath -> Accessibility -> Visual -> LLM)
        """
        self.driver = driver
        self.agent = agent
        self.timeout = timeout
        
        # Initialize strategies
        self.strategies = {
            StrategyType.CSS: CSSLocatorStrategy(driver, timeout),
            StrategyType.XPATH: XPathLocatorStrategy(driver, timeout),
            StrategyType.ACCESSIBILITY: AccessibilityLocatorStrategy(driver, timeout),
            StrategyType.VISUAL: VisualLocatorStrategy(driver, timeout),
            StrategyType.LLM: LLMGuidedLocatorStrategy(driver, agent, timeout),
        }
        
        # Set strategy order
        self.strategies_order = strategies_order or [
            StrategyType.CSS,
            StrategyType.XPATH,
            StrategyType.ACCESSIBILITY,
            StrategyType.VISUAL,
            StrategyType.LLM,
        ]
        
        # Cache for element signatures
        self.signature_cache: Dict[str, ElementSignature] = {}
        
        # History for adaptive learning
        self.strategy_success_history: Dict[StrategyType, int] = {
            strategy: 0 for strategy in StrategyType
        }
        
        # Selector robustness scorer
        self.scorer = SelectorRobustnessScorer()
        
        logger.info("AdaptiveElementLocator initialized with strategies: " + 
                   ", ".join([s.value for s in self.strategies_order]))
    
    async def locate_element(self, 
                            description: str, 
                            context: Optional[Dict[str, Any]] = None,
                            cache_key: Optional[str] = None) -> LocationResult:
        """
        Locate element using adaptive multi-strategy approach.
        
        Args:
            description: Natural language description of element to find
            context: Additional context for location
            cache_key: Optional key for caching element signature
            
        Returns:
            LocationResult with element and metadata
        """
        start_time = time.time()
        attempts = 0
        
        # Check cache first
        if cache_key and cache_key in self.signature_cache:
            cached_signature = self.signature_cache[cache_key]
            element = await self._locate_from_signature(cached_signature)
            
            if element:
                logger.info(f"Element found in cache with key: {cache_key}")
                return LocationResult(
                    element=element,
                    signature=cached_signature,
                    strategy_used=cached_signature.strategy_used,
                    confidence=cached_signature.confidence,
                    attempts=1,
                    time_taken=time.time() - start_time,
                )
        
        # Try strategies in order
        for strategy_type in self.strategies_order:
            attempts += 1
            strategy = self.strategies.get(strategy_type)
            
            if not strategy:
                continue
            
            logger.debug(f"Trying strategy: {strategy_type.value}")
            
            try:
                result = await strategy.locate(description, context)
                
                if result.element and result.confidence > 0.3:  # Minimum confidence threshold
                    # Update success history
                    self.strategy_success_history[strategy_type] += 1
                    
                    # Generate robust alternatives
                    alternatives = self.scorer.generate_robust_alternatives(result.signature)
                    result.signature.attributes["robust_alternatives"] = [
                        {"selector": alt[0], "type": alt[1], "score": alt[2]}
                        for alt in alternatives[:3]  # Keep top 3
                    ]
                    
                    # Cache the signature
                    if cache_key:
                        self.signature_cache[cache_key] = result.signature
                    
                    # Update result with attempt count
                    result.attempts = attempts
                    
                    logger.info(f"Element found with {strategy_type.value} "
                              f"(confidence: {result.confidence:.2f}, attempts: {attempts})")
                    
                    return result
                
            except Exception as e:
                logger.warning(f"Strategy {strategy_type.value} failed: {e}")
                continue
        
        # All strategies failed
        logger.warning(f"All strategies failed for description: {description}")
        
        return LocationResult(
            element=None,
            signature=ElementSignature(),
            strategy_used=StrategyType.CSS,  # Default
            confidence=0.0,
            attempts=attempts,
            time_taken=time.time() - start_time,
            error="All location strategies failed",
        )
    
    async def _locate_from_signature(self, signature: ElementSignature) -> Optional[WebElement]:
        """Try to locate element from cached signature."""
        try:
            # Try CSS selector first
            if signature.css_selector:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, signature.css_selector)
                    if element and element.is_displayed():
                        return element
                except:
                    pass
            
            # Try XPath
            if signature.xpath:
                try:
                    element = self.driver.find_element(By.XPATH, signature.xpath)
                    if element and element.is_displayed():
                        return element
                except:
                    pass
            
            # Try by accessibility attributes
            if signature.aria_label:
                try:
                    element = self.driver.find_element(
                        By.XPATH, 
                        f"//*[@aria-label='{signature.aria_label}']"
                    )
                    if element and element.is_displayed():
                        return element
                except:
                    pass
            
            # Try by ID
            if signature.attributes.get("id"):
                try:
                    element = self.driver.find_element(By.ID, signature.attributes["id"])
                    if element and element.is_displayed():
                        return element
                except:
                    pass
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to locate from signature: {e}")
            return None
    
    async def heal_selector(self, 
                           signature: ElementSignature, 
                           description: str) -> Optional[ElementSignature]:
        """
        Attempt to heal a broken selector by re-locating the element.
        
        Args:
            signature: Original element signature with broken selector
            description: Description of the element
            
        Returns:
            Updated signature with working selector, or None if healing failed
        """
        logger.info(f"Attempting to heal selector for: {description}")
        
        # Try to re-locate the element
        result = await self.locate_element(description)
        
        if result.element:
            # Update the signature with new selectors
            new_signature = result.signature
            
            # Keep original metadata but update selectors
            new_signature.timestamp = time.time()
            new_signature.confidence = result.confidence
            
            # Add healing metadata
            new_signature.attributes["healed"] = True
            new_signature.attributes["original_selectors"] = {
                "css": signature.css_selector,
                "xpath": signature.xpath,
            }
            new_signature.attributes["healing_time"] = time.time()
            
            logger.info(f"Selector healed successfully with {result.strategy_used.value}")
            
            return new_signature
        
        logger.warning("Selector healing failed")
        return None
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for each strategy."""
        total_attempts = sum(self.strategy_success_history.values())
        
        performance = {}
        for strategy, successes in self.strategy_success_history.items():
            performance[strategy.value] = {
                "successes": successes,
                "success_rate": successes / total_attempts if total_attempts > 0 else 0,
            }
        
        return performance
    
    def optimize_strategy_order(self) -> None:
        """Optimize strategy order based on historical performance."""
        # Sort strategies by success rate
        sorted_strategies = sorted(
            self.strategies_order,
            key=lambda s: self.strategy_success_history.get(s, 0),
            reverse=True,
        )
        
        self.strategies_order = sorted_strategies
        
        logger.info(f"Optimized strategy order: {[s.value for s in self.strategies_order]}")
    
    async def batch_locate(self, 
                          descriptions: List[str], 
                          context: Optional[Dict[str, Any]] = None) -> List[LocationResult]:
        """
        Locate multiple elements in batch.
        
        Args:
            descriptions: List of element descriptions
            context: Shared context for all elements
            
        Returns:
            List of LocationResults
        """
        tasks = []
        for i, desc in enumerate(descriptions):
            cache_key = f"batch_{i}_{desc[:50]}"
            task = self.locate_element(desc, context, cache_key)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch location failed for '{descriptions[i]}': {result}")
                final_results.append(LocationResult(
                    element=None,
                    signature=ElementSignature(),
                    strategy_used=StrategyType.CSS,
                    confidence=0.0,
                    error=str(result),
                ))
            else:
                final_results.append(result)
        
        return final_results


class SelfHealingElement:
    """
    Self-healing element wrapper that automatically recovers from selector failures.
    """
    
    def __init__(self, 
                 element: WebElement,
                 signature: ElementSignature,
                 locator: AdaptiveElementLocator,
                 description: str):
        """
        Initialize self-healing element.
        
        Args:
            element: The WebElement
            signature: Element signature with selectors
            locator: Adaptive element locator for healing
            description: Description of the element
        """
        self._element = element
        self._signature = signature
        self._locator = locator
        self._description = description
        self._last_successful_action = time.time()
        self._action_count = 0
        
        # Store original element properties
        self._original_tag = element.tag_name
        self._original_text = element.text
    
    async def _ensure_element(self) -> WebElement:
        """Ensure element is still valid, heal if necessary."""
        try:
            # Quick check if element is still attached and valid
            if self._element and self._element.is_displayed() and self._element.is_enabled():
                return self._element
        except:
            pass
        
        # Element is stale or invalid, attempt healing
        logger.info(f"Element stale, attempting to heal: {self._description}")
        
        new_signature = await self._locator.heal_selector(self._signature, self._description)
        
        if new_signature:
            # Try to locate with new signature
            result = await self._locator.locate_element(self._description)
            
            if result.element:
                self._element = result.element
                self._signature = new_signature
                self._last_successful_action = time.time()
                
                logger.info("Element successfully healed")
                return self._element
        
        raise Exception(f"Failed to heal element: {self._description}")
    
    async def click(self) -> None:
        """Click the element with automatic healing."""
        element = await self._ensure_element()
        element.click()
        self._action_count += 1
        self._last_successful_action = time.time()
    
    async def send_keys(self, text: str) -> None:
        """Send keys to the element with automatic healing."""
        element = await self._ensure_element()
        element.send_keys(text)
        self._action_count += 1
        self._last_successful_action = time.time()
    
    async def get_text(self) -> str:
        """Get text from the element with automatic healing."""
        element = await self._ensure_element()
        return element.text
    
    async def get_attribute(self, name: str) -> Optional[str]:
        """Get attribute from the element with automatic healing."""
        element = await self._ensure_element()
        return element.get_attribute(name)
    
    async def is_displayed(self) -> bool:
        """Check if element is displayed with automatic healing."""
        try:
            element = await self._ensure_element()
            return element.is_displayed()
        except:
            return False
    
    async def is_enabled(self) -> bool:
        """Check if element is enabled with automatic healing."""
        try:
            element = await self._ensure_element()
            return element.is_enabled()
        except:
            return False
    
    def get_signature(self) -> ElementSignature:
        """Get current element signature."""
        return self._signature
    
    def get_stats(self) -> Dict[str, Any]:
        """Get element usage statistics."""
        return {
            "description": self._description,
            "action_count": self._action_count,
            "last_action": self._last_successful_action,
            "age_seconds": time.time() - self._signature.timestamp,
            "current_selectors": {
                "css": self._signature.css_selector,
                "xpath": self._signature.xpath,
            },
            "confidence": self._signature.confidence,
        }


# Factory function for easy integration
def create_adaptive_locator(driver: WebDriver, 
                           agent: Optional["Agent"] = None,
                           **kwargs) -> AdaptiveElementLocator:
    """Create an adaptive element locator with default configuration."""
    return AdaptiveElementLocator(driver, agent, **kwargs)


# Integration with existing Page class
async def enhance_page_with_adaptive_location(page: "Page", agent: Optional["Agent"] = None) -> None:
    """
    Enhance an existing Page instance with adaptive element location capabilities.
    
    Args:
        page: Page instance to enhance
        agent: Optional LLM agent for guided exploration
    """
    if not hasattr(page, 'adaptive_locator'):
        page.adaptive_locator = create_adaptive_locator(page.driver, agent)
        
        # Add convenience methods
        async def find_element_adaptive(description: str, **kwargs) -> Optional[WebElement]:
            result = await page.adaptive_locator.locate_element(description, **kwargs)
            return result.element
        
        async def find_and_click(description: str, **kwargs) -> bool:
            result = await page.adaptive_locator.locate_element(description, **kwargs)
            if result.element:
                result.element.click()
                return True
            return False
        
        async def find_and_type(description: str, text: str, **kwargs) -> bool:
            result = await page.adaptive_locator.locate_element(description, **kwargs)
            if result.element:
                result.element.send_keys(text)
                return True
            return False
        
        # Attach methods to page
        page.find_element_adaptive = find_element_adaptive
        page.find_and_click = find_and_click
        page.find_and_type = find_and_type
        
        logger.info("Page enhanced with adaptive element location")