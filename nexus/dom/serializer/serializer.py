# @file purpose: Serializes enhanced DOM trees to string format for LLM consumption

import re
from typing import Any

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

DISABLED_ELEMENTS = {'style', 'script', 'head', 'meta', 'link', 'title'}

# SVG child elements to skip (decorative only, no interaction value)
SVG_ELEMENTS = {
    'path',
    'rect',
    'g',
    'circle',
    'ellipse',
    'line',
    'polyline',
    'polygon',
    'use',
    'defs',
    'clipPath',
    'mask',
    'pattern',
    'image',
    'text',
    'tspan',
}

# LLM attention patterns - elements prioritized for different task types
LLM_ATTENTION_PATTERNS = {
    'navigation': {
        'high_priority': {'a', 'button', 'nav', '[role="navigation"]', '[role="button"]', '[role="link"]'},
        'medium_priority': {'header', 'footer', 'main', '[role="main"]', '[role="banner"]', '[role="contentinfo"]'},
        'low_priority': {'div', 'span', 'section', 'article'},
    },
    'form_interaction': {
        'high_priority': {'input', 'select', 'textarea', 'button', '[role="textbox"]', '[role="combobox"]', '[role="listbox"]'},
        'medium_priority': {'label', 'fieldset', 'legend', '[role="group"]', '[role="radiogroup"]'},
        'low_priority': {'div', 'span', 'p'},
    },
    'content_extraction': {
        'high_priority': {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'article', 'section', 'main'},
        'medium_priority': {'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'dl', 'dt', 'dd'},
        'low_priority': {'div', 'span', 'header', 'footer', 'nav', 'aside'},
    },
    'general': {
        'high_priority': {'button', 'a', 'input', 'select', 'textarea', '[role="button"]', '[role="link"]', '[role="textbox"]'},
        'medium_priority': {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'img', '[alt]', 'label'},
        'low_priority': {'div', 'span', 'section', 'article', 'header', 'footer', 'nav'},
    }
}

# Semantic grouping patterns for hierarchical summarization
SEMANTIC_PATTERNS = {
    'navigation_group': ['nav', '[role="navigation"]', 'menu', '[role="menu"]', '[role="menubar"]'],
    'form_group': ['form', 'fieldset', '[role="form"]', '[role="group"]'],
    'content_group': ['article', 'section', 'main', '[role="main"]', '[role="article"]'],
    'interactive_group': ['button', 'a', 'input', 'select', 'textarea', '[role="button"]', '[role="link"]'],
    'media_group': ['img', 'video', 'audio', 'picture', 'figure', '[role="img"]', '[role="video"]'],
}


class DOMTreeSerializer:
    """Serializes enhanced DOM trees to string format with LLM optimization."""

    # Configuration - elements that propagate bounds to their children
    PROPAGATING_ELEMENTS = [
        {'tag': 'a', 'role': None},  # Any <a> tag
        {'tag': 'button', 'role': None},  # Any <button> tag
        {'tag': 'div', 'role': 'button'},  # <div role="button">
        {'tag': 'div', 'role': 'combobox'},  # <div role="combobox"> - dropdowns/selects
        {'tag': 'span', 'role': 'button'},  # <span role="button">
        {'tag': 'span', 'role': 'combobox'},  # <span role="combobox">
        {'tag': 'input', 'role': 'combobox'},  # <input role="combobox"> - autocomplete inputs
        {'tag': 'input', 'role': 'combobox'},  # <input type="text"> - text inputs with suggestions
        # {'tag': 'div', 'role': 'link'},     # <div role="link">
        # {'tag': 'span', 'role': 'link'},    # <span role="link">
    ]
    DEFAULT_CONTAINMENT_THRESHOLD = 0.99  # 99% containment by default

    def __init__(
        self,
        root_node: EnhancedDOMTreeNode,
        previous_cached_state: SerializedDOMState | None = None,
        enable_bbox_filtering: bool = True,
        containment_threshold: float | None = None,
        paint_order_filtering: bool = True,
        session_id: str | None = None,
        task_type: str = 'general',
        enable_llm_optimization: bool = True,
        max_tokens: int = 4000,
    ):
        self.root_node = root_node
        self._interactive_counter = 1
        self._selector_map: DOMSelectorMap = {}
        self._previous_cached_selector_map = previous_cached_state.selector_map if previous_cached_state else None
        # Add timing tracking
        self.timing_info: dict[str, float] = {}
        # Cache for clickable element detection to avoid redundant calls
        self._clickable_cache: dict[int, bool] = {}
        # Bounding box filtering configuration
        self.enable_bbox_filtering = enable_bbox_filtering
        self.containment_threshold = containment_threshold or self.DEFAULT_CONTAINMENT_THRESHOLD
        # Paint order filtering configuration
        self.paint_order_filtering = paint_order_filtering
        # Session ID for session-specific exclude attribute
        self.session_id = session_id
        
        # LLM optimization configuration
        self.task_type = task_type
        self.enable_llm_optimization = enable_llm_optimization
        self.max_tokens = max_tokens
        self._token_count = 0
        self._attention_patterns = LLM_ATTENTION_PATTERNS.get(task_type, LLM_ATTENTION_PATTERNS['general'])
        
        # Semantic grouping for hierarchical summarization
        self._semantic_groups = {}
        self._current_group_id = 0

    def _safe_parse_number(self, value_str: str, default: float) -> float:
        """Parse string to float, handling negatives and decimals."""
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return default

    def _safe_parse_optional_number(self, value_str: str | None) -> float | None:
        """Parse string to float, returning None for invalid values."""
        if not value_str:
            return None
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple heuristic: ~4 characters per token for English text
        return len(text) // 4

    def _matches_selector(self, node: EnhancedDOMTreeNode, selector: str) -> bool:
        """Check if node matches a CSS selector pattern."""
        if selector.startswith('[') and selector.endswith(']'):
            # Attribute selector like [role="button"]
            attr_match = re.match(r'\[(\w+)="([^"]+)"\]', selector)
            if attr_match:
                attr_name, attr_value = attr_match.groups()
                return node.attributes.get(attr_name) == attr_value
        else:
            # Tag selector
            return node.tag_name == selector
        return False

    def _get_element_priority(self, node: EnhancedDOMTreeNode) -> int:
        """Get element priority based on LLM attention patterns (0=ignore, 1=low, 2=medium, 3=high)."""
        # Check high priority
        for selector in self._attention_patterns['high_priority']:
            if self._matches_selector(node, selector):
                return 3
        
        # Check medium priority
        for selector in self._attention_patterns['medium_priority']:
            if self._matches_selector(node, selector):
                return 2
        
        # Check for interactive elements (always at least medium priority)
        if self._is_interactive_element(node):
            return 2
        
        # Check for visible text content
        if node.text_content and node.text_content.strip():
            return 1
        
        return 0  # Ignore

    def _is_interactive_element(self, node: EnhancedDOMTreeNode) -> bool:
        """Check if element is interactive."""
        # Check cache first
        node_id = id(node)
        if node_id in self._clickable_cache:
            return self._clickable_cache[node_id]
        
        # Check using ClickableElementDetector
        detector = ClickableElementDetector()
        is_interactive = detector.is_clickable(node)
        self._clickable_cache[node_id] = is_interactive
        return is_interactive

    def _get_semantic_group(self, node: EnhancedDOMTreeNode) -> str | None:
        """Determine semantic group for hierarchical summarization."""
        for group_name, selectors in SEMANTIC_PATTERNS.items():
            for selector in selectors:
                if self._matches_selector(node, selector):
                    return group_name
        return None

    def _create_simplified_tree(self, node: EnhancedDOMTreeNode, parent_group: str | None = None) -> SimplifiedNode | None:
        """Create simplified tree with LLM optimization."""
        # Skip disabled elements
        if node.tag_name in DISABLED_ELEMENTS:
            return None
        
        # Skip SVG decorative elements
        if node.tag_name in SVG_ELEMENTS:
            return None
        
        # Get element priority
        priority = self._get_element_priority(node)
        
        # Skip low priority elements if optimization is enabled
        if self.enable_llm_optimization and priority == 0:
            # Check if element has important children
            has_important_children = False
            for child in node.children:
                if child and self._get_element_priority(child) > 0:
                    has_important_children = True
                    break
            
            if not has_important_children:
                return None
        
        # Create simplified node
        simplified = SimplifiedNode(
            node_id=node.node_id,
            tag_name=node.tag_name,
            attributes=node.attributes or {},
            text_content=node.text_content,
            bounds=node.bounds,
            ax_node=node.ax_node,
            is_visible=node.is_visible,
            is_interactive=self._is_interactive_element(node),
            priority=priority,
        )
        
        # Determine semantic group
        current_group = self._get_semantic_group(node) or parent_group
        if current_group:
            simplified.semantic_group = current_group
        
        # Process children
        for child in node.children:
            if child:
                child_simplified = self._create_simplified_tree(child, current_group)
                if child_simplified:
                    simplified.children.append(child_simplified)
        
        # Apply hierarchical summarization for groups
        if self.enable_llm_optimization and current_group and len(simplified.children) > 3:
            simplified = self._summarize_group(simplified, current_group)
        
        return simplified

    def _summarize_group(self, node: SimplifiedNode, group_type: str) -> SimplifiedNode:
        """Apply hierarchical summarization to semantic groups."""
        # Count elements by priority in the group
        priority_counts = {3: 0, 2: 0, 1: 0, 0: 0}
        
        def count_priorities(n: SimplifiedNode):
            priority_counts[n.priority] = priority_counts.get(n.priority, 0) + 1
            for child in n.children:
                count_priorities(child)
        
        count_priorities(node)
        
        # If group has too many low-priority elements, summarize
        total_elements = sum(priority_counts.values())
        low_priority_ratio = (priority_counts[0] + priority_counts[1]) / total_elements if total_elements > 0 else 0
        
        if low_priority_ratio > 0.7 and total_elements > 5:
            # Create summarized representation
            summarized = SimplifiedNode(
                node_id=node.node_id,
                tag_name=node.tag_name,
                attributes=node.attributes,
                text_content=f"[{group_type.replace('_', ' ').title()} Group: {priority_counts[3]} interactive, {priority_counts[2]} important, {total_elements - priority_counts[3] - priority_counts[2]} other elements]",
                bounds=node.bounds,
                ax_node=node.ax_node,
                is_visible=node.is_visible,
                is_interactive=node.is_interactive,
                priority=node.priority,
                semantic_group=node.semantic_group,
                is_summarized=True,
            )
            
            # Keep only high and medium priority children
            for child in node.children:
                if child.priority >= 2:
                    summarized.children.append(child)
            
            return summarized
        
        return node

    def _add_compound_components(self, simplified: SimplifiedNode, node: EnhancedDOMTreeNode) -> None:
        """Enhance compound controls with information from their child components."""
        # Only process elements that might have compound components
        if node.tag_name not in ['input', 'select', 'details', 'audio', 'video']:
            return

        # For input elements, check for compound input types
        if node.tag_name == 'input':
            if not node.attributes or node.attributes.get('type') not in [
                'date',
                'time',
                'datetime-local',
                'month',
                'week',
                'range',
                'number',
                'color',
                'file',
            ]:
                return
        # For other elements, check if they have AX child indicators
        elif not node.ax_node or not node.ax_node.child_ids:
            return

        # Add compound component information based on element type
        element_type = node.tag_name
        input_type = node.attributes.get('type', '') if node.attributes else ''

        if element_type == 'input':
            # NOTE: For date/time inputs, we DON'T add compound components because:
            # 1. They confuse the model (seeing "Day, Month, Year" suggests DD.MM.YYYY format)
            # 2. HTML5 date/time inputs ALWAYS require ISO format (YYYY-MM-DD, HH:MM, etc.)
            # 3. The placeholder attribute clearly shows the required format
            # 4. These inputs use direct value assignment, not sequential typing
            if input_type in ['date', 'time', 'datetime-local', 'month', 'week']:
                # Skip compound components for date/time inputs - format is shown in placeholder
                pass
            elif input_type == 'range':
                # Range slider with value indicator
                min_val = node.attributes.get('min', '0') if node.attributes else '0'
                max_val = node.attributes.get('max', '100') if node.attributes else '100'

                node._compound_children.append(
                    {
                        'role': 'slider',
                        'name': 'Value',
                        'valuemin': self._safe_parse_number(min_val, 0.0),
                        'valuemax': self._safe_parse_number(max_val, 100.0),
                        'valuenow': None,
                    }
                )
                simplified.is_compound_component = True
            elif input_type == 'number':
                # Number input with increment/decrement buttons
                min_val = node.attributes.get('min') if node.attributes else None
                max_val = node.attributes.get('max') if node.attributes else None

                node._compound_children.extend(
                    [
                        {'role': 'button', 'name': 'Increment', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                        {'role': 'button', 'name': 'Decrement', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                        {
                            'role': 'textbox',
                            'name': 'Value',
                            'valuemin': self._safe_parse_optional_number(min_val),
                            'valuemax': self._safe_parse_optional_number(max_val),
                            'valuenow': None,
                        },
                    ]
                )
                simplified.is_compound_component = True
            elif input_type == 'color':
                # Color picker with components
                node._compound_children.extend(
                    [
                        {'role': 'textbox', 'name': 'Color Value', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                        {'role': 'button', 'name': 'Color Picker', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                    ]
                )
                simplified.is_compound_component = True
            elif input_type == 'file':
                # File input with components
                node._compound_children.extend(
                    [
                        {'role': 'button', 'name': 'Browse', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                        {'role': 'textbox', 'name': 'File Path', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                    ]
                )
                simplified.is_compound_component = True
        elif element_type == 'select':
            # Select dropdown with components
            node._compound_children.extend(
                [
                    {'role': 'button', 'name': 'Dropdown', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                    {'role': 'listbox', 'name': 'Options', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                ]
            )
            simplified.is_compound_component = True
        elif element_type == 'details':
            # Details/summary element
            node._compound_children.append(
                {'role': 'button', 'name': 'Toggle', 'valuemin': None, 'valuemax': None, 'valuenow': None}
            )
            simplified.is_compound_component = True
        elif element_type in ['audio', 'video']:
            # Media player controls
            node._compound_children.extend(
                [
                    {'role': 'button', 'name': 'Play/Pause', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                    {'role': 'slider', 'name': 'Seek', 'valuemin': 0.0, 'valuemax': 1.0, 'valuenow': None},
                    {'role': 'button', 'name': 'Mute', 'valuemin': None, 'valuemax': None, 'valuenow': None},
                    {'role': 'slider', 'name': 'Volume', 'valuemin': 0.0, 'valuemax': 1.0, 'valuenow': None},
                ]
            )
            simplified.is_compound_component = True

    def _optimize_tree(self, simplified_tree: SimplifiedNode | None) -> SimplifiedNode | None:
        """Optimize tree by removing unnecessary parent nodes."""
        if not simplified_tree:
            return None
        
        # If node has only one child and no meaningful content, replace with child
        if (len(simplified_tree.children) == 1 and 
            not simplified_tree.text_content and 
            not simplified_tree.is_interactive and
            simplified_tree.tag_name not in ['html', 'body', 'head']):
            
            child = simplified_tree.children[0]
            # Preserve important attributes from parent
            if simplified_tree.attributes.get('id'):
                child.attributes['id'] = simplified_tree.attributes['id']
            if simplified_tree.attributes.get('class'):
                child.attributes['class'] = simplified_tree.attributes['class']
            
            return self._optimize_tree(child)
        
        # Recursively optimize children
        optimized_children = []
        for child in simplified_tree.children:
            optimized_child = self._optimize_tree(child)
            if optimized_child:
                optimized_children.append(optimized_child)
        
        simplified_tree.children = optimized_children
        return simplified_tree

    def _apply_bounding_box_filtering(self, simplified_tree: SimplifiedNode | None) -> SimplifiedNode | None:
        """Apply bounding box filtering to remove off-screen elements."""
        if not simplified_tree:
            return None
        
        # Check if element is within viewport
        if simplified_tree.bounds and not self._is_in_viewport(simplified_tree.bounds):
            # Check if element has interactive children
            has_interactive_children = False
            for child in simplified_tree.children:
                if child.is_interactive:
                    has_interactive_children = True
                    break
            
            if not has_interactive_children:
                return None
        
        # Recursively filter children
        filtered_children = []
        for child in simplified_tree.children:
            filtered_child = self._apply_bounding_box_filtering(child)
            if filtered_child:
                filtered_children.append(filtered_child)
        
        simplified_tree.children = filtered_children
        return simplified_tree

    def _is_in_viewport(self, bounds: DOMRect) -> bool:
        """Check if element is within viewport."""
        # Simple viewport check - assumes viewport is 0,0 to 1920,1080
        # In a real implementation, you'd get actual viewport dimensions
        viewport_width = 1920
        viewport_height = 1080
        
        return (bounds.x < viewport_width and 
                bounds.x + bounds.width > 0 and
                bounds.y < viewport_height and 
                bounds.y + bounds.height > 0)

    def _assign_interactive_indices_and_mark_new_nodes(self, simplified_tree: SimplifiedNode | None) -> None:
        """Assign interactive indices to clickable elements and mark new nodes."""
        if not simplified_tree:
            return
        
        # Check if element is interactive
        if simplified_tree.is_interactive:
            # Check if this node was in previous state
            is_new = True
            if self._previous_cached_selector_map:
                for selector, node_id in self._previous_cached_selector_map.items():
                    if node_id == simplified_tree.node_id:
                        is_new = False
                        break
            
            # Assign interactive index
            simplified_tree.interactive_index = self._interactive_counter
            self._selector_map[str(self._interactive_counter)] = simplified_tree.node_id
            self._interactive_counter += 1
            
            # Mark as new if needed
            simplified_tree.is_new = is_new
        
        # Recursively process children
        for child in simplified_tree.children:
            self._assign_interactive_indices_and_mark_new_nodes(child)

    def serialize_accessible_elements(self) -> tuple[SerializedDOMState, dict[str, float]]:
        import time

        start_total = time.time()

        # Reset state
        self._interactive_counter = 1
        self._selector_map = {}
        self._semantic_groups = []
        self._clickable_cache = {}  # Clear cache for new serialization
        self._token_count = 0

        # Step 1: Create simplified tree (includes clickable element detection and LLM optimization)
        start_step1 = time.time()
        simplified_tree = self._create_simplified_tree(self.root_node)
        end_step1 = time.time()
        self.timing_info['create_simplified_tree'] = end_step1 - start_step1

        # Step 2: Remove elements based on paint order
        start_step3 = time.time()
        if self.paint_order_filtering and simplified_tree:
            PaintOrderRemover(simplified_tree).calculate_paint_order()
        end_step3 = time.time()
        self.timing_info['calculate_paint_order'] = end_step3 - start_step3

        # Step 3: Optimize tree (remove unnecessary parents)
        start_step2 = time.time()
        optimized_tree = self._optimize_tree(simplified_tree)
        end_step2 = time.time()
        self.timing_info['optimize_tree'] = end_step2 - start_step2

        # Step 4: Apply bounding box filtering (NEW)
        if self.enable_bbox_filtering and optimized_tree:
            start_step3 = time.time()
            filtered_tree = self._apply_bounding_box_filtering(optimized_tree)
            end_step3 = time.time()
            self.timing_info['bbox_filtering'] = end_step3 - start_step3
        else:
            filtered_tree = optimized_tree

        # Step 5: Assign interactive indices to clickable elements
        start_step4 = time.time()
        self._assign_interactive_indices_and_mark_new_nodes(filtered_tree)
        end_step4 = time.time()
        self.timing_info['assign_interactive_indices'] = end_step4 - start_step4

        # Step 6: Apply token limit optimization
        if self.enable_llm_optimization and filtered_tree:
            start_step5 = time.time()
            filtered_tree = self._apply_token_limit(filtered_tree)
            end_step5 = time.time()
            self.timing_info['token_limit_optimization'] = end_step5 - start_step5

        end_total = time.time()
        self.timing_info['serialize_accessible_elements_total'] = end_total - start_total

        return SerializedDOMState(_root=filtered_tree, selector_map=self._selector_map), self.timing_info

    def _apply_token_limit(self, tree: SimplifiedNode | None) -> SimplifiedNode | None:
        """Apply token limit by truncating low-priority elements."""
        if not tree:
            return None
        
        # Estimate current tokens
        current_tokens = self._estimate_tree_tokens(tree)
        
        # If within limit, return as is
        if current_tokens <= self.max_tokens:
            return tree
        
        # Otherwise, prune low-priority elements
        return self._prune_tree_by_priority(tree, current_tokens)

    def _estimate_tree_tokens(self, node: SimplifiedNode) -> int:
        """Estimate token count for entire tree."""
        tokens = 0
        
        # Estimate tokens for this node
        node_text = f"<{node.tag_name}"
        for attr, value in node.attributes.items():
            node_text += f' {attr}="{value}"'
        node_text += ">"
        
        if node.text_content:
            node_text += node.text_content
        
        tokens += self._estimate_tokens(node_text)
        
        # Recursively estimate children
        for child in node.children:
            tokens += self._estimate_tree_tokens(child)
        
        return tokens

    def _prune_tree_by_priority(self, node: SimplifiedNode, current_tokens: int) -> SimplifiedNode | None:
        """Prune tree by removing low-priority elements until under token limit."""
        if current_tokens <= self.max_tokens:
            return node
        
        # Sort children by priority (lowest first for pruning)
        sorted_children = sorted(node.children, key=lambda x: x.priority)
        
        # Prune lowest priority children first
        pruned_children = []
        for child in sorted_children:
            if child.priority >= 2:  # Keep medium and high priority
                pruned_children.append(child)
            elif current_tokens > self.max_tokens * 1.2:  # Only prune if significantly over limit
                # Recursively prune child
                pruned_child = self._prune_tree_by_priority(child, current_tokens)
                if pruned_child:
                    pruned_children.append(pruned_child)
                    current_tokens = self._estimate_tree_tokens(node)  # Recalculate
        
        node.children = pruned_children
        return node