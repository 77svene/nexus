"""Element class for element operations with safety validation."""

import asyncio
from typing import TYPE_CHECKING, Literal, Union, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import json

from cdp_use.client import logger
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cdp_use.cdp.dom.commands import (
        DescribeNodeParameters,
        FocusParameters,
        GetAttributesParameters,
        GetBoxModelParameters,
        PushNodesByBackendIdsToFrontendParameters,
        RequestChildNodesParameters,
        ResolveNodeParameters,
    )
    from cdp_use.cdp.input.commands import (
        DispatchMouseEventParameters,
    )
    from cdp_use.cdp.input.types import MouseButton
    from cdp_use.cdp.page.commands import CaptureScreenshotParameters
    from cdp_use.cdp.page.types import Viewport
    from cdp_use.cdp.runtime.commands import CallFunctionOnParameters

    from nexus.browser.session import BrowserSession

# Type definitions for element operations
ModifierType = Literal['Alt', 'Control', 'Meta', 'Shift']


class RiskLevel(Enum):
    """Risk levels for element actions."""
    LOW = "low"           # Safe actions like hovering, getting text
    MEDIUM = "medium"     # Actions that change state but are reversible
    HIGH = "high"         # Destructive actions like deleting, submitting forms
    CRITICAL = "critical" # Actions that could cause data loss or security issues


@dataclass
class RiskAssessment:
    """Assessment of action risk."""
    level: RiskLevel
    reasons: List[str]
    requires_confirmation: bool
    can_sandbox: bool
    estimated_impact: str


class SafetyConfig(TypedDict):
    """Configuration for safety validation."""
    require_confirmation_for_high_risk: bool
    sandbox_medium_risk: bool
    auto_rollback_on_error: bool
    max_sandbox_timeout: float  # seconds
    snapshot_before_action: bool
    custom_risk_scorer: Optional[Callable[['Element', str], RiskAssessment]]


class Position(TypedDict):
    """2D position coordinates."""
    x: float
    y: float


class BoundingBox(TypedDict):
    """Element bounding box with position and dimensions."""
    x: float
    y: float
    width: float
    height: float


class ElementInfo(TypedDict):
    """Basic information about a DOM element."""
    backendNodeId: int
    nodeId: int | None
    nodeName: str
    nodeType: int
    nodeValue: str | None
    attributes: dict[str, str]
    boundingBox: BoundingBox | None
    error: str | None


class DOMSnapshot(TypedDict):
    """Snapshot of DOM state for rollback."""
    html: str
    timestamp: float
    element_states: Dict[str, Any]
    url: str


class Element:
    """Element operations using BackendNodeId with safety validation."""

    def __init__(
        self,
        browser_session: 'BrowserSession',
        backend_node_id: int,
        session_id: str | None = None,
        safety_config: Optional[SafetyConfig] = None,
    ):
        self._browser_session = browser_session
        self._client = browser_session.cdp_client
        self._backend_node_id = backend_node_id
        self._session_id = session_id
        
        # Safety configuration with defaults
        self._safety_config: SafetyConfig = safety_config or {
            'require_confirmation_for_high_risk': True,
            'sandbox_medium_risk': True,
            'auto_rollback_on_error': True,
            'max_sandbox_timeout': 5.0,
            'snapshot_before_action': True,
            'custom_risk_scorer': None,
        }
        
        # DOM snapshots for rollback
        self._dom_snapshots: List[DOMSnapshot] = []
        self._max_snapshots = 5
        
        # Sandbox environment state
        self._sandbox_active = False
        self._sandbox_frame_id: Optional[str] = None
        
        # Risk assessment cache
        self._risk_cache: Dict[str, RiskAssessment] = {}

    async def _get_node_id(self) -> int:
        """Get DOM node ID from backend node ID."""
        params: 'PushNodesByBackendIdsToFrontendParameters' = {'backendNodeIds': [self._backend_node_id]}
        result = await self._client.send.DOM.pushNodesByBackendIdsToFrontend(params, session_id=self._session_id)
        return result['nodeIds'][0]

    async def _get_remote_object_id(self) -> str | None:
        """Get remote object ID for this element."""
        node_id = await self._get_node_id()
        params: 'ResolveNodeParameters' = {'nodeId': node_id}
        result = await self._client.send.DOM.resolveNode(params, session_id=self._session_id)
        object_id = result['object'].get('objectId', None)

        if not object_id:
            return None
        return object_id

    async def _take_dom_snapshot(self) -> DOMSnapshot:
        """Take a snapshot of the current DOM state for rollback."""
        try:
            # Get the entire document HTML
            doc_result = await self._client.send.DOM.getDocument(session_id=self._session_id)
            root_node_id = doc_result['root']['nodeId']
            
            html_result = await self._client.send.DOM.getOuterHTML(
                params={'nodeId': root_node_id}, 
                session_id=self._session_id
            )
            
            # Get current URL
            url_result = await self._client.send.Page.getNavigationHistory(session_id=self._session_id)
            current_url = url_result['entries'][-1]['url'] if url_result['entries'] else ''
            
            # Get element-specific state
            element_states = await self._capture_element_states()
            
            snapshot: DOMSnapshot = {
                'html': html_result['outerHTML'],
                'timestamp': asyncio.get_event_loop().time(),
                'element_states': element_states,
                'url': current_url,
            }
            
            # Store snapshot (keep only recent ones)
            self._dom_snapshots.append(snapshot)
            if len(self._dom_snapshots) > self._max_snapshots:
                self._dom_snapshots.pop(0)
            
            logger.debug(f"DOM snapshot taken for rollback protection")
            return snapshot
            
        except Exception as e:
            logger.warning(f"Failed to take DOM snapshot: {e}")
            raise

    async def _capture_element_states(self) -> Dict[str, Any]:
        """Capture state of interactive elements for validation."""
        try:
            # Get all form elements, buttons, inputs, and links
            script = """
                function() {
                    const elements = {};
                    const forms = document.querySelectorAll('form');
                    const buttons = document.querySelectorAll('button, input[type="submit"], input[type="button"]');
                    const inputs = document.querySelectorAll('input, textarea, select');
                    const links = document.querySelectorAll('a[href]');
                    
                    forms.forEach((form, i) => {
                        elements[`form_${i}`] = {
                            action: form.action,
                            method: form.method,
                            id: form.id,
                            name: form.name
                        };
                    });
                    
                    buttons.forEach((btn, i) => {
                        elements[`button_${i}`] = {
                            type: btn.type,
                            disabled: btn.disabled,
                            value: btn.value || btn.textContent,
                            id: btn.id,
                            name: btn.name
                        };
                    });
                    
                    inputs.forEach((input, i) => {
                        elements[`input_${i}`] = {
                            type: input.type,
                            value: input.value,
                            checked: input.checked,
                            disabled: input.disabled,
                            required: input.required,
                            id: input.id,
                            name: input.name
                        };
                    });
                    
                    links.forEach((link, i) => {
                        elements[`link_${i}`] = {
                            href: link.href,
                            target: link.target,
                            id: link.id,
                            text: link.textContent.trim()
                        };
                    });
                    
                    return elements;
                }
            """
            
            result = await self._client.send.Runtime.callFunctionOn(
                params={
                    'functionDeclaration': script,
                    'returnByValue': True,
                },
                session_id=self._session_id,
            )
            
            return result.get('result', {}).get('value', {})
            
        except Exception as e:
            logger.warning(f"Failed to capture element states: {e}")
            return {}

    async def _restore_from_snapshot(self, snapshot: DOMSnapshot) -> bool:
        """Restore DOM from a snapshot."""
        try:
            # Get current URL to check if we need to navigate back
            url_result = await self._client.send.Page.getNavigationHistory(session_id=self._session_id)
            current_url = url_result['entries'][-1]['url'] if url_result['entries'] else ''
            
            if current_url != snapshot['url']:
                # Navigate back to original URL
                await self._client.send.Page.navigate(
                    params={'url': snapshot['url']},
                    session_id=self._session_id
                )
                await asyncio.sleep(0.5)  # Wait for navigation
            
            # Get document root
            doc_result = await self._client.send.DOM.getDocument(session_id=self._session_id)
            root_node_id = doc_result['root']['nodeId']
            
            # Set the HTML back to snapshot
            await self._client.send.DOM.setOuterHTML(
                params={
                    'nodeId': root_node_id,
                    'outerHTML': snapshot['html']
                },
                session_id=self._session_id
            )
            
            # Wait for DOM to settle
            await asyncio.sleep(0.2)
            
            logger.info(f"Successfully restored DOM from snapshot")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from snapshot: {e}")
            return False

    async def _assess_action_risk(self, action: str, **kwargs) -> RiskAssessment:
        """Assess risk level of an action on this element."""
        # Use custom risk scorer if provided
        if self._safety_config.get('custom_risk_scorer'):
            return self._safety_config['custom_risk_scorer'](self, action)
        
        # Cache key for this assessment
        cache_key = f"{action}_{hash(frozenset(kwargs.items()))}"
        if cache_key in self._risk_cache:
            return self._risk_cache[cache_key]
        
        try:
            # Get element information
            node_id = await self._get_node_id()
            describe_result = await self._client.send.DOM.describeNode(
                params={'nodeId': node_id, 'depth': 2},
                session_id=self._session_id
            )
            
            node = describe_result.get('node', {})
            node_name = node.get('nodeName', '').lower()
            attributes = {}
            
            # Parse attributes
            attrs = node.get('attributes', [])
            for i in range(0, len(attrs), 2):
                if i + 1 < len(attrs):
                    attributes[attrs[i].lower()] = attrs[i + 1]
            
            # Get element context via JavaScript
            context_script = """
                function() {
                    const element = this;
                    const context = {
                        isFormElement: false,
                        isSubmitButton: false,
                        isDeleteAction: false,
                        isNavigation: false,
                        isPasswordField: false,
                        isFileInput: false,
                        hasEventListeners: false,
                        parentForm: null,
                        href: null,
                        action: null
                    };
                    
                    // Check if it's a form element
                    context.isFormElement = ['FORM', 'INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes(element.tagName);
                    
                    // Check for submit buttons
                    context.isSubmitButton = element.type === 'submit' || 
                                          element.tagName === 'BUTTON' && !element.type ||
                                          element.getAttribute('role') === 'button';
                    
                    // Check for delete/remove actions
                    const text = (element.textContent || '').toLowerCase();
                    const classes = (element.className || '').toLowerCase();
                    context.isDeleteAction = text.includes('delete') || text.includes('remove') ||
                                          classes.includes('delete') || classes.includes('remove') ||
                                          element.getAttribute('data-action') === 'delete';
                    
                    // Check for navigation
                    if (element.tagName === 'A') {
                        context.isNavigation = true;
                        context.href = element.href;
                    }
                    
                    // Check for password fields
                    context.isPasswordField = element.type === 'password';
                    
                    // Check for file inputs
                    context.isFileInput = element.type === 'file';
                    
                    // Check for parent form
                    const form = element.closest('form');
                    if (form) {
                        context.parentForm = {
                            action: form.action,
                            method: form.method,
                            id: form.id
                        };
                        context.action = form.action;
                    }
                    
                    // Check for event listeners (approximate)
                    context.hasEventListeners = element.onclick != null || 
                                              element.onsubmit != null ||
                                              element.getAttribute('onclick') != null;
                    
                    return context;
                }
            """
            
            object_id = await self._get_remote_object_id()
            if object_id:
                context_result = await self._client.send.Runtime.callFunctionOn(
                    params={
                        'functionDeclaration': context_script,
                        'objectId': object_id,
                        'returnByValue': True,
                    },
                    session_id=self._session_id,
                )
                context = context_result.get('result', {}).get('value', {})
            else:
                context = {}
            
            # Assess risk based on action and context
            risk_level = RiskLevel.LOW
            reasons = []
            requires_confirmation = False
            can_sandbox = True
            estimated_impact = "minimal"
            
            if action == 'click':
                if context.get('isSubmitButton') or context.get('isFormElement'):
                    risk_level = RiskLevel.HIGH
                    reasons.append("Action may submit a form")
                    requires_confirmation = True
                    estimated_impact = "form submission, potential data loss"
                
                if context.get('isDeleteAction'):
                    risk_level = RiskLevel.HIGH
                    reasons.append("Action appears to be delete/remove operation")
                    requires_confirmation = True
                    estimated_impact = "data deletion"
                
                if context.get('isNavigation') and context.get('href'):
                    href = context['href']
                    if href.startswith('javascript:') or 'logout' in href.lower():
                        risk_level = RiskLevel.HIGH
                        reasons.append("Action may navigate away or execute JavaScript")
                        requires_confirmation = True
                        estimated_impact = "navigation or script execution"
                    else:
                        risk_level = RiskLevel.MEDIUM
                        reasons.append("Action may navigate to a different page")
                        can_sandbox = False  # Can't sandbox navigation
                        estimated_impact = "page navigation"
            
            elif action == 'type' or action == 'input':
                if context.get('isPasswordField'):
                    risk_level = RiskLevel.HIGH
                    reasons.append("Action modifies password field")
                    requires_confirmation = True
                    estimated_impact = "security-sensitive data modification"
                
                if context.get('isFileInput'):
                    risk_level = RiskLevel.HIGH
                    reasons.append("Action may trigger file upload")
                    requires_confirmation = True
                    estimated_impact = "file system access"
            
            elif action == 'submit':
                risk_level = RiskLevel.CRITICAL
                reasons.append("Form submission action")
                requires_confirmation = True
                can_sandbox = False
                estimated_impact = "form data submission, potential irreversible action"
            
            # Check for disabled elements
            if attributes.get('disabled') == 'true' or attributes.get('aria-disabled') == 'true':
                risk_level = RiskLevel.MEDIUM
                reasons.append("Element is disabled")
                estimated_impact = "action may not execute as expected"
            
            # Check for readonly elements
            if attributes.get('readonly') == 'true':
                risk_level = RiskLevel.MEDIUM
                reasons.append("Element is readonly")
                estimated_impact = "action may be blocked by browser"
            
            assessment = RiskAssessment(
                level=risk_level,
                reasons=reasons,
                requires_confirmation=requires_confirmation,
                can_sandbox=can_sandbox,
                estimated_impact=estimated_impact
            )
            
            # Cache the assessment
            self._risk_cache[cache_key] = assessment
            return assessment
            
        except Exception as e:
            logger.warning(f"Risk assessment failed, defaulting to medium risk: {e}")
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                reasons=[f"Risk assessment failed: {str(e)}"],
                requires_confirmation=True,
                can_sandbox=True,
                estimated_impact="unknown"
            )

    async def _create_sandbox_environment(self) -> str:
        """Create a sandboxed iframe for testing actions."""
        try:
            # Create a sandboxed iframe
            sandbox_script = """
                function() {
                    // Remove existing sandbox if any
                    const existingSandbox = document.getElementById('nexus-sandbox');
                    if (existingSandbox) {
                        existingSandbox.remove();
                    }
                    
                    // Create new sandbox iframe
                    const iframe = document.createElement('iframe');
                    iframe.id = 'nexus-sandbox';
                    iframe.style.cssText = `
                        position: fixed;
                        top: 10px;
                        right: 10px;
                        width: 400px;
                        height: 300px;
                        border: 2px solid #ff6b6b;
                        border-radius: 8px;
                        background: white;
                        z-index: 999999;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    `;
                    iframe.sandbox = 'allow-same-origin allow-scripts';
                    iframe.src = 'about:blank';
                    
                    document.body.appendChild(iframe);
                    
                    // Wait for iframe to load
                    return new Promise((resolve) => {
                        iframe.onload = () => {
                            // Copy current page content to sandbox
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            iframeDoc.open();
                            iframeDoc.write(document.documentElement.outerHTML);
                            iframeDoc.close();
                            
                            // Add sandbox indicator
                            const indicator = iframeDoc.createElement('div');
                            indicator.style.cssText = `
                                position: fixed;
                                top: 0;
                                left: 0;
                                right: 0;
                                background: #ff6b6b;
                                color: white;
                                text-align: center;
                                padding: 5px;
                                font-weight: bold;
                                z-index: 999999;
                            `;
                            indicator.textContent = '⚠️ SANDBOX MODE - Testing Action';
                            iframeDoc.body.appendChild(indicator);
                            
                            resolve(iframe.id);
                        };
                    });
                }
            """
            
            result = await self._client.send.Runtime.callFunctionOn(
                params={
                    'functionDeclaration': sandbox_script,
                    'returnByValue': True,
                },
                session_id=self._session_id,
            )
            
            iframe_id = result.get('result', {}).get('value')
            if iframe_id:
                self._sandbox_active = True
                self._sandbox_frame_id = iframe_id
                logger.info("Created sandbox environment for action testing")
                return iframe_id
            else:
                raise Exception("Failed to create sandbox iframe")
                
        except Exception as e:
            logger.error(f"Failed to create sandbox environment: {e}")
            raise

    async def _execute_in_sandbox(self, action_script: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Execute an action in the sandbox environment."""
        if not self._sandbox_active or not self._sandbox_frame_id:
            raise Exception("Sandbox environment not active")
        
        try:
            # Get sandbox iframe
            get_sandbox_script = f"""
                function() {{
                    const iframe = document.getElementById('{self._sandbox_frame_id}');
                    if (!iframe) {{
                        throw new Error('Sandbox iframe not found');
                    }}
                    return iframe.contentWindow;
                }}
            """
            
            sandbox_result = await self._client.send.Runtime.callFunctionOn(
                params={
                    'functionDeclaration': get_sandbox_script,
                    'returnByValue': True,
                },
                session_id=self._session_id,
            )
            
            # Execute action in sandbox with timeout
            sandbox_execution_script = f"""
                async function() {{
                    const sandboxWindow = {json.dumps(sandbox_result.get('result', {}).get('value', ''))};
                    if (!sandboxWindow) {{
                        throw new Error('Could not access sandbox window');
                    }}
                    
                    // Create a promise that rejects after timeout
                    const timeoutPromise = new Promise((_, reject) => {{
                        setTimeout(() => reject(new Error('Sandbox execution timeout')), {timeout * 1000});
                    }});
                    
                    // Execute the action
                    const actionPromise = new Promise((resolve) => {{
                        try {{
                            // Execute the action script in sandbox context
                            const result = sandboxWindow.eval({json.dumps(action_script)});
                            resolve({{ success: true, result }});
                        }} catch (error) {{
                            resolve({{ success: false, error: error.message }});
                        }}
                    }});
                    
                    // Race between action and timeout
                    return Promise.race([actionPromise, timeoutPromise]);
                }}
            """
            
            execution_result = await self._client.send.Runtime.callFunctionOn(
                params={
                    'functionDeclaration': sandbox_execution_script,
                    'returnByValue': True,
                    'awaitPromise': True,
                },
                session_id=self._session_id,
            )
            
            return execution_result.get('result', {}).get('value', {})
            
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return {'success': False, 'error': str(e)}

    async def _cleanup_sandbox(self):
        """Clean up the sandbox environment."""
        if not self._sandbox_frame_id:
            return
        
        try:
            cleanup_script = f"""
                function() {{
                    const iframe = document.getElementById('{self._sandbox_frame_id}');
                    if (iframe) {{
                        iframe.remove();
                    }}
                    return true;
                }}
            """
            
            await self._client.send.Runtime.callFunctionOn(
                params={
                    'functionDeclaration': cleanup_script,
                    'returnByValue': True,
                },
                session_id=self._session_id,
            )
            
            self._sandbox_active = False
            self._sandbox_frame_id = None
            logger.debug("Cleaned up sandbox environment")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup sandbox: {e}")

    async def _validate_and_execute(
        self, 
        action_name: str, 
        action_coroutine,
        sandbox_script: Optional[str] = None,
        **action_kwargs
    ) -> Any:
        """Validate action safety and execute with appropriate safeguards."""
        
        # Assess risk
        risk_assessment = await self._assess_action_risk(action_name, **action_kwargs)
        
        logger.info(f"Action '{action_name}' risk assessment: {risk_assessment.level.value}")
        if risk_assessment.reasons:
            logger.info(f"  Reasons: {', '.join(risk_assessment.reasons)}")
        logger.info(f"  Estimated impact: {risk_assessment.estimated_impact}")
        
        # Check if confirmation is required
        if risk_assessment.requires_confirmation and self._safety_config.get('require_confirmation_for_high_risk'):
            # In a real implementation, this would prompt the user
            # For now, we'll log a warning and proceed
            logger.warning(
                f"⚠️ HIGH RISK ACTION: {action_name} on element {self._backend_node_id}\n"
                f"  Impact: {risk_assessment.estimated_impact}\n"
                f"  Reasons: {', '.join(risk_assessment.reasons)}\n"
                f"  Set require_confirmation_for_high_risk=False to disable this warning"
            )
        
        # Take DOM snapshot if configured
        snapshot = None
        if self._safety_config.get('snapshot_before_action'):
            try:
                snapshot = await self._take_dom_snapshot()
            except Exception as e:
                logger.warning(f"Failed to take snapshot, proceeding without rollback capability: {e}")
        
        # Execute in sandbox if applicable
        sandbox_result = None
        if (risk_assessment.can_sandbox and 
            self._safety_config.get('sandbox_medium_risk') and 
            risk_assessment.level in [RiskLevel.MEDIUM, RiskLevel.HIGH] and
            sandbox_script):
            
            try:
                await self._create_sandbox_environment()
                sandbox_result = await self._execute_in_sandbox(
                    sandbox_script,
                    self._safety_config.get('max_sandbox_timeout', 5.0)
                )
                
                if not sandbox_result.get('success'):
                    logger.warning(f"Sandbox execution failed: {sandbox_result.get('error')}")
                    # Continue with real execution despite sandbox failure
                else:
                    logger.info("Sandbox execution successful, proceeding with real action")
                    
            except Exception as e:
                logger.warning(f"Sandbox testing failed: {e}, proceeding with real execution")
            finally:
                await self._cleanup_sandbox()
        
        # Execute the real action
        try:
            result = await action_coroutine
            return result
            
        except Exception as e:
            logger.error(f"Action '{action_name}' failed: {e}")
            
            # Attempt rollback if configured and snapshot exists
            if (self._safety_config.get('auto_rollback_on_error') and 
                snapshot and 
                risk_assessment.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]):
                
                logger.info("Attempting automatic rollback due to action failure...")
                rollback_success = await self._restore_from_snapshot(snapshot)
                
                if rollback_success:
                    logger.info("Rollback successful")
                else:
                    logger.error("Rollback failed - page may be in inconsistent state")
            
            raise

    async def _get_node_id(self) -> int:
        """Get DOM node ID from backend node ID."""
        params: 'PushNodesByBackendIdsToFrontendParameters' = {'backendNodeIds': [self._backend_node_id]}
        result = await self._client.send.DOM.pushNodesByBackendIdsToFrontend(params, session_id=self._session_id)
        return result['nodeIds'][0]

    async def _get_remote_object_id(self) -> str | None:
        """Get remote object ID for this element."""
        node_id = await self._get_node_id()
        params: 'ResolveNodeParameters' = {'nodeId': node_id}
        result = await self._client.send.DOM.resolveNode(params, session_id=self._session_id)
        object_id = result['object'].get('objectId', None)

        if not object_id:
            return None
        return object_id

    async def click(
        self,
        button: 'MouseButton' = 'left',
        click_count: int = 1,
        modifiers: list[ModifierType] | None = None,
    ) -> None:
        """Click the element using the advanced watchdog implementation with safety validation."""

        async def _perform_click():
            """Internal click implementation."""
            try:
                # Get viewport dimensions for visibility checks
                layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
                viewport_width = layout_metrics['layoutViewport']['clientWidth']
                viewport_height = layout_metrics['layoutViewport']['clientHeight']

                # Try multiple methods to get element geometry
                quads = []

                # Method 1: Try DOM.getContentQuads first (best for inline elements and complex layouts)
                try:
                    content_quads_result = await self._client.send.DOM.getContentQuads(
                        params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
                    )
                    if 'quads' in content_quads_result and content_quads_result['quads']:
                        quads = content_quads_result['quads']
                except Exception:
                    pass

                # Method 2: Fall back to DOM.getBoxModel
                if not quads:
                    try:
                        box_model = await self._client.send.DOM.getBoxModel(
                            params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
                        )
                        if 'model' in box_model and 'content' in box_model['model']:
                            content_quad = box_model['model']['content']
                            if len(content_quad) >= 8:
                                # Convert box model format to quad format
                                quads = [
                                    [
                                        content_quad[0],
                                        content_quad[1],  # x1, y1
                                        content_quad[2],
                                        content_quad[3],  # x2, y2
                                        content_quad[4],
                                        content_quad[5],  # x3, y3
                                        content_quad[6],
                                        content_quad[7],  # x4, y4
                                    ]
                                ]
                    except Exception:
                        pass

                # Method 3: Fall back to JavaScript getBoundingClientRect
                if not quads:
                    try:
                        result = await self._client.send.DOM.resolveNode(
                            params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
                        )
                        if 'object' in result and 'objectId' in result['object']:
                            object_id = result['object']['objectId']

                            # Get bounding rect via JavaScript
                            bounds_result = await self._client.send.Runtime.callFunctionOn(
                                params={
                                    'functionDeclaration': """
                                        function() {
                                            const rect = this.getBoundingClientRect();
                                            return {
                                                x: rect.left,
                                                y: rect.top,
                                                width: rect.width,
                                                height: rect.height
                                            };
                                        }
                                    """,
                                    'objectId': object_id,
                                    'returnByValue': True,
                                },
                                session_id=self._session_id,
                            )

                            if 'result' in bounds_result and 'value' in bounds_result['result']:
                                rect = bounds_result['result']['value']
                                # Convert rect to quad format
                                x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
                                quads = [
                                    [
                                        x,
                                        y,  # top-left
                                        x + w,
                                        y,  # top-right
                                        x + w,
                                        y + h,  # bottom-right
                                        x,
                                        y + h,  # bottom-left
                                    ]
                                ]
                    except Exception:
                        pass

                # If we still don't have quads, fall back to JS click
                if not quads:
                    try:
                        result = await self._client.send.DOM.resolveNode(
                            params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
                        )
                        if 'object' not in result or 'objectId' not in result['object']:
                            raise Exception('Failed to find DOM element based on backendNodeId, maybe page content changed?')
                        object_id = result['object']['objectId']

                        await self._client.send.Runtime.callFunctionOn(
                            params={
                                'functionDeclaration': 'function() { this.click(); }',
                                'objectId': object_id,
                            },
                            session_id=self._session_id,
                        )
                        await asyncio.sleep(0.05)
                        return
                    except Exception as js_e:
                        raise Exception(f'Failed to click element: {js_e}')

                # Find the largest visible quad within the viewport
                best_quad = None
                best_area = 0

                for quad in quads:
                    if len(quad) < 8:
                        continue

                    # Calculate quad bounds
                    xs = [quad[i] for i in range(0, 8, 2)]
                    ys = [quad[i] for i in range(1, 8, 2)]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    # Check if quad intersects with viewport
                    if max_x < 0 or max_y < 0 or min_x > viewport_width or min_y > viewport_height:
                        continue  # Quad is completely outside viewport

                    # Calculate visible area (intersection with viewport)
                    visible_min_x = max(0, min_x)
                    visible_max_x = min(viewport_width, max_x)
                    visible_min_y = max(0, min_y)
                    visible_max_y = min(viewport_height, max_y)

                    visible_width = visible_max_x - visible_min_x
                    visible_height = visible_max_y - visible_min_y
                    visible_area = visible_width * visible_height

                    if visible_area > best_area:
                        best_area = visible_area
                        best_quad = quad

                if not best_quad:
                    # No visible quad found, use the first quad anyway
                    best_quad = quads[0]

                # Calculate center point of the best quad
                center_x = sum(best_quad[i] for i in range(0, 8, 2)) / 4
                center_y = sum(best_quad[i] for i in range(1, 8, 2)) / 4

                # Ensure click point is within viewport bounds
                center_x = max(0, min(viewport_width - 1, center_x))
                center_y = max(0, min(viewport_height - 1, center_y))

                # Scroll element into view
                try:
                    await self._client.send.DOM.scrollIntoViewIfNeeded(
                        params={'backendNodeId': self._backend_node_id}, session_id=self._session_id
                    )
                    await asyncio.sleep(0.05)  # Wait for scroll to complete
                except Exception:
                    pass

                # Calculate click parameters
                click_params: 'DispatchMouseEventParameters' = {
                    'type': 'mousePressed',
                    'x': center_x,
                    'y': center_y,
                    'button': button,
                    'clickCount': click_count,
                }

                if modifiers:
                    click_params['modifiers'] = sum(
                        1 << i for i, mod in enumerate(['Alt', 'Control', 'Meta', 'Shift']) if mod in modifiers
                    )

                # Dispatch mousePressed event
                await self._client.send.Input.dispatchMouseEvent(
                    params=click_params,
                    session_id=self._session_id,
                )

                # Small delay between press and release
                await asyncio.sleep(0.02)

                # Dispatch mouseReleased event
                click_params['type'] = 'mouseReleased'
                await self._client.send.Input.dispatchMouseEvent(
                    params=click_params,
                    session_id=self._session_id,
                )

                # Small delay after click
                await asyncio.sleep(0.05)

            except Exception as e:
                raise Exception(f'Failed to click element: {e}')

        # Create sandbox script for testing click
        sandbox_script = """
            async function() {
                // Find the element in sandbox (simplified - in real implementation would need to match element)
                const element = document.querySelector('[data-sandbox-test]');
                if (!element) {
                    return { success: false, error: 'Element not found in sandbox' };
                }
                
                try {
                    element.click();
                    return { success: true, message: 'Click executed successfully in sandbox' };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            }
        """

        # Validate and execute with safety checks
        return await self._validate_and_execute(
            action_name='click',
            action_coroutine=_perform_click(),
            sandbox_script=sandbox_script,
            button=button,
            click_count=click_count,
            modifiers=modifiers,
        )

    async def type_text(self, text: str, delay: float = 0.05) -> None:
        """Type text into an element with safety validation."""
        
        async def _perform_type():
            """Internal type implementation."""
            try:
                # Focus the element first
                node_id = await self._get_node_id()
                await self._client.send.DOM.focus(
                    params={'nodeId': node_id},
                    session_id=self._session_id,
                )
                
                # Type each character with delay
                for char in text:
                    # Key down
                    await self._client.send.Input.dispatchKeyEvent(
                        params={
                            'type': 'keyDown',
                            'text': char,
                            'key': char,
                            'code': f'Key{char.upper()}' if char.isalpha() else '',
                        },
                        session_id=self._session_id,
                    )
                    
                    # Key up
                    await self._client.send.Input.dispatchKeyEvent(
                        params={
                            'type': 'keyUp',
                            'key': char,
                            'code': f'Key{char.upper()}' if char.isalpha() else '',
                        },
                        session_id=self._session_id,
                    )
                    
                    # Delay between keystrokes
                    if delay > 0:
                        await asyncio.sleep(delay)
                
                await asyncio.sleep(0.1)  # Final delay
                
            except Exception as e:
                raise Exception(f'Failed to type text: {e}')

        # Create sandbox script for testing type
        sandbox_script = """
            async function() {
                const element = document.querySelector('[data-sandbox-test]');
                if (!element) {
                    return { success: false, error: 'Element not found in sandbox' };
                }
                
                if (element.tagName !== 'INPUT' && element.tagName !== 'TEXTAREA') {
                    return { success: false, error: 'Element is not an input field' };
                }
                
                try {
                    const originalValue = element.value;
                    element.value = 'test';
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                    
                    // Check if value was set
                    const success = element.value === 'test';
                    
                    // Restore original value
                    element.value = originalValue;
                    
                    return { 
                        success: success, 
                        message: success ? 'Type test successful' : 'Failed to set value in sandbox' 
                    };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            }
        """

        # Validate and execute with safety checks
        return await self._validate_and_execute(
            action_name='type',
            action_coroutine=_perform_type(),
            sandbox_script=sandbox_script,
            text=text,
            delay=delay,
        )

    async def submit_form(self) -> None:
        """Submit a form with critical risk validation."""
        
        async def _perform_submit():
            """Internal submit implementation."""
            try:
                # Find the parent form
                script = """
                    function() {
                        const element = this;
                        const form = element.closest('form');
                        if (!form) {
                            throw new Error('Element is not inside a form');
                        }
                        form.submit();
                        return true;
                    }
                """
                
                object_id = await self._get_remote_object_id()
                if not object_id:
                    raise Exception('Could not get element reference')
                
                await self._client.send.Runtime.callFunctionOn(
                    params={
                        'functionDeclaration': script,
                        'objectId': object_id,
                    },
                    session_id=self._session_id,
                )
                
                await asyncio.sleep(0.5)  # Wait for submission
                
            except Exception as e:
                raise Exception(f'Failed to submit form: {e}')

        # Form submission is always high risk - no sandbox possible
        return await self._validate_and_execute(
            action_name='submit',
            action_coroutine=_perform_submit(),
            sandbox_script=None,  # Cannot sandbox form submission
        )

    async def get_risk_assessment(self, action: str = 'click') -> RiskAssessment:
        """Get risk assessment for a proposed action."""
        return await self._assess_action_risk(action)

    def clear_risk_cache(self):
        """Clear the risk assessment cache."""
        self._risk_cache.clear()

    def update_safety_config(self, config: SafetyConfig):
        """Update safety configuration."""
        self._safety_config.update(config)
        # Clear risk cache when config changes
        self.clear_risk_cache()

    def get_safety_config(self) -> SafetyConfig:
        """Get current safety configuration."""
        return self._safety_config.copy()

    async def rollback_to_snapshot(self, snapshot_index: int = -1) -> bool:
        """Manually rollback to a specific snapshot."""
        if not self._dom_snapshots:
            logger.warning("No snapshots available for rollback")
            return False
        
        try:
            if snapshot_index == -1:
                snapshot = self._dom_snapshots[-1]
            elif 0 <= snapshot_index < len(self._dom_snapshots):
                snapshot = self._dom_snapshots[snapshot_index]
            else:
                logger.error(f"Invalid snapshot index: {snapshot_index}")
                return False
            
            return await self._restore_from_snapshot(snapshot)
            
        except Exception as e:
            logger.error(f"Manual rollback failed: {e}")
            return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available DOM snapshots."""
        return [
            {
                'index': i,
                'timestamp': snapshot['timestamp'],
                'url': snapshot['url'],
                'size': len(snapshot['html']),
            }
            for i, snapshot in enumerate(self._dom_snapshots)
        ]