"""Mouse class for mouse operations."""

import asyncio
import math
import random
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from cdp_use.cdp.input.commands import DispatchMouseEventParameters, SynthesizeScrollGestureParameters
    from cdp_use.cdp.input.types import MouseButton
    from nexus.browser.session import BrowserSession


class Mouse:
    """Mouse operations for a target with human-like behavior simulation."""

    def __init__(self, browser_session: 'BrowserSession', session_id: str | None = None, target_id: str | None = None):
        self._browser_session = browser_session
        self._client = browser_session.cdp_client
        self._session_id = session_id
        self._target_id = target_id
        
        # Human behavior simulation parameters
        self._current_x = 0
        self._current_y = 0
        self._min_move_delay = 0.01  # Minimum delay between movement steps (seconds)
        self._max_move_delay = 0.05  # Maximum delay between movement steps
        self._min_click_delay = 0.05  # Minimum delay before click
        self._max_click_delay = 0.15  # Maximum delay before click
        self._min_scroll_delay = 0.02  # Minimum delay between scroll steps
        self._max_scroll_delay = 0.08  # Maximum delay between scroll steps
        
        # Bezier curve control point variation for natural movement
        self._control_point_variation = 0.3
        
        # Typing pattern simulation (for future use)
        self._typing_speed_variation = 0.2
        self._min_typing_delay = 0.05
        self._max_typing_delay = 0.15

    def _generate_bezier_curve(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                               steps: int = 10) -> List[Tuple[float, float]]:
        """Generate points along a Bezier curve for natural mouse movement."""
        # Calculate distance for control point offset
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        control_offset = distance * self._control_point_variation
        
        # Generate random control points for natural curve
        control1_x = start_x + (end_x - start_x) * 0.3 + random.uniform(-control_offset, control_offset)
        control1_y = start_y + (end_y - start_y) * 0.3 + random.uniform(-control_offset, control_offset)
        control2_x = start_x + (end_x - start_x) * 0.7 + random.uniform(-control_offset, control_offset)
        control2_y = start_y + (end_y - start_y) * 0.7 + random.uniform(-control_offset, control_offset)
        
        points = []
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bezier curve formula
            x = (1-t)**3 * start_x + 3*(1-t)**2*t * control1_x + 3*(1-t)*t**2 * control2_x + t**3 * end_x
            y = (1-t)**3 * start_y + 3*(1-t)**2*t * control1_y + 3*(1-t)*t**2 * control2_y + t**3 * end_y
            points.append((x, y))
        
        return points

    async def _human_delay(self, min_delay: float, max_delay: float) -> None:
        """Add a human-like random delay."""
        delay = random.uniform(min_delay, max_delay)
        await asyncio.sleep(delay)

    async def _move_human_like(self, x: int, y: int, steps: int = 0) -> None:
        """Move mouse with human-like Bezier curve movement."""
        # Calculate steps based on distance if not provided
        if steps == 0:
            distance = math.sqrt((x - self._current_x)**2 + (y - self._current_y)**2)
            steps = max(5, min(20, int(distance / 50)))  # Adaptive steps based on distance
        
        # Generate Bezier curve points
        curve_points = self._generate_bezier_curve(self._current_x, self._current_y, x, y, steps)
        
        # Move through each point with human-like timing
        for i, (point_x, point_y) in enumerate(curve_points[1:], 1):  # Skip starting point
            # Add slight random variation to each point for imperfection
            varied_x = point_x + random.uniform(-2, 2)
            varied_y = point_y + random.uniform(-2, 2)
            
            # Move to point
            params: 'DispatchMouseEventParameters' = {
                'type': 'mouseMoved', 
                'x': varied_x, 
                'y': varied_y
            }
            await self._client.send.Input.dispatchMouseEvent(params, session_id=self._session_id)
            
            # Update current position
            self._current_x = varied_x
            self._current_y = varied_y
            
            # Add human-like delay between movements (except for last point)
            if i < len(curve_points) - 1:
                await self._human_delay(self._min_move_delay, self._max_move_delay)
        
        # Final move to exact target with slight delay
        await self._human_delay(0.005, 0.02)
        params: 'DispatchMouseEventParameters' = {
            'type': 'mouseMoved', 
            'x': x, 
            'y': y
        }
        await self._client.send.Input.dispatchMouseEvent(params, session_id=self._session_id)
        self._current_x = x
        self._current_y = y

    async def click(self, x: int, y: int, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
        """Click at the specified coordinates with human-like behavior."""
        # Move to target with human-like movement
        await self._move_human_like(x, y)
        
        # Random delay before clicking (human reaction time)
        await self._human_delay(self._min_click_delay, self._max_click_delay)
        
        # Mouse press
        press_params: 'DispatchMouseEventParameters' = {
            'type': 'mousePressed',
            'x': x,
            'y': y,
            'button': button,
            'clickCount': click_count,
        }
        await self._client.send.Input.dispatchMouseEvent(
            press_params,
            session_id=self._session_id,
        )
        
        # Random delay between press and release (human click duration)
        await self._human_delay(0.05, 0.12)
        
        # Mouse release
        release_params: 'DispatchMouseEventParameters' = {
            'type': 'mouseReleased',
            'x': x,
            'y': y,
            'button': button,
            'clickCount': click_count,
        }
        await self._client.send.Input.dispatchMouseEvent(
            release_params,
            session_id=self._session_id,
        )
        
        # Small delay after click
        await self._human_delay(0.02, 0.08)

    async def down(self, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
        """Press mouse button down with human-like timing."""
        await self._human_delay(0.01, 0.05)
        
        params: 'DispatchMouseEventParameters' = {
            'type': 'mousePressed',
            'x': self._current_x,
            'y': self._current_y,
            'button': button,
            'clickCount': click_count,
        }
        await self._client.send.Input.dispatchMouseEvent(
            params,
            session_id=self._session_id,
        )

    async def up(self, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
        """Release mouse button with human-like timing."""
        await self._human_delay(0.01, 0.05)
        
        params: 'DispatchMouseEventParameters' = {
            'type': 'mouseReleased',
            'x': self._current_x,
            'y': self._current_y,
            'button': button,
            'clickCount': click_count,
        }
        await self._client.send.Input.dispatchMouseEvent(
            params,
            session_id=self._session_id,
        )

    async def move(self, x: int, y: int, steps: int = 0) -> None:
        """Move mouse to the specified coordinates with human-like behavior."""
        await self._move_human_like(x, y, steps)

    async def scroll(self, x: int = 0, y: int = 0, delta_x: int | None = None, delta_y: int | None = None) -> None:
        """Scroll the page with human-like behavior."""
        if not self._session_id:
            raise RuntimeError('Session ID is required for scroll operations')

        # Move mouse to scroll position with human-like movement
        if x > 0 or y > 0:
            await self._move_human_like(x, y)

        # Calculate total scroll distance
        total_dx = delta_x or 0
        total_dy = delta_y or 0
        
        # Break scroll into human-like steps with variable speed
        scroll_steps = max(3, min(10, int(abs(total_dy) / 100) + 1))
        
        # Variable scroll speed pattern (accelerates then decelerates)
        for step in range(scroll_steps):
            # Calculate progress (0 to 1)
            progress = step / (scroll_steps - 1) if scroll_steps > 1 else 1
            
            # Ease-in-out curve for natural scrolling
            if progress < 0.5:
                # Accelerating
                speed_factor = 2 * progress * progress
            else:
                # Decelerating
                speed_factor = 1 - 2 * (progress - 1) * (progress - 1)
            
            # Calculate step deltas with some randomness
            step_dx = total_dx * speed_factor / scroll_steps * random.uniform(0.8, 1.2)
            step_dy = total_dy * speed_factor / scroll_steps * random.uniform(0.8, 1.2)
            
            # Method 1: Try mouse wheel event
            try:
                layout_metrics = await self._client.send.Page.getLayoutMetrics(session_id=self._session_id)
                viewport_width = layout_metrics['layoutViewport']['clientWidth']
                viewport_height = layout_metrics['layoutViewport']['clientHeight']
                
                scroll_x = x if x > 0 else viewport_width / 2
                scroll_y = y if y > 0 else viewport_height / 2
                
                await self._client.send.Input.dispatchMouseEvent(
                    params={
                        'type': 'mouseWheel',
                        'x': scroll_x,
                        'y': scroll_y,
                        'deltaX': step_dx,
                        'deltaY': step_dy,
                    },
                    session_id=self._session_id,
                )
                
                # Human-like delay between scroll steps
                if step < scroll_steps - 1:
                    await self._human_delay(self._min_scroll_delay, self._max_scroll_delay)
                
            except Exception:
                # Method 2: Fallback to synthesizeScrollGesture
                try:
                    params: 'SynthesizeScrollGestureParameters' = {
                        'x': x, 
                        'y': y, 
                        'xDistance': step_dx, 
                        'yDistance': step_dy
                    }
                    await self._client.send.Input.synthesizeScrollGesture(
                        params,
                        session_id=self._session_id,
                    )
                    
                    if step < scroll_steps - 1:
                        await self._human_delay(self._min_scroll_delay, self._max_scroll_delay)
                        
                except Exception:
                    # Method 3: JavaScript fallback
                    scroll_js = f'window.scrollBy({step_dx}, {step_dy})'
                    await self._client.send.Runtime.evaluate(
                        params={'expression': scroll_js, 'returnByValue': True},
                        session_id=self._session_id,
                    )
                    
                    if step < scroll_steps - 1:
                        await self._human_delay(self._min_scroll_delay, self._max_scroll_delay)
        
        # Micro-pause after scrolling completes
        await self._human_delay(0.05, 0.15)

    async def move_and_click(self, x: int, y: int, button: 'MouseButton' = 'left', click_count: int = 1) -> None:
        """Convenience method for moving and clicking with human-like behavior."""
        await self.click(x, y, button, click_count)

    def set_human_behavior_params(self, 
                                 move_delay_range: Tuple[float, float] = (0.01, 0.05),
                                 click_delay_range: Tuple[float, float] = (0.05, 0.15),
                                 scroll_delay_range: Tuple[float, float] = (0.02, 0.08),
                                 control_point_variation: float = 0.3) -> None:
        """Configure human behavior simulation parameters."""
        self._min_move_delay, self._max_move_delay = move_delay_range
        self._min_click_delay, self._max_click_delay = click_delay_range
        self._min_scroll_delay, self._max_scroll_delay = scroll_delay_range
        self._control_point_variation = control_point_variation

    def update_position(self, x: int, y: int) -> None:
        """Update the current mouse position (for synchronization)."""
        self._current_x = x
        self._current_y = y