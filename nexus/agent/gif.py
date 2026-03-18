from __future__ import annotations

import base64
import io
import json
import logging
import os
import platform
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nexus.agent.views import AgentHistoryList
from nexus.browser.views import PLACEHOLDER_4PX_SCREENSHOT
from nexus.config import CONFIG

if TYPE_CHECKING:
    from PIL import Image, ImageFont

logger = logging.getLogger(__name__)


class VisualDebugger:
    """Visual debugging and replay system with ring buffer for recording agent actions."""
    
    def __init__(self, max_buffer_size: int = 100):
        self.max_buffer_size = max_buffer_size
        self.ring_buffer = deque(maxlen=max_buffer_size)
        self.current_step = 0
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        
    def record_step(
        self,
        step_number: int,
        action: str,
        url: str,
        screenshot_before: Optional[str] = None,
        screenshot_after: Optional[str] = None,
        dom_snapshot: Optional[str] = None,
        dom_diff: Optional[Dict[str, Any]] = None,
        network_requests: Optional[List[Dict[str, Any]]] = None,
        console_logs: Optional[List[Dict[str, Any]]] = None,
        agent_reasoning: Optional[str] = None,
        timestamp: Optional[float] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a step in the ring buffer with all debugging information."""
        
        if timestamp is None:
            timestamp = time.time()
            
        step_data = {
            "step_number": step_number,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "action": action,
            "url": url,
            "screenshot_before": screenshot_before,
            "screenshot_after": screenshot_after,
            "dom_snapshot": dom_snapshot,
            "dom_diff": dom_diff,
            "network_requests": network_requests or [],
            "console_logs": console_logs or [],
            "agent_reasoning": agent_reasoning,
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
        }
        
        self.ring_buffer.append(step_data)
        self.current_step = step_number
        
    def get_step(self, step_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific step from the ring buffer."""
        for step in self.ring_buffer:
            if step["step_number"] == step_number:
                return step
        return None
    
    def get_all_steps(self) -> List[Dict[str, Any]]:
        """Get all steps from the ring buffer in order."""
        return list(self.ring_buffer)
    
    def get_failed_steps(self) -> List[Dict[str, Any]]:
        """Get all failed steps from the ring buffer."""
        return [step for step in self.ring_buffer if not step["success"]]
    
    def generate_debug_bundle(
        self,
        output_path: str = "debug_bundle.zip",
        include_screenshots: bool = True,
        include_dom: bool = True,
        include_network: bool = True,
        include_console: bool = True,
        include_reasoning: bool = True,
    ) -> str:
        """Generate a shareable debug bundle with all recorded data."""
        
        bundle_dir = Path(f"debug_bundle_{self.session_id}")
        bundle_dir.mkdir(exist_ok=True)
        
        # Create manifest
        manifest = {
            "session_id": self.session_id,
            "session_start": self.session_start_time.isoformat(),
            "total_steps": len(self.ring_buffer),
            "max_buffer_size": self.max_buffer_size,
            "generated_at": datetime.now().isoformat(),
        }
        
        # Save manifest
        with open(bundle_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Save all steps as JSON
        steps_data = self.get_all_steps()
        with open(bundle_dir / "steps.json", "w") as f:
            json.dump(steps_data, f, indent=2, default=str)
        
        # Save screenshots
        if include_screenshots:
            screenshots_dir = bundle_dir / "screenshots"
            screenshots_dir.mkdir(exist_ok=True)
            
            for step in steps_data:
                if step["screenshot_before"]:
                    self._save_screenshot(
                        step["screenshot_before"],
                        screenshots_dir / f"step_{step['step_number']:04d}_before.png"
                    )
                if step["screenshot_after"]:
                    self._save_screenshot(
                        step["screenshot_after"],
                        screenshots_dir / f"step_{step['step_number']:04d}_after.png"
                    )
        
        # Save DOM snapshots
        if include_dom:
            dom_dir = bundle_dir / "dom_snapshots"
            dom_dir.mkdir(exist_ok=True)
            
            for step in steps_data:
                if step["dom_snapshot"]:
                    with open(dom_dir / f"step_{step['step_number']:04d}_dom.html", "w") as f:
                        f.write(step["dom_snapshot"])
                if step["dom_diff"]:
                    with open(dom_dir / f"step_{step['step_number']:04d}_diff.json", "w") as f:
                        json.dump(step["dom_diff"], f, indent=2)
        
        # Save network requests
        if include_network:
            network_dir = bundle_dir / "network"
            network_dir.mkdir(exist_ok=True)
            
            for step in steps_data:
                if step["network_requests"]:
                    with open(network_dir / f"step_{step['step_number']:04d}_network.json", "w") as f:
                        json.dump(step["network_requests"], f, indent=2, default=str)
        
        # Save console logs
        if include_console:
            console_dir = bundle_dir / "console"
            console_dir.mkdir(exist_ok=True)
            
            for step in steps_data:
                if step["console_logs"]:
                    with open(console_dir / f"step_{step['step_number']:04d}_console.json", "w") as f:
                        json.dump(step["console_logs"], f, indent=2, default=str)
        
        # Save agent reasoning
        if include_reasoning:
            reasoning_dir = bundle_dir / "reasoning"
            reasoning_dir.mkdir(exist_ok=True)
            
            for step in steps_data:
                if step["agent_reasoning"]:
                    with open(reasoning_dir / f"step_{step['step_number']:04d}_reasoning.txt", "w") as f:
                        f.write(step["agent_reasoning"])
        
        # Create HTML viewer
        self._create_html_viewer(bundle_dir)
        
        # Create ZIP file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in bundle_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(bundle_dir.parent))
        
        # Cleanup temporary directory
        import shutil
        shutil.rmtree(bundle_dir)
        
        logger.info(f"Debug bundle created at: {output_path}")
        return output_path
    
    def _save_screenshot(self, screenshot_b64: str, output_path: Path) -> None:
        """Save a base64 screenshot to a file."""
        try:
            img_data = base64.b64decode(screenshot_b64)
            with open(output_path, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            logger.warning(f"Failed to save screenshot: {e}")
    
    def _create_html_viewer(self, bundle_dir: Path) -> None:
        """Create an HTML viewer for the debug bundle."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Browser-Use Debug Viewer - {self.session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .step {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .step.success {{ border-left: 4px solid #4CAF50; }}
        .step.failed {{ border-left: 4px solid #f44336; }}
        .step-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
        .step-number {{ font-weight: bold; font-size: 1.2em; }}
        .step-time {{ color: #666; }}
        .step-action {{ font-family: monospace; background: #f5f5f5; padding: 5px; margin: 5px 0; }}
        .step-url {{ color: #1a73e8; word-break: break-all; }}
        .screenshots {{ display: flex; gap: 10px; margin: 10px 0; }}
        .screenshot {{ max-width: 300px; border: 1px solid #ddd; }}
        .tabs {{ display: flex; border-bottom: 1px solid #ddd; margin-bottom: 10px; }}
        .tab {{ padding: 10px 15px; cursor: pointer; border: 1px solid transparent; }}
        .tab.active {{ border: 1px solid #ddd; border-bottom: 1px solid white; margin-bottom: -1px; }}
        .tab-content {{ display: none; padding: 10px; border: 1px solid #ddd; border-top: none; }}
        .tab-content.active {{ display: block; }}
        .error {{ color: #f44336; background: #ffebee; padding: 10px; margin: 10px 0; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Browser-Use Debug Viewer</h1>
        <p>Session: {self.session_id} | Steps: {len(self.ring_buffer)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div id="steps-container">
            <!-- Steps will be loaded here -->
        </div>
    </div>
    
    <script>
        // Load steps data
        fetch('steps.json')
            .then(response => response.json())
            .then(steps => {
                const container = document.getElementById('steps-container');
                
                steps.forEach(step => {{
                    const stepDiv = document.createElement('div');
                    stepDiv.className = `step ${{step.success ? 'success' : 'failed'}}`;
                    
                    stepDiv.innerHTML = `
                        <div class="step-header">
                            <span class="step-number">Step ${{step.step_number}}</span>
                            <span class="step-time">${{step.datetime}}</span>
                        </div>
                        <div class="step-action">${{step.action}}</div>
                        <div class="step-url">${{step.url}}</div>
                        ${{step.error ? `<div class="error">Error: ${{step.error}}</div>` : ''}}
                        
                        <div class="tabs">
                            <div class="tab active" data-tab="screenshots-${{step.step_number}}">Screenshots</div>
                            <div class="tab" data-tab="network-${{step.step_number}}">Network</div>
                            <div class="tab" data-tab="console-${{step.step_number}}">Console</div>
                            <div class="tab" data-tab="reasoning-${{step.step_number}}">Reasoning</div>
                        </div>
                        
                        <div id="screenshots-${{step.step_number}}" class="tab-content active">
                            <div class="screenshots">
                                ${{step.screenshot_before ? `<img src="screenshots/step_${{String(step.step_number).padStart(4, '0')}}_before.png" class="screenshot" alt="Before">` : ''}}
                                ${{step.screenshot_after ? `<img src="screenshots/step_${{String(step.step_number).padStart(4, '0')}}_after.png" class="screenshot" alt="After">` : ''}}
                            </div>
                        </div>
                        
                        <div id="network-${{step.step_number}}" class="tab-content">
                            <pre>${{JSON.stringify(step.network_requests, null, 2)}}</pre>
                        </div>
                        
                        <div id="console-${{step.step_number}}" class="tab-content">
                            <pre>${{JSON.stringify(step.console_logs, null, 2)}}</pre>
                        </div>
                        
                        <div id="reasoning-${{step.step_number}}" class="tab-content">
                            <pre>${{step.agent_reasoning || 'No reasoning recorded'}}</pre>
                        </div>
                    `;
                    
                    container.appendChild(stepDiv);
                }});
                
                // Add tab switching functionality
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.addEventListener('click', function() {
                        const tabId = this.getAttribute('data-tab');
                        const stepNumber = tabId.split('-')[1];
                        
                        // Remove active class from all tabs and contents in this step
                        document.querySelectorAll(`[data-tab*="-${{stepNumber}}"]`).forEach(t => t.classList.remove('active'));
                        document.querySelectorAll(`[id*="-${{stepNumber}}"]`).forEach(c => c.classList.remove('active'));
                        
                        // Add active class to clicked tab and corresponding content
                        this.classList.add('active');
                        document.getElementById(tabId).classList.add('active');
                    });
                });
            });
    </script>
</body>
</html>
        """
        
        with open(bundle_dir / "viewer.html", "w") as f:
            f.write(html_content)
    
    def time_travel_debug(self, target_step: int) -> Optional[Dict[str, Any]]:
        """Jump to a specific step for time-travel debugging."""
        return self.get_step(target_step)
    
    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze failures in the recorded steps."""
        failed_steps = self.get_failed_steps()
        
        analysis = {
            "total_failures": len(failed_steps),
            "failure_rate": len(failed_steps) / len(self.ring_buffer) if self.ring_buffer else 0,
            "common_errors": {},
            "failed_actions": {},
            "failed_urls": {},
        }
        
        for step in failed_steps:
            # Count error types
            error = step.get("error", "Unknown error")
            analysis["common_errors"][error] = analysis["common_errors"].get(error, 0) + 1
            
            # Count failed actions
            action = step.get("action", "Unknown action")
            analysis["failed_actions"][action] = analysis["failed_actions"].get(action, 0) + 1
            
            # Count failed URLs
            url = step.get("url", "Unknown URL")
            analysis["failed_urls"][url] = analysis["failed_urls"].get(url, 0) + 1
        
        return analysis
    
    def get_reproduction_steps(self, step_number: int) -> List[Dict[str, Any]]:
        """Get all steps leading up to a specific step for bug reproduction."""
        reproduction_steps = []
        for step in self.ring_buffer:
            if step["step_number"] <= step_number:
                reproduction_steps.append(step)
            else:
                break
        return reproduction_steps


def decode_unicode_escapes_to_utf8(text: str) -> str:
    """Handle decoding any unicode escape sequences embedded in a string (needed to render non-ASCII languages like chinese or arabic in the GIF overlay text)"""

    if r'\u' not in text:
        # doesn't have any escape sequences that need to be decoded
        return text

    try:
        # Try to decode Unicode escape sequences
        return text.encode('latin1').decode('unicode_escape')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # logger.debug(f"Failed to decode unicode escape sequences while generating gif text: {text}")
        return text


def create_history_gif(
    task: str,
    history: AgentHistoryList,
    #
    output_path: str = 'agent_history.gif',
    duration: int = 3000,
    show_goals: bool = True,
    show_task: bool = True,
    show_logo: bool = False,
    font_size: int = 40,
    title_font_size: int = 56,
    goal_font_size: int = 44,
    margin: int = 40,
    line_spacing: float = 1.5,
    visual_debugger: Optional[VisualDebugger] = None,
) -> None:
    """Create a GIF from the agent's history with overlaid task and goal text."""
    if not history.history:
        logger.warning('No history to create GIF from')
        return

    from PIL import Image, ImageFont

    images = []

    # if history is empty, we can't create a gif
    if not history.history:
        logger.warning('No history to create GIF from')
        return

    # Get all screenshots from history (including None placeholders)
    screenshots = history.screenshots(return_none_if_not_screenshot=True)

    if not screenshots:
        logger.warning('No screenshots found in history')
        return

    # Find the first non-placeholder screenshot
    # A screenshot is considered a placeholder if:
    # 1. It's the exact 4px placeholder for about:blank pages, OR
    # 2. It comes from a new tab page (chrome://newtab/, about:blank, etc.)
    first_real_screenshot = None
    for screenshot in screenshots:
        if screenshot and screenshot != PLACEHOLDER_4PX_SCREENSHOT:
            first_real_screenshot = screenshot
            break

    if not first_real_screenshot:
        logger.warning('No valid screenshots found (all are placeholders or from new tab pages)')
        return

    # Try to load nicer fonts
    try:
        # Try different font options in order of preference
        # ArialUni is a font that comes with Office and can render most non-alphabet characters
        font_options = [
            'PingFang',
            'STHeiti Medium',
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',  # 黑体
            'SimSun',  # 宋体
            'Noto Sans CJK SC',  # 思源黑体
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'Helvetica',
            'Arial',
            'DejaVuSans',
            'Verdana',
        ]
        font_loaded = False

        for font_name in font_options:
            try:
                if platform.system() == 'Windows':
                    # Need to specify the abs font path on Windows
                    font_name = os.path.join(CONFIG.WIN_FONT_DIR, font_name + '.ttf')
                regular_font = ImageFont.truetype(font_name, font_size)
                title_font = ImageFont.truetype(font_name, title_font_size)
                font_loaded = True
                break
            except OSError:
                continue

        if not font_loaded:
            raise OSError('No preferred fonts found')

    except OSError:
        regular_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Load logo if requested
    logo = None
    if show_logo:
        try:
            logo = Image.open('./static/nexus.png')
            # Resize logo to be small (e.g., 40px height)
            logo_height = 150
            aspect_ratio = logo.width / logo.height
            logo_width = int(logo_height * aspect_ratio)
            logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        except Exception as e:
            logger.warning(f'Could not load logo: {e}')

    # Create task frame if requested
    if show_task and task:
        # Find the first non-placeholder screenshot for the task frame
        first_real_screenshot = None
        for item in history.history:
            screenshot_b64 = item.state.get_screenshot()
            if screenshot_b64 and screenshot_b64 != PLACEHOLDER_4PX_SCREENSHOT:
                first_real_screenshot = screenshot_b64
                break

        if first_real_screenshot:
            task_frame = _create_task_frame(
                task,
                first_real_screenshot,
                title_font,  # type: ignore
                regular_font,  # type: ignore
                logo,
                line_spacing,
            )
            images.append(task_frame)
        else:
            logger.warning('No real screenshots found for task frame, skipping task frame')

    # Process each history item with its corresponding screenshot
    for i, (item, screenshot) in enumerate(zip(history.history, screenshots), 1):
        if not screenshot:
            continue

        # Skip placeholder screenshots from about:blank pages
        # These are 4x4 white PNGs encoded as a specific base64 string
        if screenshot == PLACEHOLDER_4PX_SCREENSHOT:
            logger.debug(f'Skipping placeholder screenshot from about:blank page at step {i}')
            continue

        # Skip screenshots from new tab pages
        from nexus.utils import is_new_tab_page

        if is_new_tab_page(item.state.url):
            logger.debug(f'Skipping screenshot from new tab page ({item.state.url}) at step {i}')
            continue

        # Convert base64 screenshot to PIL Image
        img_data = base64.b64decode(screenshot)
        image = Image.open(io.BytesIO(img_data))

        if show_goals and item.model_output:
            image = _add_overlay_to_image(
                image=image,
                step_number=i,
                goal_text=item.model_output.current_state.next_goal,
                regular_font=regular_font,  # type: ignore
                title_font=title_font,  # type: ignore
                margin=margin,
                logo=logo,
            )

        images.append(image)

    if images:
        # Save the GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        logger.info(f'Created GIF at {output_path}')
        
        # Generate debug bundle if visual debugger is provided
        if visual_debugger:
            debug_bundle_path = output_path.replace('.gif', '_debug.zip')
            visual_debugger.generate_debug_bundle(debug_bundle_path)
            logger.info(f'Created debug bundle at {debug_bundle_path}')
    else:
        logger.warning('No images found in history to create GIF')


def _create_task_frame(
    task: str,
    first_screenshot: str,
    title_font: ImageFont.FreeTypeFont,
    regular_font: ImageFont.FreeTypeFont,
    logo: Image.Image | None = None,
    line_spacing: float = 1.5,
) -> Image.Image:
    """Create initial frame showing the task."""
    from PIL import Image, ImageDraw, ImageFont

    img_data = base64.b64decode(first_screenshot)
    template = Image.open(io.BytesIO(img_data))
    image = Image.new('RGB', template.size, (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Calculate vertical center of image
    center_y = image.height // 2

    # Draw task text with dynamic font size based on task length
    margin = 140  # Increased margin
    max_width = image.width - (2 * margin)

    # Dynamic font size calculation based on task length
    # Start with base font size (regular + 16)
    base_font_size = regular_font.size + 16
    min_font_size = max(regular_font.size - 10, 16)  # Don't go below 16pt
    # Calculate dynamic font size based on text length and complexity
    # Longer texts get progressively smaller fonts
    text_length = len(task)
    if text_length > 200:
        # For very long text, reduce font size logarithmically
        font_size = max(base_font_size - int(10 * (text_length / 200)), min_font_size)
    else:
        font_size = base_font_size

    # Try to create a larger font, but fall back to regular font if it fails
    try:
        larger_font = ImageFont.truetype(regular_font.path, font_size)  # type: ignore
    except (OSError, AttributeError):
        # Fall back to regular font if .path is not available or font loading fails
        larger_font = regular_font

    # Generate wrapped text with the calculated font size
    wrapped_text = _wrap_text(task, larger_font, max_width)

    # Calculate line height with spacing
    line_height = larger_font.size * line_spacing

    # Split text into lines and draw with custom spacing
    lines = wrapped_text.split('\n')
    total_height = line_height * len(lines)

    # Start position for first line
    text_y = center_y - (total_height / 2) + 50  # Shifted down slightly

    for line in lines:
        # Get line width for centering
        line_bbox = draw.textbbox((0, 0), line, font=larger_font)
        line_width = line_bbox[2] - line_bbox[0]
        
        # Center the line
        text_x = (image.width - line_width) // 2
        
        # Draw text with shadow for better visibility
        draw.text((text_x + 2, text_y + 2), line, fill=(0, 0, 0), font=larger_font)
        draw.text((text_x, text_y), line, fill=(255, 255, 255), font=larger_font)
        
        text_y += line_height

    # Add logo if provided
    if logo:
        logo_x = (image.width - logo.width) // 2
        logo_y = 50  # Position at top
        image.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)

    return image


def _add_overlay_to_image(
    image: Image.Image,
    step_number: int,
    goal_text: str,
    regular_font: ImageFont.FreeTypeFont,
    title_font: ImageFont.FreeTypeFont,
    margin: int = 40,
    logo: Image.Image | None = None,
) -> Image.Image:
    """Add overlay with step number and goal text to an image."""
    from PIL import ImageDraw, ImageFont

    # Create a copy of the image to draw on
    overlay_image = image.copy()
    draw = ImageDraw.Draw(overlay_image)

    # Add semi-transparent overlay at the bottom
    overlay_height = 200
    overlay = Image.new('RGBA', (overlay_image.width, overlay_height), (0, 0, 0, 180))
    overlay_image.paste(overlay, (0, overlay_image.height - overlay_height), overlay)

    # Draw step number
    step_text = f"Step {step_number}"
    draw.text((margin, overlay_image.height - overlay_height + margin), 
              step_text, fill=(255, 255, 255), font=title_font)

    # Draw goal text with word wrap
    goal_y = overlay_image.height - overlay_height + margin + title_font.size + 20
    max_text_width = overlay_image.width - (2 * margin)
    wrapped_goal = _wrap_text(goal_text, regular_font, max_text_width)
    
    # Split into lines and draw
    lines = wrapped_goal.split('\n')
    for i, line in enumerate(lines):
        draw.text((margin, goal_y + i * (regular_font.size * 1.2)), 
                  line, fill=(255, 255, 255), font=regular_font)

    # Add logo if provided
    if logo:
        logo_x = overlay_image.width - logo.width - margin
        logo_y = margin
        overlay_image.paste(logo, (logo_x, logo_y), logo if logo.mode == 'RGBA' else None)

    return overlay_image


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Wrap text to fit within a given width."""
    from PIL import ImageDraw
    
    # Create a dummy draw object to measure text
    dummy_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_image)
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Try adding the word to the current line
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            # Line is full, start a new one
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    # Add the last line
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)