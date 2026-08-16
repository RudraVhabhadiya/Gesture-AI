"""System command execution engine.

Maps recognized gestures to OS-level actions including mouse/keyboard
control, volume/brightness adjustment, media controls, and app launching.
"""

import logging
import os
import platform
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import pyautogui
    pyautogui.PAUSE = 0.001  # Minimize delay for responsive cursor tracking
    pyautogui.FAILSAFE = False  # Prevent fail-safe exceptions when tracking near screen edges
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Windows-specific volume control
HAS_VOLUME_CONTROL = False
try:
    if platform.system() == 'Windows':
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        HAS_VOLUME_CONTROL = True
except ImportError:
    pass

# Brightness control
HAS_BRIGHTNESS = False
try:
    import screen_brightness_control as sbc
    HAS_BRIGHTNESS = True
except ImportError:
    pass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger(__name__)


class CommandExecutor:
    """Executes system commands mapped from gesture recognition results.
    
    Supports mouse/keyboard control, volume/brightness adjustment,
    media controls, and application launching.
    """

    def __init__(self, cooldown_ms: int = 300):
        self._screen_width = 1920
        self._screen_height = 1080
        self._mouse_tracking_enabled = False
        self._mouse_smoothing_factor = 0.5  # Lower = smoother but more lag
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._volume_interface = None
        self._paused = False  # Whether gesture recognition is paused
        self._custom_actions: Dict[str, Dict[str, str]] = {}  # name -> {type, data, description}
        self._cooldown_ms = cooldown_ms  # Min ms between command executions
        self._last_exec_time: Dict[str, float] = {}  # command_key -> last execution timestamp
        
        # Initialize screen size
        if HAS_PYAUTOGUI:
            self._screen_width, self._screen_height = pyautogui.size()
            # Fix Windows DPI scaling issues
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        
        # Initialize volume control (Windows)
        if HAS_VOLUME_CONTROL:
            try:
                devices = AudioUtilities.GetSpeakers()
                # Support both old and new pycaw API
                if hasattr(devices, 'EndpointVolume'):
                    # New pycaw API (AudioDevice wrapper)
                    self._volume_interface = devices.EndpointVolume
                elif hasattr(devices, 'Activate'):
                    # Legacy pycaw API (raw COM interface)
                    interface = devices.Activate(
                        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                    )
                    self._volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
                logger.info("Volume control initialized (pycaw)")
            except Exception as e:
                logger.warning(f"Failed to initialize volume control: {e}")
        
        # Build command dispatch table
        self._commands: Dict[str, Callable] = {
            'mouse_move': self._mouse_move,
            'mouse_click': self._mouse_click,
            'mouse_right_click': self._mouse_right_click,
            'mouse_scroll': self._mouse_scroll,
            'volume_up': self._volume_up,
            'volume_down': self._volume_down,
            'volume_mute': self._volume_mute,
            'brightness_up': self._brightness_up,
            'brightness_down': self._brightness_down,
            'media_play_pause': self._media_play_pause,
            'media_next': self._media_next,
            'media_prev': self._media_prev,
            'slide_next': self._slide_next,
            'slide_prev': self._slide_prev,
            'zoom_in': self._zoom_in,
            'zoom_out': self._zoom_out,
            'screenshot': self._screenshot,
            'launch_browser': self._launch_browser,
            'launch_terminal': self._launch_terminal,
            'launch_notepad': self._launch_notepad,
            'pause_resume': self._pause_resume,
            'none': self._noop,
        }
        
        logger.info(f"CommandExecutor initialized: pyautogui={HAS_PYAUTOGUI}, "
                    f"volume={HAS_VOLUME_CONTROL}, brightness={HAS_BRIGHTNESS}, "
                    f"screen={self._screen_width}x{self._screen_height}")

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def cooldown_ms(self) -> int:
        return self._cooldown_ms

    def set_cooldown(self, ms: int) -> None:
        """Set the cooldown between gesture command executions in milliseconds."""
        self._cooldown_ms = max(0, ms)
        logger.info(f"Cooldown set to {self._cooldown_ms}ms")

    def execute(self, command_key: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Execute a command by key.
        
        Args:
            command_key: The command identifier (e.g., 'volume_up')
            context: Optional dict with extra data (e.g., hand landmarks for mouse tracking)
            
        Returns:
            (success: bool, message: str)
        """
        if command_key not in self._commands:
            logger.warning(f"Unknown command: {command_key}")
            return False, f"Unknown command: {command_key}"
        
        # Don't execute commands while paused (except pause_resume itself)
        if self._paused and command_key != 'pause_resume':
            return False, "Recognition paused"
        
        # Cooldown check (skip for continuous commands like mouse_move)
        now = time.time() * 1000  # current time in ms
        if command_key != 'mouse_move' and self._cooldown_ms > 0:
            last = self._last_exec_time.get(command_key, 0)
            if (now - last) < self._cooldown_ms:
                return False, "Cooldown active"
        
        try:
            result = self._commands[command_key](context)
            self._last_exec_time[command_key] = now
            return True, result or f"Executed: {command_key}"
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI fail-safe triggered (mouse moved to corner)")
            return False, "Fail-safe triggered"
        except Exception as e:
            logger.error(f"Command execution failed [{command_key}]: {e}")
            return False, f"Error: {str(e)}"

    def set_mouse_tracking(self, enabled: bool) -> None:
        """Enable or disable continuous mouse tracking."""
        self._mouse_tracking_enabled = enabled
        logger.info(f"Mouse tracking {'enabled' if enabled else 'disabled'}")

    # ── Mouse Commands ──
    
    def _mouse_move(self, context: Optional[Dict] = None) -> str:
        """Move mouse cursor to index fingertip position."""
        if not HAS_PYAUTOGUI or context is None:
            return "Mouse move skipped"
        
        hand = context.get('hand')
        if hand is None:
            return "No hand data"
        
        # Get index fingertip pixel position (landmark 8)
        from core.hand_detector import HandDetector
        ix, iy = hand.pixel_landmarks[HandDetector.INDEX_TIP]
        
        # Scale from camera frame to screen coordinates
        frame_w = context.get('frame_width', FRAME_WIDTH)
        frame_h = context.get('frame_height', FRAME_HEIGHT)
        
        screen_x = int(np.interp(ix, [0, frame_w], [0, self._screen_width]))
        screen_y = int(np.interp(iy, [0, frame_h], [0, self._screen_height]))
        
        # Clamp to screen bounds (avoid fail-safe trigger)
        screen_x = max(5, min(screen_x, self._screen_width - 5))
        screen_y = max(5, min(screen_y, self._screen_height - 5))
        
        # Apply smoothing
        smooth_x = int(self._last_mouse_x + 
                       self._mouse_smoothing_factor * (screen_x - self._last_mouse_x))
        smooth_y = int(self._last_mouse_y + 
                       self._mouse_smoothing_factor * (screen_y - self._last_mouse_y))
        
        pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
        self._last_mouse_x = smooth_x
        self._last_mouse_y = smooth_y
        
        return f"Mouse → ({smooth_x}, {smooth_y})"

    def _mouse_click(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.click()
            return "Left click"
        return "PyAutoGUI not available"

    def _mouse_right_click(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.rightClick()
            return "Right click"
        return "PyAutoGUI not available"

    def _mouse_scroll(self, context: Optional[Dict] = None) -> str:
        if not HAS_PYAUTOGUI:
            return "PyAutoGUI not available"
        
        # Scroll direction based on hand movement if available
        direction = 3  # Default: scroll up
        if context and 'scroll_direction' in context:
            direction = context['scroll_direction']
        
        pyautogui.scroll(direction)
        return f"Scroll {'up' if direction > 0 else 'down'}"

    # ── Volume Commands ──
    
    def _volume_up(self, context: Optional[Dict] = None) -> str:
        if self._volume_interface:
            try:
                current = self._volume_interface.GetMasterVolumeLevelScalar()
                new_level = min(1.0, current + 0.1)
                self._volume_interface.SetMasterVolumeLevelScalar(new_level, None)
                return f"Volume: {int(new_level * 100)}%"
            except Exception as e:
                logger.error(f"Volume up failed: {e}")
        elif HAS_PYAUTOGUI:
            pyautogui.press('volumeup')
            return "Volume up (key)"
        return "Volume control not available"

    def _volume_down(self, context: Optional[Dict] = None) -> str:
        if self._volume_interface:
            try:
                current = self._volume_interface.GetMasterVolumeLevelScalar()
                new_level = max(0.0, current - 0.1)
                self._volume_interface.SetMasterVolumeLevelScalar(new_level, None)
                return f"Volume: {int(new_level * 100)}%"
            except Exception as e:
                logger.error(f"Volume down failed: {e}")
        elif HAS_PYAUTOGUI:
            pyautogui.press('volumedown')
            return "Volume down (key)"
        return "Volume control not available"

    def _volume_mute(self, context: Optional[Dict] = None) -> str:
        if self._volume_interface:
            try:
                is_muted = self._volume_interface.GetMute()
                self._volume_interface.SetMute(not is_muted, None)
                return f"{'Unmuted' if is_muted else 'Muted'}"
            except Exception as e:
                logger.error(f"Mute toggle failed: {e}")
        elif HAS_PYAUTOGUI:
            pyautogui.press('volumemute')
            return "Mute toggled (key)"
        return "Volume control not available"

    # ── Brightness Commands ──
    
    def _brightness_up(self, context: Optional[Dict] = None) -> str:
        if HAS_BRIGHTNESS:
            try:
                current = sbc.get_brightness(display=0)
                if isinstance(current, list):
                    current = current[0]
                new_level = min(100, current + 10)
                sbc.set_brightness(new_level, display=0)
                return f"Brightness: {new_level}%"
            except Exception as e:
                logger.error(f"Brightness up failed: {e}")
                return f"Brightness error: {e}"
        return "Brightness control not available"

    def _brightness_down(self, context: Optional[Dict] = None) -> str:
        if HAS_BRIGHTNESS:
            try:
                current = sbc.get_brightness(display=0)
                if isinstance(current, list):
                    current = current[0]
                new_level = max(0, current - 10)
                sbc.set_brightness(new_level, display=0)
                return f"Brightness: {new_level}%"
            except Exception as e:
                logger.error(f"Brightness down failed: {e}")
                return f"Brightness error: {e}"
        return "Brightness control not available"

    # ── Media Commands ──
    
    def _media_play_pause(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.press('playpause')
            return "Media: Play/Pause"
        return "PyAutoGUI not available"

    def _media_next(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.press('nexttrack')
            return "Media: Next Track"
        return "PyAutoGUI not available"

    def _media_prev(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.press('prevtrack')
            return "Media: Previous Track"
        return "PyAutoGUI not available"

    # ── Presentation Commands ──
    
    def _slide_next(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.press('right')
            return "Slide: Next"
        return "PyAutoGUI not available"

    def _slide_prev(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.press('left')
            return "Slide: Previous"
        return "PyAutoGUI not available"

    # ── Zoom Commands ──
    
    def _zoom_in(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.hotkey('ctrl', 'plus')
            return "Zoom In (Ctrl +)"
        return "PyAutoGUI not available"

    def _zoom_out(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            pyautogui.hotkey('ctrl', 'minus')
            return "Zoom Out (Ctrl -)"
        return "PyAutoGUI not available"

    # ── Utility Commands ──
    
    def _screenshot(self, context: Optional[Dict] = None) -> str:
        if HAS_PYAUTOGUI:
            try:
                from config import BASE_DIR
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filepath = BASE_DIR / 'data' / f'screenshot_{timestamp}.png'
                img = pyautogui.screenshot()
                img.save(str(filepath))
                return f"Screenshot saved: {filepath.name}"
            except Exception as e:
                logger.error(f"Screenshot failed: {e}")
                return f"Screenshot error: {e}"
        return "PyAutoGUI not available"

    def _launch_browser(self, context: Optional[Dict] = None) -> str:
        try:
            webbrowser.open('https://www.google.com')
            return "Browser launched"
        except Exception as e:
            return f"Browser launch failed: {e}"

    def _launch_terminal(self, context: Optional[Dict] = None) -> str:
        try:
            if platform.system() == 'Windows':
                # Try Windows Terminal first, fall back to cmd
                try:
                    subprocess.Popen('wt.exe', creationflags=subprocess.CREATE_NEW_CONSOLE)
                except FileNotFoundError:
                    subprocess.Popen('cmd.exe', creationflags=subprocess.CREATE_NEW_CONSOLE)
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', '-a', 'Terminal'])
            else:
                subprocess.Popen(['x-terminal-emulator'])
            return "Terminal launched"
        except Exception as e:
            return f"Terminal launch failed: {e}"

    def _launch_notepad(self, context: Optional[Dict] = None) -> str:
        try:
            if platform.system() == 'Windows':
                subprocess.Popen('notepad.exe')
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', '-a', 'TextEdit'])
            else:
                subprocess.Popen(['gedit'])
            return "Notepad launched"
        except Exception as e:
            return f"Notepad launch failed: {e}"

    def _pause_resume(self, context: Optional[Dict] = None) -> str:
        self._paused = not self._paused
        status = 'PAUSED' if self._paused else 'RESUMED'
        logger.info(f"Recognition {status}")
        return f"Recognition {status}"

    def _noop(self, context: Optional[Dict] = None) -> str:
        return "No action"

    # ── Custom Action Support ──

    def register_custom_action(self, action_name: str, action_type: str,
                                action_data: str, description: str = '') -> None:
        """Register a custom action that can be mapped to gestures.
        
        Args:
            action_name: Unique key like 'custom_open_discord'
            action_type: One of 'hotkey', 'run_program', 'open_url', 'type_text'
            action_data: The action payload (e.g., 'ctrl+shift+t', 'C:\\app.exe', 'https://...')
            description: Human-readable description
        """
        self._custom_actions[action_name] = {
            'type': action_type,
            'data': action_data,
            'description': description
        }
        self._commands[action_name] = self._execute_custom_action
        logger.info(f"Custom action registered: {action_name} ({action_type}: {action_data})")

    def unregister_custom_action(self, action_name: str) -> None:
        """Remove a custom action."""
        self._custom_actions.pop(action_name, None)
        self._commands.pop(action_name, None)
        logger.info(f"Custom action unregistered: {action_name}")

    def get_custom_actions(self) -> Dict[str, Dict[str, str]]:
        """Return all registered custom actions."""
        return dict(self._custom_actions)

    def _execute_custom_action(self, context: Optional[Dict] = None) -> str:
        """Execute a custom action based on the command_key passed through the dispatch."""
        # Find which custom action is being executed by checking the call stack
        import inspect
        frame = inspect.currentframe()
        # The command_key is passed through execute() which calls self._commands[command_key]
        # We need to find it from the caller
        caller_locals = frame.f_back.f_locals if frame.f_back else {}
        command_key = caller_locals.get('command_key', '')
        
        if command_key not in self._custom_actions:
            return f"Unknown custom action: {command_key}"
        
        action = self._custom_actions[command_key]
        action_type = action['type']
        action_data = action['data']
        
        try:
            if action_type == 'hotkey':
                # Parse hotkey string like 'ctrl+shift+t' or 'alt+f4'
                keys = [k.strip() for k in action_data.split('+')]
                if HAS_PYAUTOGUI:
                    pyautogui.hotkey(*keys)
                    return f"Hotkey: {action_data}"
                return "PyAutoGUI not available"
            
            elif action_type == 'run_program':
                subprocess.Popen(action_data, shell=True)
                return f"Launched: {action_data}"
            
            elif action_type == 'open_url':
                webbrowser.open(action_data)
                return f"Opened: {action_data}"
            
            elif action_type == 'type_text':
                if HAS_PYAUTOGUI:
                    pyautogui.typewrite(action_data, interval=0.02)
                    return f"Typed: {action_data}"
                return "PyAutoGUI not available"
            
            elif action_type == 'press_key':
                if HAS_PYAUTOGUI:
                    pyautogui.press(action_data)
                    return f"Key pressed: {action_data}"
                return "PyAutoGUI not available"
            
            else:
                return f"Unknown action type: {action_type}"
        
        except Exception as e:
            logger.error(f"Custom action failed [{command_key}]: {e}")
            return f"Error: {e}"
