"""Global configuration for AI Gesture Control System."""
import os
from pathlib import Path

# ── Project Paths ──
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "gesture_ai.db"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure directories exist
for d in [MODELS_DIR, DATA_DIR, ASSETS_DIR]:
    d.mkdir(exist_ok=True)

# ── Camera Settings ──
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# ── MediaPipe Settings ──
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2
MODEL_COMPLEXITY = 1

# ── Gesture Recognition ──
GESTURE_CONFIDENCE_THRESHOLD = 0.75
GESTURE_COOLDOWN_MS = 300
SWIPE_THRESHOLD = 0.15       # Min wrist displacement for swipe detection
SWIPE_FRAMES = 15            # Sliding window size for dynamic gesture tracking
PINCH_THRESHOLD = 0.04       # Max thumb-index distance for pinch
ZOOM_DELTA_THRESHOLD = 0.02  # Min pinch distance change for zoom gesture

# ── Authentication ──
HASH_ITERATIONS = 100_000
SESSION_TIMEOUT_MINUTES = 60

# ── Performance ──
FPS_ROLLING_WINDOW = 30
EVENT_HISTORY_SIZE = 100
LOG_BATCH_SIZE = 20  # Flush logs to DB every N events

# ── Built-in Gesture Names ──
class Gestures:
    FIST = "Fist"
    OPEN_PALM = "Open Palm"
    PINCH = "Pinch"
    PEACE = "Peace"
    THUMBS_UP = "Thumbs Up"
    POINT = "Point"
    SWIPE_LEFT = "Swipe Left"
    SWIPE_RIGHT = "Swipe Right"
    SWIPE_UP = "Swipe Up"
    SWIPE_DOWN = "Swipe Down"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    UNKNOWN = "Unknown"

# ── Default Gesture → Command Mappings ──
DEFAULT_GESTURE_MAPPINGS = {
    Gestures.OPEN_PALM: "pause_resume",
    Gestures.FIST: "mouse_click",
    Gestures.POINT: "mouse_move",
    Gestures.PINCH: "mouse_scroll",
    Gestures.PEACE: "screenshot",
    Gestures.THUMBS_UP: "volume_up",
    Gestures.SWIPE_LEFT: "slide_prev",
    Gestures.SWIPE_RIGHT: "slide_next",
    Gestures.SWIPE_UP: "brightness_up",
    Gestures.SWIPE_DOWN: "brightness_down",
    Gestures.ZOOM_IN: "zoom_in",
    Gestures.ZOOM_OUT: "zoom_out",
}

# ── Available Commands (for UI display) ──
AVAILABLE_COMMANDS = {
    "mouse_move": "Mouse Move (track index finger)",
    "mouse_click": "Mouse Left Click",
    "mouse_right_click": "Mouse Right Click",
    "mouse_scroll": "Mouse Scroll",
    "volume_up": "Volume Up (+10%)",
    "volume_down": "Volume Down (-10%)",
    "volume_mute": "Toggle Mute",
    "brightness_up": "Brightness Up (+10%)",
    "brightness_down": "Brightness Down (-10%)",
    "media_play_pause": "Media Play/Pause",
    "media_next": "Media Next Track",
    "media_prev": "Media Previous Track",
    "slide_next": "Next Slide (Right Arrow)",
    "slide_prev": "Previous Slide (Left Arrow)",
    "zoom_in": "Zoom In (Ctrl +)",
    "zoom_out": "Zoom Out (Ctrl -)",
    "screenshot": "Take Screenshot",
    "launch_browser": "Launch Browser",
    "launch_terminal": "Launch Terminal",
    "launch_notepad": "Launch Notepad",
    "pause_resume": "Pause/Resume Recognition",
    "none": "No Action",
}

# ── GUI Theme ──
APP_TITLE = "Gesture AI — Hand Gesture Control System"
APP_VERSION = "1.0.0"
