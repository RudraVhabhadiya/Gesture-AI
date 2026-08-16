"""Video capture and frame preprocessing pipeline.

Handles webcam initialization, frame capture, mirroring,
resizing, and color-space conversions for the gesture pipeline.
"""

import cv2
import logging
import numpy as np
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET

logger = logging.getLogger(__name__)


class VideoCapture:
    """Manages webcam video capture with preprocessing.
    
    Provides a clean interface for capturing frames from a webcam
    with automatic mirroring, resizing, and color conversion.
    
    Supports context manager protocol for safe resource management.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX,
                 width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT,
                 fps: int = FPS_TARGET, mirror: bool = True):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        logger.info(f"VideoCapture configured: camera={camera_index}, "
                    f"resolution={width}x{height}, fps={fps}")

    def open(self) -> bool:
        """Open the webcam. Returns True if successful."""
        try:
            # Use DirectShow backend on Windows for faster initialization
            import platform
            if platform.system() == 'Windows':
                self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self.camera_index)
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Read actual values (camera may not support requested settings)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
            
            self._is_opened = True
            logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture and preprocess a single frame.
        
        Returns:
            (success, frame) tuple. Frame is BGR, mirrored if enabled.
        """
        if not self._is_opened or self._cap is None:
            return False, None
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        
        # Resize if frame doesn't match target dimensions
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Mirror horizontally for natural interaction
        if self.mirror:
            frame = cv2.flip(frame, 1)
        
        return True, frame

    @staticmethod
    def to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB for MediaPipe processing."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Return (width, height) of captured frames."""
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.width, self.height

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._is_opened = False
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
