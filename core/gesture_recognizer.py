"""Gesture recognition engine with hybrid rule-based and ML classification.

Combines geometric heuristics for built-in gestures, trained ML models
for custom gestures, and temporal tracking for dynamic gestures.
"""

import logging
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import joblib
except ImportError:
    joblib = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    Gestures, GESTURE_CONFIDENCE_THRESHOLD, GESTURE_COOLDOWN_MS,
    SWIPE_THRESHOLD, SWIPE_FRAMES, PINCH_THRESHOLD, ZOOM_DELTA_THRESHOLD,
    MODELS_DIR
)
from core.hand_detector import HandDetector, HandLandmarks

logger = logging.getLogger(__name__)


class GestureType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    ML_PREDICTED = "ml_predicted"


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture_name: str
    confidence: float
    gesture_type: GestureType
    hand_label: str = "Unknown"  # Left or Right


class DynamicGestureTracker:
    """Tracks hand movement over time for dynamic gesture detection.
    
    Maintains a sliding window of landmark positions and computes
    velocity/displacement to detect swipes and zoom gestures.
    """

    def __init__(self, window_size: int = SWIPE_FRAMES):
        self.window_size = window_size
        self._wrist_history: deque = deque(maxlen=window_size)
        self._pinch_distance_history: deque = deque(maxlen=window_size)
        self._timestamp_history: deque = deque(maxlen=window_size)

    def update(self, hand: HandLandmarks) -> None:
        """Add current frame's landmarks to the tracking history."""
        wrist = hand.landmarks[HandDetector.WRIST]
        self._wrist_history.append((wrist[0], wrist[1]))
        
        # Track pinch distance (thumb tip to index tip)
        thumb_tip = np.array(hand.landmarks[HandDetector.THUMB_TIP])
        index_tip = np.array(hand.landmarks[HandDetector.INDEX_TIP])
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        self._pinch_distance_history.append(pinch_dist)
        
        self._timestamp_history.append(time.time())

    def detect_swipe(self) -> Optional[str]:
        """Detect swipe gestures from wrist displacement over the window.
        
        Returns:
            Gesture name string or None
        """
        if len(self._wrist_history) < self.window_size // 2:
            return None
        
        # Compare first and last positions in the window
        start = self._wrist_history[0]
        end = self._wrist_history[-1]
        
        dx = end[0] - start[0]  # Horizontal displacement
        dy = end[1] - start[1]  # Vertical displacement
        
        # Check if displacement exceeds threshold
        if abs(dx) > SWIPE_THRESHOLD and abs(dx) > abs(dy):
            # Horizontal swipe dominates
            if dx > 0:
                return Gestures.SWIPE_RIGHT
            else:
                return Gestures.SWIPE_LEFT
        elif abs(dy) > SWIPE_THRESHOLD and abs(dy) > abs(dx):
            # Vertical swipe dominates
            if dy > 0:
                return Gestures.SWIPE_DOWN
            else:
                return Gestures.SWIPE_UP
        
        return None

    def detect_zoom(self) -> Optional[str]:
        """Detect zoom gestures from pinch distance changes.
        
        Returns:
            Gesture name string or None
        """
        if len(self._pinch_distance_history) < self.window_size // 2:
            return None
        
        # Compare early vs recent pinch distances
        early = list(self._pinch_distance_history)[:5]
        recent = list(self._pinch_distance_history)[-5:]
        
        avg_early = np.mean(early)
        avg_recent = np.mean(recent)
        delta = avg_recent - avg_early
        
        if abs(delta) > ZOOM_DELTA_THRESHOLD:
            if delta > 0:
                return Gestures.ZOOM_IN  # Fingers spreading apart
            else:
                return Gestures.ZOOM_OUT  # Fingers coming together
        
        return None

    def reset(self) -> None:
        """Clear tracking history."""
        self._wrist_history.clear()
        self._pinch_distance_history.clear()
        self._timestamp_history.clear()


class GestureRecognizer:
    """Main gesture recognition engine.
    
    Combines rule-based static gesture detection, ML model inference,
    and dynamic temporal gesture tracking.
    """

    def __init__(self, user_id: Optional[int] = None):
        self._ml_model = None
        self._ml_model_classes: List[str] = []
        self._dynamic_tracker = DynamicGestureTracker()
        self._last_gesture_time: float = 0.0
        self._last_gesture_name: str = ""
        self._user_id = user_id
        
        # Try to load user-specific ML model
        self._load_ml_model(user_id)
        
        logger.info(f"GestureRecognizer initialized (user_id={user_id}, "
                    f"ml_model={'loaded' if self._ml_model else 'none'})")

    def _load_ml_model(self, user_id: Optional[int]) -> bool:
        """Load trained ML model for the given user."""
        if joblib is None:
            logger.warning("joblib not available — ML prediction disabled")
            return False
        
        if user_id is not None:
            model_path = MODELS_DIR / f"gesture_model_{user_id}.joblib"
        else:
            model_path = MODELS_DIR / "gesture_model_default.joblib"
        
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                self._ml_model = model_data['model']
                self._ml_model_classes = list(model_data.get('classes', []))
                logger.info(f"Loaded ML model from {model_path} "
                            f"({len(self._ml_model_classes)} gestures)")
                return True
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
        
        return False

    def reload_model(self, user_id: Optional[int] = None) -> bool:
        """Reload the ML model (e.g., after training new gestures)."""
        uid = user_id if user_id is not None else self._user_id
        return self._load_ml_model(uid)

    def _detect_static_gesture(self, hand: HandLandmarks) -> Optional[GestureResult]:
        """Rule-based detection of built-in static gestures."""
        fingers = HandDetector.get_extended_fingers(hand)
        thumb, index, middle, ring, pinky = fingers
        
        # Count extended fingers
        extended_count = sum(fingers)
        
        # ── PINCH: thumb and index tips very close ──
        pinch_dist = HandDetector.get_landmark_distance(
            hand, HandDetector.THUMB_TIP, HandDetector.INDEX_TIP
        )
        if pinch_dist < PINCH_THRESHOLD:
            return GestureResult(
                gesture_name=Gestures.PINCH,
                confidence=max(0.5, 1.0 - (pinch_dist / PINCH_THRESHOLD)),
                gesture_type=GestureType.STATIC,
                hand_label=hand.handedness
            )
        
        # ── FIST: all fingers curled ──
        if extended_count == 0:
            return GestureResult(
                gesture_name=Gestures.FIST,
                confidence=0.95,
                gesture_type=GestureType.STATIC,
                hand_label=hand.handedness
            )
        
        # ── OPEN PALM: all fingers extended ──
        if extended_count == 5:
            return GestureResult(
                gesture_name=Gestures.OPEN_PALM,
                confidence=0.95,
                gesture_type=GestureType.STATIC,
                hand_label=hand.handedness
            )
        
        # ── POINT: only index extended ──
        if not thumb and index and not middle and not ring and not pinky:
            return GestureResult(
                gesture_name=Gestures.POINT,
                confidence=0.90,
                gesture_type=GestureType.STATIC,
                hand_label=hand.handedness
            )
        
        # ── PEACE: index + middle extended, others curled ──
        if not thumb and index and middle and not ring and not pinky:
            return GestureResult(
                gesture_name=Gestures.PEACE,
                confidence=0.90,
                gesture_type=GestureType.STATIC,
                hand_label=hand.handedness
            )
        
        # ── THUMBS UP: only thumb extended, hand oriented upward ──
        if thumb and not index and not middle and not ring and not pinky:
            # Verify thumb is pointing upward (thumb tip y < thumb cmc y)
            if hand.landmarks[HandDetector.THUMB_TIP][1] < hand.landmarks[HandDetector.THUMB_CMC][1]:
                return GestureResult(
                    gesture_name=Gestures.THUMBS_UP,
                    confidence=0.88,
                    gesture_type=GestureType.STATIC,
                    hand_label=hand.handedness
                )
        
        return None

    def _detect_ml_gesture(self, hand: HandLandmarks) -> Optional[GestureResult]:
        """ML model-based gesture classification."""
        if self._ml_model is None:
            return None
        
        try:
            feature_vector = HandDetector.extract_feature_vector(hand)
            features = feature_vector.reshape(1, -1)
            
            # Get probability distribution
            probabilities = self._ml_model.predict_proba(features)[0]
            max_idx = np.argmax(probabilities)
            max_prob = float(probabilities[max_idx])
            
            if max_prob >= GESTURE_CONFIDENCE_THRESHOLD:
                predicted_class = self._ml_model_classes[max_idx]
                return GestureResult(
                    gesture_name=predicted_class,
                    confidence=max_prob,
                    gesture_type=GestureType.ML_PREDICTED,
                    hand_label=hand.handedness
                )
        except Exception as e:
            logger.debug(f"ML prediction error: {e}")
        
        return None

    def _detect_dynamic_gesture(self, hand: HandLandmarks) -> Optional[GestureResult]:
        """Detect dynamic gestures (swipes, zoom) from temporal tracking."""
        # Update tracker with current frame
        self._dynamic_tracker.update(hand)
        
        # Check for swipe gestures
        swipe = self._dynamic_tracker.detect_swipe()
        if swipe:
            self._dynamic_tracker.reset()  # Reset after detection to avoid repeats
            return GestureResult(
                gesture_name=swipe,
                confidence=0.85,
                gesture_type=GestureType.DYNAMIC,
                hand_label=hand.handedness
            )
        
        # Check for zoom gestures
        zoom = self._dynamic_tracker.detect_zoom()
        if zoom:
            self._dynamic_tracker.reset()
            return GestureResult(
                gesture_name=zoom,
                confidence=0.80,
                gesture_type=GestureType.DYNAMIC,
                hand_label=hand.handedness
            )
        
        return None

    def _apply_cooldown(self, gesture_name: str) -> bool:
        """Check if cooldown period has elapsed since the last identical gesture.
        
        Returns True if the gesture should be suppressed (still in cooldown).
        """
        now = time.time() * 1000  # Convert to milliseconds
        if (gesture_name == self._last_gesture_name and
                now - self._last_gesture_time < GESTURE_COOLDOWN_MS):
            return True  # Suppress — still in cooldown
        return False

    def recognize(self, hand: HandLandmarks) -> Optional[GestureResult]:
        """Run the full gesture recognition pipeline on a single hand.
        
        Priority order:
        1. Dynamic gestures (swipes, zoom) — highest priority
        2. ML model prediction (if trained model available)
        3. Rule-based static gestures (built-in fallback)
        
        Args:
            hand: HandLandmarks from the detector
            
        Returns:
            GestureResult or None if no gesture confidently detected
        """
        # 1. Dynamic gestures (always check for temporal patterns)
        dynamic_result = self._detect_dynamic_gesture(hand)
        if dynamic_result and not self._apply_cooldown(dynamic_result.gesture_name):
            self._last_gesture_time = time.time() * 1000
            self._last_gesture_name = dynamic_result.gesture_name
            return dynamic_result
        
        # 2. ML-based prediction (custom trained gestures)
        ml_result = self._detect_ml_gesture(hand)
        if ml_result and not self._apply_cooldown(ml_result.gesture_name):
            self._last_gesture_time = time.time() * 1000
            self._last_gesture_name = ml_result.gesture_name
            return ml_result
        
        # 3. Rule-based static gestures (built-in)
        static_result = self._detect_static_gesture(hand)
        if static_result and not self._apply_cooldown(static_result.gesture_name):
            self._last_gesture_time = time.time() * 1000
            self._last_gesture_name = static_result.gesture_name
            return static_result
        
        return None

    def reset_tracker(self) -> None:
        """Reset the dynamic gesture tracker."""
        self._dynamic_tracker.reset()
        self._last_gesture_time = 0.0
        self._last_gesture_name = ""
