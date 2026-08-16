"""Hand detection and landmark tracking using Google MediaPipe.

Extracts 21-keypoint hand landmarks, computes normalized feature vectors,
and provides drawing utilities for visualization.
"""

import cv2
import logging
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, MAX_NUM_HANDS, MODEL_COMPLEXITY

logger = logging.getLogger(__name__)

# MediaPipe solutions references
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class HandLandmarks:
    """Container for detected hand landmark data.
    
    Attributes:
        landmarks: List of 21 (x, y, z) normalized coordinates [0.0-1.0]
        pixel_landmarks: List of 21 (px, py) integer pixel coordinates
        handedness: 'Left' or 'Right' hand classification
        raw_landmarks: Original MediaPipe landmark protobuf object
    """
    landmarks: List[Tuple[float, float, float]]
    pixel_landmarks: List[Tuple[int, int]]
    handedness: str
    raw_landmarks: object = field(repr=False)


class HandDetector:
    """Detects hands and extracts 21-point landmarks using MediaPipe."""

    # Landmark index constants for readability
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # Fingertip indices
    FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    # PIP joint indices (for finger extension detection)
    FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    # MCP joint indices
    FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

    def __init__(self, max_hands: int = MAX_NUM_HANDS,
                 detection_confidence: float = MIN_DETECTION_CONFIDENCE,
                 tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
                 model_complexity: int = MODEL_COMPLEXITY):
        self.max_hands = max_hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        logger.info(f"HandDetector initialized: max_hands={max_hands}, "
                    f"det_conf={detection_confidence}, track_conf={tracking_confidence}")

    def detect(self, rgb_frame: np.ndarray) -> List[HandLandmarks]:
        """Detect hands and extract landmarks from an RGB frame.
        
        Args:
            rgb_frame: Input frame in RGB color space
            
        Returns:
            List of HandLandmarks for each detected hand (up to max_hands)
        """
        results_list = []
        h, w, _ = rgb_frame.shape
        
        # MediaPipe optimization: mark image as non-writeable
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if not results.multi_hand_landmarks:
            return results_list
        
        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            # Extract handedness (Left/Right)
            handedness = "Unknown"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
            
            # Extract normalized and pixel coordinates
            landmarks = []
            pixel_landmarks = []
            for lm in hand_lm.landmark:
                landmarks.append((lm.x, lm.y, lm.z))
                pixel_landmarks.append((int(lm.x * w), int(lm.y * h)))
            
            results_list.append(HandLandmarks(
                landmarks=landmarks,
                pixel_landmarks=pixel_landmarks,
                handedness=handedness,
                raw_landmarks=hand_lm
            ))
        
        return results_list

    @staticmethod
    def extract_feature_vector(hand: HandLandmarks) -> np.ndarray:
        """Extract a normalized feature vector from hand landmarks.
        
        Creates a 78-dimensional feature vector:
        - 63 values: 21 landmarks × 3 (x,y,z), normalized relative to wrist
        - 15 values: key inter-landmark distances (fingertip-to-wrist, 
          fingertip-to-fingertip pairs)
        
        Normalization: All coordinates are shifted so wrist = (0,0,0),
        then scaled by the max distance from wrist to any landmark.
        
        Args:
            hand: HandLandmarks object
            
        Returns:
            78-element numpy float32 array
        """
        lm = np.array(hand.landmarks, dtype=np.float32)  # Shape: (21, 3)
        
        # Center on wrist (landmark 0)
        wrist = lm[0].copy()
        lm_centered = lm - wrist
        
        # Scale normalization: divide by max distance from wrist
        distances_from_wrist = np.linalg.norm(lm_centered, axis=1)
        max_dist = np.max(distances_from_wrist)
        if max_dist > 0:
            lm_normalized = lm_centered / max_dist
        else:
            lm_normalized = lm_centered
        
        # Flatten to 63 values
        coords = lm_normalized.flatten()  # (63,)
        
        # Compute 15 inter-landmark distances
        tip_indices = [4, 8, 12, 16, 20]  # Fingertips
        inter_distances = []
        
        # 5 distances: each fingertip to wrist
        for tip in tip_indices:
            d = np.linalg.norm(lm_normalized[tip] - lm_normalized[0])
            inter_distances.append(d)
        
        # 10 distances: all pairwise fingertip-to-fingertip
        for i in range(len(tip_indices)):
            for j in range(i + 1, len(tip_indices)):
                d = np.linalg.norm(
                    lm_normalized[tip_indices[i]] - lm_normalized[tip_indices[j]]
                )
                inter_distances.append(d)
        
        feature_vector = np.concatenate([coords, np.array(inter_distances, dtype=np.float32)])
        return feature_vector  # Shape: (78,)

    @staticmethod
    def is_finger_extended(hand: HandLandmarks, finger_tip_idx: int, 
                           finger_pip_idx: int, finger_mcp_idx: int,
                           is_thumb: bool = False) -> bool:
        """Check if a specific finger is extended (straight).
        
        For thumb: compares tip.x distance from MCP (accounts for thumb's lateral movement)
        For other fingers: checks if tip.y < pip.y (tip is above pip in image coords)
        """
        lm = hand.landmarks
        if is_thumb:
            # Thumb: check if tip is further from palm center than IP joint
            # Use x-axis for right hand, inverted for left
            if hand.handedness == "Right":
                return lm[finger_tip_idx][0] < lm[finger_pip_idx][0]
            else:
                return lm[finger_tip_idx][0] > lm[finger_pip_idx][0]
        else:
            # Other fingers: tip should be above (lower y) than PIP joint
            return lm[finger_tip_idx][1] < lm[finger_pip_idx][1]

    @classmethod
    def get_extended_fingers(cls, hand: HandLandmarks) -> List[bool]:
        """Get extension state of all 5 fingers.
        
        Returns:
            List of 5 booleans: [thumb, index, middle, ring, pinky]
        """
        return [
            cls.is_finger_extended(hand, cls.THUMB_TIP, cls.THUMB_IP, cls.THUMB_MCP, is_thumb=True),
            cls.is_finger_extended(hand, cls.INDEX_TIP, cls.INDEX_PIP, cls.INDEX_MCP),
            cls.is_finger_extended(hand, cls.MIDDLE_TIP, cls.MIDDLE_PIP, cls.MIDDLE_MCP),
            cls.is_finger_extended(hand, cls.RING_TIP, cls.RING_PIP, cls.RING_MCP),
            cls.is_finger_extended(hand, cls.PINKY_TIP, cls.PINKY_PIP, cls.PINKY_MCP),
        ]

    @staticmethod
    def get_landmark_distance(hand: HandLandmarks, idx1: int, idx2: int) -> float:
        """Calculate Euclidean distance between two landmarks."""
        lm = hand.landmarks
        p1 = np.array(lm[idx1])
        p2 = np.array(lm[idx2])
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def draw_landmarks(frame: np.ndarray, hand: HandLandmarks,
                       draw_connections: bool = True) -> np.ndarray:
        """Draw hand landmarks and connections on a BGR frame.
        
        Args:
            frame: BGR image to draw on (modified in-place)
            hand: HandLandmarks with raw_landmarks
            draw_connections: Whether to draw bone connections
            
        Returns:
            The annotated frame
        """
        if hand.raw_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame,
                hand.raw_landmarks,
                mp_hands.HAND_CONNECTIONS if draw_connections else None,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return frame

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            logger.info("HandDetector closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
