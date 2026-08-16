"""Performance monitoring and diagnostics module.

Tracks real-time FPS, maintains gesture event history,
provides on-screen display (OSD) rendering, and manages
activity logging to the database.
"""

import cv2
import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FPS_ROLLING_WINDOW, EVENT_HISTORY_SIZE, LOG_BATCH_SIZE, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class GestureEvent:
    """Record of a single gesture detection event."""
    timestamp: float
    gesture_name: str
    action_executed: str
    confidence: float
    fps: float
    hand_label: str = "Unknown"


class PerformanceMonitor:
    """Tracks performance metrics and renders diagnostic overlays.
    
    Provides:
    - Real-time FPS calculation (rolling average)
    - Gesture event history logging
    - On-screen display (OSD) rendering on video frames
    - Database activity logging with batched writes
    - Session statistics aggregation
    """

    def __init__(self, db=None, user_id: Optional[int] = None):
        """Initialize the performance monitor.
        
        Args:
            db: Optional DatabaseManager for persistent logging
            user_id: Current user ID for log attribution
        """
        self.db = db
        self.user_id = user_id
        
        # FPS tracking
        self._frame_times: deque = deque(maxlen=FPS_ROLLING_WINDOW)
        self._last_frame_time: float = time.perf_counter()
        self._current_fps: float = 0.0
        
        # Event history
        self._event_history: deque = deque(maxlen=EVENT_HISTORY_SIZE)
        self._pending_log_entries: List = []
        
        # Session tracking
        self._session_start: float = time.time()
        self._total_gestures: int = 0
        self._gesture_counts: Dict[str, int] = {}
        self._confidence_sum: float = 0.0
        
        # OSD state
        self._last_gesture_display: str = ""
        self._last_confidence_display: float = 0.0
        self._gesture_display_time: float = 0.0
        self._gesture_display_duration: float = 2.0  # Show gesture for 2 seconds
        
        logger.info("PerformanceMonitor initialized")

    def update_fps(self) -> float:
        """Update and return current FPS.
        
        Call this once per frame in the main loop.
        
        Returns:
            Current rolling average FPS
        """
        now = time.perf_counter()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        
        if delta > 0:
            self._frame_times.append(delta)
        
        if len(self._frame_times) > 0:
            avg_delta = sum(self._frame_times) / len(self._frame_times)
            self._current_fps = 1.0 / avg_delta if avg_delta > 0 else 0.0
        
        return self._current_fps

    @property
    def fps(self) -> float:
        """Current FPS value."""
        return self._current_fps

    def log_event(self, gesture_name: str, action: str,
                  confidence: float, hand_label: str = "Unknown") -> None:
        """Log a gesture detection event.
        
        Args:
            gesture_name: Detected gesture name
            action: Command that was executed
            confidence: Detection confidence [0-1]
            hand_label: Left or Right hand
        """
        event = GestureEvent(
            timestamp=time.time(),
            gesture_name=gesture_name,
            action_executed=action,
            confidence=confidence,
            fps=self._current_fps,
            hand_label=hand_label
        )
        
        self._event_history.append(event)
        self._total_gestures += 1
        self._confidence_sum += confidence
        self._gesture_counts[gesture_name] = self._gesture_counts.get(gesture_name, 0) + 1
        
        # Update OSD display
        self._last_gesture_display = gesture_name
        self._last_confidence_display = confidence
        self._gesture_display_time = time.time()
        
        # Queue for database batch write
        if self.db and self.user_id:
            self._pending_log_entries.append((
                self.user_id, gesture_name, action,
                confidence, self._current_fps, None
            ))
            
            # Flush batch if threshold reached
            if len(self._pending_log_entries) >= LOG_BATCH_SIZE:
                self._flush_logs()

    def _flush_logs(self) -> None:
        """Write pending log entries to database."""
        if not self._pending_log_entries or not self.db:
            return
        
        try:
            self.db.log_activities_batch(self._pending_log_entries)
            self._pending_log_entries.clear()
        except Exception as e:
            logger.error(f"Failed to flush logs to database: {e}")

    def flush(self) -> None:
        """Public method to force flush pending logs."""
        self._flush_logs()

    def get_recent_events(self, count: int = 5) -> List[GestureEvent]:
        """Get the most recent gesture events.
        
        Args:
            count: Number of events to return
            
        Returns:
            List of recent GestureEvent objects (newest first)
        """
        events = list(self._event_history)
        return events[-count:][::-1]

    def get_session_stats(self) -> Dict:
        """Get aggregate statistics for the current session.
        
        Returns:
            Dict with session metrics
        """
        elapsed = time.time() - self._session_start
        avg_conf = (self._confidence_sum / self._total_gestures 
                    if self._total_gestures > 0 else 0.0)
        
        most_common = ""
        if self._gesture_counts:
            most_common = max(self._gesture_counts, key=self._gesture_counts.get)
        
        return {
            'session_duration_sec': elapsed,
            'session_duration_str': self._format_duration(elapsed),
            'total_gestures': self._total_gestures,
            'avg_confidence': round(avg_conf, 3),
            'avg_fps': round(self._current_fps, 1),
            'most_common_gesture': most_common,
            'gesture_counts': dict(self._gesture_counts),
            'events_in_buffer': len(self._event_history),
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def draw_overlay(self, frame, gesture_name: str = None,
                     confidence: float = None, paused: bool = False) -> None:
        """Draw performance overlay on video frame.
        
        Renders:
        - Top-left: FPS counter (color-coded green/yellow/red)
        - Top-right: Current gesture + confidence bar
        - Bottom: Recent event history (last 3 events)
        - Pause indicator if recognition is paused
        
        Args:
            frame: BGR video frame (modified in-place)
            gesture_name: Currently detected gesture (overrides stored)
            confidence: Detection confidence (overrides stored)
            paused: Whether recognition is paused
        """
        h, w = frame.shape[:2]
        
        # Use stored values if not provided
        if gesture_name is None:
            if time.time() - self._gesture_display_time < self._gesture_display_duration:
                gesture_name = self._last_gesture_display
                confidence = self._last_confidence_display
            else:
                gesture_name = ""
                confidence = 0.0
        if confidence is None:
            confidence = 0.0
        
        # ── Semi-transparent header background ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # ── FPS Counter (top-left) ──
        fps_val = self._current_fps
        if fps_val >= 25:
            fps_color = (0, 255, 100)     # Green
        elif fps_val >= 15:
            fps_color = (0, 220, 255)     # Yellow
        else:
            fps_color = (0, 80, 255)      # Red
        
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2, cv2.LINE_AA)
        
        # Session time
        elapsed = time.time() - self._session_start
        time_str = self._format_duration(elapsed)
        cv2.putText(frame, time_str, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # ── Gesture + Confidence (top-right) ──
        if gesture_name:
            # Gesture name
            text = f"{gesture_name}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = w - text_size[0] - 20
            cv2.putText(frame, text, (text_x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Confidence bar
            bar_width = 150
            bar_x = w - bar_width - 20
            bar_y = 42
            bar_h = 12
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h),
                          (60, 60, 60), -1)
            # Filled portion
            fill_w = int(bar_width * confidence)
            if confidence >= 0.8:
                bar_color = (0, 255, 100)
            elif confidence >= 0.6:
                bar_color = (0, 220, 255)
            else:
                bar_color = (0, 80, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          bar_color, -1)
            
            # Percentage text
            pct_text = f"{confidence * 100:.0f}%"
            cv2.putText(frame, pct_text, (bar_x + bar_width + 5, bar_y + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # ── Gesture Count (top-center) ──
        count_text = f"Gestures: {self._total_gestures}"
        count_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(frame, count_text, ((w - count_size[0]) // 2, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        
        # ── Recent Events (bottom) ──
        recent = self.get_recent_events(3)
        if recent:
            # Semi-transparent footer
            footer_h = 25 + len(recent) * 22
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - footer_h), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, "Recent Events:", (15, h - footer_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
            
            for i, event in enumerate(recent):
                y_pos = h - footer_h + 38 + i * 22
                ts = datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')
                event_text = f"{ts}  {event.gesture_name:<15s} -> {event.action_executed}"
                cv2.putText(frame, event_text, (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # ── Pause indicator ──
        if paused:
            # Large centered PAUSED text with semi-transparent background
            overlay3 = frame.copy()
            cv2.rectangle(overlay3, (w//2 - 120, h//2 - 30), 
                          (w//2 + 120, h//2 + 30), (0, 0, 150), -1)
            cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "PAUSED", (w//2 - 75, h//2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # ── Help hint ──
        help_text = "Q: Quit | M: Mouse Track | P: Pause | H: Help"
        cv2.putText(frame, help_text, (w // 2 - 200, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    def export_session_log(self, filepath: Path = None) -> Optional[Path]:
        """Export session event history to CSV.
        
        Args:
            filepath: Target CSV path (auto-generated if None)
            
        Returns:
            Path to created file, or None on failure
        """
        if filepath is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = DATA_DIR / f"session_log_{timestamp}.csv"
        
        try:
            events = list(self._event_history)
            if not events:
                return None
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'gesture', 'action', 'confidence', 'fps', 'hand'])
                for event in events:
                    writer.writerow([
                        datetime.fromtimestamp(event.timestamp).isoformat(),
                        event.gesture_name,
                        event.action_executed,
                        f"{event.confidence:.3f}",
                        f"{event.fps:.1f}",
                        event.hand_label
                    ])
            
            logger.info(f"Session log exported to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export session log: {e}")
            return None

    def reset(self) -> None:
        """Reset all monitoring state for a new session."""
        self._flush_logs()
        self._frame_times.clear()
        self._event_history.clear()
        self._session_start = time.time()
        self._total_gestures = 0
        self._gesture_counts.clear()
        self._confidence_sum = 0.0
        self._last_gesture_display = ""
        logger.info("PerformanceMonitor reset")
