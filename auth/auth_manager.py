"""User authentication and profile management module.

Handles user registration, login/logout, session management,
and per-user gesture mapping configuration.
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_GESTURE_MAPPINGS, HASH_ITERATIONS, SESSION_TIMEOUT_MINUTES
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class AuthManager:
    """Manages user authentication, sessions, and profile settings."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._current_user: Optional[Dict] = None
        self._session_start: Optional[datetime] = None
        logger.info("AuthManager initialized")

    # ── Password Hashing ──
    
    def _hash_password(self, password: str, salt: bytes = None) -> Tuple[str, str]:
        """Hash password using PBKDF2-HMAC-SHA256.
        
        Args:
            password: Plain text password
            salt: Optional salt bytes (generated if None)
            
        Returns:
            Tuple of (password_hash_hex, salt_hex)
        """
        if salt is None:
            salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations=HASH_ITERATIONS
        )
        return key.hex(), salt.hex()

    def _verify_password(self, password: str, stored_hash: str, salt_hex: str) -> bool:
        """Verify password against stored hash.
        
        Uses hmac.compare_digest for timing-attack resistance.
        """
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations=HASH_ITERATIONS
        )
        return hmac.compare_digest(key.hex(), stored_hash)

    # ── Registration ──
    
    def register(self, username: str, password: str) -> Tuple[bool, str]:
        """Register a new user.
        
        Args:
            username: Desired username (3-50 chars, alphanumeric + underscore)
            password: Password (min 6 chars)
            
        Returns:
            (success: bool, message: str)
        """
        # Validate inputs
        if not username or len(username.strip()) < 3:
            return False, "Username must be at least 3 characters."
        if len(username) > 50:
            return False, "Username must be 50 characters or less."
        if not username.replace('_', '').isalnum():
            return False, "Username can only contain letters, numbers, and underscores."
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters."
        
        username = username.strip().lower()
        
        # Check if user exists
        existing = self.db.get_user(username)
        if existing:
            return False, "Username already exists."
        
        # Hash password and create user
        try:
            password_hash, salt_hex = self._hash_password(password)
            user_id = self.db.add_user(username, password_hash, salt_hex)
            
            # Initialize default gesture mappings
            self.db.initialize_default_mappings(user_id)
            
            logger.info(f"User registered successfully: {username} (ID: {user_id})")
            return True, f"Registration successful! Welcome, {username}."
        except Exception as e:
            logger.error(f"Registration failed for {username}: {e}")
            return False, f"Registration failed: {str(e)}"

    # ── Login / Logout ──
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate user and start session.
        
        Returns:
            (success: bool, message: str)
        """
        if not username or not password:
            return False, "Username and password are required."
        
        username = username.strip().lower()
        
        user = self.db.get_user(username)
        if not user:
            return False, "Invalid username or password."
        
        if not self._verify_password(password, user['password_hash'], user['salt']):
            return False, "Invalid username or password."
        
        # Set session
        self._current_user = {
            'user_id': user['user_id'],
            'username': user['username'],
            'preferences': json.loads(user['preferences'] or '{}'),
            'created_at': user['created_at'],
        }
        self._session_start = datetime.now()
        
        # Update last login timestamp
        self.db.update_last_login(user['user_id'])
        
        logger.info(f"User logged in: {username}")
        return True, f"Welcome back, {username}!"

    def logout(self) -> None:
        """End current user session."""
        if self._current_user:
            logger.info(f"User logged out: {self._current_user['username']}")
        self._current_user = None
        self._session_start = None

    def is_logged_in(self) -> bool:
        """Check if a user is currently logged in and session hasn't expired."""
        if self._current_user is None:
            return False
        if self._session_start:
            elapsed = datetime.now() - self._session_start
            if elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                logger.info("Session expired")
                self.logout()
                return False
        return True

    def get_current_user(self) -> Optional[Dict]:
        """Get current logged-in user info."""
        if self.is_logged_in():
            return self._current_user
        return None

    def get_current_user_id(self) -> Optional[int]:
        """Get current user's ID or None."""
        user = self.get_current_user()
        return user['user_id'] if user else None

    # ── Profile & Preferences ──
    
    def update_preferences(self, preferences: Dict) -> Tuple[bool, str]:
        """Update current user's preferences."""
        if not self.is_logged_in():
            return False, "Not logged in."
        try:
            prefs_json = json.dumps(preferences)
            self.db.update_user_preferences(self._current_user['user_id'], prefs_json)
            self._current_user['preferences'] = preferences
            return True, "Preferences updated."
        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")
            return False, str(e)

    # ── Gesture Mappings ──
    
    def get_gesture_mappings(self) -> Dict[str, str]:
        """Get current user's gesture→command mappings as a dict.
        
        Falls back to default mappings if no user is logged in.
        """
        if not self.is_logged_in():
            return dict(DEFAULT_GESTURE_MAPPINGS)
        
        rows = self.db.get_gesture_mappings(self._current_user['user_id'])
        if not rows:
            return dict(DEFAULT_GESTURE_MAPPINGS)
        return {row['gesture_name']: row['command'] for row in rows}

    def save_gesture_mapping(self, gesture_name: str, command: str) -> Tuple[bool, str]:
        """Save or update a gesture mapping for the current user."""
        if not self.is_logged_in():
            return False, "Not logged in."
        try:
            self.db.save_gesture_mapping(
                self._current_user['user_id'], gesture_name, command
            )
            return True, f"Mapping saved: {gesture_name} → {command}"
        except Exception as e:
            logger.error(f"Failed to save mapping: {e}")
            return False, str(e)

    def delete_gesture_mapping(self, gesture_name: str) -> Tuple[bool, str]:
        """Delete a gesture mapping for the current user."""
        if not self.is_logged_in():
            return False, "Not logged in."
        try:
            self.db.delete_gesture_mapping(
                self._current_user['user_id'], gesture_name
            )
            return True, f"Mapping deleted: {gesture_name}"
        except Exception as e:
            logger.error(f"Failed to delete mapping: {e}")
            return False, str(e)

    def reset_default_mappings(self) -> Tuple[bool, str]:
        """Reset gesture mappings to system defaults for current user."""
        if not self.is_logged_in():
            return False, "Not logged in."
        try:
            self.db.initialize_default_mappings(self._current_user['user_id'])
            return True, "Default mappings restored."
        except Exception as e:
            logger.error(f"Failed to reset default mappings: {e}")
            return False, str(e)

    # ── Custom Actions ──

    def save_custom_action(self, action_name: str, action_type: str,
                           action_data: str, description: str = '') -> Tuple[bool, str]:
        """Save a custom action for the current user."""
        if not self.is_logged_in():
            return False, "Not logged in."
        try:
            self.db.save_custom_action(
                self._current_user['user_id'], action_name,
                action_type, action_data, description
            )
            return True, f"Custom action saved: {action_name}"
        except Exception as e:
            logger.error(f"Failed to save custom action: {e}")
            return False, str(e)

    def get_custom_actions(self) -> list:
        """Get all custom actions for the current user."""
        if not self.is_logged_in():
            return []
        return self.db.get_custom_actions(self._current_user['user_id'])

    def delete_custom_action(self, action_name: str) -> Tuple[bool, str]:
        """Delete a custom action for the current user."""
        if not self.is_logged_in():
            return False, "Not logged in."
        try:
            self.db.delete_custom_action(
                self._current_user['user_id'], action_name
            )
            return True, f"Custom action deleted: {action_name}"
        except Exception as e:
            logger.error(f"Failed to delete custom action: {e}")
            return False, str(e)
