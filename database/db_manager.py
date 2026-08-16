import sqlite3
import logging
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import sys

# Add parent directory to sys.path to allow importing from config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import DB_PATH, DEFAULT_GESTURE_MAPPINGS
except ImportError:
    # Fallback configuration if config module is unavailable
    DB_PATH = str(Path(__file__).parent.parent / "gesture_system.db")
    DEFAULT_GESTURE_MAPPINGS = {
        "Fist": "Play/Pause",
        "Palm": "Stop",
        "Thumbs Up": "Volume Up",
        "Thumbs Down": "Volume Down"
    }

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Singleton SQLite database manager for the AI Gesture Control System.
    Manages connections and provides interface for data operations.
    """
    _instance = None
    
    def __new__(cls, db_path: Optional[str] = None):
        """Implement Singleton pattern to ensure only one database manager instance exists."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database connection and configure SQLite parameters.
        
        Args:
            db_path: Path to the SQLite database file. Defaults to config.DB_PATH.
        """
        if self._initialized:
            return
            
        self.db_path = db_path if db_path is not None else DB_PATH
        
        # Ensure the directory for the database exists
        db_file_path = Path(self.db_path)
        if db_file_path.parent:
            db_file_path.parent.mkdir(parents=True, exist_ok=True)
            
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.create_tables()
            self._initialized = True
            logger.info(f"Database connection established successfully at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
            
    def create_tables(self) -> None:
        """Create the necessary database tables if they do not exist."""
        users_schema = '''
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login    TIMESTAMP,
                preferences   TEXT DEFAULT '{}'
            );
        '''
        
        mappings_schema = '''
            CREATE TABLE IF NOT EXISTS gesture_mappings (
                mapping_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                gesture_name TEXT NOT NULL,
                command      TEXT NOT NULL,
                is_active    BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, gesture_name)
            );
        '''
        
        datasets_schema = '''
            CREATE TABLE IF NOT EXISTS gesture_datasets (
                sample_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                gesture_name TEXT NOT NULL,
                landmark_data TEXT NOT NULL,
                recorded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        '''
        
        logs_schema = '''
            CREATE TABLE IF NOT EXISTS activity_logs (
                log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER,
                timestamp    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                gesture      TEXT,
                action       TEXT,
                confidence   REAL,
                fps          REAL,
                details      TEXT
            );
        '''
        
        custom_actions_schema = '''
            CREATE TABLE IF NOT EXISTS custom_actions (
                action_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                action_name  TEXT NOT NULL,
                action_type  TEXT NOT NULL,
                action_data  TEXT NOT NULL,
                description  TEXT DEFAULT '',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, action_name)
            );
        '''
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(users_schema)
            cursor.execute(mappings_schema)
            cursor.execute(datasets_schema)
            cursor.execute(logs_schema)
            cursor.execute(custom_actions_schema)
            self.conn.commit()
            logger.info("Database tables verified/created successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Safely close the database connection."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
                self._initialized = False
                logger.info("Database connection closed successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")

    # ── User Operations ──

    def add_user(self, username: str, password_hash: str, salt: str) -> int:
        """
        Add a new user to the database.
        
        Args:
            username: Unique username.
            password_hash: Hashed password.
            salt: Password salt.
            
        Returns:
            int: The new user's ID.
        """
        query = "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (username, password_hash, salt))
            self.conn.commit()
            logger.info(f"User added: {username}")
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to add user '{username}': {e}")
            self.conn.rollback()
            return -1

    def get_user(self, username: str) -> Optional[sqlite3.Row]:
        """
        Retrieve a user by their username.
        
        Args:
            username: The username to look up.
            
        Returns:
            sqlite3.Row representing the user, or None if not found.
        """
        query = "SELECT * FROM users WHERE username = ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (username,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch user '{username}': {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[sqlite3.Row]:
        """
        Retrieve a user by their user ID.
        
        Args:
            user_id: The unique ID of the user.
            
        Returns:
            sqlite3.Row representing the user, or None if not found.
        """
        query = "SELECT * FROM users WHERE user_id = ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch user by ID '{user_id}': {e}")
            return None

    def update_last_login(self, user_id: int) -> None:
        """
        Update the last login timestamp for a given user.
        
        Args:
            user_id: The ID of the user.
        """
        query = "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?"
        try:
            self.conn.execute(query, (user_id,))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update last login for user {user_id}: {e}")
            self.conn.rollback()

    def update_user_preferences(self, user_id: int, preferences: str) -> None:
        """
        Update the JSON-serialized preferences for a given user.
        
        Args:
            user_id: The ID of the user.
            preferences: JSON string representing user preferences.
        """
        query = "UPDATE users SET preferences = ? WHERE user_id = ?"
        try:
            self.conn.execute(query, (preferences, user_id))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update preferences for user {user_id}: {e}")
            self.conn.rollback()

    # ── Gesture Mapping Operations ──

    def save_gesture_mapping(self, user_id: int, gesture_name: str, command: str) -> None:
        """
        Save or update a gesture-to-command mapping for a user.
        
        Args:
            user_id: The user's ID.
            gesture_name: The name of the gesture.
            command: The command mapped to the gesture.
        """
        query = """
            INSERT INTO gesture_mappings (user_id, gesture_name, command)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, gesture_name) DO UPDATE SET command = excluded.command
        """
        try:
            self.conn.execute(query, (user_id, gesture_name, command))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save mapping for user {user_id}, gesture '{gesture_name}': {e}")
            self.conn.rollback()

    def get_gesture_mappings(self, user_id: int) -> List[sqlite3.Row]:
        """
        Retrieve all active gesture mappings for a user.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            List of sqlite3.Row containing gesture mapping data.
        """
        query = "SELECT * FROM gesture_mappings WHERE user_id = ? AND is_active = 1"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve mappings for user {user_id}: {e}")
            return []

    def delete_gesture_mapping(self, user_id: int, gesture_name: str) -> None:
        """
        Delete a specific gesture mapping for a user.
        
        Args:
            user_id: The user's ID.
            gesture_name: The name of the gesture mapping to delete.
        """
        query = "DELETE FROM gesture_mappings WHERE user_id = ? AND gesture_name = ?"
        try:
            self.conn.execute(query, (user_id, gesture_name))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to delete mapping '{gesture_name}' for user {user_id}: {e}")
            self.conn.rollback()

    def initialize_default_mappings(self, user_id: int) -> None:
        """
        Initialize the default gesture-to-command mappings for a newly created user.
        
        Args:
            user_id: The user's ID.
        """
        try:
            for gesture, command in DEFAULT_GESTURE_MAPPINGS.items():
                self.save_gesture_mapping(user_id, gesture, command)
            logger.info(f"Default mappings initialized for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize default mappings for user {user_id}: {e}")

    # ── Gesture Dataset Operations ──

    def save_gesture_sample(self, user_id: int, gesture_name: str, landmark_data: str) -> int:
        """
        Save a single gesture landmark data sample.
        
        Args:
            user_id: The user's ID.
            gesture_name: Name of the gesture.
            landmark_data: Serialized landmark data.
            
        Returns:
            int: The ID of the newly inserted sample.
        """
        query = "INSERT INTO gesture_datasets (user_id, gesture_name, landmark_data) VALUES (?, ?, ?)"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id, gesture_name, landmark_data))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to save gesture sample for user {user_id}: {e}")
            self.conn.rollback()
            return -1

    def save_gesture_samples_batch(self, samples: List[Tuple[int, str, str]]) -> None:
        """
        Save a batch of gesture samples efficiently.
        
        Args:
            samples: A list of tuples, each containing (user_id, gesture_name, landmark_data).
        """
        query = "INSERT INTO gesture_datasets (user_id, gesture_name, landmark_data) VALUES (?, ?, ?)"
        try:
            self.conn.executemany(query, samples)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to batch insert gesture samples: {e}")
            self.conn.rollback()

    def get_gesture_samples(self, user_id: int, gesture_name: Optional[str] = None) -> List[sqlite3.Row]:
        """
        Retrieve gesture dataset samples for a user, optionally filtered by gesture name.
        
        Args:
            user_id: The user's ID.
            gesture_name: Optional filter by gesture name.
            
        Returns:
            List of sqlite3.Row representing the datasets.
        """
        try:
            cursor = self.conn.cursor()
            if gesture_name:
                query = "SELECT * FROM gesture_datasets WHERE user_id = ? AND gesture_name = ?"
                cursor.execute(query, (user_id, gesture_name))
            else:
                query = "SELECT * FROM gesture_datasets WHERE user_id = ?"
                cursor.execute(query, (user_id,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve gesture samples for user {user_id}: {e}")
            return []

    def get_gesture_sample_counts(self, user_id: int) -> Dict[str, int]:
        """
        Get the count of collected samples for each gesture for a user.
        
        Args:
            user_id: The user's ID.
            
        Returns:
            Dictionary mapping gesture_name to sample count.
        """
        query = "SELECT gesture_name, COUNT(*) as count FROM gesture_datasets WHERE user_id = ? GROUP BY gesture_name"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            return {row['gesture_name']: row['count'] for row in results}
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve sample counts for user {user_id}: {e}")
            return {}

    def delete_gesture_samples(self, user_id: int, gesture_name: str) -> int:
        """
        Delete all dataset samples for a specific gesture.
        
        Args:
            user_id: The user's ID.
            gesture_name: The name of the gesture to delete samples for.
            
        Returns:
            int: The number of deleted samples.
        """
        query = "DELETE FROM gesture_datasets WHERE user_id = ? AND gesture_name = ?"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id, gesture_name))
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Failed to delete samples for gesture '{gesture_name}': {e}")
            self.conn.rollback()
            return 0

    # ── Activity Log Operations ──

    def log_activity(self, user_id: int, gesture: str, action: str, 
                     confidence: float, fps: float, details: Optional[str] = None) -> None:
        """
        Insert a single activity log entry.
        
        Args:
            user_id: The ID of the user.
            gesture: The detected gesture.
            action: The executed action/command.
            confidence: The recognition confidence score.
            fps: Frame rate at the time.
            details: Optional extra details.
        """
        query = """
            INSERT INTO activity_logs (user_id, gesture, action, confidence, fps, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            self.conn.execute(query, (user_id, gesture, action, confidence, fps, details))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log activity for user {user_id}: {e}")
            self.conn.rollback()

    def log_activities_batch(self, entries: List[Tuple[int, str, str, float, float, Optional[str]]]) -> None:
        """
        Batch insert activity log entries.
        
        Args:
            entries: List of tuples corresponding to the columns (user_id, gesture, action, confidence, fps, details).
        """
        query = """
            INSERT INTO activity_logs (user_id, gesture, action, confidence, fps, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            self.conn.executemany(query, entries)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to batch log activities: {e}")
            self.conn.rollback()

    def get_activity_logs(self, user_id: Optional[int] = None, limit: int = 100, offset: int = 0) -> List[sqlite3.Row]:
        """
        Retrieve paginated activity logs, optionally filtered by user.
        
        Args:
            user_id: The user's ID, or None for system-wide logs.
            limit: Maximum number of rows to return.
            offset: Offset for pagination.
            
        Returns:
            List of sqlite3.Row representing the activity logs.
        """
        try:
            cursor = self.conn.cursor()
            if user_id is not None:
                query = "SELECT * FROM activity_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                cursor.execute(query, (user_id, limit, offset))
            else:
                query = "SELECT * FROM activity_logs ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                cursor.execute(query, (limit, offset))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve activity logs: {e}")
            return []

    def get_activity_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Compute and return aggregate statistics of user activities.
        
        Args:
            user_id: The ID of the user.
            
        Returns:
            Dictionary containing stats like total_gestures, most_common_gesture, avg_confidence.
        """
        stats: Dict[str, Any] = {
            "total_gestures": 0,
            "most_common_gesture": None,
            "avg_confidence": 0.0
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Total gestures
            cursor.execute("SELECT COUNT(*) FROM activity_logs WHERE user_id = ?", (user_id,))
            total = cursor.fetchone()[0]
            stats["total_gestures"] = total or 0
            
            if stats["total_gestures"] > 0:
                # Most common gesture
                cursor.execute("""
                    SELECT gesture, COUNT(*) as cnt 
                    FROM activity_logs 
                    WHERE user_id = ? 
                    GROUP BY gesture 
                    ORDER BY cnt DESC 
                    LIMIT 1
                """, (user_id,))
                most_common = cursor.fetchone()
                if most_common:
                    stats["most_common_gesture"] = most_common['gesture']
                    
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM activity_logs WHERE user_id = ?", (user_id,))
                avg_conf = cursor.fetchone()[0]
                if avg_conf is not None:
                    stats["avg_confidence"] = float(avg_conf)
                    
            return stats
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve activity stats for user {user_id}: {e}")
            return stats

    def clear_activity_logs(self, user_id: Optional[int] = None) -> int:
        """
        Clear activity logs.
        
        Args:
            user_id: User ID to clear logs for, or None to clear all logs.
            
        Returns:
            int: The number of deleted log entries.
        """
        try:
            cursor = self.conn.cursor()
            if user_id is not None:
                query = "DELETE FROM activity_logs WHERE user_id = ?"
                cursor.execute(query, (user_id,))
            else:
                query = "DELETE FROM activity_logs"
                cursor.execute(query)
            
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Failed to clear activity logs: {e}")
            self.conn.rollback()
            return 0

    # ── Custom Action Operations ──

    def save_custom_action(self, user_id: int, action_name: str, action_type: str,
                           action_data: str, description: str = '') -> int:
        """Save or update a custom action for a user."""
        query = """
            INSERT INTO custom_actions (user_id, action_name, action_type, action_data, description)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, action_name) DO UPDATE SET
                action_type = excluded.action_type,
                action_data = excluded.action_data,
                description = excluded.description
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id, action_name, action_type, action_data, description))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to save custom action '{action_name}': {e}")
            self.conn.rollback()
            return -1

    def get_custom_actions(self, user_id: int) -> List[sqlite3.Row]:
        """Get all custom actions for a user."""
        query = "SELECT * FROM custom_actions WHERE user_id = ? ORDER BY action_name"
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (user_id,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Failed to get custom actions for user {user_id}: {e}")
            return []

    def delete_custom_action(self, user_id: int, action_name: str) -> None:
        """Delete a custom action."""
        query = "DELETE FROM custom_actions WHERE user_id = ? AND action_name = ?"
        try:
            self.conn.execute(query, (user_id, action_name))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to delete custom action '{action_name}': {e}")
            self.conn.rollback()
