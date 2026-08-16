"""Custom gesture training suite.

Provides interactive data collection from camera feed and automated
ML model training pipeline using scikit-learn classifiers.
"""

import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS_DIR, DATA_DIR
from database.db_manager import DatabaseManager
from core.hand_detector import HandDetector, HandLandmarks

logger = logging.getLogger(__name__)


class GestureTrainer:
    """Handles custom gesture data collection and model training.
    
    Workflow:
    1. User names a gesture and holds the pose in front of camera
    2. System records N samples of landmark feature vectors
    3. Samples are stored in the database
    4. User can train an ML model on all collected samples
    5. Best model (RF vs MLP) is saved for real-time inference
    """

    def __init__(self, db: DatabaseManager, user_id: int):
        """Initialize the gesture trainer.
        
        Args:
            db: DatabaseManager instance for persistence
            user_id: Current user's ID for scoping data
        """
        self.db = db
        self.user_id = user_id
        self._model_path = MODELS_DIR / f"gesture_model_{user_id}.joblib"
        logger.info(f"GestureTrainer initialized for user {user_id}")

    def record_sample(self, gesture_name: str, hand: HandLandmarks) -> bool:
        """Record a single training sample from current hand landmarks.
        
        Args:
            gesture_name: Label for this gesture
            hand: Detected hand landmarks
            
        Returns:
            True if sample was saved successfully
        """
        try:
            feature_vector = HandDetector.extract_feature_vector(hand)
            landmark_json = json.dumps(feature_vector.tolist())
            self.db.save_gesture_sample(
                self.user_id, gesture_name, landmark_json
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record sample: {e}")
            return False

    def record_samples_batch(self, gesture_name: str, 
                              feature_vectors: List[np.ndarray]) -> int:
        """Record multiple training samples at once.
        
        Args:
            gesture_name: Label for all samples
            feature_vectors: List of feature vector arrays
            
        Returns:
            Number of successfully saved samples
        """
        samples = []
        for fv in feature_vectors:
            landmark_json = json.dumps(fv.tolist())
            samples.append((self.user_id, gesture_name, landmark_json))
        
        try:
            self.db.save_gesture_samples_batch(samples)
            logger.info(f"Saved {len(samples)} samples for '{gesture_name}'")
            return len(samples)
        except Exception as e:
            logger.error(f"Batch save failed: {e}")
            return 0

    def get_gesture_list(self) -> Dict[str, int]:
        """Get all gesture names and their sample counts for current user.
        
        Returns:
            Dict mapping gesture_name -> sample_count
        """
        return self.db.get_gesture_sample_counts(self.user_id)

    def delete_gesture(self, gesture_name: str) -> int:
        """Delete all samples for a specific gesture.
        
        Args:
            gesture_name: Name of gesture to delete
            
        Returns:
            Number of deleted samples
        """
        count = self.db.delete_gesture_samples(self.user_id, gesture_name)
        logger.info(f"Deleted {count} samples for '{gesture_name}'")
        return count

    def _load_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Load all training samples from database.
        
        Returns:
            (X features array, y labels array, list of unique class names)
            Returns (None, None, []) if insufficient data
        """
        samples = self.db.get_gesture_samples(self.user_id)
        if not samples:
            logger.warning("No training samples found")
            return None, None, []
        
        X_list = []
        y_list = []
        
        for sample in samples:
            try:
                features = json.loads(sample['landmark_data'])
                X_list.append(features)
                y_list.append(sample['gesture_name'])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted sample {sample['sample_id']}: {e}")
                continue
        
        if len(X_list) < 10:
            logger.warning(f"Insufficient samples ({len(X_list)}). Need at least 10.")
            return None, None, []
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        classes = sorted(list(set(y_list)))
        
        logger.info(f"Loaded {len(X_list)} samples across {len(classes)} gestures")
        return X, y, classes

    def train_model(self) -> Dict:
        """Train gesture classification models and save the best one.
        
        Trains both RandomForest and MLP classifiers, evaluates on a
        hold-out test set, and saves the best performer.
        
        Returns:
            Dict with training results:
            {
                'success': bool,
                'message': str,
                'best_model': str,
                'accuracy': float,
                'report': str,
                'classes': list,
                'num_samples': int
            }
        """
        if not HAS_SKLEARN:
            return {
                'success': False,
                'message': 'scikit-learn is not installed. Run: pip install scikit-learn',
                'best_model': None, 'accuracy': 0.0, 'report': '',
                'classes': [], 'num_samples': 0
            }
        
        if joblib is None:
            return {
                'success': False,
                'message': 'joblib is not installed. Run: pip install joblib',
                'best_model': None, 'accuracy': 0.0, 'report': '',
                'classes': [], 'num_samples': 0
            }
        
        # Load data
        X, y, classes = self._load_training_data()
        if X is None:
            return {
                'success': False,
                'message': 'Insufficient training data. Record at least 10 samples.',
                'best_model': None, 'accuracy': 0.0, 'report': '',
                'classes': classes, 'num_samples': 0
            }
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails (too few samples per class), split without
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train models
        models = {}
        
        # 1. Random Forest
        try:
            rf = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test))
            models['RandomForest'] = (rf, rf_acc)
            logger.info(f"RandomForest accuracy: {rf_acc:.4f}")
        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")
        
        # 2. MLP Neural Network
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
            mlp.fit(X_train, y_train)
            mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
            models['MLP'] = (mlp, mlp_acc)
            logger.info(f"MLP accuracy: {mlp_acc:.4f}")
        except Exception as e:
            logger.error(f"MLP training failed: {e}")
        
        if not models:
            return {
                'success': False,
                'message': 'All model training attempts failed.',
                'best_model': None, 'accuracy': 0.0, 'report': '',
                'classes': classes, 'num_samples': len(X)
            }
        
        # Select best model
        best_name = max(models, key=lambda k: models[k][1])
        best_model, best_acc = models[best_name]
        
        # Generate classification report
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        # Save model
        model_data = {
            'model': best_model,
            'classes': classes,
            'model_type': best_name,
            'accuracy': best_acc,
            'feature_dim': X.shape[1],
            'num_samples': len(X),
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, self._model_path, compress=3)
        logger.info(f"Best model ({best_name}) saved to {self._model_path}")
        
        return {
            'success': True,
            'message': f'{best_name} model trained successfully!',
            'best_model': best_name,
            'accuracy': best_acc,
            'report': report,
            'classes': classes,
            'num_samples': len(X),
            'all_results': {name: acc for name, (_, acc) in models.items()}
        }

    def get_model_info(self) -> Optional[Dict]:
        """Get information about the currently saved model.
        
        Returns:
            Dict with model metadata or None if no model exists
        """
        if not self._model_path.exists() or joblib is None:
            return None
        
        try:
            model_data = joblib.load(self._model_path)
            return {
                'model_type': model_data.get('model_type', 'Unknown'),
                'accuracy': model_data.get('accuracy', 0.0),
                'classes': model_data.get('classes', []),
                'feature_dim': model_data.get('feature_dim', 0),
                'num_samples': model_data.get('num_samples', 0),
                'trained_at': model_data.get('trained_at', 'Unknown'),
                'path': str(self._model_path)
            }
        except Exception as e:
            logger.error(f"Failed to load model info: {e}")
            return None

    def delete_model(self) -> bool:
        """Delete the trained model file."""
        try:
            if self._model_path.exists():
                self._model_path.unlink()
                logger.info(f"Model deleted: {self._model_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
        return False

    def export_dataset_csv(self, filepath: Path = None) -> Optional[Path]:
        """Export all training samples to a CSV file.
        
        Args:
            filepath: Target CSV path (auto-generated if None)
            
        Returns:
            Path to the created CSV file, or None on failure
        """
        if filepath is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = DATA_DIR / f"gesture_dataset_{self.user_id}_{timestamp}.csv"
        
        samples = self.db.get_gesture_samples(self.user_id)
        if not samples:
            logger.warning("No samples to export")
            return None
        
        try:
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header: gesture_name, feature_0, feature_1, ..., feature_77
                first_features = json.loads(samples[0]['landmark_data'])
                header = ['gesture_name'] + [f'f_{i}' for i in range(len(first_features))]
                writer.writerow(header)
                
                for sample in samples:
                    features = json.loads(sample['landmark_data'])
                    writer.writerow([sample['gesture_name']] + features)
            
            logger.info(f"Exported {len(samples)} samples to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return None
