"""
Recommendation Model
Handles ML model training, evaluation, and recommendations
"""
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter

from utils.logger import setup_logger
from utils.errors import DataLoadingException, RecommendationException
from config.settings import active_config

logger = setup_logger(__name__)


class RecommendationModel:
    """
    Handles skincare product recommendations using ML models
    """
    
    def __init__(self):
        self.ensemble_model = None
        self.knn_model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = ['skin type', 'concern', 'concern 2', 'concern 3']
        self.data = None
        self.model_metrics = {}
        self.feature_importance = {}
        self.is_trained = False
    
    def load_dataset(self, filepath):
        """
        Load and preprocess dataset
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            DataLoadingException: If loading fails
        """
        try:
            logger.info(f"Loading dataset from {filepath}")
            df = pd.read_csv(filepath)
            
            # Drop rows with missing skin type
            df = df.dropna(subset=['skin type'])
            
            # Impute missing concern values
            imputer = SimpleImputer(strategy='most_frequent')
            concern_cols = ['concern', 'concern 2', 'concern 3']
            df[concern_cols] = imputer.fit_transform(df[concern_cols])
            
            logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            self.data = df
            return df
            
        except FileNotFoundError:
            raise DataLoadingException(f"Dataset file not found: {filepath}", filepath)
        except Exception as e:
            raise DataLoadingException(f"Error loading dataset: {str(e)}", filepath)
    
    def encode_features(self, df, fit=True):
        """
        Encode categorical features
        
        Args:
            df: DataFrame with raw values
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            np.ndarray: Encoded features
        """
        df_encoded = df.copy()
        
        for col in self.feature_columns:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Encoded '{col}': {len(le.classes_)} unique values")
            else:
                le = self.label_encoders[col]
                df_encoded[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return df_encoded
    
    def train(self, train_data=None, test_size=0.2, random_state=42):
        """
        Train the ensemble model
        
        Args:
            train_data: DataFrame with training data (uses self.data if None)
            test_size: Proportion for test set
            random_state: Random state for reproducibility
            
        Returns:
            dict: Training metrics
        """
        try:
            if train_data is None:
                if self.data is None:
                    raise ValueError("No training data provided")
                train_data = self.data
            
            print("\n" + "="*60)
            print("🚀 STARTING MODEL TRAINING")
            print("="*60)
            logger.info("Starting model training...")
            
            # Encode features
            print("\n📊 Step 1/5: Encoding features...")
            X = self.encode_features(train_data[self.feature_columns], fit=True)
            y = train_data['label']
            print(f"   ✓ Encoded {len(self.feature_columns)} features")
            
            # Split data
            print("\n✂️  Step 2/5: Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print(f"   ✓ Train set: {len(X_train)} samples")
            print(f"   ✓ Test set: {len(X_test)} samples")
            logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Handle class imbalance
            print("\n⚖️  Step 3/5: Balancing classes with SMOTE...")
            try:
                min_samples = min(Counter(y_train).values())
                k_neighbors = min(5, min_samples - 1)
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"   ✓ Applied SMOTE: {len(X_train)} samples after balancing")
                logger.info(f"Applied SMOTE: {len(X_train)} samples")
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {str(e)}")
                logger.warning(f"SMOTE failed: {str(e)}")
            
            # Train KNN
            print("\n🔍 Step 4a/5: Training KNN model (Grid Search)...")
            knn_params = {
                'n_neighbors': [5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
            knn = KNeighborsClassifier()
            knn_grid = GridSearchCV(
                knn,
                knn_params,
                cv=StratifiedKFold(n_splits=3),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            knn_grid.fit(X_train, y_train)
            self.knn_model = knn_grid.best_estimator_
            print(f"   ✓ KNN trained with best params: {knn_grid.best_params_}")
            logger.info(f"Best KNN params: {knn_grid.best_params_}")
            
            # Train Random Forest
            print("\n🌲 Step 4b/5: Training Random Forest model (Grid Search)...")
            rf_params = {
                'n_estimators': [100],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5]
            }
            rf = RandomForestClassifier(random_state=random_state)
            rf_grid = GridSearchCV(
                rf,
                rf_params,
                cv=StratifiedKFold(n_splits=3),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            rf_grid.fit(X_train, y_train)
            rf_model = rf_grid.best_estimator_
            print(f"   ✓ Random Forest trained with best params: {rf_grid.best_params_}")
            
            # Store feature importance
            self.feature_importance = dict(zip(
                self.feature_columns,
                rf_model.feature_importances_[:len(self.feature_columns)]
            ))
            print(f"   ✓ Feature importance calculated")
            
            # Create ensemble
            print("\n🎭 Step 5/5: Creating ensemble model...")
            self.ensemble_model = VotingClassifier(
                estimators=[('knn', self.knn_model), ('rf', rf_model)],
                voting='soft'
            )
            self.ensemble_model.fit(X_train, y_train)
            print(f"   ✓ Ensemble model created")
            
            # Evaluate
            print("\n📈 Evaluating model performance...")
            y_pred = self.ensemble_model.predict(X_test)
            self.model_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            self.is_trained = True
            
            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE")
            print("="*60)
            print(f"\n📊 Model Metrics:")
            print(f"   • Accuracy:  {self.model_metrics['accuracy']:.4f} ({self.model_metrics['accuracy']*100:.2f}%)")
            print(f"   • Precision: {self.model_metrics['precision']:.4f}")
            print(f"   • Recall:    {self.model_metrics['recall']:.4f}")
            print(f"   • F1-Score:  {self.model_metrics['f1']:.4f}")
            print(f"\n🎯 Model Status: {'READY FOR PRODUCTION' if self.model_metrics['accuracy'] > 0.7 else 'NEEDS IMPROVEMENT'}")
            print("="*60 + "\n")
            
            logger.info(f"Training complete. Accuracy: {self.model_metrics['accuracy']:.4f}")
            
            return self.model_metrics
            
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def recommend(self, skin_type, concern_1, concern_2, concern_3, top_n=10):
        """
        Get product recommendations
        
        Args:
            skin_type: User's skin type
            concern_1, concern_2, concern_3: User's concerns
            top_n: Number of products to recommend
            
        Returns:
            pd.DataFrame: Recommended products
            
        Raises:
            RecommendationException: If recommendation fails
        """
        try:
            if not self.is_trained:
                raise RecommendationException("Model not trained")
            
            # Prepare input
            input_values = [skin_type, concern_1, concern_2, concern_3]
            encoded_values = []
            
            for col, value in zip(self.feature_columns, input_values):
                le = self.label_encoders[col]
                if value not in le.classes_:
                    available = sorted(le.classes_.tolist())[:5]
                    raise RecommendationException(
                        f"Invalid {col}: '{value}'",
                        {'available': available}
                    )
                encoded_values.append(le.transform([value])[0])
            
            X_input = np.array(encoded_values).reshape(1, -1)
            
            # Get predictions
            if hasattr(self.ensemble_model, 'predict_proba'):
                proba = self.ensemble_model.predict_proba(X_input)[0]
                top_indices = np.argsort(proba)[-top_n:][::-1]
                
                recommendations = []
                for rank, idx in enumerate(top_indices, 1):
                    if idx < len(self.data):
                        product = self.data.iloc[idx]
                        recommendations.append({
                            'rank': rank,
                            'label': product['label'],
                            'brand': product['brand'],
                            'name': product['name'],
                            'price': product['price'],
                            'confidence': float(proba[idx])
                        })
                
                return pd.DataFrame(recommendations)
            else:
                raise RecommendationException("Model not configured for probability predictions")
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            raise
    
    def save(self, filepath):
        """Save model to disk"""
        try:
            print(f"\n💾 Saving model to {filepath}...")
            data = {
                'ensemble_model': self.ensemble_model,
                'knn_model': self.knn_model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'metrics': self.model_metrics,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'timestamp': datetime.now()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"✅ Model saved successfully!")
            print(f"   Location: {filepath}")
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load(self, filepath):
        """Load model from disk"""
        try:
            logger.info(f"Attempting to load model from {filepath}")
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.ensemble_model = data.get('ensemble_model')
            self.knn_model = data.get('knn_model')
            self.label_encoders = data.get('label_encoders', {})
            self.scaler = data.get('scaler')
            self.feature_columns = data.get('feature_columns', self.feature_columns)
            self.model_metrics = data.get('metrics', {})
            self.feature_importance = data.get('feature_importance', {})
            self.is_trained = data.get('is_trained', False)
            
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Model timestamp: {data.get('timestamp', 'N/A')}")
            logger.info(f"Label encoders loaded: {len(self.label_encoders)} encoders")
            logger.info(f"Model trained: {self.is_trained}")
            
            # Verify label encoders have data
            for col in self.feature_columns:
                if col in self.label_encoders:
                    n_classes = len(self.label_encoders[col].classes_)
                    logger.info(f"  - {col}: {n_classes} classes")
            
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            return False


# Global model instance
model = RecommendationModel()
