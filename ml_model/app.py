# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    cross_val_score,
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    make_scorer
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import os
from datetime import datetime
import warnings
import json
import logging
from collections import Counter

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models and encoders
ensemble_model = None
knn_model = None  # Keep separate for backward compatibility
label_encoders = {}
scaler = None
feature_selector = None
feature_columns = ['skin type', 'concern', 'concern 2', 'concern 3']
data = None
model_metrics = {}
feature_importance = {}

class RobustRecommendationSystem:
    """Enhanced recommendation system with robust metrics and multiple models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.metrics = {}
        self.feature_importance = {}
        
    def create_ensemble_model(self):
        """Create an ensemble of multiple models for robust predictions"""
        models = [
            ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ]
        return VotingClassifier(estimators=models, voting='soft')
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics including precision, recall, F1"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 for each class and weighted average
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Macro averages (treating all classes equally)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Micro averages (global calculation)
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Per-class metrics
        unique_classes = np.unique(y_true)
        metrics['per_class_precision'] = {}
        metrics['per_class_recall'] = {}
        metrics['per_class_f1'] = {}
        
        for class_label in unique_classes[:10]:  # Limit to first 10 for display
            class_mask = y_true == class_label
            if np.sum(class_mask) > 0:
                metrics['per_class_precision'][str(class_label)] = precision_score(
                    y_true == class_label, y_pred == class_label, zero_division=0
                )
                metrics['per_class_recall'][str(class_label)] = recall_score(
                    y_true == class_label, y_pred == class_label, zero_division=0
                )
                metrics['per_class_f1'][str(class_label)] = f1_score(
                    y_true == class_label, y_pred == class_label, zero_division=0
                )
        
        # Support (number of samples per class)
        metrics['support'] = dict(Counter(y_true))
        
        return metrics

recommender = RobustRecommendationSystem()

def load_and_preprocess_data(filepath="ml_model/unused/result.csv"):
    """
    Enhanced data loading with additional preprocessing and validation
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file '{filepath}' not found!")
        
        logger.info("Loading dataset...")
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data validation
        logger.info("Validating data...")
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values more intelligently
        logger.info("Handling missing values...")
        
        # For concern columns, use KNN imputation if possible
        concern_cols = ['concern', 'concern 2', 'concern 3']
        
        # Simple imputation for now (can be enhanced with KNNImputer)
        imputer = SimpleImputer(strategy='most_frequent')
        df[concern_cols] = imputer.fit_transform(df[concern_cols])
        
        # Drop rows with missing critical features
        df = df.dropna(subset=['skin type', 'label'])
        
        # Data quality checks
        logger.info("Performing data quality checks...")
        
        # Check class distribution
        label_counts = df['label'].value_counts()
        min_samples_per_class = 2
        
        # Remove classes with too few samples
        valid_labels = label_counts[label_counts >= min_samples_per_class].index
        df = df[df['label'].isin(valid_labels)]
        
        logger.info(f"Final dataset size: {len(df)} rows")
        logger.info(f"Number of unique products: {df['label'].nunique()}")
        
        # Add feature engineering
        df = engineer_features(df)
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def engineer_features(df):
    """
    Create additional features for better recommendations
    """
    try:
        # Create combination features
        df['skin_concern_combo'] = df['skin type'].astype(str) + '_' + df['concern'].astype(str)
        
        # Count total concerns (non-null)
        concern_cols = ['concern', 'concern 2', 'concern 3']
        df['concern_count'] = df[concern_cols].notna().sum(axis=1)
        
        # Create concern similarity score (how similar are the concerns)
        df['concern_diversity'] = df[concern_cols].nunique(axis=1)
        
        logger.info("Feature engineering completed")
        return df
    except Exception as e:
        logger.warning(f"Feature engineering failed: {str(e)}")
        return df

def encode_features_robust(df, fit=True):
    """
    Enhanced encoding with additional features and robustness
    """
    global label_encoders, scaler
    
    df_encoded = df.copy()
    
    # Encode categorical features
    for col in feature_columns:
        if fit:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Encoded '{col}': {len(le.classes_)} unique values")
        else:
            le = label_encoders[col]
            df_encoded[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Add engineered features if they exist
    if 'skin_concern_combo' in df.columns:
        if fit:
            le = LabelEncoder()
            df_encoded['skin_concern_combo'] = le.fit_transform(df['skin_concern_combo'].astype(str))
            label_encoders['skin_concern_combo'] = le
        else:
            if 'skin_concern_combo' in label_encoders:
                le = label_encoders['skin_concern_combo']
                df_encoded['skin_concern_combo'] = df['skin_concern_combo'].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    # Scale numerical features
    numerical_features = ['concern_count', 'concern_diversity'] if 'concern_count' in df.columns else []
    if numerical_features:
        if fit:
            scaler = StandardScaler()
            df_encoded[numerical_features] = scaler.fit_transform(df[numerical_features])
        else:
            if scaler:
                df_encoded[numerical_features] = scaler.transform(df[numerical_features])
    
    return df_encoded

def train_robust_model(X_train, y_train, X_val=None, y_val=None):
    """
    Train multiple models and select the best based on comprehensive metrics
    """
    global ensemble_model, knn_model, model_metrics, feature_importance
    
    logger.info("Training robust ensemble model...")
    
    # Handle class imbalance with SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_train).values())-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        logger.info(f"Applied SMOTE: {len(X_train)} -> {len(X_train_balanced)} samples")
    except:
        X_train_balanced, y_train_balanced = X_train, y_train
        logger.warning("SMOTE failed, using original data")
    
    # Train KNN with GridSearch
    logger.info("Training KNN model...")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn_base = KNeighborsClassifier()
    knn_grid = GridSearchCV(
        knn_base, 
        knn_params, 
        cv=StratifiedKFold(n_splits=5),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    knn_grid.fit(X_train_balanced, y_train_balanced)
    knn_model = knn_grid.best_estimator_
    
    logger.info(f"Best KNN params: {knn_grid.best_params_}")
    
    # Train Random Forest
    logger.info("Training Random Forest model...")
    rf_params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(
        rf_base,
        rf_params,
        cv=StratifiedKFold(n_splits=3),
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train_balanced, y_train_balanced)
    rf_model = rf_grid.best_estimator_
    
    # Store feature importance
    feature_importance['random_forest'] = dict(zip(
        feature_columns, 
        rf_model.feature_importances_[:len(feature_columns)]
    ))
    
    # Create ensemble
    logger.info("Creating ensemble model...")
    ensemble_model = VotingClassifier(
        estimators=[
            ('knn', knn_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate all models
    if X_val is not None and y_val is not None:
        logger.info("Evaluating models on validation set...")
        
        # KNN metrics
        knn_pred = knn_model.predict(X_val)
        knn_metrics = recommender.calculate_comprehensive_metrics(y_val, knn_pred)
        
        # RF metrics
        rf_pred = rf_model.predict(X_val)
        rf_metrics = recommender.calculate_comprehensive_metrics(y_val, rf_pred)
        
        # Ensemble metrics
        ensemble_pred = ensemble_model.predict(X_val)
        ensemble_metrics = recommender.calculate_comprehensive_metrics(y_val, ensemble_pred)
        
        # Store metrics
        model_metrics = {
            'knn': knn_metrics,
            'random_forest': rf_metrics,
            'ensemble': ensemble_metrics
        }
        
        # Display comparison
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON (Validation Set)")
        logger.info("="*60)
        
        for model_name, metrics in model_metrics.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision_weighted']:.4f} (weighted)")
            logger.info(f"  Recall:    {metrics['recall_weighted']:.4f} (weighted)")
            logger.info(f"  F1-Score:  {metrics['f1_weighted']:.4f} (weighted)")
    
    return ensemble_model

def evaluate_model_comprehensive(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive evaluation with detailed metrics
    """
    logger.info(f"\nEvaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate comprehensive metrics
    metrics = recommender.calculate_comprehensive_metrics(y_test, y_pred)
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info(f"{model_name.upper()} EVALUATION RESULTS")
    logger.info("="*60)
    
    logger.info(f"\nOVERALL METRICS:")
    logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"\nWEIGHTED AVERAGES (considering class imbalance):")
    logger.info(f"  Precision:          {metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall:             {metrics['recall_weighted']:.4f}")
    logger.info(f"  F1-Score:           {metrics['f1_weighted']:.4f}")
    
    logger.info(f"\nMACRO AVERAGES (treating all classes equally):")
    logger.info(f"  Precision:          {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:             {metrics['recall_macro']:.4f}")
    logger.info(f"  F1-Score:           {metrics['f1_macro']:.4f}")
    
    logger.info(f"\nMICRO AVERAGES (global calculation):")
    logger.info(f"  Precision:          {metrics['precision_micro']:.4f}")
    logger.info(f"  Recall:             {metrics['recall_micro']:.4f}")
    logger.info(f"  F1-Score:           {metrics['f1_micro']:.4f}")
    
    # Classification Report
    logger.info("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Display top performing classes
    if 'per_class_f1' in metrics:
        sorted_classes = sorted(
            metrics['per_class_f1'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        logger.info("\nTOP 5 BEST PERFORMING CLASSES (F1-Score):")
        for class_label, f1 in sorted_classes:
            logger.info(f"  {class_label}: {f1:.4f}")
    
    return metrics

def save_model_enhanced(models_dict, encoders, metrics, filepath="skincare_model_enhanced.pkl"):
    """
    Save all models and metadata
    """
    try:
        model_data = {
            'ensemble_model': models_dict.get('ensemble'),
            'knn_model': models_dict.get('knn'),
            'encoders': encoders,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'feature_columns': feature_columns,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'timestamp': datetime.now()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
        
        # Save metrics to JSON for easy reading
        metrics_filepath = filepath.replace('.pkl', '_metrics.json')
        with open(metrics_filepath, 'w') as f:
            json.dump(
                {k: v for k, v in metrics.items() if k != 'per_class_precision' and k != 'per_class_recall' and k != 'per_class_f1'},
                f, 
                indent=2,
                default=str
            )
        logger.info(f"Metrics saved to {metrics_filepath}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")

def load_model_enhanced(filepath="skincare_model_enhanced.pkl"):
    """
    Load enhanced model with all components
    """
    global ensemble_model, knn_model, label_encoders, scaler, feature_selector, model_metrics, feature_importance
    
    try:
        # Try enhanced model first
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            ensemble_model = model_data.get('ensemble_model')
            knn_model = model_data.get('knn_model')
            label_encoders = model_data['encoders']
            scaler = model_data.get('scaler')
            feature_selector = model_data.get('feature_selector')
            model_metrics = model_data.get('metrics', {})
            feature_importance = model_data.get('feature_importance', {})
            
            logger.info(f"Enhanced model loaded from {filepath}")
            logger.info(f"Model trained on: {model_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if model_metrics:
                logger.info(f"Model performance: F1-Score = {model_metrics.get('f1_weighted', 'N/A'):.4f}")
            
            return True
        
        # Fall back to original model
        original_filepath = "skincare_model.pkl"
        if os.path.exists(original_filepath):
            with open(original_filepath, 'rb') as f:
                model_data = pickle.load(f)
            knn_model = model_data['model']
            label_encoders = model_data['encoders']
            ensemble_model = None  # No ensemble in original
            logger.info(f"Original model loaded from {original_filepath}")
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def recommend_top_products_enhanced(skin_type, concern_1, concern_2, concern_3, top_n=10):
    """
    Enhanced recommendation with confidence scores and diversity
    """
    global ensemble_model, knn_model, label_encoders, data
    
    try:
        # Use ensemble if available, otherwise fall back to KNN
        model = ensemble_model if ensemble_model is not None else knn_model
        
        if model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        logger.info(f"Processing recommendation request...")
        logger.info(f"Input: skin_type='{skin_type}', concerns=['{concern_1}', '{concern_2}', '{concern_3}']")
        
        input_values = [skin_type, concern_1, concern_2, concern_3]
        
        # Encode user input
        encoded_input = []
        for col, value in zip(feature_columns, input_values):
            le = label_encoders[col]
            
            if value not in le.classes_:
                available_values = ', '.join(sorted(le.classes_)[:10])
                raise ValueError(
                    f"Invalid {col}: '{value}'. Available options: {available_values}..."
                )
            
            encoded_value = le.transform([value])[0]
            encoded_input.append(encoded_value)
        
        # Add engineered features if available
        if 'skin_concern_combo' in label_encoders:
            combo = f"{skin_type}_{concern_1}"
            le = label_encoders['skin_concern_combo']
            if combo in le.classes_:
                encoded_input.append(le.transform([combo])[0])
        
        encoded_input = np.array(encoded_input).reshape(1, -1)
        
        # Get predictions with probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(encoded_input)[0]
            top_indices = np.argsort(probabilities)[-top_n:][::-1]
            
            # Get unique products
            unique_products = data['label'].unique()
            recommendations = []
            
            for idx in top_indices:
                if idx < len(unique_products):
                    product_label = unique_products[idx]
                    product_data = data[data['label'] == product_label].iloc[0]
                    recommendations.append({
                        'label': product_label,
                        'brand': product_data['brand'],
                        'name': product_data['name'],
                        'price': product_data['price'],
                        'confidence_score': float(probabilities[idx]),
                        'rank': len(recommendations) + 1
                    })
            
            recommended_products = pd.DataFrame(recommendations)
            
        else:
            # Fall back to KNN-style recommendations
            if hasattr(model, 'kneighbors'):
                distances, indices = model.kneighbors(encoded_input, n_neighbors=min(top_n, len(data)))
            else:
                # For ensemble without kneighbors
                prediction = model.predict(encoded_input)[0]
                similar_products = data[data['label'] == prediction].head(top_n)
                
                recommended_products = similar_products[['label', 'brand', 'name', 'price']].copy()
                recommended_products['confidence_score'] = 1.0
                recommended_products['rank'] = range(1, len(recommended_products) + 1)
                
                return recommended_products
            
            recommended_products = data.iloc[indices[0]][['label', 'brand', 'name', 'price']].copy()
            similarity_scores = 1 / (1 + distances[0])
            recommended_products['confidence_score'] = similarity_scores
            recommended_products['rank'] = range(1, len(recommended_products) + 1)
        
        # Add diversity to recommendations (avoid all same brand)
        if len(recommended_products) > 5:
            unique_brands = recommended_products['brand'].unique()
            if len(unique_brands) < 3:
                logger.info("Adding diversity to recommendations...")
                # Add some diversity logic here if needed
        
        recommended_products = recommended_products.reset_index(drop=True)
        logger.info(f"Top recommendation: {recommended_products.iloc[0]['name']} (confidence: {recommended_products.iloc[0]['confidence_score']:.4f})")
        
        return recommended_products
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return {"error": f"Recommendation failed: {str(e)}"}

# ==================== FLASK API ROUTES ====================

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "Enhanced Skincare Product Recommendation API",
        "version": "3.0",
        "features": [
            "Ensemble model with KNN, Random Forest",
            "Comprehensive metrics (Precision, Recall, F1-Score)",
            "Class imbalance handling with SMOTE",
            "Feature engineering and importance",
            "Confidence scores for recommendations"
        ],
        "endpoints": {
            "/": "GET - Health check and API info",
            "/recommend": "POST - Get product recommendations",
            "/categories": "GET - Get available categories for inputs",
            "/model_info": "GET - Get model information and statistics",
            "/metrics": "GET - Get detailed model performance metrics"
        },
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """API endpoint for product recommendations - ORIGINAL FORMAT MAINTAINED"""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        user_input = request.json
        logger.info(f"Received recommendation request: {user_input}")
        
        # Validate required fields
        required_fields = ['skin_type', 'concern_1', 'concern_2', 'concern_3']
        missing_fields = [field for field in required_fields if field not in user_input]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "required_fields": required_fields
            }), 400
        
        top_n = user_input.get('top_n', 10)
        
        if not isinstance(top_n, int) or top_n < 1 or top_n > 50:
            return jsonify({"error": "top_n must be an integer between 1 and 50"}), 400
        
        # Get recommendations
        recommendations = recommend_top_products_enhanced(
            user_input['skin_type'],
            user_input['concern_1'],
            user_input['concern_2'],
            user_input['concern_3'],
            top_n=top_n
        )
        
        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify(recommendations), 400
        
        # Convert to original response format
        response_data = {
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully returned {len(recommendations)} recommendations")
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Error in /recommend endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available categories - ORIGINAL FORMAT MAINTAINED"""
    try:
        if not label_encoders:
            return jsonify({
                "error": "Model not trained yet. Please wait for initialization."
            }), 503
        
        categories = {}
        all_concerns = set()
        
        for col in feature_columns:
            classes = sorted(label_encoders[col].classes_.tolist())
            categories[col] = classes
            
            if 'concern' in col:
                all_concerns.update(classes)
        
        summary = {
            "total_skin_types": len(categories.get('skin type', [])),
            "total_concerns": sum(
                len(categories.get(col, [])) 
                for col in feature_columns if 'concern' in col
            ),
            "unique_concerns": len(all_concerns)
        }
        
        return jsonify({
            "status": "success",
            "categories": categories,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in /categories endpoint: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve categories",
            "message": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information - ENHANCED VERSION"""
    try:
        model = ensemble_model if ensemble_model is not None else knn_model
        
        if model is None:
            return jsonify({
                "error": "Model not trained yet. Please wait for initialization."
            }), 503
        
        info = {
            "model_type": "Ensemble (KNN + Random Forest)" if ensemble_model else "KNeighborsClassifier",
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "dataset_size": len(data) if data is not None else 0,
            "n_categories": {
                col: len(label_encoders[col].classes_) 
                for col in feature_columns
            },
            "performance_metrics": {
                "accuracy": model_metrics.get('accuracy', 'N/A'),
                "precision_weighted": model_metrics.get('precision_weighted', 'N/A'),
                "recall_weighted": model_metrics.get('recall_weighted', 'N/A'),
                "f1_weighted": model_metrics.get('f1_weighted', 'N/A')
            },
            "feature_importance": feature_importance,
            "model_trained_at": model_metrics.get('timestamp', 'N/A')
        }
        
        # Add KNN specific info if using KNN
        if knn_model and hasattr(knn_model, 'n_neighbors'):
            info["knn_parameters"] = {
                "n_neighbors": knn_model.n_neighbors,
                "weights": knn_model.weights,
                "metric": knn_model.metric
            }
        
        return jsonify({
            "status": "success",
            "model_info": info,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve model info",
            "message": str(e)
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """NEW ENDPOINT: Get detailed model performance metrics"""
    try:
        if not model_metrics:
            return jsonify({
                "error": "No metrics available. Model might not be trained with enhanced metrics."
            }), 503
        
        # Prepare metrics for response
        metrics_response = {
            "overall_metrics": {
                "accuracy": model_metrics.get('accuracy', 'N/A'),
                "weighted_averages": {
                    "precision": model_metrics.get('precision_weighted', 'N/A'),
                    "recall": model_metrics.get('recall_weighted', 'N/A'),
                    "f1_score": model_metrics.get('f1_weighted', 'N/A')
                },
                "macro_averages": {
                    "precision": model_metrics.get('precision_macro', 'N/A'),
                    "recall": model_metrics.get('recall_macro', 'N/A'),
                    "f1_score": model_metrics.get('f1_macro', 'N/A')
                },
                "micro_averages": {
                    "precision": model_metrics.get('precision_micro', 'N/A'),
                    "recall": model_metrics.get('recall_micro', 'N/A'),
                    "f1_score": model_metrics.get('f1_micro', 'N/A')
                }
            },
            "feature_importance": feature_importance.get('random_forest', {}),
            "class_distribution": model_metrics.get('support', {}),
            "model_comparison": {
                model_name: {
                    "accuracy": metrics.get('accuracy', 'N/A'),
                    "f1_weighted": metrics.get('f1_weighted', 'N/A')
                }
                for model_name, metrics in model_metrics.items()
                if isinstance(metrics, dict) and 'accuracy' in metrics
            }
        }
        
        return jsonify({
            "status": "success",
            "metrics": metrics_response,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in /metrics endpoint: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve metrics",
            "message": str(e)
        }), 500

# ==================== SYSTEM INITIALIZATION ====================

def initialize_system():
    """Initialize the enhanced recommendation system"""
    global ensemble_model, knn_model, data, model_metrics
    
    print("\n" + "="*80)
    print(" " * 15 + "ENHANCED SKINCARE RECOMMENDATION SYSTEM v3.0")
    print("="*80)
    print("\nFeatures:")
    print("✓ Ensemble Learning (KNN + Random Forest)")
    print("✓ Comprehensive Metrics (Precision, Recall, F1-Score)")
    print("✓ Class Imbalance Handling (SMOTE)")
    print("✓ Feature Engineering & Importance Analysis")
    print("✓ Confidence Scores for Recommendations")
    print("="*80 + "\n")
    
    # Try to load existing enhanced model
    if load_model_enhanced():
        logger.info("✓ Using pre-trained enhanced model")
        data = load_and_preprocess_data()
        
        # Display loaded metrics if available
        if model_metrics:
            logger.info("\nLOADED MODEL PERFORMANCE:")
            logger.info(f"  Accuracy:  {model_metrics.get('accuracy', 'N/A')}")
            logger.info(f"  Precision: {model_metrics.get('precision_weighted', 'N/A')}")
            logger.info(f"  Recall:    {model_metrics.get('recall_weighted', 'N/A')}")
            logger.info(f"  F1-Score:  {model_metrics.get('f1_weighted', 'N/A')}")
    else:
        logger.info("✗ No pre-trained model found. Training new enhanced model...\n")
        
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Prepare features and target
        X = data[feature_columns].copy()
        y = data['label']
        
        # Encode features with robust encoding
        X_encoded = encode_features_robust(X, fit=True)
        
        # Split data with stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"\nData Split:")
        logger.info(f"  Training set:   {len(X_train)} samples")
        logger.info(f"  Validation set: {len(X_val)} samples")
        logger.info(f"  Test set:       {len(X_test)} samples")
        logger.info(f"  Number of classes: {len(y.unique())}")
        
        # Check class distribution
        logger.info("\nClass Distribution Analysis:")
        class_counts = Counter(y_train)
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        logger.info(f"  Min samples per class: {min_class}")
        logger.info(f"  Max samples per class: {max_class}")
        logger.info(f"  Imbalance ratio: {max_class/min_class:.2f}")
        
        # Train robust model
        ensemble_model = train_robust_model(X_train, y_train, X_val, y_val)
        
        # Comprehensive evaluation on test set
        test_metrics = evaluate_model_comprehensive(ensemble_model, X_test, y_test, "Ensemble Model")
        model_metrics = test_metrics
        
        # Also evaluate KNN separately for comparison
        if knn_model:
            knn_metrics = evaluate_model_comprehensive(knn_model, X_test, y_test, "KNN Model")
            model_metrics['knn'] = knn_metrics
            model_metrics['ensemble'] = test_metrics
        
        # Cross-validation for robustness check
        logger.info("\nPerforming 5-fold cross-validation for robustness...")
        cv_scores = cross_val_score(
            ensemble_model, X_encoded, y, 
            cv=StratifiedKFold(n_splits=5), 
            scoring='f1_weighted',
            n_jobs=-1
        )
        logger.info(f"Cross-validation F1-scores: {cv_scores}")
        logger.info(f"Mean CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save enhanced model
        save_model_enhanced(
            {'ensemble': ensemble_model, 'knn': knn_model},
            label_encoders,
            model_metrics
        )
    
    # Display system information
    logger.info("\n" + "="*80)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*80)
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total products: {len(data['label'].unique())}")
    logger.info(f"  Total samples: {len(data)}")
    logger.info(f"  Features used: {len(feature_columns)}")
    
    logger.info("\nAvailable Categories:")
    for col in feature_columns:
        if col in label_encoders:
            classes = label_encoders[col].classes_
            logger.info(f"  {col.upper()}: {len(classes)} options")
            sample_classes = sorted(classes)[:5]
            logger.info(f"    Sample: {', '.join(sample_classes)}{'...' if len(classes) > 5 else ''}")
    
    if feature_importance:
        logger.info("\nFeature Importance (Random Forest):")
        if 'random_forest' in feature_importance:
            for feature, importance in sorted(
                feature_importance['random_forest'].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                logger.info(f"  {feature}: {importance:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info(" " * 20 + "✓ System initialized successfully!")
    logger.info("="*80 + "\n")

# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    # Initialize system
    try:
        initialize_system()
        
        # Display startup information
        print("\n" + "="*80)
        print("Starting Enhanced Flask Server...")
        print("="*80)
        print(f"\n🌐 API URL: http://localhost:5000")
        print("\n📋 Available Endpoints:")
        print("  ├─ GET  /              → Health check and API info")
        print("  ├─ POST /recommend     → Get product recommendations")
        print("  ├─ GET  /categories    → Get available input options")
        print("  ├─ GET  /model_info    → Get model details")
        print("  └─ GET  /metrics       → Get performance metrics")
        print("\n🚀 New Features in v3.0:")
        print("  • Ensemble model with multiple algorithms")
        print("  • Comprehensive precision, recall, and F1 scores")
        print("  • SMOTE for handling class imbalance")
        print("  • Feature importance analysis")
        print("  • Confidence scores for recommendations")
        print("\n⚠️  Press CTRL+C to stop the server\n")
        print("="*80 + "\n")
        
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        print("\n❌ System initialization failed!")
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("1. Dataset file exists at 'ml_model/unused/result.csv'")
        print("2. All required packages are installed")
        print("3. Python version is compatible (3.7+)")
        import traceback
        traceback.print_exc()