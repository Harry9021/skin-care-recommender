# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for React frontend

# Global variables for model and encoders
knn = None
label_encoders = {}
feature_columns = ['skin type', 'concern', 'concern 2', 'concern 3']
data = None

def load_and_preprocess_data(filepath="ml_model/unused/result.csv"):
    """
    Load and preprocess the dataset with proper error handling
    
    Args:
        filepath: Path to CSV dataset
        
    Returns:
        Cleaned pandas DataFrame
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file '{filepath}' not found!")
        
        # Load dataset
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset...")
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display initial info
        print("\nInitial missing values:")
        print(df[feature_columns + ['label']].isnull().sum())
        
        # Handle missing values in concern columns
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Handling missing values...")
        imputer = SimpleImputer(strategy='most_frequent')
        concern_cols = ['concern', 'concern 2', 'concern 3']
        df[concern_cols] = imputer.fit_transform(df[concern_cols])
        
        # Drop rows where 'skin type' is missing (critical feature)
        rows_before = len(df)
        df = df.dropna(subset=['skin type'])
        rows_after = len(df)
        data_loss = ((rows_before - rows_after) / rows_before) * 100
        print(f"Dropped {rows_before - rows_after} rows with missing skin type ({data_loss:.2f}% data loss)")
        
        # Drop rows where label is missing
        df = df.dropna(subset=['label'])
        print(f"Final dataset size: {len(df)} rows")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def encode_features(df, fit=True):
    """
    Encode categorical features using LabelEncoder
    
    Args:
        df: DataFrame with features to encode
        fit: If True, fit new encoders. If False, use existing encoders
    
    Returns:
        Encoded DataFrame
    """
    global label_encoders
    
    df_encoded = df.copy()
    
    for col in feature_columns:
        if fit:
            # Create and fit new encoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded '{col}': {len(le.classes_)} unique values")
            print(f"  Classes: {list(le.classes_)[:10]}...")  # Show first 10
        else:
            # Use existing encoder
            le = label_encoders[col]
            # Handle unseen categories
            df_encoded[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    return df_encoded

def train_model(X_train, y_train):
    """
    Train KNN model with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained KNN model
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training model...")
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn_base = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn_base, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Accuracy score
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return accuracy

def save_model(model, encoders, filepath="skincare_model.pkl"):
    """
    Save trained model and encoders to disk
    
    Args:
        model: Trained KNN model
        encoders: Dictionary of LabelEncoders
        filepath: Path to save the model
    """
    try:
        model_data = {
            'model': model,
            'encoders': encoders,
            'feature_columns': feature_columns,
            'timestamp': datetime.now()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model(filepath="skincare_model.pkl"):
    """
    Load saved model and encoders from disk
    
    Args:
        filepath: Path to saved model
        
    Returns:
        True if loaded successfully, False otherwise
    """
    global knn, label_encoders
    
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            knn = model_data['model']
            label_encoders = model_data['encoders']
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model loaded from {filepath}")
            print(f"Model trained on: {model_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def recommend_top_products(skin_type, concern_1, concern_2, concern_3, top_n=10):
    """
    Recommend top N products based on user input
    
    Args:
        skin_type: User's skin type (string)
        concern_1: First skin concern (string)
        concern_2: Second skin concern (string)
        concern_3: Third skin concern (string)
        top_n: Number of recommendations to return
    
    Returns:
        DataFrame with recommended products or error dictionary
    """
    global knn, label_encoders, data
    
    try:
        # Validate model is loaded
        if knn is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing recommendation request...")
        print(f"Input: skin_type='{skin_type}', concerns=['{concern_1}', '{concern_2}', '{concern_3}']")
        
        # Use exact values from frontend (no normalization needed if frontend sends dataset values)
        input_values = [skin_type, concern_1, concern_2, concern_3]
        
        print(f"Processing: skin_type='{input_values[0]}', concerns=['{input_values[1]}', '{input_values[2]}', '{input_values[3]}']")
        
        # Encode user input
        encoded_input = []
        
        for col, value in zip(feature_columns, input_values):
            le = label_encoders[col]
            
            # Check if the value exists in trained categories
            if value not in le.classes_:
                available_values = ', '.join(sorted(le.classes_)[:10])
                total_options = len(le.classes_)
                raise ValueError(
                    f"Invalid {col}: '{value}'. "
                    f"Available options ({total_options} total): {available_values}"
                    f"{'...' if total_options > 10 else ''}"
                )
            
            # Encode the value
            encoded_value = le.transform([value])[0]
            encoded_input.append(encoded_value)
            print(f"  Encoded '{value}' → {encoded_value}")
        
        # Reshape for prediction
        encoded_input = np.array(encoded_input).reshape(1, -1)
        
        # Get nearest neighbors
        distances, indices = knn.kneighbors(encoded_input, n_neighbors=min(top_n, len(data)))
        
        print(f"Found {len(indices[0])} recommendations")
        
        # Get recommended products
        recommended_products = data.iloc[indices[0]][['label', 'brand', 'name', 'price']].copy()
        
        # Add similarity score
        similarity_scores = 1 / (1 + distances[0])
        recommended_products['similarity_score'] = similarity_scores
        recommended_products['rank'] = range(1, len(recommended_products) + 1)
        
        # Reset index for clean output
        recommended_products = recommended_products.reset_index(drop=True)
        
        print(f"Top recommendation: {recommended_products.iloc[0]['name']} (similarity: {similarity_scores[0]:.4f})")
        
        return recommended_products
    
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        return {"error": str(ve)}
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        return {"error": f"Recommendation failed: {str(e)}"}

# ==================== FLASK API ROUTES ====================

@app.route('/', methods=['GET'])
def home():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "online",
        "message": "Skincare Product Recommendation API",
        "version": "2.0",
        "endpoints": {
            "/": "GET - Health check and API info",
            "/recommend": "POST - Get product recommendations",
            "/categories": "GET - Get available categories for inputs",
            "/model_info": "GET - Get model information and statistics"
        },
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for product recommendations
    
    Expected JSON format:
    {
        "skin_type": "Oily",
        "concern_1": "Acne or Blemishes",
        "concern_2": "Dark Spots",
        "concern_3": "Oil Control",
        "top_n": 10  (optional, defaults to 10)
    }
    """
    try:
        # Validate request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        user_input = request.json
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Received recommendation request")
        print(f"Request data: {user_input}")
        
        # Validate required fields
        required_fields = ['skin_type', 'concern_1', 'concern_2', 'concern_3']
        missing_fields = [field for field in required_fields if field not in user_input]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "required_fields": required_fields
            }), 400
        
        # Get top_n parameter (default 10)
        top_n = user_input.get('top_n', 10)
        
        # Validate top_n
        if not isinstance(top_n, int) or top_n < 1 or top_n > 50:
            return jsonify({"error": "top_n must be an integer between 1 and 50"}), 400
        
        # Get recommendations
        recommendations = recommend_top_products(
            user_input['skin_type'],
            user_input['concern_1'],
            user_input['concern_2'],
            user_input['concern_3'],
            top_n=top_n
        )
        
        # Check if error occurred
        if isinstance(recommendations, dict) and 'error' in recommendations:
            return jsonify(recommendations), 400
        
        # Convert DataFrame to dict and return
        response_data = {
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Successfully returned {len(recommendations)} recommendations")
        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"Error in /recommend endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """
    Get available categories for each feature
    Returns all valid options from the dataset for dropdowns
    
    Response format:
    {
        "status": "success",
        "categories": {
            "skin type": ["Dry", "Oily", "Combination", "Normal", ...],
            "concern": ["Acne or Blemishes", "Dark Spots", ...],
            "concern 2": ["Anti-Pollution", "Hydration", ...],
            "concern 3": ["Oil Control", "Pore Care", ...]
        },
        "summary": {
            "total_skin_types": 5,
            "total_concerns": 25,
            "unique_concerns": 25
        }
    }
    """
    try:
        if not label_encoders:
            return jsonify({
                "error": "Model not trained yet. Please wait for initialization."
            }), 503
        
        categories = {}
        all_concerns = set()
        
        for col in feature_columns:
            # Get all classes (sorted for better UX)
            classes = sorted(label_encoders[col].classes_.tolist())
            categories[col] = classes
            
            # Collect all concerns
            if 'concern' in col:
                all_concerns.update(classes)
        
        # Create summary statistics
        summary = {
            "total_skin_types": len(categories.get('skin type', [])),
            "total_concerns": sum(
                len(categories.get(col, [])) 
                for col in feature_columns if 'concern' in col
            ),
            "unique_concerns": len(all_concerns)
        }
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Categories endpoint accessed")
        print(f"Returning {summary['total_skin_types']} skin types and {summary['unique_concerns']} unique concerns")
        
        return jsonify({
            "status": "success",
            "categories": categories,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        print(f"Error in /categories endpoint: {str(e)}")
        return jsonify({
            "error": "Failed to retrieve categories",
            "message": str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get model information and statistics
    Useful for debugging and monitoring
    """
    try:
        if knn is None:
            return jsonify({
                "error": "Model not trained yet. Please wait for initialization."
            }), 503
        
        info = {
            "model_type": "KNeighborsClassifier",
            "n_neighbors": knn.n_neighbors,
            "weights": knn.weights,
            "metric": knn.metric,
            "n_samples_fit": knn.n_samples_fit_,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "dataset_size": len(data) if data is not None else 0,
            "n_categories": {
                col: len(label_encoders[col].classes_) 
                for col in feature_columns
            },
            "product_labels": sorted(data['label'].unique().tolist()) if data is not None else []
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

# ==================== SYSTEM INITIALIZATION ====================

def initialize_system():
    """
    Initialize the recommendation system
    Tries to load existing model, otherwise trains a new one
    """
    global knn, data
    
    print("=" * 80)
    print(" " * 20 + "SKINCARE PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Try to load existing model
    if load_model():
        print("\n✓ Using pre-trained model")
        # Still need to load data for recommendations
        data = load_and_preprocess_data()
    else:
        print("\n✗ No pre-trained model found. Training new model...\n")
        
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Prepare features and target
        X = data[feature_columns].copy()
        y = data['label']
        
        # Encode features
        X_encoded = encode_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Number of classes: {len(y.unique())}")
        
        # Train model
        knn = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(knn, X_test, y_test)
        
        # Save model
        save_model(knn, label_encoders)
    
    # Display category information
    print("\n" + "=" * 80)
    print("AVAILABLE CATEGORIES:")
    print("=" * 80)
    for col in feature_columns:
        classes = label_encoders[col].classes_
        print(f"\n{col.upper()}: {len(classes)} options")
        print(f"  {', '.join(sorted(classes)[:5])}..." if len(classes) > 5 else f"  {', '.join(sorted(classes))}")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "✓ System initialized successfully!")
    print("=" * 80 + "\n")

# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Display startup information
    print("\n" + "=" * 80)
    print("Starting Flask server...")
    print("=" * 80)
    print(f"\n🌐 API URL: http://localhost:5000")
    print(f"🌐 Network URL: http://0.0.0.0:5000")
    print("\n📋 Available Endpoints:")
    print("  ├─ GET  /              → Health check and API info")
    print("  ├─ POST /recommend     → Get product recommendations")
    print("  ├─ GET  /categories    → Get available input options")
    print("  └─ GET  /model_info    → Get model details")
    print("\n💡 Example request:")
    print('  curl -X POST http://localhost:5000/recommend \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"skin_type":"Oily","concern_1":"Acne or Blemishes","concern_2":"Dark Spots","concern_3":"Oil Control"}\'')
    print("\n⚠️  Press CTRL+C to stop the server\n")
    print("=" * 80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)