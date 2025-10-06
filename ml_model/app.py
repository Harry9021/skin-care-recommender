# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and encoders
knn = None
label_encoders = {}
feature_columns = ['skin type', 'concern', 'concern 2', 'concern 3']
data = None
X_train_indices = None

def load_and_preprocess_data(filepath="ml_model/to_be_use_dataset.csv"):
    """
    Load and preprocess the dataset with proper error handling
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file '{filepath}' not found!")
        
        # Load dataset
        print(f"[{datetime.now()}] Loading dataset...")
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display initial info
        print("\nInitial missing values:")
        print(df[feature_columns + ['label']].isnull().sum())
        
        # Step 1: Handle missing values in concern columns
        print("\n[{datetime.now()}] Handling missing values...")
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
    """
    print(f"\n[{datetime.now()}] Training model...")
    
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
    """
    print(f"\n[{datetime.now()}] Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return accuracy

def save_model(model, encoders, filepath="skincare_model.pkl"):
    """
    Save trained model and encoders
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
        print(f"\n[{datetime.now()}] Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def load_model(filepath="skincare_model.pkl"):
    """
    Load saved model and encoders
    """
    global knn, label_encoders
    
    try:
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            knn = model_data['model']
            label_encoders = model_data['encoders']
            print(f"[{datetime.now()}] Model loaded from {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def recommend_top_products(skin_type, concern_1, concern_2, concern_3, top_n=10):
    """
    Recommend top N products based on user input
    
    Args:
        skin_type: User's skin type
        concern_1, concern_2, concern_3: User's skin concerns
        top_n: Number of recommendations to return
    
    Returns:
        DataFrame with recommended products
    """
    global knn, label_encoders, data
    
    try:
        # Validate model is loaded
        if knn is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Encode user input
        encoded_input = []
        input_values = [skin_type, concern_1, concern_2, concern_3]
        
        for col, value in zip(feature_columns, input_values):
            le = label_encoders[col]
            if value not in le.classes_:
                # Handle unseen categories
                available_values = ', '.join(le.classes_[:5])
                raise ValueError(
                    f"Invalid {col}: '{value}'. Available options include: {available_values}..."
                )
            encoded_input.append(le.transform([value])[0])
        
        # Reshape for prediction
        encoded_input = np.array(encoded_input).reshape(1, -1)
        
        # Get nearest neighbors
        distances, indices = knn.kneighbors(encoded_input, n_neighbors=min(top_n, len(data)))
        
        # Get recommended products
        recommended_products = data.iloc[indices[0]][['label', 'brand', 'name', 'price']].copy()
        
        # Add similarity score (inverse of distance, normalized)
        similarity_scores = 1 / (1 + distances[0])
        recommended_products['similarity_score'] = similarity_scores
        recommended_products['rank'] = range(1, len(recommended_products) + 1)
        
        # Reset index for clean output
        recommended_products = recommended_products.reset_index(drop=True)
        
        return recommended_products
    
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

# Flask API Routes

@app.route('/', methods=['GET'])
def home():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "online",
        "message": "Skincare Product Recommendation API",
        "endpoints": {
            "/recommend": "POST - Get product recommendations",
            "/categories": "GET - Get available categories",
            "/model_info": "GET - Get model information"
        }
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for product recommendations
    
    Expected JSON format:
    {
        "skin_type": "oily",
        "concern_1": "acne",
        "concern_2": "dark spots",
        "concern_3": "aging",
        "top_n": 10  (optional, defaults to 10)
    }
    """
    try:
        # Validate request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        user_input = request.json
        
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
        return jsonify({
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations.to_dict('records')
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """
    Get available categories for each feature
    """
    try:
        if not label_encoders:
            return jsonify({"error": "Model not trained yet"}), 400
        
        categories = {}
        for col in feature_columns:
            categories[col] = label_encoders[col].classes_.tolist()
        
        return jsonify({
            "status": "success",
            "categories": categories
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get model information and statistics
    """
    try:
        if knn is None:
            return jsonify({"error": "Model not trained yet"}), 400
        
        info = {
            "model_type": "KNeighborsClassifier",
            "n_neighbors": knn.n_neighbors,
            "weights": knn.weights,
            "metric": knn.metric,
            "n_samples_fit": knn.n_samples_fit_,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
            "dataset_size": len(data) if data is not None else 0
        }
        
        return jsonify({
            "status": "success",
            "model_info": info
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main training and server startup
def initialize_system():
    """
    Initialize the recommendation system
    """
    global knn, data
    
    print("="*60)
    print("SKINCARE PRODUCT RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Try to load existing model
    if load_model():
        print("Using pre-trained model")
        # Still need to load data for recommendations
        data = load_and_preprocess_data()
    else:
        print("Training new model...")
        
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
        
        # Train model
        knn = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(knn, X_test, y_test)
        
        # Save model
        save_model(knn, label_encoders)
    
    print("\n" + "="*60)
    print("System initialized successfully!")
    print("="*60 + "\n")

if __name__ == '__main__':
    # Initialize system (train or load model)
    initialize_system()
    
    # Run Flask app
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /              - Health check")
    print("  POST /recommend     - Get recommendations")
    print("  GET  /categories    - Get available options")
    print("  GET  /model_info    - Get model details")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
