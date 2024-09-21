# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
file_path = 'skin-care-recommender/ml_model/to_be_use_dataset.csv'
data = pd.read_csv(file_path)

# Step 1: Handle missing values
# Fill missing values in 'concern', 'concern 2', 'concern 3' with the most frequent value (mode) 
# and drop rows with missing skin types
imputer = SimpleImputer(strategy='most_frequent')

# Impute missing values for the concern columns
data['concern'] = imputer.fit_transform(data[['concern']])
data['concern 2'] = imputer.fit_transform(data[['concern 2']])
data['concern 3'] = imputer.fit_transform(data[['concern 3']])
data = data.dropna(subset=['skin type'])  # Drop rows where 'skin type' is missing

# Step 2: Feature set to include 'skin type' and all three concerns
X = data[['skin type', 'concern', 'concern 2', 'concern 3']]  # Use skin type and all concerns as features
y = data['label']  # Assuming 'label' is the product category or target

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training using KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors
knn.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of KNN model with skin type and all concerns: {accuracy * 100:.2f}%')

# Step 6: Recommendation Function for Top 10 Products
def recommend_top_products(skin_type, concern_1, concern_2, concern_3, top_n=10):
    user_input = [[skin_type, concern_1, concern_2, concern_3]]  # Skin type and three concerns as input
    distances, indices = knn.kneighbors(user_input, n_neighbors=top_n)  # Get top N neighbors
    
    # Return top products with label, brand, name, and price
    recommended_products = data.iloc[indices[0]][['label', 'brand', 'name', 'price']]  # Get relevant columns
    return recommended_products

# Flask route to handle POST requests for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get input data from request
    user_input = request.json
    skin_type = user_input['skin_type']
    concern_1 = user_input['concern_1']
    concern_2 = user_input['concern_2']
    concern_3 = user_input['concern_3']

    # Get recommendations based on input
    recommendations = recommend_top_products(skin_type, concern_1, concern_2, concern_3)
    
    # Convert DataFrame to JSON and return the response
    recommendations_json = recommendations.to_json(orient='records')
    return jsonify(recommendations_json)

if __name__ == '__main__':
    # Run Flask app
    app.run(port=5000, debug=True)
