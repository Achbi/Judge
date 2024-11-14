from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)
CORS(app)  # This allows all origins

MODEL_FILE = 'gb_classifier.joblib'
FEATURE_NAMES_FILE = 'feature_names.joblib'

def train_model():
    # Load the dataset
    df = pd.read_csv('a.csv')

    # Feature Engineering
    df['risk_score'] = df['risk_of_escape'].astype(int) + df['risk_of_influence'].astype(int) + df['served_half_term'].astype(int)
    df['penalty_severity'] = df['penalty'].map({"Fine": 1, "Imprisonment": 2, "Both": 3}) * df['imprisonment_duration_served']

    # Prepare features and target
    X = df.drop(["bail_eligibility", "case_id", "penalty", "imprisonment_duration_served", "risk_of_escape", "risk_of_influence", "served_half_term"], axis=1)
    y = df["bail_eligibility"]

    # Convert categorical variables into numerical format
    X = pd.get_dummies(X, drop_first=True)

    # Save feature names
    joblib.dump(X.columns.tolist(), FEATURE_NAMES_FILE)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    joblib.dump(model, MODEL_FILE)
    
    return model, X.columns.tolist()

def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(FEATURE_NAMES_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            feature_names = joblib.load(FEATURE_NAMES_FILE)
            return model, feature_names
        except:
            print("Error loading model or feature names. Training a new one.")
    return train_model()

# Load or train the model when the app starts
model, feature_names = load_or_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])
    
    # Perform the same preprocessing as in the training
    input_df['risk_score'] = (input_df['risk_of_escape'].astype(int) + 
                              input_df['risk_of_influence'].astype(int) + 
                              input_df['served_half_term'].astype(int))
    
    input_df['penalty_severity'] = input_df['penalty'].map({"Fine": 1, "Imprisonment": 2, "Both": 3}) * input_df['imprisonment_duration_served']
    
    # Drop unnecessary columns
    input_df = input_df.drop(["penalty", "imprisonment_duration_served", "risk_of_escape", "risk_of_influence", "served_half_term"], axis=1)
    
    # Convert categorical variables into numerical format
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Align the input data with the training data columns
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    return jsonify({'bail_eligibility': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)