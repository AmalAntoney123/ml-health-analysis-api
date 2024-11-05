import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class HealthPredictor:
    def __init__(self, model_path='health_predictor_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, data_file):
        # Check if model already exists
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.load_model()
            return
            
        print("Training new model...")
        # Load data
        df = pd.read_csv(data_file)
        
        # Separate features and target
        X = df.drop('diabetes_risk', axis=1)
        y = df['diabetes_risk']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump((self.model, self.scaler), self.model_path)
        
    def load_model(self):
        if not hasattr(self, 'model') or self.model is None:
            self.model, self.scaler = joblib.load(self.model_path)
    
    def predict_risk(self, user_data):
        self.load_model()
        
        # Scale user data
        user_data_scaled = self.scaler.transform([user_data])
        
        # Get prediction and probability
        prediction = self.model.predict(user_data_scaled)[0]
        probability = self.model.predict_proba(user_data_scaled)[0]
        
        return prediction, probability

def get_sample_data():
    # Sample data array [age, bmi, glucose, blood_pressure, insulin, exercise, family_history]
    sample_data = [
        [45, 28.5, 120, 80, 100, 3, 0],  # Low risk sample
        [55, 34.2, 165, 92, 180, 1, 1],   # High risk sample
    ]
    return sample_data

def main():
    # Initialize predictor
    predictor = HealthPredictor()
    
    try:
        # Train or load model
        predictor.train('health_data.csv')
        
        # Use sample data instead of user input
        sample_data = get_sample_data()
        
        for i, user_data in enumerate(sample_data, 1):
            print(f"\n=== Sample {i} Assessment ===")
            print(f"Input data: {user_data}")
            
            # Make prediction
            prediction, probability = predictor.predict_risk(user_data)
            
            # Display results
            print("\n=== Assessment Results ===")
            risk_level = "High" if prediction == 1 else "Low"
            risk_percentage = probability[1] * 100 if prediction == 1 else probability[0] * 100
            
            print(f"Risk Level: {risk_level}")
            print(f"Confidence: {risk_percentage:.1f}%")
            
            # Provide recommendations
            print("\nRecommendations:")
            if prediction == 1:
                print("- Consult with a healthcare provider")
                print("- Monitor blood glucose regularly")
                print("- Maintain a healthy diet")
                print("- Increase physical activity")
                print("- Regular health check-ups")
            else:
                print("- Maintain healthy lifestyle")
                print("- Regular exercise")
                print("- Balanced diet")
                print("- Annual health check-ups")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()