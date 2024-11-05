from flask import Flask, request, jsonify
from ml_health import HealthPredictor
import numpy as np

app = Flask(__name__)
predictor = HealthPredictor()

# Initialize the model when starting the server
try:
    predictor.train('health_data.csv')
except Exception as e:
    print(f"Error initializing model: {str(e)}")

@app.route('/health/predict', methods=['POST'])
def predict_health_risk():
    try:
        data = request.get_json()
        
        # Extract features from request
        user_data = [
            data.get('age', 0),
            data.get('bmi', 0),
            data.get('glucose', 0),
            data.get('blood_pressure', 0),
            data.get('insulin', 0),
            data.get('exercise', 0),
            data.get('family_history', 0)
        ]

        # Make prediction
        prediction, probability = predictor.predict_risk(user_data)
        
        # Prepare response
        risk_level = "High" if prediction == 1 else "Low"
        risk_percentage = probability[1] * 100 if prediction == 1 else probability[0] * 100
        
        # Get recommendations based on risk level
        recommendations = get_recommendations(prediction)

        response = {
            'risk_level': risk_level,
            'confidence': f"{risk_percentage:.1f}%",
            'recommendations': recommendations
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_recommendations(prediction):
    if prediction == 1:
        return [
            "Consult with a healthcare provider",
            "Monitor blood glucose regularly",
            "Maintain a healthy diet",
            "Increase physical activity",
            "Regular health check-ups"
        ]
    else:
        return [
            "Maintain healthy lifestyle",
            "Regular exercise",
            "Balanced diet",
            "Annual health check-ups"
        ]

@app.route('/health/sample', methods=['GET'])
def get_sample_prediction():
    try:
        sample_data = [45, 28.5, 120, 80, 100, 3, 0]  # Low risk sample
        prediction, probability = predictor.predict_risk(sample_data)
        
        risk_level = "High" if prediction == 1 else "Low"
        risk_percentage = probability[1] * 100 if prediction == 1 else probability[0] * 100
        
        response = {
            'sample_data': {
                'age': sample_data[0],
                'bmi': sample_data[1],
                'glucose': sample_data[2],
                'blood_pressure': sample_data[3],
                'insulin': sample_data[4],
                'exercise': sample_data[5],
                'family_history': sample_data[6]
            },
            'risk_level': risk_level,
            'confidence': f"{risk_percentage:.1f}%",
            'recommendations': get_recommendations(prediction)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
