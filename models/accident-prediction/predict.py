import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from typing import Dict, Tuple, List, Any

def load_model_components() -> Tuple:
    """Load all saved model components"""
    model = tf.keras.models.load_model("final/final_accident_model.keras")
    scaler = joblib.load("artifacts/scaler.pkl")
    label_encoders = joblib.load("artifacts/label_encoders.pkl")
    input_features = joblib.load("artifacts/input_features.pkl")
    cat_cols = joblib.load("artifacts/cat_cols.pkl")
    
    return model, scaler, label_encoders, input_features, cat_cols

def preprocess_input(
    input_data: Dict[str, Any],
    input_features: List[str],
    cat_cols: List[str],
    label_encoders: Dict,
    scaler
) -> np.ndarray:
    """Preprocess input data for prediction"""
    # Create DataFrame with single row from input
    input_df = pd.DataFrame([input_data])
    
    # Process datetime if provided as a string
    if 'datetime' in input_data:
        dt = pd.to_datetime(input_data['datetime'])
        input_df['hour'] = dt.hour
        input_df['day'] = dt.dayofweek
        input_df['month'] = dt.month
        input_df['year'] = dt.year
        input_df['is_weekend'] = int(dt.dayofweek >= 5)
        input_df['is_night'] = int((dt.hour < 6) | (dt.hour > 20))
    
    # Handle boolean values
    for col in input_df.columns:
        if input_df[col].dtype == bool:
            input_df[col] = input_df[col].astype(int)
    
    # Fill missing columns that are in the model but not in input
    for col in input_features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Keep only the columns that the model was trained on
    input_df = input_df[input_features]
    
    # Encode categorical columns
    for col in cat_cols:
        if col in input_df.columns:
            # Convert to string before encoding
            input_df[col] = input_df[col].astype(str)
            # Check if all values in this column are in the label encoder's classes
            if set(input_df[col]).issubset(set(label_encoders[col].classes_)):
                input_df[col] = label_encoders[col].transform(input_df[col])
            else:
                # Handle unknown categories by setting to the most common class (0)
                print(f"Warning: Unknown category in column '{col}'. Using default value.")
                input_df[col] = 0
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    return input_scaled

def predict_accident_severity(
    input_data: Dict[str, Any]
) -> Tuple[int, np.ndarray]:
    """
    Predict accident severity based on input data
    
    Args:
        input_data: Dictionary containing input features
        
    Returns:
        Tuple of (predicted severity class, probability array)
    """
    # Load model components
    model, scaler, label_encoders, input_features, cat_cols = load_model_components()
    
    # Preprocess input
    input_scaled = preprocess_input(
        input_data, input_features, cat_cols, label_encoders, scaler
    )
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Get predicted class (add 1 to shift back to original labels)
    predicted_class = int(np.argmax(prediction[0])) + 1
    
    return predicted_class, prediction[0]

def format_prediction_result(severity: int, probabilities: np.ndarray) -> Dict:
    """Format prediction result as a dictionary with human-readable descriptions"""
    severity_descriptions = {
        1: "Minor - Little or no physical damage, no injuries",
        2: "Moderate - Moderate damage, possible minor injuries",
        3: "Serious - Significant damage, possible serious injuries",
        4: "Severe - Major damage, high likelihood of severe injuries or fatalities"
    }
    
    result = {
        "severity_level": severity,
        "severity_description": severity_descriptions.get(severity, "Unknown"),
        "confidence": float(probabilities[severity-1]),
        "all_probabilities": {
            f"Level {i+1}": float(prob) for i, prob in enumerate(probabilities)
        }
    }
    
    return result

def main():
    """Example usage of prediction function"""
    # Example input data with required columns
    example = {
        'temperature': 68,
        'humidity': 0.6,
        'wind_speed': 12,
        'city': 'New York',
        'state': 'NY',
        'weather_condition': 'Clear',
        'side': 'R',
        'roundabout': False,
        'hour': 15,
        'day': 2,
        'month': 5,
        'year': 2023,
        'is_weekend': 0,
        'is_night': 0,
        'duration_minutes': 25
    }
    
    # Alternative example using datetime instead of separate time fields
    datetime_example = {
        'datetime': '2023-05-03 15:30:00',  # Will be processed to extract time features
        'temperature': 48,
        'humidity': 0.2,
        'wind_speed': 2,
        'city': 'Atlanta',
        'state': 'GA',
        'weather_condition': 'Cloudy',
        'side': 'L',
        'roundabout': True,
        'duration_minutes': 65
    }
    
    # Make prediction
    severity, probabilities = predict_accident_severity(example)
    
    # Format and display results
    result = format_prediction_result(severity, probabilities)
    print("\nPrediction Results:")
    print(f"Severity Level: {result['severity_level']} - {result['severity_description']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("Probabilities for all severity levels:")
    for level, prob in result['all_probabilities'].items():
        print(f"  {level}: {prob:.2%}")

if __name__ == "__main__":
    main()