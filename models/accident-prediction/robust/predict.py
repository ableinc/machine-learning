import os
import argparse
import pandas as pd
import numpy as np
import json
import joblib
import tensorflow as tf
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('accident_predictor')

# Configuration
CONFIG = {
    'paths': {
        'models': 'models',
        'artifacts': 'artifacts',
        'logs': 'logs'
    }
}

class AccidentSeverityPredictor:
    """Class for predicting accident severity using the trained ensemble model"""
    
    def __init__(self, models_dir: str = None, artifacts_dir: str = None):
        """
        Initialize the predictor by loading models and preprocessing artifacts
        
        Args:
            models_dir: Directory containing the trained models
            artifacts_dir: Directory containing preprocessing artifacts
        """
        self.models_dir = models_dir or CONFIG['paths']['models']
        self.artifacts_dir = artifacts_dir or CONFIG['paths']['artifacts']
        
        logger.info("Initializing Accident Severity Predictor...")
        self._load_artifacts()
        self._load_models()
        logger.info("Predictor initialized successfully")
    
    def _load_artifacts(self):
        """Load preprocessing artifacts"""
        try:
            logger.info("Loading preprocessing artifacts...")
            
            # Load feature list and preprocessing objects
            self.selected_features = joblib.load(os.path.join(self.artifacts_dir, "selected_features.pkl"))
            self.scaler = joblib.load(os.path.join(self.artifacts_dir, "scaler.pkl"))
            self.label_encoders = joblib.load(os.path.join(self.artifacts_dir, "label_encoders.pkl"))
            self.categorical_columns = joblib.load(os.path.join(self.artifacts_dir, "categorical_columns.pkl"))
            
            # Load metadata if available
            try:
                self.metadata = joblib.load(os.path.join(self.artifacts_dir, "model_metadata.pkl"))
                logger.info(f"Model trained on: {self.metadata.get('training_date', 'Unknown')}")
            except FileNotFoundError:
                logger.warning("Model metadata not found")
                self.metadata = {}
            
            logger.info(f"Loaded {len(self.selected_features)} selected features")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def _load_models(self):
        """Load trained models"""
        try:
            logger.info("Loading trained models...")
            
            # Try loading the ensemble predictor first
            ensemble_path = os.path.join(self.models_dir, "ensemble_predictor.pkl")
            if os.path.exists(ensemble_path):
                logger.info("Loading ensemble predictor...")
                self.ensemble_predictor = joblib.load(ensemble_path)
                self.using_ensemble = True
            else:
                logger.warning("Ensemble predictor not found, loading individual models...")
                
                # Load DNN model
                dnn_path = os.path.join(self.models_dir, "dnn_accident_model.keras")
                logger.info(f"Loading DNN model from {dnn_path}")
                self.dnn_model = tf.keras.models.load_model(dnn_path)
                
                # Load XGBoost model
                xgb_path = os.path.join(self.models_dir, "xgboost_accident_model.joblib")
                logger.info(f"Loading XGBoost model from {xgb_path}")
                self.xgb_model = joblib.load(xgb_path)
                
                # Define weights for manual ensemble
                self.ensemble_weights = {
                    'dnn': 0.6,
                    'xgb': 0.4
                }
                self.using_ensemble = False
                
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Preprocessed features as numpy array
        """
        logger.info("Preprocessing input data...")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Extract time features if datetime is provided
        if 'timestamp' in input_df and 'hour' not in input_df:
            try:
                dt = pd.to_datetime(input_df['timestamp'].iloc[0])
                input_df['hour'] = dt.hour
                input_df['day'] = dt.dayofweek
                input_df['month'] = dt.month
                input_df['year'] = dt.year
                input_df['is_weekend'] = int(dt.dayofweek >= 5)
                input_df['is_night'] = int((dt.hour < 6) or (dt.hour > 20))
                input_df['is_rush_hour'] = int(((dt.hour >= 7) and (dt.hour <= 9)) or 
                                             ((dt.hour >= 16) and (dt.hour <= 18)))
                
                # Create season feature
                input_df['season'] = input_df['month'].apply(lambda x: 1 if x in [12, 1, 2] else  # Winter
                                                         2 if x in [3, 4, 5] else  # Spring
                                                         3 if x in [6, 7, 8] else  # Summer
                                                         4)  # Fall
                
                # Identify holidays (simplified approach)
                holidays = [(1, 1), (7, 4), (12, 25)]  # New Year, Independence Day, Christmas
                input_df['is_holiday'] = int((dt.month, dt.day) in holidays)
                
            except Exception as e:
                logger.warning(f"Error processing timestamp: {e}")
        
        # Handle categorical features
        for col in self.categorical_columns:
            if col in input_df:
                # Convert to string first
                input_df[col] = input_df[col].astype(str)
                
                # Apply label encoder if available
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    if input_df[col].iloc[0] in le.classes_:
                        input_df[col] = le.transform(input_df[col])
                    else:
                        logger.warning(f"Unknown category '{input_df[col].iloc[0]}' for feature '{col}'. Using most frequent class.")
                        # Use most frequent class as fallback
                        most_frequent_idx = np.argmax(np.bincount([le.transform([c])[0] for c in le.classes_]))
                        input_df[col] = most_frequent_idx
        
        # Feature engineering: create interaction terms
        if all(x in input_df for x in ['temperature', 'humidity']):
            input_df['temp_humidity'] = input_df['temperature'] * input_df['humidity']
        
        if all(x in input_df for x in ['visibility', 'is_night']):
            input_df['visibility_night'] = input_df['visibility'] * input_df['is_night']
        
        if all(x in input_df for x in ['wind_speed', 'precipitation']):
            input_df['wind_precip'] = input_df['wind_speed'] * input_df['precipitation']
        
        # Ensure all required features are present
        for feature in self.selected_features:
            if feature not in input_df:
                logger.warning(f"Missing feature '{feature}'. Using default value 0.")
                input_df[feature] = 0
        
        # Keep only selected features in the right order
        input_df = input_df[self.selected_features]
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        return input_scaled
    
    def predict(self, input_data: Dict[str, Any]) -> Dict:
        """
        Predict accident severity from input features
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        logger.info("Making severity prediction...")
        
        # Preprocess input
        input_scaled = self._preprocess_input(input_data)
        
        # Make prediction
        if self.using_ensemble:
            # Use the ensemble predictor directly
            prediction = self.ensemble_predictor(input_data)
        else:
            # Manual ensemble prediction
            dnn_pred = self.dnn_model.predict(input_scaled)[0]
            xgb_pred = self.xgb_model.predict_proba(input_scaled)[0]
            
            # Weighted ensemble
            ensemble_pred = (self.ensemble_weights['dnn'] * dnn_pred + 
                           self.ensemble_weights['xgb'] * xgb_pred)
            predicted_class = int(np.argmax(ensemble_pred))
            
            # Add 1 to shift back to original labels [1,2,3,4]
            severity_level = predicted_class + 1
            
            # Create result dictionary
            severity_descriptions = {
                1: "Minor - Little or no physical damage, no injuries",
                2: "Moderate - Moderate damage, possible minor injuries",
                3: "Serious - Significant damage, possible serious injuries",
                4: "Severe - Major damage, high likelihood of severe injuries or fatalities"
            }
            
            prediction = {
                "severity_level": severity_level,
                "severity_description": severity_descriptions.get(severity_level, "Unknown"),
                "confidence": float(ensemble_pred[predicted_class]),
                "all_probabilities": {
                    f"Level {i+1}": float(prob) for i, prob in enumerate(ensemble_pred)
                },
                "model_contributions": {
                    "dnn": float(dnn_pred[predicted_class]),
                    "xgboost": float(xgb_pred[predicted_class])
                }
            }
        
        logger.info(f"Predicted severity: Level {prediction['severity_level']} with confidence {prediction['confidence']:.4f}")
        return prediction
    
    def batch_predict(self, input_data_list: List[Dict[str, Any]]) -> List[Dict]:
        """
        Predict accident severity for multiple inputs
        
        Args:
            input_data_list: List of dictionaries with feature values
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Processing batch of {len(input_data_list)} predictions...")
        results = []
        
        for i, input_data in enumerate(input_data_list):
            logger.info(f"Processing item {i+1}/{len(input_data_list)}")
            result = self.predict(input_data)
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for the XGBoost model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self, 'xgb_model'):
            logger.warning("XGBoost model not loaded, cannot get feature importance")
            return {}
            
        importance_scores = self.xgb_model.feature_importances_
        return {feature: float(score) for feature, score in zip(self.selected_features, importance_scores)}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict accident severity')
    parser.add_argument('--input', '-i', type=str, help='JSON file with input features')
    parser.add_argument('--output', '-o', type=str, help='Output file for prediction results')
    parser.add_argument('--models-dir', type=str, default='final', help='Directory containing trained models')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts', help='Directory containing model artifacts')
    parser.add_argument('--batch', action='store_true', help='Process input as batch (list of examples)')
    return parser.parse_args()


def load_input_data(input_file: str, is_batch: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Load input data from JSON file"""
    logger.info(f"Loading input data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if is_batch and not isinstance(data, list):
        # Convert single example to list for batch processing
        data = [data]
    elif not is_batch and isinstance(data, list):
        # Take first example if batch mode is not enabled
        logger.warning("Input contains multiple examples but batch mode is not enabled. Using only the first example.")
        data = data[0]
    
    return data


def save_results(results: Union[Dict, List[Dict]], output_file: Optional[str] = None):
    """Save prediction results to file or print to console"""
    if output_file:
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


def demo_interactive():
    """Run an interactive demo for accident severity prediction"""
    predictor = AccidentSeverityPredictor()
    
    print("\n=== Accident Severity Prediction Interactive Demo ===")
    print("Enter values for accident features (press Enter to use default):")
    
    # Get user input for basic features
    features = {}
    
    features['temperature'] = float(input("Temperature (°F) [70]: ") or 70)
    features['humidity'] = float(input("Humidity (0-1) [0.5]: ") or 0.5)
    features['wind_speed'] = float(input("Wind speed (mph) [10]: ") or 10)
    features['visibility'] = float(input("Visibility (miles) [10]: ") or 10)
    features['precipitation'] = float(input("Precipitation (inches) [0]: ") or 0)
    features['weather_condition'] = input("Weather condition [Clear]: ") or "Clear"
    features['city'] = input("City [New York]: ") or "New York"
    features['state'] = input("State [NY]: ") or "NY"
    features['side'] = input("Road side (L/R) [R]: ") or "R"
    
    # Use current time if not specified
    use_current_time = input("Use current time? (y/n) [y]: ").lower() != 'n'
    if use_current_time:
        now = datetime.now()
        features['hour'] = now.hour
        features['day'] = now.weekday()
        features['month'] = now.month
        features['year'] = now.year
        features['is_weekend'] = int(now.weekday() >= 5)
        features['is_night'] = int((now.hour < 6) or (now.hour > 20))
    else:
        features['hour'] = int(input("Hour (0-23) [12]: ") or 12)
        features['day'] = int(input("Day of week (0-6, 0=Monday) [1]: ") or 1)
        features['month'] = int(input("Month (1-12) [6]: ") or 6)
        features['year'] = int(input("Year [2025]: ") or 2025)
        features['is_weekend'] = int(input("Is weekend (0/1) [0]: ") or 0)
        features['is_night'] = int(input("Is night (0/1) [0]: ") or 0)
    
    # Make prediction
    result = predictor.predict(features)
    
    # Display results
    print("\n=== Prediction Results ===")
    print(f"Severity Level: {result['severity_level']} - {result['severity_description']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbabilities:")
    for level, prob in result['all_probabilities'].items():
        print(f"  {level}: {prob:.2%}")
    
    # Show top contributing features if available
    try:
        importance = predictor.get_feature_importance()
        if importance:
            print("\nTop 5 contributing features:")
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, score in top_features:
                print(f"  {feature}: {score:.4f}")
    except:
        pass


def main():
    """Main function for command-line usage"""
    args = parse_args()
    
    # Initialize predictor
    predictor = AccidentSeverityPredictor(
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir
    )
    
    if args.input:
        # Load input data
        input_data = load_input_data(args.input, args.batch)
        
        # Make prediction
        if args.batch:
            results = predictor.batch_predict(input_data)
        else:
            results = predictor.predict(input_data)
        
        # Save or print results
        save_results(results, args.output)
    else:
        # Run interactive demo
        demo_interactive()


if __name__ == "__main__":
    main()