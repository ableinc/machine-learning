# train.py - Improved Accident Severity Prediction Model
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from datasets import load_dataset
import shap

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configuration
CONFIG = {
    'data': {
        'local_path': 'datasets/accident_data.csv',
        'test_size': 0.2,
        'validation_size': 0.1,
        'nan_threshold': 0.8  # Keep rows with at least 80% non-NaN values
    },
    'model': {
        'dnn': {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 10,
            'patience': 5
        },
        'xgboost': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    },
    'paths': {
        'checkpoints': 'checkpoints',
        'models': 'final',
        'artifacts': 'artifacts',
        'logs': 'logs'
    }
}

# Create directories if they don't exist
for path in CONFIG['paths'].values():
    os.makedirs(path, exist_ok=True)

class Logger:
    """Simple logging utility"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            with open(log_file, 'w') as f:
                f.write(f"=== Training Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")

# Initialize logger
logger = Logger(os.path.join(CONFIG['paths']['logs'], 'training.log'))

def load_data() -> pd.DataFrame:
    """Load accident dataset from local file or download from Huggingface"""
    logger.log("Loading accident data...")
    
    if os.path.isfile(CONFIG['data']['local_path']):
        logger.log(f"Loading data from local file: {CONFIG['data']['local_path']}")
        df = pd.read_csv(CONFIG['data']['local_path'])
    else:
        logger.log("Local file not found. Downloading from Huggingface...")
        dataset = load_dataset("nateraw/us-accidents")
        df = pd.DataFrame(dataset['train'])
        
        # Save dataset locally
        os.makedirs(os.path.dirname(CONFIG['data']['local_path']), exist_ok=True)
        df.to_csv(CONFIG['data']['local_path'], index=False)
        logger.log(f"Data saved to {CONFIG['data']['local_path']}")
    
    logger.log(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocess the accident data"""
    logger.log("Preprocessing data...")
    
    # Store preprocessing artifacts
    artifacts = {}
    
    # Drop rows with too many NaNs
    original_rows = df.shape[0]
    df = df.dropna(thresh=int(len(df.columns) * CONFIG['data']['nan_threshold']))
    logger.log(f"Dropped {original_rows - df.shape[0]} rows with too many NaNs")
    
    # Process timestamps
    logger.log("Processing timestamps...")
    df['start_time'] = pd.to_datetime(df['start_time'], format='ISO8601')
    df['end_time'] = pd.to_datetime(df['end_time'], format='ISO8601')
    
    # Extract time features
    df['hour'] = df['start_time'].dt.hour
    df['day'] = df['start_time'].dt.dayofweek
    df['month'] = df['start_time'].dt.month
    df['year'] = df['start_time'].dt.year
    df['is_weekend'] = df['day'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) | 
                         ((df['hour'] >= 16) & (df['hour'] <= 18))).astype(int)
    
    # Calculate accident duration
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    
    # Create season feature
    df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else  # Winter
                                   2 if x in [3, 4, 5] else  # Spring
                                   3 if x in [6, 7, 8] else  # Summer
                                   4)  # Fall
    
    # Identify holidays (simplified approach)
    holidays = [(1, 1), (7, 4), (12, 25)]  # New Year, Independence Day, Christmas
    df['is_holiday'] = df.apply(lambda x: int((x['month'], x['day']) in holidays), axis=1)
    
    # Drop unused columns
    drop_cols = ['id', 'description', 'start_time', 'end_time', 'weather_timestamp']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    artifacts['categorical_columns'] = cat_cols
    
    # Handle categorical features
    logger.log(f"Encoding {len(cat_cols)} categorical columns...")
    label_encoders = {}
    for col in cat_cols:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    artifacts['label_encoders'] = label_encoders
    
    # Feature engineering: interaction terms
    logger.log("Creating interaction features...")
    df['temp_humidity'] = df['temperaturef'] * df['humidity']
    df['visibility_night'] = df['visibilitymi'] * df['is_night']
    df['wind_precip'] = df['wind_speedmph'] * df['precipitationin']
    
    # Fill remaining NaNs
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Log shape after preprocessing
    logger.log(f"Data shape after preprocessing: {df.shape}")
    
    return df, artifacts

def perform_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], n_features: int = None) -> Tuple[np.ndarray, List[str]]:
    """Perform feature selection using Random Forest feature importance"""
    logger.log("Performing feature selection...")
    
    # Train a random forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Log top 10 features
    logger.log("Top 10 features by importance:")
    for i in range(min(10, len(feature_names))):
        logger.log(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Select top features if n_features is specified
    if n_features:
        logger.log(f"Selecting top {n_features} features")
        selected_indices = indices[:n_features]
        X_selected = X[:, selected_indices]
        selected_features = [feature_names[i] for i in selected_indices]
        return X_selected, selected_features
    
    # Otherwise, use SelectFromModel with mean threshold
    selector = SelectFromModel(rf, threshold='mean', prefit=True)
    X_selected = selector.transform(X)
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    logger.log(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
    return X_selected, selected_features

def prepare_data_for_training(df: pd.DataFrame) -> Tuple:
    """Prepare data for model training"""
    logger.log("Preparing data for training...")
    
    # Split features and target
    X = df.drop('severity', axis=1)
    y = df['severity']
    
    # Adjust severity labels to be 0-indexed
    y = y - 1  # Shift from [1, 2, 3, 4] to [0, 1, 2, 3]
    num_classes = len(y.unique())
    logger.log(f"Target distribution: {np.bincount(y)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = X.columns.tolist()
    
    # Perform feature selection
    X_selected, selected_features = perform_feature_selection(X_scaled, y, feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=CONFIG['data']['test_size'], 
        random_state=RANDOM_SEED, stratify=y
    )
    
    # Handle class imbalance with SMOTE
    logger.log("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.log(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    logger.log(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")
    
    # Calculate class weights for training
    class_weights = dict(enumerate(class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )))
    
    return (X_train_resampled, X_test, y_train_resampled, y_test, 
            scaler, selected_features, num_classes, class_weights)

def build_dnn_model(input_shape: int, num_classes: int, config: Dict) -> tf.keras.Model:
    """Build a deep neural network model"""
    logger.log("Building DNN model...")
    
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for units in config['hidden_layers']:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(config['dropout_rate']))
    
    # Output layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary(print_fn=logger.log)
    return model

def train_dnn_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    num_classes: int, class_weights: Dict,
    config: Dict
) -> Tuple[tf.keras.Model, Dict]:
    """Train DNN model with early stopping and learning rate scheduling"""
    logger.log("Training DNN model...")
    
    # Create checkpoint directory
    checkpoint_path = os.path.join(CONFIG['paths']['checkpoints'], "accident_model.weights.h5")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(CONFIG['paths']['logs'], 'tensorboard'),
            histogram_freq=1
        )
    ]
    
    # Detect multi-GPU setup
    strategy = tf.distribute.get_strategy()
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 1:
        logger.log(f"Multi-GPU training with {len(devices)} GPUs")
        strategy = tf.distribute.MirroredStrategy()
    
    # Build and train model within strategy scope
    with strategy.scope():
        model = build_dnn_model(X_train.shape[1], num_classes, config)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=CONFIG['data']['validation_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.log(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(CONFIG['paths']['models'], "dnn_accident_model.keras")
    model.save(model_path)
    logger.log(f"DNN model saved to {model_path}")
    
    return model, history.history

def train_xgboost_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    config: Dict
) -> XGBClassifier:
    """Train XGBoost model as an alternative approach"""
    logger.log("Training XGBoost model...")
    
    # Initialize and train XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        learning_rate=config['learning_rate'],
        subsample=config['subsample'],
        colsample_bytree=config['colsample_bytree'],
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Evaluate model
    y_pred = xgb_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    logger.log(f"XGBoost test accuracy: {accuracy:.4f}")
    
    # Save model
    model_path = os.path.join(CONFIG['paths']['models'], "xgboost_accident_model.joblib")
    joblib.dump(xgb_model, model_path)
    logger.log(f"XGBoost model saved to {model_path}")
    
    return xgb_model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, is_xgboost: bool = False) -> Dict:
    """Evaluate model performance with multiple metrics"""
    logger.log("Evaluating model performance...")
    
    # Get predictions
    if is_xgboost:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    else:
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.log("\nClassification Report:")
    logger.log(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC score (one-vs-rest)
    roc_auc = roc_auc_score(
        tf.keras.utils.to_categorical(y_test), 
        y_pred_proba, 
        multi_class='ovr'
    )
    logger.log(f"ROC AUC Score (OVR): {roc_auc:.4f}")
    
    # F1 score (weighted)
    f1 = f1_score(y_test, y_pred, average='weighted')
    logger.log(f"F1 Score (weighted): {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(CONFIG['paths']['artifacts'], 'confusion_matrix.png'))
    
    # Calculate per-class metrics
    results = {
        'accuracy': np.mean(y_pred == y_test),
        'f1_weighted': f1,
        'roc_auc': roc_auc,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return results

def create_ensembled_predictor(dnn_model, xgb_model, selected_features: List[str], scaler):
    """Create an ensemble model that combines DNN and XGBoost predictions"""
    logger.log("Creating ensemble predictor...")
    
    # Define ensemble weights (can be tuned)
    weights = {
        'dnn': 0.6,
        'xgb': 0.4
    }
    
    def predict_severity(input_data: Dict[str, Any]) -> Dict:
        """Predict severity using ensemble of models"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in selected_features:
            if feature not in input_df:
                input_df[feature] = 0
        
        # Keep only selected features in the right order
        input_df = input_df[selected_features]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Get predictions from both models
        dnn_pred = dnn_model.predict(input_scaled)[0]
        xgb_pred = xgb_model.predict_proba(input_scaled)[0]
        
        # Weighted ensemble
        ensemble_pred = weights['dnn'] * dnn_pred + weights['xgb'] * xgb_pred
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
        
        result = {
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
        
        return result
    
    return predict_severity

def explain_model(model, X_train: np.ndarray, X_test: np.ndarray, feature_names: List[str]):
    """Generate SHAP explanations for model predictions"""
    logger.log("Generating model explanations with SHAP...")
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
    
    # Calculate SHAP values for a subset of test data
    shap_values = explainer.shap_values(X_test[:100])
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    plt.savefig(os.path.join(CONFIG['paths']['artifacts'], 'shap_summary.png'))
    plt.close()
    
    # Plot impact on model output
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, plot_type="bar")
    plt.savefig(os.path.join(CONFIG['paths']['artifacts'], 'shap_importance.png'))
    plt.close()
    
    logger.log("SHAP analysis completed. Plots saved to artifacts directory.")

def save_model_artifacts(artifacts: Dict, selected_features: List[str]):
    """Save all model artifacts for later use"""
    logger.log("Saving model artifacts...")
    
    artifact_path = CONFIG['paths']['artifacts']
    
    # Save artifacts
    joblib.dump(artifacts['scaler'], os.path.join(artifact_path, "scaler.pkl"))
    joblib.dump(artifacts['label_encoders'], os.path.join(artifact_path, "label_encoders.pkl"))
    joblib.dump(selected_features, os.path.join(artifact_path, "selected_features.pkl"))
    joblib.dump(artifacts['categorical_columns'], os.path.join(artifact_path, "categorical_columns.pkl"))
    
    # Save model metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': selected_features,
        'categorical_features': artifacts['categorical_columns'],
        'metrics': artifacts['evaluation_results'],
        'config': CONFIG
    }
    joblib.dump(metadata, os.path.join(artifact_path, "model_metadata.pkl"))
    
    logger.log("All artifacts saved successfully")

def main():
    """Main training function"""
    start_time = datetime.now()
    logger.log(f"Starting training pipeline at {start_time}")
    
    # Load data
    df = load_data()
    
    # Preprocess data
    df, preprocessing_artifacts = preprocess_data(df)
    
    # Prepare data for training
    X_train, X_test, y_train, y_test, scaler, selected_features, num_classes, class_weights = prepare_data_for_training(df)
    
    # Train DNN model
    dnn_model, history = train_dnn_model(
        X_train, y_train, X_test, y_test, 
        num_classes, class_weights, 
        CONFIG['model']['dnn']
    )
    
    # Train XGBoost model
    xgb_model = train_xgboost_model(
        X_train, y_train, X_test, y_test,
        CONFIG['model']['xgboost']
    )
    
    # Evaluate models
    dnn_results = evaluate_model(dnn_model, X_test, y_test)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, is_xgboost=True)
    
    # Model explanation with SHAP
    explain_model(dnn_model, X_train, X_test, selected_features)
    
    # Create ensemble predictor
    ensemble_predictor = create_ensembled_predictor(
        dnn_model, xgb_model, selected_features, scaler
    )
    
    # Save ensemble predictor
    ensemble_path = os.path.join(CONFIG['paths']['models'], "ensemble_predictor.pkl")
    joblib.dump(ensemble_predictor, ensemble_path)
    logger.log(f"Ensemble predictor saved to {ensemble_path}")
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'label_encoders': preprocessing_artifacts['label_encoders'],
        'categorical_columns': preprocessing_artifacts['categorical_columns'],
        'evaluation_results': {
            'dnn': dnn_results,
            'xgboost': xgb_results
        }
    }
    save_model_artifacts(artifacts, selected_features)
    
    # Print execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.log(f"Training completed in {execution_time}")
    logger.log(f"DNN model accuracy: {dnn_results['accuracy']:.4f}")
    logger.log(f"XGBoost model accuracy: {xgb_results['accuracy']:.4f}")
    
    # Test predictor with sample data
    example = {
        'temperature': 68,
        'humidity': 0.6,
        'wind_speed': 12,
        'visibility': 10,
        'precipitation': 0.01,
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
    
    prediction = ensemble_predictor(example)
    logger.log(f"Sample prediction: Severity Level {prediction['severity_level']} - {prediction['severity_description']}")
    logger.log(f"Confidence: {prediction['confidence']:.4f}")

if __name__ == "__main__":
    main()