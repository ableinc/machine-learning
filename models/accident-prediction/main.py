import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from datasets import load_dataset

# Load CSV
# dataset: https://huggingface.co/datasets/nateraw/us-accidents
df: DataFrame = None
if os.path.isfile('datasets/accident_data.csv'):
    df = pd.read_csv('datasets/accident_data.csv')
else:
    # Huggingface datasets save path: ~/.cache/huggingface/datasets
    dataset = load_dataset("nateraw/us-accidents")
    df = pd.DataFrame(dataset['train'])
    # df.to_csv("datasets/accident_data.csv", index=True) # Save dataset locally

# Ensure dataset has been loaded
if df is None:
    raise AttributeError('Unable to load dataset. Expect pandas DataFrame, but got None.')

# Drop rows with too many NaNs
df = df.dropna(thresh=len(df.columns) * 0.8)

# Convert times to datetime and extract time-series features
df['start_time'] = pd.to_datetime(df['start_time'])
df['hour'] = df['start_time'].dt.hour
df['day'] = df['start_time'].dt.dayofweek
df['month'] = df['start_time'].dt.month
df['year'] = df['start_time'].dt.year
df['is_weekend'] = df['day'] >= 5
df['is_night'] = (df['hour'] < 6) | (df['hour'] > 20)

# Duration in minutes
df['end_time'] = pd.to_datetime(df['end_time'])
df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

# Drop unused columns
drop_cols = ['id', 'description', 'start_time', 'end_time', 'weather_timestamp']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Encode categorical
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
label_encoders = {}
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Fill NaNs
df.fillna(df.mean(numeric_only=True), inplace=True)

# Split
X = df.drop('severity', axis=1)
y = df['severity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model builder for hyperparameter tuning
def build_model(hp_units=64, hp_layers=2, hp_learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for _ in range(hp_layers):
        model.add(tf.keras.layers.Dense(hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Checkpoint path
checkpoint_path = "checkpoints/accident_model_checkpoint"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Detect multi-GPU strategy
devices = tf.config.list_logical_devices('GPU')
if len(devices) > 1:
    print(f"🚀 Multi-GPU detected: {len(devices)} GPUs. Using MirroredStrategy.")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("⚙️ Single GPU or CPU detected. Using default strategy.")
    strategy = tf.distribute.get_strategy()

# Train model within strategy scope
with strategy.scope():
    model = build_model()
    if os.path.exists(checkpoint_path + ".index"):
        print("🔄 Restoring from last checkpoint...")
        model.load_weights(checkpoint_path)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler_callback]
    )

    model.save("final_accident_model.keras")
    print("✅ Final model saved to 'final_accident_model.keras'")

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
print(classification_report(y_test, np.argmax(model.predict(X_test), axis=1)))

# Prediction with arbitrary input
input_features = X.columns

def predict_accident_severity(input_data: dict):
    input_df = pd.DataFrame([input_data])
    for col in input_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[input_features]

    # Encode categorical
    for col in cat_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col].astype(str))

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return int(np.argmax(prediction)), prediction[0]

# Example
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
result = predict_accident_severity(example)
print("Predicted Severity:", result[0])
print("Probabilities:", result[1])
