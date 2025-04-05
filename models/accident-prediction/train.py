import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from datasets import load_dataset

# Load CSV
# dataset: https://huggingface.co/datasets/nateraw/us-accidents
df: DataFrame = None
if os.path.isfile('datasets/accident_data.csv'):
    print('datasets/accident_data.csv file found locally..')
    df = pd.read_csv('datasets/accident_data.csv')
else:
    # Huggingface datasets save path: ~/.cache/huggingface/datasets
    print('datasets/accident_data.csv file NOT found locally. Downloading from Huggingface.\nNote: You will need to update the header using the clean_header.py or update the df[] methods to use the correct letter case.')
    dataset = load_dataset("nateraw/us-accidents")
    df = pd.DataFrame(dataset['train'])
    # df.to_csv("datasets/accident_data.csv", index=True) # Save dataset locally

# Ensure dataset has been loaded
if df is None:
    raise AttributeError('Unable to load dataset. Expect pandas DataFrame, but got None.')

# Drop rows with too many NaNs
df = df.dropna(thresh=len(df.columns) * 0.8)

# Convert times to datetime and extract time-series features
df['start_time'] = pd.to_datetime(df['start_time'], format="ISO8601")
df['hour'] = df['start_time'].dt.hour
df['day'] = df['start_time'].dt.dayofweek
df['month'] = df['start_time'].dt.month
df['year'] = df['start_time'].dt.year
df['is_weekend'] = df['day'] >= 5
df['is_night'] = (df['hour'] < 6) | (df['hour'] > 20)

# Duration in minutes
df['end_time'] = pd.to_datetime(df['end_time'], format="ISO8601")
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
y = y - 1  # Shift severity labels from [1, 2, 3, 4] to [0, 1, 2, 3]
num_classes = df['severity'].nunique()
# print("Unique severity labels:", df['severity'].unique())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save after training
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(label_encoders, "artifacts/label_encoders.pkl")
joblib.dump(list(X.columns), "artifacts/input_features.pkl")
joblib.dump(list(cat_cols), "artifacts/cat_cols.pkl")

# Define model builder for hyperparameter tuning
def build_model(hp_units=64, hp_layers=2, hp_learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    for _ in range(hp_layers):
        model.add(tf.keras.layers.Dense(hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Checkpoint path
checkpoint_path = "checkpoints/accident_model_checkpoint.weights.h5"
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
devices = tf.config.list_physical_devices('GPU')
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

    model.save("final/final_accident_model.keras")
    print("✅ Final model saved to 'final/final_accident_model.keras'")

# Evaluation
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
print(classification_report(y_test, np.argmax(model.predict(X_test), axis=1)))

print("Run python models/accident-prediction/predict.py to make accurate predictions")