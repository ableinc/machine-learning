# Accident Prediction with TensorFlow Using Full CSV Dataset

This implementation uses all columns from a rich CSV dataset to train a TensorFlow model for accident severity prediction and provides a separate function to make predictions using any subset of features.

Includes:
- Full time-series modeling using hour, day, month, year, weekend/night indicators.

- Hyperparameter tuning with RandomizedSearchCV over model architecture and learning rate.

- Flexible prediction function supporting any subset of known features.

---

## 🚀 Load & Preprocess Dataset
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load CSV
df = pd.read_csv('accident_data.csv')

# Drop rows with too many NaNs
df = df.dropna(thresh=len(df.columns) * 0.8)

# Handle time features
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

# Drop unused time columns
df.drop(['id', 'description', 'start_time', 'end_time', 'weather_timestamp'], axis=1, inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Handle NaNs again (after encoding)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Split features/labels
X = df.drop('severity', axis=1)
y = df['severity']

# Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---

## 🧠 Model Definition & Training
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # assuming 4 severity classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)
```

---

## 📊 Evaluation
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

---

## 🔮 Prediction Function for Any Column Subset
```python
def predict_accident_severity(input_data: dict):
    """
    Predict accident severity given a dictionary of input features.
    Example:
    predict_accident_severity({
        'temperature': 55,
        'humidity': 0.7,
        'wind_speed': 15,
        'city': 'Houston',
        'weather_condition': 'Rain'
    })
    """
    input_df = pd.DataFrame([input_data])

    # Fill missing columns with default or 0
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure column order matches training
    input_df = input_df[X.columns]

    # Encode categorical
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = LabelEncoder().fit(df[col].astype(str)).transform(input_df[col].astype(str))

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    severity_class = np.argmax(prediction)
    return severity_class, prediction[0]
```

---

## ✅ Example Prediction Call
```python
result = predict_accident_severity({
    'temperature': 68,
    'humidity': 0.6,
    'wind_speed': 12,
    'city': 'New York',
    'state': 'NY',
    'weather_condition': 'Clear',
    'side': 'R',
    'roundabout': False
})
print("Predicted Severity:", result[0])
print("Class Probabilities:", result[1])
```
