# Accident Severity Prediction Model

This repository contains a TensorFlow-based deep learning model to predict **accident severity** using historical accident data, weather, time of day, location, and more. It includes support for **time-series features**, **GPU acceleration**, **multi-GPU training**, **checkpoint recovery**, **early stopping**, and **learning rate scheduling**.

---

## 🧠 Problem Statement

The model aims to predict the **severity** of a road accident based on features such as:

- Date & time (hour, day of week, month, etc.)
- Location (city, state, zip code, latitude/longitude)
- Weather conditions (temperature, wind speed, visibility, etc.)
- Road characteristics (traffic signals, crossings, roundabouts, etc.)
- Time of day (day/night)
- Duration of incident

---

## 📁 Dataset

The model expects a CSV dataset (`accident_data.csv`) or it will pull the data from huggingface: https://huggingface.co/datasets/nateraw/us-accidents . The CSV file should have the following columns:

```
id, severity, start_time, end_time, start_lat, start_lng, end_lat, end_lng, distance, description, number, street, side, city, country, state, zipcode, timezone, airport_code, weather_timestamp, temperature, wind_chill, humidity, pressure, visibility, wind_direction, wind_speed, precipitation, weather_condition, amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, traffic_calming, traffic_signal, turning_loop, sunrise_sunset, civil_twilight, nautical_twilight, astronomical_twilight
```

📌 `severity` is the target label (0–3), representing the impact of the accident.

---

## 🚀 Features

- ✅ Automatic preprocessing (missing values, encoding, normalization)
- 🧠 Neural network model with tunable layers, neurons, and learning rate
- 🕒 Time-based feature engineering
- 💾 Checkpoint saving with resume-on-crash
- 📉 Early stopping to avoid overfitting
- 🔁 ReduceLROnPlateau scheduler for adaptive learning rate
- 🖥️ GPU + Multi-GPU support with TensorFlow's MirroredStrategy

---

## 🛠️ How to Run

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

You will need: ***tensorflow, pandas, numpy, scikit-learn***

## Prepare the Data (if you're not downloading the dataset from Huggingface)

Place your cleaned dataset as ```accident_data.csv``` in the same directory.

## Run the Script

```
python accident_model.py
```

Training will begin and resume from the last checkpoint if interrupted.
