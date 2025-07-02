import pandas as pd
import numpy as np

# Load your generated dataset
df = pd.read_csv('voltage_datasets/voltage_data_10pct_anomalies.csv')

# Use the same min/max as your training
v_min = df['voltage'].min()
v_max = df['voltage'].max()

# Sample 400 normal and 100 anomaly voltages for calibration
normal = df[df['anomaly'] == 0]['voltage'].sample(475, random_state=42)
anomaly = df[df['anomaly'] == 1]['voltage'].sample(25, random_state=42)
calib_voltages = np.concatenate([normal.values, anomaly.values]).reshape(-1, 1).astype(np.float32)

# Normalize
calib_voltages_norm = (calib_voltages - v_min) / (v_max - v_min)

# Save for Hailo calibration
np.save('calib_data.npy', calib_voltages_norm)
print("Saved calibration data to calib_data.npy")