"""
This script generates and saves a calibration dataset for voltage anomaly detection.
Steps performed:
1. Loads a CSV file containing voltage readings and anomaly labels.
2. Determines the minimum and maximum voltage values for normalization.
3. Samples 475 normal and 25 anomaly voltage values to create a calibration set.
4. Normalizes the sampled voltages using min-max scaling.
5. Saves the normalized calibration data as a NumPy array for use in Hailo calibration.
Outputs:
- 'calib_data.npy': NumPy file containing the normalized calibration voltages.
Usage:
- Ensure the input CSV file ('voltage_datasets/voltage_data_10pct_anomalies.csv') exists and contains 'voltage' and 'anomaly' columns.
- Run the script to generate and save the calibration data.
"""

import numpy as np
import pandas as pd 

# Load your generated dataset
df = pd.read_csv('voltage_datasets/voltage_data_1pct_anomalies.csv')

# Use the same min/max as your training
v_min = df['voltage'].min()
v_max = df['voltage'].max()

# Sample 400 normal and 100 anomaly voltages for calibration
normal = df[df['anomaly'] == 0]['voltage'].sample(495, random_state=42)
anomaly = df[df['anomaly'] == 1]['voltage'].sample(5, random_state=42)
calib_voltages = np.concatenate([normal.values, anomaly.values]).reshape(-1, 1).astype(np.float32)

# Normalize
calib_voltages_norm = (calib_voltages - v_min) / (v_max - v_min)

# Save for Hailo calibration
np.save('calib_data_1pct.npy', calib_voltages_norm)
print("Saved calibration data to calib_data_1pct.npy")