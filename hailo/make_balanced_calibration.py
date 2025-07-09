"""
This script generates and saves a balanced calibration dataset for voltage anomaly detection.
Steps performed:
1. Loads a CSV file containing voltage readings and anomaly labels.
2. Determines the minimum and maximum voltage values for normalization.
3. Samples equal numbers of normal and anomaly voltage values to create a calibration set.
4. Normalizes the sampled voltages using min-max scaling.
5. Saves the normalized calibration data as a NumPy array for use in Hailo calibration.
Outputs:
- 'calib_data_balanced.npy': NumPy file containing the normalized calibration voltages.
Usage:
- Ensure the input CSV file ('voltage_datasets/voltage_data_15pct_anomalies.csv') exists and contains 'voltage' and 'anomaly' columns.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import time

# Load your generated dataset
df = pd.read_csv('voltage_datasets/voltage_data_15pct_anomalies.csv')

# Use the same min/max as your training
v_min = df['voltage'].min()
v_max = df['voltage'].max()

# Determine the number of samples for each class
num_samples = 500  # Total calibration samples
num_per_class = num_samples // 2  # Equal distribution between normal and anomalies

# Sample equal numbers of normal and anomaly voltages
normal = df[df['anomaly'] == 0]['voltage'].sample(num_per_class, random_state=42)
anomaly = df[df['anomaly'] == 1]['voltage'].sample(num_per_class, random_state=42)
calib_voltages = np.concatenate([normal.values, anomaly.values]).reshape(-1, 1).astype(np.float32)

# Normalize
calib_voltages_norm = (calib_voltages - v_min) / (v_max - v_min)

# Save for Hailo calibration
np.save('calib_data_balanced.npy', calib_voltages_norm)
print(f"Saved balanced calibration data to 'calib_data_balanced.npy' with shape {calib_voltages_norm.shape}") 
