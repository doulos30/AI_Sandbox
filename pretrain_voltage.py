# simple script to ake pretrain data for line voltage anomoli detection
import os
import datetime 
#import serial 
import time 
import random 


# simulate serial data with timestamp and voltage values  for 120v  lines with edge cases 
# 0V anomoly, 120V normal, lower than 120 threshold, higher than 120 threshold also anomoly
# low percentage of anomolies, 5% of data is anomoly 

# need csv file with 80,000 columns: timestamp, voltage, anomaly
def generate_voltage_data(filename, num_samples=80000, anomaly_rate=0.10):
    with open(filename, 'w') as f:
        f.write("timestamp,voltage,anomaly\n")
        for _ in range(num_samples):
            # Generate a timestamp
            timestamp = datetime.datetime.now().isoformat()
            # Simulate voltage values
            voltage = random.uniform(110, 124)  # Normal range
            anomaly = 0  # Default to no anomaly
            
            # Introduce anomalies 
            if random.random() < anomaly_rate:  # Configurable chance of anomaly
                if random.random() < 0.5:
                    voltage = 0  # Anomaly: 0V
                else:
                    voltage = random.uniform(90, 125)  # Anomaly: outside normal range
                anomaly = 1
            
            f.write(f"{timestamp},{voltage},{anomaly}\n")
            time.sleep(0.01)  # Simulate time delay between readings

def count_anomalies(filename):
    """Count the number of anomalies in a dataset file"""
    total_samples = 0
    anomaly_count = 0
    
    with open(filename, 'r') as f:
        next(f)  # Skip header
        for line in f:
            total_samples += 1
            if line.strip().endswith(',1'):
                anomaly_count += 1
    
    return total_samples, anomaly_count, (anomaly_count / total_samples) * 100

def main():
    # Create output directory if it doesn't exist
    output_dir = "voltage_datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate dataset with 10% anomaly rate
    anomaly_rate = 0.10
    sample_size = 80000  # Full sample size for training
    
    filename = f"{output_dir}/voltage_data_{int(anomaly_rate*100)}pct_anomalies.csv"
    print(f"Generating dataset with {anomaly_rate*100}% anomaly rate: {filename}")
    generate_voltage_data(filename, num_samples=sample_size, anomaly_rate=anomaly_rate)
    
    # Verify the actual anomaly rate
    total, anomalies, percentage = count_anomalies(filename)
    print(f"Dataset generated: {total} samples with {anomalies} anomalies ({percentage:.2f}%)")
    print("-" * 50)

if __name__ == "__main__":
    main()



