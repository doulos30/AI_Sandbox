import os
import datetime
import random

def generate_voltage_data(filename, num_samples=80000, anomaly_rate=0.15):
    with open(filename, 'w') as f:
        f.write("timestamp,voltage,anomaly\n")
        
        # Track previous values for creating sequences
        prev_anomaly = False
        sequence_length = 0
        
        for _ in range(num_samples):
            timestamp = datetime.datetime.now().isoformat()
            
            # Continue anomaly sequences with higher probability if already in sequence
            if prev_anomaly and random.random() < 0.7 and sequence_length < 10:
                anomaly = 1
                sequence_length += 1
            else:
                # Start new sample
                sequence_length = 0
                # Determine if this sample is an anomaly
                anomaly = 1 if random.random() < anomaly_rate else 0
            
            # Set voltage based on anomaly status
            if anomaly == 1:
                anomaly_type = random.choice(["high_spike", "low_dip", "zero", "oscillation", 
                                             "sustained_high", "sustained_low"])
                
                if anomaly_type == "high_spike":
                    # More variation in spikes
                    severity = random.choice(["mild", "moderate", "severe"])
                    if severity == "mild":
                        voltage = random.uniform(126, 130)
                    elif severity == "moderate":
                        voltage = random.uniform(130, 135)
                    else:
                        voltage = random.uniform(135, 140)
                        
                elif anomaly_type == "low_dip":
                    # More variation in dips
                    severity = random.choice(["mild", "moderate", "severe"])
                    if severity == "mild":
                        voltage = random.uniform(110, 114)
                    elif severity == "moderate":
                        voltage = random.uniform(100, 110)
                    else:
                        voltage = random.uniform(80, 100)
                        
                elif anomaly_type == "zero":
                    # Complete outage
                    voltage = 0
                    
                elif anomaly_type == "oscillation":
                    # Rapid fluctuation
                    voltage = random.choice([random.uniform(126, 135), random.uniform(90, 114)])
                    
                elif anomaly_type == "sustained_high":
                    # Sustained high voltage
                    voltage = random.uniform(126, 132)
                    
                elif anomaly_type == "sustained_low":
                    # Sustained low voltage
                    voltage = random.uniform(100, 114)
            else:
                # Normal range with natural variation
                voltage = random.uniform(115, 125)
            
            prev_anomaly = (anomaly == 1)
            f.write(f"{timestamp},{voltage},{anomaly}\n")


def main():
    output_dir = "voltage_datasets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    anomaly_rate = 0.15
    sample_size = 80000
    filename = f"{output_dir}/voltage_data_{int(anomaly_rate*100)}pct_anomalies.csv"
    print(f"Generating dataset with {anomaly_rate*100}% anomaly rate: {filename}")
    generate_voltage_data(filename, num_samples=sample_size, anomaly_rate=anomaly_rate)
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()