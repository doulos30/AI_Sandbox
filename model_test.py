import onnxruntime as ort
import numpy as np 
import random 
import matplotlib.pyplot as plt

# Load the ONNX model
onnx_model_path = "voltage_model2.onnx"  # Path to your ONNX file
session = ort.InferenceSession(onnx_model_path)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Test a voltage value
def test_voltage_onnx(voltage, v_min=0, v_max=140):
    voltage = round(voltage, 2)  # Round to 2 decimal places
    print(f"\n--- Testing Voltage: {voltage}V ---")
    voltage_norm = (voltage - v_min) / (v_max - v_min)  # Normalize voltage
    print(f"Normalization range: v_min={v_min}, v_max={v_max}")
    input_data = np.array([[voltage_norm]], dtype=np.float32)  # Shape: (1, 1)

    # Run inference
    output = session.run([output_name], {input_name: input_data})
    model_prediction = output[0][0][0]  # Extract prediction

    # Display results
    #print(f"Voltage: {voltage}V (normalized: {voltage_norm:.4f})")
    print(f"Model prediction: {model_prediction:.4f}")
    print(f"Model detection: {'Anomaly' if model_prediction > 0.5 else 'Normal'}")

def test_random_voltages(num_samples=1000, normal_range=(115, 125), anomaly_ranges=[(0, 80), (126, 140)]):
    """
    Generates random voltage values, including normal and anomalous voltages, and tests them.
    
    Parameters:
    - num_samples: Number of random voltages to generate.
    - normal_range: Tuple specifying the range of normal voltages.
    - anomaly_ranges: List of tuples specifying the ranges of anomalous voltages.
    """
    print(f"\n--- Testing {num_samples} Random Voltages ---")
    for _ in range(num_samples):
        # Randomly decide if the voltage is normal or anomalous
        if random.random() < 0.7:  # 70% chance for normal voltage
            voltage = random.uniform(*normal_range)
        else:  # 30% chance for anomalous voltage
            anomaly_range = random.choice(anomaly_ranges)
            voltage = random.uniform(*anomaly_range)
        
        # Test the voltage using the existing function
        test_voltage_onnx(voltage, v_min=0, v_max=140)

# Run the random voltage test
test_random_voltages()

