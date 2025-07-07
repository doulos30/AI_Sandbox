"""
Voltage Anomaly Detection using Hailo Platform and Rule-based Logic
This script provides a simple voltage anomaly detection system that combines a rule-based approach with inference from a Hailo neural network model. It includes:
- `is_voltage_anomaly(voltage, v_min=110.0, v_max=125.0)`: 
    Determines if a given voltage value is considered anomalous based on predefined minimum and maximum thresholds.
- `test_voltage(voltage, v_min=110.0, v_max=125.0)`:
    Tests a given voltage value by:
        - Normalizing the input voltage.
        - Running inference using a Hailo neural network model loaded from "voltage_model.hef".
        - Comparing the model's prediction with the rule-based anomaly detection.
        - Printing detailed results, including raw and normalized model outputs, and the final anomaly decision.
The script demonstrates usage by testing several voltage values, both within and outside the normal range, and prints the results for each test case.
Dependencies:
    - numpy
    - hailo_platform (as hpf)
    - A valid Hailo HEF file ("voltage_model.hef") for inference
Example usage:
    test_voltage(110.0)  # Test a normal voltage value
    test_voltage(130.0)  # Test an anomalous voltage value
"""

import numpy as np
import hailo_platform as hpf 

# Simple rule-based voltage anomaly detector
def is_voltage_anomaly(voltage, v_min=110.0, v_max=125.0):
    return voltage < v_min or voltage > v_max

def test_voltage(voltage, v_min=90.0, v_max=125.0):
    print(f"\n--- Testing Voltage: {voltage}V ---")
    
    hef = hpf.HEF("voltage_model.hef")

    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                voltage_norm = (voltage - v_min) / (v_max - v_min)
                input_shape = input_vstream_info.shape
                input_data = {input_vstream_info.name: np.array([voltage_norm], dtype=np.float32).reshape(input_shape)}

                results = infer_pipeline.infer(input_data)
                output_data = results[output_vstream_info.name]

                # Get model prediction
                normalized_output = output_data / 255.0  # Normalize to [0, 1]
                model_prediction = float(normalized_output.flatten()[0])
                
                # Get rule-based prediction
                rule_based_anomaly = is_voltage_anomaly(voltage)
                
                # Display results
                print(f"Voltage: {voltage}V (normalized: {voltage_norm:.4f})")
                print(f"Raw model output: {output_data}")
                print(f"Normalized output: {normalized_output}")
                print(f"Model prediction: {model_prediction:.4f}")
                print(f"Model detection: {'Yes' if model_prediction > 0.5 else 'No'}")
                print(f"Rule-based detection: {'Yes' if rule_based_anomaly else 'No'}")
                
                # Final decision (use rule-based for now)
                if rule_based_anomaly:
                    print("FINAL DECISION: ⚠️ ANOMALY DETECTED ⚠️")
                else:
                    print("FINAL DECISION: ✓ Normal voltage")

# Test normal values

test_voltage(110.0)  # Mid-range normal
test_voltage(125.0)  # Maximum normal

# Test anomalies
test_voltage(90.0)   # Minimum normal
test_voltage(0.0)    # Zero voltage (anomaly)
test_voltage(85.0)   # Below minimum (anomaly)
test_voltage(130.0)  # Above maximum (anomaly)