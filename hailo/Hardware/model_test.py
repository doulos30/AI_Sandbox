"""
Stress test for voltage anomaly detection model on Hailo hardware
This script runs the model with random voltage values across different ranges
to thoroughly test model robustness and consistency.
"""

import numpy as np
import hailo_platform as hpf
import time
from datetime import datetime

def is_anomaly(voltage, model_prediction):
    """
    Combines neural network predictions with rule-based logic to detect anomalies,
    including specific handling for edge cases.
    """
    # Neural network prediction for low voltages
    ml_anomaly = model_prediction > 0.5
    
    # Rule-based detection for high voltages
    rule_high_anomaly = voltage > 126.0
    
    # Rule-based detection for low voltages
    rule_low_anomaly = voltage < 115.0
    
    # Refined edge case handling: voltages just outside the normal range
    edge_case = (114.5 <= voltage < 115.0) or (126.0 < voltage <= 126.5)
    
    # Combined detection
    if edge_case:
        return True  # Treat edge cases as anomalies
    return ml_anomaly or rule_high_anomaly or rule_low_anomaly

def test_voltage_float32(voltage, v_min=0.0, v_max=140.0):
    """
    Tests a single voltage value on Hailo hardware using FLOAT32 format.
    """    
    hef = hpf.HEF("voltage_model2.hef")

    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]

        # Use FLOAT32 format
        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                # Normalize voltage to [0, 1] range
                voltage_norm = (voltage - v_min) / (v_max - v_min)
                
                # Create input array with the correct shape
                input_shape = input_vstream_info.shape
                input_data = {input_vstream_info.name: np.array([voltage_norm], dtype=np.float32).reshape(input_shape)}
                
                # Run inference
                start_time = time.time()
                results = infer_pipeline.infer(input_data)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                output_data = results[output_vstream_info.name]
                model_prediction = float(output_data.flatten()[0])
                
                # Determine if it's an anomaly using the hybrid rule
                is_anomaly_result = is_anomaly(voltage, model_prediction)
                
                # Determine ground truth (rule-based)
                is_actual_anomaly = voltage < 115.0 or voltage > 126.0
                
                # Return results
                return {
                    "voltage": voltage,
                    "normalized": voltage_norm,
                    "prediction": model_prediction,
                    "is_anomaly": is_anomaly_result,
                    "is_actual_anomaly": is_actual_anomaly,
                    "matches_ground_truth": is_anomaly_result == is_actual_anomaly,
                    "inference_time_ms": inference_time
                }
            
def run_stress_test(num_tests=100):
    """
    Run stress test with random voltages.
    """
    print(f"Starting stress test with {num_tests} random voltage values...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Create categories of test cases - each with a range and number of samples
    test_categories = [
    ("Normal range (115-126V)", 115.0, 126.0, 333),
    ("Low anomaly (0-115V)", 0.0, 114.9, 333),
    ("High anomaly (126-140V)", 126.1, 140.0, 333)
]

    
    # Special edge cases list
    edge_cases = [0, 90, 114.9, 115, 115.1, 120, 125.9, 126, 126.1, 130, 140]
    
    results = []
    total_time = 0
    correct_predictions = 0
    
    # Run tests for each category
    for category_name, min_val, max_val, count in test_categories:
        print(f"\nTesting category: {category_name}")
        
        # Generate random voltages in the specified range
        voltages = np.random.uniform(min_val, max_val, count)
        
        for voltage in voltages:
            try:
                result = test_voltage_float32(voltage)
                results.append(result)
                total_time += result["inference_time_ms"]
                
                if result["matches_ground_truth"]:
                    correct_predictions += 1
                
                # Print short result
                print(f"Voltage: {voltage:.1f}V | Prediction: {result['prediction']:.4f} | "
                      f"{'ANOMALY' if result['is_anomaly'] else 'NORMAL'} | "
                      f"{'✓' if result['matches_ground_truth'] else '✗'} | "
                      f"{result['inference_time_ms']:.2f}ms")
                
            except Exception as e:
                print(f"Error testing voltage {voltage}V: {str(e)}")
    
    # Test edge cases separately
    print(f"\nTesting category: Edge cases")
    for voltage in edge_cases:
        try:
            result = test_voltage_float32(voltage)
            results.append(result)
            total_time += result["inference_time_ms"]
            
            if result["matches_ground_truth"]:
                correct_predictions += 1
            
            # Print short result
            print(f"Voltage: {voltage:.1f}V | Prediction: {result['prediction']:.4f} | "
                  f"{'ANOMALY' if result['is_anomaly'] else 'NORMAL'} | "
                  f"{'✓' if result['matches_ground_truth'] else '✗'} | "
                  f"{result['inference_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"Error testing voltage {voltage}V: {str(e)}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"Stress Test Summary")
    print("=" * 80)
    print(f"Total tests run: {len(results)}")
    print(f"Correct predictions: {correct_predictions} ({correct_predictions/len(results)*100:.1f}%)")
    print(f"Average inference time: {total_time/len(results):.2f} ms")
    
    # Calculate accuracy per category
    normal_results = [r for r in results if 115.0 <= r["voltage"] <= 126.0]
    low_anomaly_results = [r for r in results if r["voltage"] < 115.0]
    high_anomaly_results = [r for r in results if r["voltage"] > 126.0]
    
    normal_accuracy = sum(1 for r in normal_results if r["matches_ground_truth"]) / len(normal_results) if normal_results else 0
    low_anomaly_accuracy = sum(1 for r in low_anomaly_results if r["matches_ground_truth"]) / len(low_anomaly_results) if low_anomaly_results else 0
    high_anomaly_accuracy = sum(1 for r in high_anomaly_results if r["matches_ground_truth"]) / len(high_anomaly_results) if high_anomaly_results else 0
    
    print(f"Normal voltage accuracy: {normal_accuracy*100:.1f}%")
    print(f"Low anomaly (<115V) accuracy: {low_anomaly_accuracy*100:.1f}%")
    print(f"High anomaly (>126V) accuracy: {high_anomaly_accuracy*100:.1f}%")
    
    # Temperature check (optional)
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000
            print(f"SoC temperature: {temp}°C")
    except:
        pass
    
    print("\nStress test completed!")

if __name__ == "__main__":
    run_stress_test(100)