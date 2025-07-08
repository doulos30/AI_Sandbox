#running on HailoRT v4.20.0, Raspberry Pi 5 AI HAT (Hailo8, python 3.11)

"""
Runs inference on a Hailo8 device using a specified HEF model file.
This script demonstrates how to:
- Load a Hailo Executable File (HEF) for a neural network model.
- Configure the Hailo device (e.g., Raspberry Pi 5 AI HAT with Hailo8) for inference using the HailoRT Python API.
- Retrieve input and output stream information and shapes from the model.
- Set up input and output virtual stream parameters for floating-point (FLOAT32) data.
- Activate the network group and create an inference pipeline.
- Perform inference 10 times using randomly generated input data matching the model's input shape.
- Print the input and output shapes, as well as the inference results for each iteration.
Requirements:
- HailoRT v4.19.0
- Hailo8 hardware (e.g., Raspberry Pi 5 AI HAT)
- Python 3.10
- The HEF model file ("my_model.hef") must be present in the working directory.
Note:
- This script is intended for demonstration and testing purposes.
- Replace "my_model.hef" with the path to your actual HEF file as needed.
"""


import numpy as np
import hailo_platform as hpf

hef = hpf.HEF("my_model.hef") 



with hpf.VDevice() as target:
    configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]

    input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
    output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

    input_shape = input_vstream_info.shape
    output_shape = output_vstream_info.shape

    print(f"Input shape: {input_shape}, Output shape: {output_shape}")

    with network_group.activate(network_group_params):
        with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            for _ in range(10):
                random_input = np.random.rand(*input_shape).astype(np.float32)
                input_data = {input_vstream_info.name: np.expand_dims(random_input, axis=0)}
                results = infer_pipeline.infer(input_data)
                output_data = results[output_vstream_info.name]
                print(f"Inference output: {output_data}")