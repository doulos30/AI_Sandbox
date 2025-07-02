import onnx

model = onnx.load("voltage_model.onnx")
for input_tensor in model.graph.input:
    name = input_tensor.name
    dims = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input name: {name}, shape: {dims}")