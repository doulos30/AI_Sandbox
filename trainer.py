'''
I had to scale the voltage data to a 0-1 range for better training performance.
after processing the data and trim the dataset two 2 decimal places.
This code trains a simple neural network to detect anomalies in voltage data
using PyTorch and exports the model to ONNX format for deployment on Hailo-8. 
''' 

# import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Load and preprocess data
#df = pd.read_csv('voltage_data_anomalies.csv') 

df = pd.read_csv('voltage_datasets/voltage_data_1pct_anomalies.csv')



# Normalize voltage to 0-1 range for neural network training
v_min = df['voltage'].min()
v_max = df['voltage'].max()
df['voltage_norm'] = (df['voltage'] - v_min) / (v_max - v_min)

X = df[['voltage_norm']].values.astype(np.float32)
y = df['anomaly'].values.astype(np.float32) 


# Ensure data is in the correct shape 
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"Normal samples: {len(df[df['anomaly'] == 0])}")
print(f"Anomaly samples: {len(df[df['anomaly'] == 1])}")
print(f"Voltage range: {v_min:.2f}V to {v_max:.2f}V\n")


# Convert to tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).unsqueeze(1)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a simple neural network
# add more neurons if the model struggles with anomaly detection
class VoltageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8), # increase neurons
            nn.ReLU(),
            nn.Linear(8, 8), # add hidden layer
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

model = VoltageNet()

# 3. Train the model
# Uses binary cross-entropy loss (BCELoss) for anomaly detection:
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(30):  # Increase epochs for better results
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4. Export to ONNX for Hailo-8
torch.onnx.export(model, X_tensor[:1], "voltage_model2.onnx", input_names=['input'], output_names=['output'], opset_version=11)
# Save PyTorch model state
torch.save(model.state_dict(), "voltage_model2.pth")
print("Exported model to voltage_model2.onnx")
print("Saved model state to voltage_model2.pth")