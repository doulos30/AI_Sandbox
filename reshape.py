import numpy as np

calib_data = np.load("calib_data_1pct.npy").reshape(-1, 1)  # Reshape to (-1, 1)
np.save("calib_data_1pct_reshaped.npy", calib_data)
print("Reshaped and saved as (500, 1, 1)") 

calib_data = np.load("calib_data_1pct_reshaped.npy")
print(calib_data.shape)  # Should print (500, 1 )