import numpy as np

calib_data = np.load("calib_data.npy").reshape(500, 1, 1)
np.save("calib_data.npy", calib_data)
print("Reshaped and saved as (500, 1, 1)")