### Filter using mask

import numpy as np

# Sample 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)

# Define a mask
mask = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool)

# Apply the mask to the array
arr_masked = np.where(mask, arr, np.nan)
print(arr_masked)

# Mask the NaN values and flatten the array
masked_array = arr_masked[~np.isnan(arr_masked)]

# Create a new array without NaN values
counts = np.count_nonzero(mask, axis=1)
split_indices = np.cumsum(counts)[:-1]
cleaned_array = np.split(masked_array, split_indices)

print(cleaned_array)
# [array([1., 3.]), array([4., 5., 6.]), array([8.])]
