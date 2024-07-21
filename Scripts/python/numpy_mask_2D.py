import numpy as np

# Sample 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr)

# Define a mask
mask = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
arr = np.where(mask.astype(bool), arr, np.nan)
print(arr)


# Mask the NaN values and flatten the array
masked_array = arr[~np.isnan(arr)]

# Create a new array without NaN values
cleaned_array = np.split(
    masked_array, np.cumsum(np.count_nonzero(~np.isnan(arr), axis=1))[:-1]
)

print(cleaned_array)
# [array([1., 3.]), array([4., 5., 6.]), array([8.])]
