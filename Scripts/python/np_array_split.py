### Filter using splip

import numpy as np

# arr = np.arange(50).reshape(-1, 10)
arr = np.arange(10)
print(arr)

size_arr = arr.shape[0]

n_arr = np.tile(arr, 5)

print(n_arr)


mask = [[0, 2], [2, 4], [4, 6], [6, 10]]
indexes_split = np.array(
    [
        [interval[0] + (idx * size_arr), interval[1] + 1 + (idx * size_arr)]
        for idx, interval in enumerate(mask)
    ]
)
print(indexes_split)

indexes_split = indexes_split.flatten()
print(indexes_split)

# cleaned_array = np.split(n_arr, [1, 3, 10, 2 + 10, 4 + 1 + 10])
cleaned_array = np.split(n_arr, indexes_split)
print(cleaned_array)

cleaned_array = cleaned_array[1::2]
print(cleaned_array)
