import numpy as np
import pandas as pd

for name, obj in np.ndarray.__dict__.items():
    if not name.startswith("_"):
        print(name, type(obj))

# argmin <class 'method_descriptor'>
# argsort <class 'method_descriptor'>
# ...

lst = sorted(
    {
        name: type(obj)
        for name, obj in np.ndarray.__dict__.items()
        if not name.startswith("_")
    }
)
lst[:10]
###
print(np.ndarray.argsort.__doc__)

# a.argsort(axis=-1, kind=None, order=None)
#     Returns the indices that would sort this array.
#     Refer to `numpy.argsort` for full documentation.
#     See Also
#     --------
#     numpy.argsort : equivalent function

for name, obj in pd.DataFrame.__dict__.items():
    if not name.startswith("_"):
        print(name, type(obj))

print(pd.DataFrame.groupby.__doc__)
