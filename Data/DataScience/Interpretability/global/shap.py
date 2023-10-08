'''
WARNING: The script is not correct. 
'''

import numpy as np
from sklearn.linear_model import LinearRegression
import itertools
import math

# Create a synthetic dataset
np.random.seed(0)  # For reproducibility
X = 2 * np.random.rand(100, 2)
y = 3 + 4 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

# Train a linear regression model
model = LinearRegression().fit(X, y)


# Define the model prediction function
def model_predict(instance):
    return model.predict(instance)


# Define the instance for which to calculate the SHAP values
instance = X[0]
print("instance: ", instance)

# Define the reference (baseline) instance as the mean of the dataset
reference = X.mean(axis=0)
print("reference: ", reference)


# Implement the SHAP values function with the modification to handle empty S
def shap_values(model_predict, instance, reference=None):
    if reference is None:
        reference = np.zeros(instance.shape)

    n_features = len(instance)
    shap_values = np.zeros(n_features)

    f_x = model_predict(instance.reshape(1, -1))[0]
    f_0 = model_predict(reference.reshape(1, -1))[0]

    for i in range(n_features):
        print("\ni:", i)
        for S in itertools.combinations(range(n_features), i):
            print("i, S:", i , S)
            mask = np.zeros(n_features, dtype=bool)
            if i > 0:
                mask[np.array(S, dtype=int)] = True

            x_S = instance.copy()
            print("x_S", x_S)
            x_S[~mask] = reference[~mask]
            print("x_S", x_S)

            x_S_union_i = x_S.copy()
            print("x_S_union_i", x_S_union_i)
            x_S_union_i[i] = instance[i]
            print("x_S_union_i", x_S_union_i)

            weight = (math.factorial(i) * math.factorial(n_features - i - 1)
                      ) / math.factorial(n_features)
            shap_values[i] += weight * (
                model_predict(x_S_union_i.reshape(1, -1))[0] -
                model_predict(x_S.reshape(1, -1))[0])

        shap_values[i] /= n_features

    # Debugging print statements
    print("\nf_x:", f_x)
    print("f_0:", f_0)
    print("SHAP values sum:", shap_values.sum())
    print("f_x - f_0:", f_x - f_0)
    print("SHAP values:", shap_values)

    # assert np.abs(f_x - f_0 - shap_values.sum(
    # )) < 1e-6, "SHAP values do not sum up to the difference in model outputs"

    return shap_values


# Calculate and display the SHAP values
shap_values_example = shap_values(model_predict, instance, reference)
shap_values_example
