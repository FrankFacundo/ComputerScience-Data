import numpy as np


def levenshtein_distance(str1, str2):
    # Get the lengths of the two strings
    len_str1, len_str2 = len(str1), len(str2)

    # Initialize a (len_str1+1) x (len_str2+1) matrix to store distances
    distance_matrix = np.zeros((len_str1 + 1, len_str2 + 1), dtype=int)

    # Fill the first row and first column with index values
    # These represent converting a string to/from an empty string
    for i in range(len_str1 + 1):
        distance_matrix[i][0] = i  # Cost of deletions
    for j in range(len_str2 + 1):
        distance_matrix[0][j] = j  # Cost of insertions

    # Calculate distances by iterating over each character in str1 and str2
    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            # Check if characters are the same
            if str1[i - 1] == str2[j - 1]:
                # No cost if characters are the same, inherit diagonal value
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            else:
                # Calculate cost as 1 + minimum of deletion, insertion, or substitution
                delete_cost = distance_matrix[i - 1][j]  # Cost of deletion
                insert_cost = distance_matrix[i][j - 1]  # Cost of insertion
                substitute_cost = distance_matrix[i - 1][j - 1]  # Cost of substitution
                distance_matrix[i][j] = 1 + min(
                    delete_cost, insert_cost, substitute_cost
                )

    # The final element of the matrix is the Levenshtein distance
    return distance_matrix[len_str1][len_str2]


# Test cases
print(
    "Distance between 'kitten' and 'sitting':",
    levenshtein_distance("kitten", "sitting"),
)
print("Distance between 'flaw' and 'lawn':", levenshtein_distance("flaw", "lawn"))
print(
    "Distance between 'intention' and 'execution':",
    levenshtein_distance("intention", "execution"),
)
print("Distance between 'apple' and 'apple':", levenshtein_distance("apple", "apple"))
print(
    "Distance between 'distance' and 'editing':",
    levenshtein_distance("distance", "editing"),
)
