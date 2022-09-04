# Iteration by n elements
import itertools

def chunker_longest(iterable, chunksize):
    return itertools.zip_longest(*[iter(iterable)] * chunksize)

n = 3
l = [1, 2, 3, 4, 5, 6, 7]
for chunk in chunker_longest(l, n):
    print(chunk)