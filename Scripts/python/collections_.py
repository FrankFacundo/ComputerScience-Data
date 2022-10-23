from collections import namedtuple, defaultdict


Point = namedtuple("Point", ["x", "y"])
point1 = Point(1, 5)
point2 = Point(2, 4)

print(point1.x)
print(point2.y)
""" print console
1
4
"""