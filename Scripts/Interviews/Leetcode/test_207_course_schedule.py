"""
207. Course Schedule
Medium
Topics
premium lock icon
Companies
Hint
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
 

Constraints:

1 <= numCourses <= 2000
0 <= prerequisites.length <= 5000
prerequisites[i].length == 2
0 <= ai, bi < numCourses
All the pairs prerequisites[i] are unique.
"""

import random
from collections import defaultdict, deque

import pytest


# --- code under test (or import from your module) ---
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    indeg=in-degree
    """
    g = defaultdict(list)
    indeg = [0] * numCourses
    for a, b in prerequisites:
        g[b].append(a)
        indeg[a] += 1

    q = deque([i for i, d in enumerate(indeg) if d == 0])
    taken = 0

    while q:
        u = q.popleft()
        taken += 1
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return taken == numCourses


# -----------------------------------------------


@pytest.mark.parametrize(
    "numCourses,prereq,expected",
    [
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], True),  # Diamond DAG
        (5, [[1, 0], [2, 1], [3, 2], [4, 3]], True),  # Simple chain
        (2, [[1, 0]], True),  # Example 1
        (2, [[1, 0], [0, 1]], False),  # Example 2 (cycle)
        (3, [], True),  # No prerequisites
        (1, [], True),  # Single course
        (3, [[0, 0]], False),  # Self-dependency (cycle)
        (5, [[1, 0], [0, 1], [3, 2]], False),  # One cyclic component blocks all
    ],
)
def test_parametrized(numCourses, prereq, expected):
    assert canFinish(numCourses, prereq) is expected


def test_disconnected_components():
    # Two components: Component A is acyclic (0->1), Component B is acyclic (2->3), node 4 isolated
    assert canFinish(5, [[1, 0], [3, 2]]) is True


def test_large_random_dag_is_finishable():
    """
    Build a random DAG by only adding edges from lower index -> higher index.
    This ensures acyclicity; should always be finishable.
    """
    n = 200
    edges = []
    random.seed(123)
    for u in range(n):
        # connect to a few higher-index nodes
        for v in range(u + 1, min(n, u + 6)):
            if random.random() < 0.3:
                edges.append([v, u])  # u -> v (prereq format is [a, b] meaning b -> a)
    assert canFinish(n, edges) is True


def test_detect_cycle_in_large_graph():
    """
    Create a large acyclic graph then add a back-edge to form a cycle.
    """
    n = 100
    edges = [[i + 1, i] for i in range(n - 1)]  # chain 0->1->2->...->99
    edges.append(
        [10, 20]
    )  # 20 -> 10 forms a back edge 0->...->20 -> 10 -> ... -> 20 (cycle)
    assert canFinish(n, edges) is False
