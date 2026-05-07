"""NumPy-only decision tree classifier."""

from .config import DecisionTreeConfig
from .tree import NumpyDecisionTreeClassifier

__all__ = ["DecisionTreeConfig", "NumpyDecisionTreeClassifier"]
