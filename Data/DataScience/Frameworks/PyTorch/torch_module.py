"""
Root Mean Square Norm. The same idea when calculating voltage in alternative current.
"""

import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


norm = RMSNorm(3, 1e-05)

# Creating a test input tensor
# test_input = torch.randn(3, 2, 5)
test_input = torch.randn(3)
print("input : \n", test_input)

# # Passing the input through the RMSNorm instance
# output = norm(test_input)

# print("output : \n", output)

o = test_input.pow(2)
print(o)
o = o.mean(-1, keepdim=True)
print(o)
o = o + 1e-05
print(o)
# reciprocal of the square-root of each of the elements of input. 1/sqrt(input_i)
o = torch.rsqrt(o)
print(o)
o = test_input * o
print(o)
