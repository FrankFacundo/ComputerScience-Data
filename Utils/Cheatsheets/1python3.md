# Utils

## Environment Variables

```python
import os

# Set environment variables
os.environ['API_USER'] = 'username'
os.environ['API_PASSWORD'] = 'secret'

# Get environment variables
USER = os.getenv('API_USER')
PASSWORD = os.environ.get('API_PASSWORD')

# Getting non-existent keys
FOO = os.getenv('FOO') # None
BAR = os.environ.get('BAR') # None
BAZ = os.environ['BAZ'] # KeyError: key does not exist.
```

## Decorators

Decorators are useful for logging,timing, Authentication and Authorization, Input Validation, Caching, Rate Limiting. 

```python
import functools

def binary_param_decorator(log):
    def binary_func(func):
        @functools.wraps(func)
        def binary_func_inner(*args, **named_args):
            if args:
                num1 = args[0]
                num2 = args[1]
            else:
                num1 = 1
                num2 = 2
            if log:
                print(f"Number 1: {num1}")
                print(f"Number 2: {num2}")
            if not isinstance(num1, int) or not isinstance(num2, int):
                raise TypeError("Invalid input. Arguments must be integers.")
            return func(num1, num2)
        return binary_func_inner
    return binary_func

@binary_param_decorator(log=True)
def custom_sum(num1, num2):
    addition = num1 + num2
    return addition


print(custom_sum())
print(custom_sum.__name__)

```

###

https://perso.limsi.fr/pointal/python:memento

