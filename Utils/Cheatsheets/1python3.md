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

Decorators are useful for logging, timing, Authentication and Authorization, Input Validation, Caching, Rate Limiting.

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


print(custom_sum(4, 5)) # Output: 9
print(custom_sum()) # Output: 3
print(custom_sum.__name__) # Output: custom_sum (This is thanks to the line "@functools.wraps(func)")


```

## Iterators

Class that implements `__iter__` and `__next__` methods.
Examples:

```python
# This is an exaple of Asynchronous iterator.
class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result

```

```python
l = [1, 2, 3]
iterator = iter(l)
```

## Generators

Python function that returns with `yield`.

In Python, when you define a function with the yield keyword, it automatically becomes a generator function. This means that when you call the function, it doesn't run the code inside it but instead returns a generator object. This generator object is a special kind of iterator.

The generator object created by calling a generator function is equipped with the methods __iter__() and __next__() by the Python runtime, which allows it to conform to the iterator protocol. You don't need to define these methods explicitly; they are implicitly provided.

Here's what happens under the hood:

- The __iter__() method simply returns the generator object itself.
- The __next__() method is where the generator function's code starts executing, and it runs until it hits a yield statement, at which point it returns the yielded value and pauses. The next time __next__() is called, the generator resumes execution right after the yield and continues until it hits another yield, and so on.

When the generator function's code runs to completion without hitting another yield, the generator automatically raises a StopIteration exception, signaling that the iteration is complete.

Example:

```python
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

counter = count_up_to(5)

# Using next to get the next item from the generator
try:
    while True:
        number = next(counter)
        print(number)
except StopIteration:
    # The generator is exhausted and will raise StopIteration
    pass
```

So, in the case of your count_up_to function, each time you call next(counter), the generator resumes where it left off, runs until it hits the yield statement, yields the next value, and then pauses again until the next call to next().

## Asynchronous

In Python, asynchronous programming is a broad concept that allows for the concurrent execution of tasks without using multiple threads or processes. It's particularly useful for I/O-bound and high-level structured network code. Here are some key properties and concepts associated with asynchronous programming in Python:

1. Event Loop: The core of the asynchronous model is the event loop. It's responsible for managing and distributing the execution of different tasks. It runs in a loop, waiting for and dispatching events or tasks.

2. Coroutines: These are special functions that can be paused and resumed, allowing other code to run in the meantime. They are defined using async def and are the building blocks of asynchronous Python code.

3. Awaitables: These are objects that can be used in an await expression. There are three main types of awaitable objects: coroutines, Tasks, and Futures.

4. Tasks: These are scheduled coroutines that are wrapped into a Task object, which is a subclass of Future. They are used to schedule coroutines concurrently.

5. Futures: A Future is a special low-level awaitable object that represents an eventual result of an asynchronous operation. When a Future object is awaited, it means that the coroutine will wait until the Future is resolved in some other place.

6. Non-blocking I/O: Asynchronous code allows for non-blocking I/O operations. This means that the system can start an I/O operation, and while waiting for the result, do something else instead of blocking the execution.

7. Concurrency: Asynchronous programming provides a way to write concurrent code. Note that concurrency is not parallelism. Concurrency involves multiple tasks making progress without necessarily running at the same instant, for example, by interleaving their execution.

8. Async/Await Syntax: Introduced in Python 3.5, this syntax is used to define and call asynchronous functions. You use async def to define a coroutine and await to call a coroutine function and wait for its result.

9. Asynchronous Context Managers and Iterators: Using the async with and async for statements, you can define context managers and iterators that allow for asynchronous operations within their scope.

10. Asynchronous Libraries and Frameworks: There are many libraries that support asynchronous programming, such as asyncio (included in the standard library), aiohttp for HTTP client/server, aiomysql and aiopg for database interaction, and frameworks like FastAPI and Sanic for web applications.

Asynchronous programming can make your code more complex, but it's a powerful tool for writing efficient programs, especially when dealing with I/O-bound operations or when building scalable network services.


## Questions:

Question: a coroutine is a function can be paused and resume and is defined by async def. on the other hand, a generator is a function that returns an special kind of iterator. Given this context I assumed that a generator can be a coroutine since they can be paused and resumed but it is not the case. Could you explain me what is wrong or missing in my reasoning? 

GPT4:
Your understanding of coroutines and generators is on the right track, but there are subtle differences that set them apart, especially in how they are used and what they are designed for.

Generators: In Python, a generator is a special type of iterator. You create a generator using a function that has at least one yield statement. When a generator function is called, it returns an iterator known as a generator. The key aspect of generators is that they use yield to produce a series of values over time. This process can indeed be "paused" and "resumed," but this pausing and resuming is specifically in the context of value generation. Each time yield is encountered, the generator's state is frozen, and a value is sent out. The next time you pull from the generator, it resumes where it left off.

Coroutines: Coroutines, introduced with the async def syntax in Python, are more general than generators. They are used for asynchronous programming, allowing you to write code that can perform "awaitable" operations, like I/O operations, without blocking the entire program. A coroutine can be paused and resumed as well, but the key difference is the context in which this happens. Pausing and resuming in coroutines are tied to the asynchronous execution of tasks, not just the generation of values.

The confusion often arises because both generators and coroutines can be "paused" and "resumed," but they do so for different reasons and in different contexts. Generators are for producing a sequence of results lazily, while coroutines are for handling asynchronous operations.

Furthermore, it's important to note that while all coroutines in Python 3.5 and newer are indeed implemented using a form of generators (hence the yield and await syntax), not all generators are coroutines. A coroutine must be declared with async def and is designed to work with await expressions, whereas a generator is defined with a regular def and uses yield.

In summary, while generators and coroutines share some similarities in their "pause and resume" mechanisms, they serve different purposes and are used in different contexts in Python programming.


###

https://perso.limsi.fr/pointal/python:memento

