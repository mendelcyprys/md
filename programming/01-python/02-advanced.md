# Python Introduction: Advanced Python Topics

## Table of Contents

1. [Generators and Iterators](#generators-and-iterators)
2. [Decorators](#decorators)
3. [Context Managers](#context-managers)
4. [Advanced Functions](#advanced-functions)
5. [Memory Management](#memory-management)

## Generators and Iterators

### Iterators

An iterator is an object that implements the iterator protocol through `__iter__()` and `__next__()` methods.

```python
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Using the iterator
for num in CountDown(5):
    print(num)  # Prints: 5, 4, 3, 2, 1
```

### Generators

Generators provide a more concise way to create iterators using the `yield` keyword.

```python
def countdown(start):
    while start > 0:
        yield start
        start -= 1

# Using the generator
for num in countdown(5):
    print(num)  # Prints: 5, 4, 3, 2, 1
```

### Generator Expressions

Similar to list comprehensions but they produce values on-demand.

```python
# List comprehension (creates the entire list in memory)
squares_list = [x**2 for x in range(1000000)]

# Generator expression (creates values on-demand)
squares_gen = (x**2 for x in range(1000000))
```

The generator version uses much less memory

### Generator Methods

Generators have special methods: `send()`, `throw()`, and `close()`.

```python
def echo_generator():
    value = yield "Ready"
    while True:
        value = yield value

gen = echo_generator()
print(next(gen))          # "Ready"
print(gen.send("Hello"))  # "Hello"
print(gen.send(42))       # 42
gen.close()               # Terminate the generator
```

`send(value)` is like `next()`, but with a bonus: it resumes the generator and sends a value back into the generator, which becomes the result of the previous `yield` expression.

This generator first yields `"Ready"` and pauses, waiting to receive a value. Each subsequent `send()` resumes the generator, passes a value into it, and yields that value back at the next `yield`. Calling `next()` after the first yield is equivalent to `send(None)`, assigning `None` to `value`. When `close()` is called, a `GeneratorExit` is raised at the paused yield, and the generator exits silently unless `GeneratorExit` is explicitly handled or a finally block is used for cleanup.

The `throw()` method in a generator allows you to raise an exception at the point where the generator is paused, enabling controlled error handling within the generator's execution flow:

```python
def my_generator():
    try:
        yield 1
    except ValueError as e:
        print(f"Caught: {e}")
        yield 2

gen = my_generator()
print(next(gen))  # Outputs: 1
gen.throw(ValueError, "An error occurred")  # Caught: An error occurred
print(next(gen))  # Outputs: 2
```

### Chaining Generators

Generators can be combined using yield from.

```python
def chain_generators(*iterables):
    for it in iterables:
        yield from it

combined = chain_generators([1, 2, 3], [4, 5, 6])
list(combined)  # [1, 2, 3, 4, 5, 6]
```

The `yield from` statement in Python allows a generator to delegate yielding to another generator or iterable, automatically propagating exceptions raised in the sub-generator and capturing any return value from it, thereby simplifying code composition and error handling.

## Decorators

### Basic Decorators

Decorators are functions that modify the behavior of other functions.

```python
def log_function_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

@log_function_call
def add(a, b):
    return a + b

add(3, 5)  # Logs the call and result
```

When a function is decorated using the `@decorator_name` syntax, the original function's name in the namespace is replaced by the wrapper function returned by the decorator, effectively modifying its behavior.

### Decorators with Arguments

Decorators can accept their own arguments.

```python
def repeat(n=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(n):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    return f"Hello, {name}!"

greet("Alice")  # Returns ["Hello, Alice!", "Hello, Alice!", "Hello, Alice!"]
```

Note that this involves two function calls at the time of definition, to get the wrapper function that will actually be called at the time of invocation.

### Class Decorators

A decorator can also be defined using a class that implements the `__call__` method, allowing instances of the class to be used as decorators. To persist state in a function-defined decorator, you can use the `nonlocal` keyword for variables in enclosing scopes or utilize mutable objects like lists or dictionaries.

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} has been called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    return "Hello!"

say_hello()  # "say_hello has been called 1 times"
say_hello()  # "say_hello has been called 2 times"
```

### Preserving Metadata

Use `functools.wraps` to maintain the original function's metadata.

```python
from functools import wraps

def log_function_call(func):
    @wraps(func)  # Preserves func's metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_function_call
def add(a, b):
    """Add two numbers."""
    return a + b

print(add.__name__)  # "add" (not "wrapper")
print(add.__doc__)   # "Add two numbers."
```

### Stacking Decorators

Multiple decorators can be applied to a single function.

```python
@decorator1
@decorator2
def function():
    pass

# Equivalent to:
# function = decorator1(decorator2(function))
```

## Context Managers

### Using Context Managers

Context managers control resource acquisition and release.

```python
# File handling with context manager
with open("file.txt", "r") as file:
    content = file.read()
# File is automatically closed when exiting the with block
```

### Creating Context Managers using Classes

Implement `__enter__` and `__exit__` methods.

```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end = time.time()
        print(f"Elapsed time: {self.end - self.start:.2f} seconds")
        # Return False to propagate exceptions, True to suppress
        return False

# Using the custom context manager
with Timer():
    # Code to measure
    import time
    time.sleep(1)
```

### Creating Context Managers using contextlib

The `contextlib` module provides utilities for working with context managers.

```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    try:
        yield  # This is where the with-block's code executes
               # You can also yield a value (with ... as ...)
    finally:
        end = time.time()
        print(f"Elapsed time: {end - start:.2f} seconds")

# Using the context manager
with timer():
    import time
    time.sleep(1)
```

### Nested Context Managers

Context managers can be nested.

```python
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())
```

### Context Manager for Database Transactions

```python
class DatabaseTransaction:
    def __init__(self, connection):
        self.connection = connection
    
    def __enter__(self):
        # Begin transaction
        self.connection.begin()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # No exceptions, commit transaction
            self.connection.commit()
        else:
            # Exception occurred, rollback transaction
            self.connection.rollback()
        return False  # Propagate any exceptions
```

## Advanced Functions

### Partial Functions

Create new functions with pre-filled arguments.

```python
from functools import partial

def multiply(x, y):
    return x * y

# Create a new function that multiplies by 2
double = partial(multiply, 2)
double(5)  # 10
```

### Closures

Functions that capture and remember the environment they were created in.

```python
def make_counter():
    count = 0
    
    def counter():
        nonlocal count
        count += 1
        return count
    
    return counter

counter1 = make_counter()
counter1()  # 1
counter1()  # 2

counter2 = make_counter()
counter2()  # 1 (independent from counter1)
```

### Higher-Order Functions
Functions that take functions as arguments or return functions.

```python
def compose(f, g):
    """Create a function that applies f after g."""
    return lambda x: f(g(x))

def square(x):
    return x * x

def increment(x):
    return x + 1

square_after_increment = compose(square, increment)
square_after_increment(3)  # (3+1)² = 16
```

### Function Attributes
Functions are objects and can have attributes.

```python
def greet(name):
    return f"Hello, {name}!"

greet.default_name = "World"
greet.count = 0

def smart_greet(name=None):
    smart_greet.count += 1
    if name is None:
        name = smart_greet.default_name
    return f"Hello, {name}! Called {smart_greet.count} times."

smart_greet.default_name = "World"
smart_greet.count = 0

smart_greet()  # "Hello, World! Called 1 times."
smart_greet("Alice")  # "Hello, Alice! Called 2 times."
```

### Single Dispatch
Implement function overloading based on the type of the first argument.

```python
from functools import singledispatch

@singledispatch
def process(data):
    raise NotImplementedError("Cannot process this type")

@process.register
def _(data: str):
    return f"Processing string: {data}"

@process.register
def _(data: int):
    return f"Processing integer: {data + 10}"

@process.register(list)  # Alternative syntax
def _(data):
    return f"Processing list with {len(data)} items"

process("hello")  # "Processing string: hello"
process(42)       # "Processing integer: 52"
process([1, 2, 3])  # "Processing list with 3 items"
```

## Memory Management

### Memory Model
Python uses reference counting and garbage collection for memory management.

```python
import sys

# Check reference count
x = [1, 2, 3]
sys.getrefcount(x) - 1  # Subtract 1 for the getrefcount parameter

# Multiple references
y = x
sys.getrefcount(x) - 1  # Increased by 1
```

### Weak References
References that don't prevent garbage collection.

```python
import weakref

class MyClass:
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return f"MyClass({self.name})"

obj = MyClass("test")
weak_ref = weakref.ref(obj)

print(weak_ref())  # MyClass(test)

# If obj is deleted, weak_ref will return None
del obj
print(weak_ref())  # None
```

### Memory Profiling
Tools to analyze memory usage.

```python
# Using memory_profiler
# pip install memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    big_list = [0] * 10000000
    result = sum(big_list)
    del big_list
    return result

memory_intensive_function()
```

### Object Size
Calculate the size of Python objects.

```python
import sys

# Size of basic objects
sys.getsizeof(1)  # Int size
sys.getsizeof("hello")  # String size
sys.getsizeof([1, 2, 3])  # List size
```

### Circular References
Garbage collection handles circular references.

```python
import gc

# Create circular reference
class Node:
    def __init__(self, name):
        self.name = name
        self.next = None

node1 = Node("node1")
node2 = Node("node2")
node1.next = node2
node2.next = node1

# Delete references
del node1
del node2

# Force garbage collection
gc.collect()
```

### Using slots to reduce memory
Optimize memory usage by defining __slots__.

```python
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
regular = RegularClass(1, 2)
slotted = SlottedClass(1, 2)

print(sys.getsizeof(regular))  # Larger
print(sys.getsizeof(slotted))  # Smaller
```