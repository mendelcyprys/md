# Advanced Python Topics Part 2: Continued Mastery

## Table of Contents
1. [Metaclasses](#metaclasses)
2. [Descriptors](#descriptors)
3. [Abstract Base Classes](#abstract-base-classes)
4. [Concurrency](#concurrency)
5. [Typing](#typing)

## Metaclasses

### Understanding Metaclasses
Metaclasses are the "classes of classes" that define how classes behave.

```python
# The default metaclass for all classes
type(object)  # <class 'type'>

# Creating a class dynamically
MyClass = type('MyClass', (object,), {'x': 5, 'say_hello': lambda self: 'Hello'})
obj = MyClass()
obj.x  # 5
obj.say_hello()  # 'Hello'
```

### Custom Metaclasses
Create custom metaclasses by subclassing `type`.

```python
class LoggingMeta(type):
    def __new__(mcs, name, bases, attrs):
        print(f"Creating class: {name}")
        return super().__new__(mcs, name, bases, attrs)
    
    def __init__(cls, name, bases, attrs):
        print(f"Initializing class: {name}")
        super().__init__(name, bases, attrs)

class MyClass(metaclass=LoggingMeta):
    pass  # Output: "Creating class: MyClass" followed by "Initializing class: MyClass"
```

### Metaclass Methods
Key methods that metaclasses can implement.

```python
class TraceMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Modify class attributes before creation
        attrs['created_by'] = 'TraceMeta'
        return super().__new__(mcs, name, bases, attrs)
    
    def __call__(cls, *args, **kwargs):
        # Called when the class is instantiated
        print(f"Creating instance of {cls.__name__}")
        instance = super().__call__(*args, **kwargs)
        print(f"Instance created: {instance}")
        return instance

class Traced(metaclass=TraceMeta):
    def __init__(self, name):
        self.name = name

obj = Traced("test")  # Triggers __call__ in TraceMeta
print(obj.created_by)  # 'TraceMeta'
```

### Singleton Pattern with Metaclasses
Implement the Singleton design pattern using metaclasses.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Initializing database connection")

# Only initializes once
db1 = Database()  # Prints "Initializing database connection"
db2 = Database()  # No output
print(db1 is db2)  # True
```

### Registry Pattern
Create class registries for plugins or extensions.

```python
class PluginRegistry(type):
    plugins = {}
    
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if bases:  # Only register non-abstract classes
            mcs.plugins[name] = cls
        return cls

class Plugin(metaclass=PluginRegistry):
    """Base plugin class"""
    pass

class AudioPlugin(Plugin):
    """Audio processing plugin"""
    pass

class VideoPlugin(Plugin):
    """Video processing plugin"""
    pass

print(PluginRegistry.plugins)  # {'AudioPlugin': <class 'AudioPlugin'>, 'VideoPlugin': <class 'VideoPlugin'>}
```

## Descriptors

### Basic Descriptors
Descriptors control attribute access through `__get__`, `__set__`, and `__delete__` methods.

```python
class Descriptor:
    def __get__(self, instance, owner):
        print(f"Getting from {instance} with class {owner}")
        return self.value
    
    def __set__(self, instance, value):
        print(f"Setting {value} to {instance}")
        self.value = value
    
    def __delete__(self, instance):
        print(f"Deleting from {instance}")
        del self.value

class MyClass:
    attr = Descriptor()

obj = MyClass()
obj.attr = 42  # Setting 42 to <__main__.MyClass object at 0x...>
print(obj.attr)  # Getting from <__main__.MyClass object at 0x...> with class <class '__main__.MyClass'>
                 # 42
del obj.attr  # Deleting from <__main__.MyClass object at 0x...>
```

### Data Validation with Descriptors
Use descriptors to validate attributes.

```python
class Positive:
    def __init__(self):
        self.name = None
    
    def __set_name__(self, owner, name):
        # Store the attribute name
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]
    
    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        instance.__dict__[self.name] = value

class Product:
    price = Positive()
    quantity = Positive()
    
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity
    
    def total(self):
        return self.price * self.quantity

# This works
product = Product(10, 5)
print(product.total())  # 50

# This raises ValueError
try:
    product = Product(-10, 5)
except ValueError as e:
    print(e)  # price must be positive
```

### Property vs Descriptors
The `property` built-in is actually a descriptor.

```python
class Person:
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        """Get the person's name."""
        return self._name
    
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value

# Equivalent descriptor implementation
class NameDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._name
    
    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        instance._name = value

class Person2:
    name = NameDescriptor()
    
    def __init__(self, name):
        self.name = name
```

### Method Descriptors
Methods in Python are descriptors too.

```python
class MyClass:
    def method(self, x):
        return x * 2

obj = MyClass()

# Method is a descriptor bound to obj
print(obj.method)  # <bound method MyClass.method of <__main__.MyClass object at 0x...>>

# Unbound method from the class
print(MyClass.method)  # <function MyClass.method at 0x...>
```

### Lazy Properties
Create attributes that are computed only when needed.

```python
class LazyProperty:
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.func(instance)
        # Replace with computed value
        setattr(instance, self.func.__name__, value)
        return value

class ExpensiveData:
    def __init__(self, filename):
        self.filename = filename
    
    @LazyProperty
    def data(self):
        """Load data only when accessed"""
        print(f"Loading data from {self.filename}...")
        # Simulate expensive operation
        import time
        time.sleep(1)
        return [1, 2, 3, 4, 5]

# Data is loaded only when needed
ed = ExpensiveData("large_file.dat")
print("Object created")  # Doesn't load data yet
print(ed.data)  # Loads data the first time
print(ed.data)  # Uses cached data the second time
```

## Abstract Base Classes

### Creating Abstract Base Classes
Define interfaces that derived classes must implement.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        """Calculate the area of the shape"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape"""
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        import math
        return 2 * math.pi * self.radius

# This would raise TypeError: Can't instantiate abstract class Shape with abstract methods area, perimeter
# shape = Shape()

# This works
circle = Circle(5)
print(circle.area())  # ~78.54
```

### Abstract Properties
Define properties that must be implemented.

```python
class Vehicle(ABC):
    @property
    @abstractmethod
    def wheels(self):
        """Number of wheels"""
        pass
    
    @abstractmethod
    def drive(self):
        """Drive the vehicle"""
        pass

class Car(Vehicle):
    @property
    def wheels(self):
        return 4
    
    def drive(self):
        return "Driving car"

class Motorcycle(Vehicle):
    @property
    def wheels(self):
        return 2
    
    def drive(self):
        return "Riding motorcycle"

car = Car()
print(car.wheels)  # 4
print(car.drive())  # Driving car
```

### Abstract Class Methods
Define class methods that must be implemented.

```python
class DatabaseConnector(ABC):
    @classmethod
    @abstractmethod
    def get_connection_string(cls):
        pass
    
    @abstractmethod
    def connect(self):
        pass

class MySQLConnector(DatabaseConnector):
    @classmethod
    def get_connection_string(cls):
        return "mysql://user:pass@localhost/db"
    
    def connect(self):
        conn_str = self.get_connection_string()
        print(f"Connecting to {conn_str}")
        # Actual connection code would go here
        return "Connected"

mysql = MySQLConnector()
print(mysql.connect())  # Connecting to mysql://user:pass@localhost/db
                         # Connected
```

### Virtual Subclasses
Register classes as "virtual" subclasses of ABCs.

```python
from abc import ABC

class Drawable(ABC):
    @abstractmethod
    def draw(self):
        pass

# A class that doesn't explicitly inherit from Drawable
class SVG:
    def __init__(self, content):
        self.content = content
    
    def draw(self):
        return f"Drawing SVG: {self.content}"

# Register as a virtual subclass
Drawable.register(SVG)

svg = SVG("<circle>")
print(isinstance(svg, Drawable))  # True
```

### Standard Library ABCs
Python's collections.abc module provides many useful ABCs.

```python
from collections.abc import Sequence, Mapping, MutableMapping

# Check if objects implement interfaces
print(isinstance([1, 2, 3], Sequence))  # True
print(isinstance({"a": 1}, Mapping))    # True
print(isinstance({"a": 1}, MutableMapping))  # True
print(isinstance("abc", MutableMapping))  # False
```

## Concurrency

### Threading
Run functions concurrently in separate threads.

```python
import threading
import time

def task(name, delay):
    print(f"{name} started")
    time.sleep(delay)
    print(f"{name} completed")

# Create threads
thread1 = threading.Thread(target=task, args=("Thread-1", 2))
thread2 = threading.Thread(target=task, args=("Thread-2", 1))

# Start threads
thread1.start()
thread2.start()

# Wait for threads to complete
thread1.join()
thread2.join()

print("All threads completed")
```

### Thread Safety
Protect shared resources using locks.

```python
import threading
import time

counter = 0
lock = threading.Lock()

def increment(amount):
    global counter
    local_counter = counter
    local_counter += amount
    time.sleep(0.1)  # Simulate some work
    counter = local_counter

# Without lock - race condition
threads = []
for i in range(5):
    thread = threading.Thread(target=increment, args=(1,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print(f"Counter without lock: {counter}")  # Likely less than 5

# Reset counter
counter = 0

# With lock - thread safe
def safe_increment(amount):
    global counter
    with lock:
        local_counter = counter
        local_counter += amount
        time.sleep(0.1)  # Simulate some work
        counter = local_counter

threads = []
for i in range(5):
    thread = threading.Thread(target=safe_increment, args=(1,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print(f"Counter with lock: {counter}")  # 5
```

### Multiprocessing
Run functions in separate processes.

```python
import multiprocessing
import time
import os

def worker(name):
    print(f"{name} started with PID {os.getpid()}")
    time.sleep(1)
    print(f"{name} completed")
    return name

if __name__ == "__main__":
    # Create a pool of workers
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(worker, ["Process-1", "Process-2", "Process-3", "Process-4"])
        print(f"Results: {results}")
```

### Asyncio
Write asynchronous code using async/await syntax.

```python
import asyncio

async def task(name, delay):
    print(f"{name} started")
    await asyncio.sleep(delay)  # Non-blocking sleep
    print(f"{name} completed")
    return f"{name} result"

async def main():
    # Create and gather tasks
    results = await asyncio.gather(
        task("Task-1", 2),
        task("Task-2", 1),
        task("Task-3", 3)
    )
    print(f"Results: {results}")

# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())
```

### Asyncio HTTP Requests
Make non-blocking HTTP requests with asyncio.

```python
import asyncio
import aiohttp
import time

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

async def main():
    urls = [
        'http://example.com',
        'http://example.org',
        'http://example.net',
    ]
    
    start = time.time()
    results = await fetch_all(urls)
    end = time.time()
    
    print(f"Fetched {len(results)} URLs in {end - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
```

### Combining Asyncio with Threading
Use threads for CPU-bound tasks and asyncio for I/O-bound tasks.

```python
import asyncio
import concurrent.futures
import time

def cpu_bound_task(x):
    """Simulate CPU-intensive calculation"""
    result = 0
    for i in range(10**7):
        result += i * x
    return result

async def main():
    print("Starting...")
    
    # Create a thread pool
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    # Schedule CPU-bound tasks in the thread pool
    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(executor, cpu_bound_task, i)
        for i in range(1, 5)
    ]
    
    # While CPU-bound tasks are running, do some asyncio tasks
    await asyncio.gather(
        asyncio.sleep(0.1),
        asyncio.sleep(0.1)
    )
    
    # Wait for CPU-bound results
    results = await asyncio.gather(*futures)
    print(f"Results: {results}")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    print(f"Completed in {time.time() - start:.2f} seconds")
```

## Typing

### Basic Type Annotations
Add type hints to variables and functions.

```python
def greeting(name: str) -> str:
    return f"Hello, {name}"

age: int = 30
ratio: float = 0.8
active: bool = True
```

### Complex Types
Use typing module for more complex type hints.

```python
from typing import List, Dict, Tuple, Set, Optional, Union

# Container types
names: List[str] = ["Alice", "Bob", "Charlie"]
scores: Dict[str, int] = {"Alice": 95, "Bob": 87}
point: Tuple[int, int] = (10, 20)
unique_numbers: Set[int] = {1, 2, 3}

# Optional and Union types
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # Could be str or None

def process_input(data: Union[str, bytes]) -> str:
    if isinstance(data, bytes):
        return data.decode('utf-8')
    return data
```

### Callable Types
Define function type signatures.

```python
from typing import Callable, TypeVar, Generic

# Simple callable annotation
Handler = Callable[[str], bool]

def process_with_handler(text: str, handler: Handler) -> bool:
    return handler(text)

# Type variable for generic functions
T = TypeVar('T')

def first(items: List[T]) -> Optional[T]:
    return items[0] if items else None
```

### Type Aliases
Create aliases for complex types.

```python
from typing import Dict, List, Tuple, TypeAlias

# Type aliases make complex types more readable
Point: TypeAlias = Tuple[float, float]
Polygon: TypeAlias = List[Point]
PolygonDict: TypeAlias = Dict[str, Polygon]

def area(polygon: Polygon) -> float:
    # Calculate area of polygon
    return 0.0

# More complex example
JSON: TypeAlias = Union[Dict[str, 'JSON'], List['JSON'], str, int, float, bool, None]

def parse_json(data: str) -> JSON:
    import json
    return json.loads(data)
```

### Protocols
Define structural subtyping with protocols.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Canvas:
    def draw(self) -> str:
        return "Drawing canvas"

class Button:
    def draw(self) -> str:
        return "Drawing button"

def render(item: Drawable) -> None:
    print(item.draw())

# Works because Canvas has a draw method, even though it doesn't inherit from Drawable
render(Canvas())  # Drawing canvas
render(Button())  # Drawing button

# Check at runtime
print(isinstance(Canvas(), Drawable))  # True
```

### Type Guards
Custom functions to narrow types.

```python
from typing import TypeGuard, Union, List

def is_string_list(val: List[object]) -> TypeGuard[List[str]]:
    """Determine if all objects in the list are strings"""
    return all(isinstance(x, str) for x in val)

def process_strings(values: Union[List[str], List[int]]) -> None:
    if is_string_list(values):
        # Within this block, values is known to be List[str]
        print("Length of first string:", len(values[0]))
    else:
        # Here, values is known to be List[int]
        print("First integer plus 5:", values[0] + 5)
```

### Final and ClassVar
Declare constants and class variables.

```python
from typing import Final, ClassVar

# Constants
PI: Final = 3.14159
API_KEY: Final = "abc123"

class Config:
    # Class variables
    DEBUG: ClassVar[bool] = False
    VERSION: ClassVar[str] = "1.0.0"
    
    # Instance variable
    def __init__(self, name: str):
        self.name = name

# This would be flagged by type checkers
# PI = 3.0  # Error: Cannot assign to Final name "PI"
```

### Literal Types
Restrict values to specific literals.

```python
from typing import Literal, Union

# Function that only accepts specific string values
def align_text(text: str, align: Literal["left", "center", "right"]) -> str:
    if align == "left":
        return text.ljust(20)
    elif align == "center":
        return text.center(20)
    else:  # right
        return text.rjust(20)

# Type checkers would flag this
# align_text("Hello", "middle")  # Error: Argument 2 has incompatible type

# Literal with multiple types
Mode = Literal[1, 2, 3, "debug", "release"]

def set_mode(mode: Mode) -> None:
    print(f"Setting mode to {mode}")
```

### Type Checking
Use mypy to check your type annotations.

```python
# Save as example.py
def add_numbers(a: int, b: int) -> int:
    return a + b

result = add_numbers("5", 10)  # Type error
```

Run mypy to find type errors:
```
$ mypy example.py
example.py:4: error: Argument 1 to "add_numbers" has incompatible type "str"; expected "int"
```

### Forward References
Reference types that aren't defined yet.

```python
class Tree:
    def __init__(self, value: int, left: Optional['Tree'] = None, right: Optional['Tree'] = None):
        self.value = value
        self.left = left
        self.right = right

# With Python 3.10+ using PEP 563
from __future__ import annotations

class LinkedList:
    def __init__(self, value: int, next: Optional[LinkedList] = None):
        self.value = value
        self.next = next
```