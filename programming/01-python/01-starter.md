# Python Introduction: From Beginner to Intermediate

## Table of Contents

1. [Introduction to Python](#introduction-to-python)
2. [Setting Up Python](#setting-up-python)
3. [Python Basics](#python-basics)
4. [Data Types and Variables](#data-types-and-variables)
5. [Operators](#operators)
6. [Control Flow](#control-flow)
7. [Functions](#functions)
8. [Data Structures](#data-structures)
9. [Modules and Packages](#modules-and-packages)
10. [File Operations](#file-operations)
11. [Error Handling](#error-handling)
12. [Object-Oriented Programming](#object-oriented-programming)
13. [Common Libraries](#common-libraries)

## Introduction to Python

Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace.

### Key Features

- **Readability**: Clean syntax that's easy to learn and understand
- **Versatility**: Used in web development, data science, AI, automation, etc.
- **Interpreted**: Code executes line by line (no compilation needed)
- **Dynamically Typed**: No need to declare variable types
- **Extensive Libraries**: Rich ecosystem of packages for various tasks

## Setting Up Python

### Installation

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Install with "Add Python to PATH" option checked
3. Verify installation by opening a terminal/command prompt and typing:

   ```zsh
   python --version
   ```

   Note that this may be `python3` instead of `python`, depending on the install.

### Development Environments

- **IDLE**: Built-in basic editor (comes with Python)
- **VS Code**: Popular, lightweight editor with Python extensions
- **PyCharm**: Full-featured Python IDE
- **Jupyter Notebooks**: Great for data science and learning

## Python Basics

### Running Python

- **Interactive Mode**: Type `python` in terminal to launch REPL
- **Script Mode**: Create a `.py` file and run with `python filename.py`

### Hello World

```python
print("Hello, World!")
```

### Comments

```python
# This is a single-line comment

"""
This is a
multi-line comment
"""
```

### Indentation

Python uses indentation (typically 4 spaces) to define code blocks:

```python
if True:
    print("This is indented")
    if True:
        print("Another level of indentation")
```

## Data Types and Variables

### Variables

```python
name = "John"  # No declaration keyword needed
age = 30
```

### Basic Data Types

```python
# Numeric types
x = 10          # int
y = 3.14        # float
z = 1 + 2j      # complex

# Text type
name = "Python" # str

# Boolean type
is_valid = True # bool

# None type
result = None   # NoneType
```

### Type Conversion

```python
# Explicit conversion
x = int("10")      # String to integer
y = float(10)      # Integer to float
z = str(3.14)      # Float to string
```

### Checking Types

```python
type(x)            # Returns the type
isinstance(x, int) # Checks if x is an integer
```

`type(x) == int` checks for an exact type match, while `isinstance(x, int)` checks for `int` or its subclasses, making the latter more versatile.

## Operators

### Arithmetic Operators

```python
a + b    # Addition
a - b    # Subtraction
a * b    # Multiplication
a / b    # Division (float result)
a // b   # Floor division (integer result)
a % b    # Modulus (remainder)
a ** b   # Exponentiation (power)
```

### Comparison Operators

```python
a == b   # Equal to
a != b   # Not equal to
a > b    # Greater than
a < b    # Less than
a >= b   # Greater than or equal to
a <= b   # Less than or equal to
```

### Logical Operators

```python
a and b  # True if both are true
a or b   # True if at least one is true
not a    # True if a is false
```

### Assignment Operators

```python
a = 5     # Assign value
a += 2    # Equivalent to a = a + 2
a -= 2    # Equivalent to a = a - 2
a *= 2    # Equivalent to a = a * 2
a /= 2    # Equivalent to a = a / 2
```

## Control Flow

### Conditional Statements

```python
# if statement
if condition:
    # code block
elif another_condition:
    # code block
else:
    # code block

# Ternary operator
result = value_if_true if condition else value_if_false
```

Note you can use the keyword `pass` to hold the place of a code block that hasn't been implemented yet.

### Loops

```python
# For loop
for item in iterable:
    # code block

# For loop with range
for i in range(5):  # 0, 1, 2, 3, 4
    # code block

# While loop
while condition:
    # code block
    
# Loop control
break    # Exit the loop
continue # Skip to the next iteration
```

## Functions

### Defining Functions

```python
def greet(name):
    """Docstring: This function greets the person"""
    return f"Hello, {name}!"

# Call the function
message = greet("Alice")
```

### Parameters and Arguments

```python
# Default parameter
def greet(name="World"):
    return f"Hello, {name}!"

# Keyword arguments
def describe_person(name, age, city):
    return f"{name} is {age} years old and lives in {city}"

describe_person(age=30, name="John", city="New York")

# Variable number of arguments
def sum_all(*args):
    return sum(args)

sum_all(1, 2, 3, 4)  # 10

# Variable number of keyword arguments
def person_details(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

person_details(name="John", age=30, city="New York")
```

The `*` prefix in Python is also used for unpacking iterables into lists or function arguments, as shown in the following examples:

1. **Unpacking a List**:

   ```python
   my_list = [1, 2, 3]
   new_list = [0, *my_list, 4]  # new_list becomes [0, 1, 2, 3, 4]
   ```

2. **Merging Lists**:

   ```python
   list1 = [1, 2]
   list2 = [3, 4]
   merged = [*list1, *list2]  # merged becomes [1, 2, 3, 4]
   ```

3. **Function Calls**:

   ```python
   def add(a, b):
       return a + b

   args = (1, 2)
   result = add(*args)  # result becomes 3
   ```

4. **Dictionary Unpacking** (using `**`):

   ```python
   dict1 = {'a': 1, 'b': 2}
   dict2 = {'c': 3}
   merged_dict = {**dict1, **dict2}  # merged_dict becomes {'a': 1, 'b': 2, 'c': 3}
   ```

### Lambda Functions

```python
# Anonymous functions
square = lambda x: x * x
square(5)  # 25
```

## Data Structures

### Lists

```python
# Ordered, mutable collection
fruits = ["apple", "banana", "cherry"]
fruits[0]                # Access by index
fruits[-1]               # Negative indexing (from end)
fruits[1:3]              # Slicing [start:end] (end exclusive)
fruits.append("date") .  # Add item
fruits.remove("banana")  # Remove item
len(fruits)              # Length
"apple" in fruits        # Membership test
```

### Tuples

```python
# Ordered, immutable collection
coordinates = (10, 20)
coordinates[0]      # Access by index
len(coordinates)    # Length
```

Note that a singleton tuple has to be written `(x,)` because `(x)` is just `x`.

### Dictionaries

```python
# Key-value pairs, mutable
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
person["name"]          # Access by key, KeyError if not found
person.get("age")       # Safe access with get()
person.get("age", 18)   # get() with a default value
person["email"] = "john@example.com"  # Add/update item
person.pop("city")      # Remove item
list(person.keys())     # All keys
list(person.values())   # All values
list(person.items())    # All key-value pairs
```

### Sets

```python
# Unordered collection of unique items
fruits = {"apple", "banana", "cherry"}
fruits.add("date")      # Add item
fruits.remove("banana") # Remove item
"apple" in fruits       # Membership test
```

### Comprehensions

```python
# List comprehension
squares = [x**2 for x in range(10)]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}

# Set comprehension
even_squares = {x**2 for x in range(10) if x % 2 == 0}
```

## Modules and Packages

### Importing Modules

```python
# Basic import
import math
math.sqrt(16)  # 4.0

# Import specific functions
from math import sqrt, pi
sqrt(16)  # 4.0

# Import with alias
import math as m
m.sqrt(16)  # 4.0

# Import all (not recommended)
from math import *
```

### Creating Modules

```python
# In mymodule.py
def greet(name):
    return f"Hello, {name}!"

# In another file
import mymodule
mymodule.greet("Alice")
```

### Common Standard Library Modules

- `math`: Mathematical functions
- `random`: Random number generation
- `datetime`: Date and time operations
- `os`: Operating system interface
- `sys`: System-specific parameters and functions
- `json`: JSON encoder and decoder
- `re`: Regular expressions

## File Operations

### Reading Files

```python
# Using with statement (context manager)
with open("file.txt", "r") as file:
    content = file.read()       # Read entire file
    
with open("file.txt", "r") as file:
    lines = file.readlines()    # Read all lines into a list
    
with open("file.txt", "r") as file:
    for line in file:           # Read line by line
        print(line.strip())
```

### Writing Files

```python
with open("file.txt", "w") as file:
    file.write("Hello, World!")  # Write string
    
with open("file.txt", "a") as file:
    file.write("\nAppended text")  # Append to file
```

### Working with CSV

```python
import csv

# Reading CSV
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Writing CSV
with open("data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 30])
```

### Working with JSON

```python
import json

# Reading JSON
with open("data.json", "r") as file:
    data = json.load(file)

# Writing JSON
with open("data.json", "w") as file:
    json.dump(data, file, indent=4)
```

## Error Handling

### Try-Except Blocks

```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    # Handle specific exception
    print("Cannot divide by zero")
except Exception as e:
    # Handle any other exception
    print(f"Error: {e}")
else:
    # Executes if no exception
    print("No error occurred")
finally:
    # Always executes
    print("This will always run")
```

The `else` block runs **only if no exception is raised** in the `try` block. It's used for code that should **only run when the `try` succeeds** and will **not run if an exception is raised**, even if that exception is handled. This helps separate **risky code** from **safe, follow-up logic**.

The `finally` block **always runs**, even if an exception is raised and **not handled**. It executes **before the program crashes or exits** due to the unhandled exception. This makes it ideal for cleanup tasks like closing files or releasing resources.

### Raising Exceptions

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

## Object-Oriented Programming

### Classes and Objects

```python
class Person:
    # Class variable
    species = "Homo sapiens"
    
    # Constructor method
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
    
    # Instance method
    def greet(self):
        return f"Hello, my name is {self.name}"
    
    # Static method
    @staticmethod
    def is_adult(age):
        return age >= 18

# Creating objects
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# Accessing attributes and methods
print(person1.name)          # Alice
print(person1.greet())       # Hello, my name is Alice
print(Person.species)        # Homo sapiens
print(Person.is_adult(20))   # True
```

Note that when calling `person1.greet()`, Python automatically passes the object as the first argument to the method (here the first argument is caught as `self`, which is used to access the instance variables).

Note that we can also call `person1.is_adult(20)` without Python automatically passing `self` to the method, because `is_adult` is marked as a static method.

If a class and instance variable share the same name, the instance variable overrides (or shadows) the class variable for that specific object, without affecting the class or other instances.

```python
class MyClass:
    x = 10

obj = MyClass()
obj.x = 99  # overrides class variable for this instance

print(MyClass.x)  # 10
print(obj.x)      # 99
del obj.x         # deletes the instance variable
print(obj.x)      # The class variable is still accessible
```

### Inheritance

```python
class Student(Person):
    def __init__(self, name, age, student_id):
        # Call parent constructor
        super().__init__(name, age)
        self.student_id = student_id
    
    # Override method
    def greet(self):
        return f"{super().greet()} and I'm a student"
```

Python supports multiple inheritance, allowing a class to inherit from more than one parent. For example, if class `C` inherits from both `A` and `B`, it can use methods from both: `class C(A, B): pass`. If both parents define the same method, Python uses the Method Resolution Order (MRO) to resolve conflicts, checking classes from left to right as listed in the inheritance: in `class C(A, B)`, `A`'s method will override `B`'s if they share the same name.

```python
class A:
    def whoami(self):
        return "A"

class B:
    def whoami(self):
        return "B"

class C(A, B):
    pass

obj = C()
print(obj.whoami())  # Output: A
```

### Special Methods

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representation
    def __str__(self):
        return f"Point({self.x}, {self.y})"
    
    # Addition
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    # Equality
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
```

#### Initialization & Representation

| Method      | Purpose                                   |
|-------------|-------------------------------------------|
| `__init__`  | Object constructor                        |
| `__str__`   | Human-readable string (`str(obj)`)        |
| `__repr__`  | Debug string (`repr(obj)`)                |

#### Arithmetic & Unary Operators

| Method       | Operator      |
|--------------|----------------|
| `__add__`    | `+`           |
| `__sub__`    | `-`           |
| `__mul__`    | `*`           |
| `__truediv__`| `/`           |
| `__floordiv__`| `//`         |
| `__mod__`    | `%`           |
| `__pow__`    | `**`          |
| `__neg__`    | Unary `-obj`  |
| `__abs__`    | `abs(obj)`    |

Right-hand versions like `__radd__` exist as well. Priority is left-to-right, so `__add__` is called first.

#### Special Comparison Operators Methods

| Method     | Operator        |
|------------|------------------|
| `__eq__`   | `==`             |
| `__ne__`   | `!=`             |
| `__lt__`   | `<`              |
| `__le__`   | `<=`             |
| `__gt__`   | `>`              |
| `__ge__`   | `>=`             |

#### Container-Like Behavior

| Method         | Purpose                       |
|----------------|-------------------------------|
| `__len__`      | `len(obj)`                    |
| `__getitem__`  | Indexing (`obj[key]`)         |
| `__setitem__`  | Assignment (`obj[key] = val`) |
| `__delitem__`  | Deletion (`del obj[key]`)     |
| `__contains__` | Membership (`x in obj`)       |
| `__iter__`, `__next__` | Iteration              |

#### Callable & Context Managers

| Method       | Purpose                          |
|--------------|----------------------------------|
| `__call__`   | Makes object callable (`obj()`)  |
| `__enter__`, `__exit__` | `with` statement support |

#### Attribute & Identity

| Method         | Purpose                       |
|----------------|-------------------------------|
| `__getattr__`  | Handle missing attributes      |
| `__setattr__`  | Intercept attribute setting    |
| `__eq__`, `__hash__` | Used for dict/set keys    |
| `__bool__`     | Boolean check (`bool(obj)`)    |

An **item** is a value stored inside a container-like object and accessed using `[]`.
An **attribute** is part of an object and accessed using dot notation `.`.

Override `__setattr__` when you want to control or monitor how attributes are set — just remember to use `super().__setattr__` to actually store the value, otherwise you end in an infinite recursion.

There are also internal attributes like `__dict__` and `__name__` that you can use to access the internal state of an object.

## Common Libraries

### NumPy (Numerical Computing)

```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
arr + 10        # Element-wise addition
arr * 2         # Element-wise multiplication
np.sqrt(arr)    # Element-wise square root
```

### Pandas (Data Analysis)

```python
import pandas as pd

# Create DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)

# Data manipulation
df['Age'].mean()            # Calculate mean
df[df['Age'] > 30]          # Filter data
df.sort_values('Age')       # Sort data
```

### Matplotlib (Data Visualization)

```python
import matplotlib.pyplot as plt

# Line plot
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]
plt.plot(x, y)
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

### Requests (HTTP Requests)

```python
import requests

# GET request
response = requests.get('https://api.example.com/data')
data = response.json()

# POST request
payload = {'key': 'value'}
response = requests.post('https://api.example.com/submit', json=payload)
```
