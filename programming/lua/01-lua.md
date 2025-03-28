# Lua Syntax Guide for Beginners

## Table of Contents

- [Lua Syntax Guide for Beginners](#lua-syntax-guide-for-beginners)
  - [Table of Contents](#table-of-contents)
  - [Running Lua](#running-lua)
  - [Comments](#comments)
  - [Variables and Data Types](#variables-and-data-types)
  - [Operators](#operators)
  - [Control Structures](#control-structures)
  - [Functions](#functions)
  - [Tables](#tables)
  - [Strings](#strings)
  - [Scopes](#scopes)
  - [Metatables](#metatables)
  - [Error Handling](#error-handling)
  - [Modules](#modules)
  - [File I/O](#file-io)
  - [Standard Libraries](#standard-libraries)

## Running Lua

Run an interactive session:

```bash
lua -i
```

Execute a Lua statement:

```bash
lua -e "print('Hello, World!')"
```

Run a Lua file:

```bash
lua filename.lua
```

## Comments

Single line comment:

```lua
-- This is a single line comment
```

Multi-line comment:

```lua
--[[
This is a
multi-line comment
--]]
```

## Variables and Data Types

Lua is dynamically typed with 8 basic types:

```lua
-- nil (absence of value)
local a = nil

-- boolean
local b = true
local c = false

-- number (double-precision floating-point)
local d = 42       -- integer
local e = 3.14     -- float
local f = 2.5e6    -- scientific notation

-- string
local g = "Hello"
local h = 'World'
local i = [[
Multiline
string
]]

-- function
local j = function() return 1 end

-- table (the only data structure in Lua)
local k = {}

-- userdata (for C data)
-- not directly usable in pure Lua

-- thread (coroutines)
local l = coroutine.create(function() print("Hi") end)
```

Variable declaration:

```lua
-- Global variables (accessible everywhere)
global_var = 10

-- Local variables (limited scope)
local local_var = 20

-- Multiple assignment
local a, b, c = 1, 2, 3
```

Note that the use of the keyword `local` with assignments will seem to not work in the REPL, this is because each line is executed in its own scope. Use global variables or wrap in a `do ... end` block, with `print()` to log results to the console:

```lua
do local a, b, c = 1, 2, 3 print(b) end
```

## Operators

Arithmetic operators:

```lua
local a = 10
local b = 3

print(a + b)    -- Addition: 13
print(a - b)    -- Subtraction: 7
print(a * b)    -- Multiplication: 30
print(a / b)    -- Division: 3.3333...
print(a % b)    -- Modulo: 1
print(a ^ b)    -- Exponentiation: 1000
```

Relational operators:

```lua
print(a == b)   -- Equal to: false
print(a ~= b)   -- Not equal to: true
print(a < b)    -- Less than: false
print(a > b)    -- Greater than: true
print(a <= b)   -- Less than or equal to: false
print(a >= b)   -- Greater than or equal to: true
```

Logical operators:

```lua
print(true and false)   -- Logical AND: false
print(true or false)    -- Logical OR: true
print(not true)         -- Logical NOT: false
```

Concatenation operator:

```lua
print("Hello" .. " " .. "World")   -- Hello World
```

Length operator:

```lua
print(#"Hello")   -- 5 (length of string)
print(#{1, 2, 3}) -- 3 (size of table)
```

## Control Structures

If statement:

```lua
local age = 18

if age < 18 then
  print("Minor")
elseif age == 18 then
  print("Just turned adult")
else
  print("Adult")
end
```

While loop:

```lua
local count = 1
while count <= 5 do
  print(count)
  count = count + 1
end
```

Repeat-until loop:

```lua
local count = 1
repeat
  print(count)
  count = count + 1
until count > 5
```

Numeric for loop:

```lua
-- for var=start,end,step do
for i = 1, 5, 1 do  -- step is optional (default 1)
  print(i)
end
```

Generic for loop:

```lua
local fruits = {"apple", "banana", "orange"}
for index, value in ipairs(fruits) do  -- iterates with index
  print(index, value)
end

local person = {name="John", age=30}
for key, value in pairs(person) do  -- iterates key-value pairs
  print(key, value)
end
```

Break statement:

```lua
for i = 1, 10 do
  if i > 5 then
    break  -- exits the loop
  end
  print(i)
end
```

## Functions

Function declaration:

```lua
-- Basic function
function sayHello()
  print("Hello!")
end
sayHello()  -- Call the function

-- Function with parameters
function greet(name)
  print("Hello, " .. name .. "!")
end
greet("Alice")  -- Hello, Alice!

-- Function with multiple parameters and return value
function add(a, b)
  return a + b
end
print(add(2, 3))  -- 5

-- Multiple return values
function getNameAndAge()
  return "John", 30
end
local name, age = getNameAndAge()
print(name, age)  -- John 30
```

Anonymous functions:

```lua
local multiply = function(a, b)
  return a * b
end
print(multiply(4, 5))  -- 20
```

Functions as arguments:

```lua
function applyOperation(func, a, b)
  return func(a, b)
end

print(applyOperation(function(x, y) return x + y end, 2, 3))  -- 5
```

Variadic functions:

```lua
function sum(...)
  local total = 0
  for _, v in ipairs({...}) do
    total = total + v
  end
  return total
end

print(sum(1, 2, 3, 4))  -- 10
```

## Tables

Tables are the only data structure in Lua and can be used as arrays, dictionaries, objects, etc.

Array-like tables:

```lua
local fruits = {"apple", "banana", "orange"}
print(fruits[1])  -- apple (Lua arrays are 1-indexed)
print(fruits[2])  -- banana
print(#fruits)    -- 3 (length)

-- Iterating arrays
for i = 1, #fruits do
  print(i, fruits[i])
end

-- Using ipairs (safer for arrays)
for i, fruit in ipairs(fruits) do
  print(i, fruit)
end
```

Dictionary-like tables:

```lua
local person = {
  name = "John",
  age = 30,
  ["favorite color"] = "blue"  -- Keys with spaces need brackets
}

print(person.name)            -- John (dot notation)
print(person["age"])          -- 30 (bracket notation)
print(person["favorite color"])  -- blue (must use brackets for keys with spaces)

-- Add or modify entries
person.job = "Engineer"
person["age"] = 31

-- Iterating all key-value pairs
for key, value in pairs(person) do
  print(key, value)
end
```

Nested tables:

```lua
local student = {
  name = "Alice",
  grades = {math = 90, science = 85, history = 95},
  hobbies = {"reading", "hiking"}
}

print(student.grades.math)  -- 90
print(student.hobbies[1])   -- reading
```

Table constructors:

```lua
-- Array constructor with explicit indices
local days = {
  [1] = "Monday",
  [2] = "Tuesday",
  [3] = "Wednesday"
}

-- Mixed constructor
local mixed = {
  10,                -- [1] = 10
  20,                -- [2] = 20
  name = "Mixed",    -- ["name"] = "Mixed"
  [3*2] = "Six"      -- [6] = "Six"
}
```

## Strings

String declaration:

```lua
local s1 = "Double quotes"
local s2 = 'Single quotes'
local s3 = [[
  Multi-line
  string
]]
```

String operations:

```lua
-- Concatenation
local fullName = "John" .. " " .. "Doe"  -- John Doe

-- Length
print(#"Hello")  -- 5

-- Substring (string.sub(s, start, end))
local s = "Hello, World!"
print(string.sub(s, 1, 5))    -- Hello (from position 1 to 5)
print(string.sub(s, 8, 12))   -- World (from position 8 to 12)
print(string.sub(s, 8))       -- World! (from position 8 to end)
```

String functions:

```lua
-- Convert to uppercase/lowercase
print(string.upper("hello"))  -- HELLO
print(string.lower("WORLD"))  -- world

-- Find pattern (string.find(s, pattern))
local s = "Hello, World!"
local start, stop = string.find(s, "World")
print(start, stop)  -- 8 12

-- Replace (string.gsub(s, pattern, replacement))
local new_s, count = string.gsub("Hello, World!", "World", "Lua")
print(new_s)  -- Hello, Lua!

-- Match pattern (string.match(s, pattern))
print(string.match("Age: 30", "%d+"))  -- 30 (matches digits)

-- Format (string.format)
print(string.format("Name: %s, Age: %d", "Alice", 25))  -- Name: Alice, Age: 25
```

## Scopes

Lua has lexical scoping with two main types of scope:

```lua
-- Global scope
global_var = 10

function example()
  -- Function scope
  local local_var = 20
  
  print(global_var)  -- 10 (accessible from anywhere)
  print(local_var)   -- 20 (only accessible within this function)
  
  if true then
    -- Block scope
    local block_var = 30
    print(block_var)  -- 30 (only accessible within this block)
  end
  
  -- print(block_var)  -- error: block_var is not accessible here
end

example()
print(global_var)  -- 10
-- print(local_var)  -- error: local_var is not accessible here
```

## Metatables

Metatables allow you to change the behavior of tables:

```lua
-- Create a basic metatable
local mt = {
  __add = function(a, b)
    return {value = a.value + b.value}
  end
}

-- Create tables with the metatable
local a = {value = 10}
local b = {value = 20}
setmetatable(a, mt)
setmetatable(b, mt)

-- Use overloaded operator
local result = a + b
print(result.value)  -- 30
```

Common metamethods:
```lua
local mt = {
  __add = function(a, b) return {value = a.value + b.value} end,    -- a + b
  __sub = function(a, b) return {value = a.value - b.value} end,    -- a - b
  __mul = function(a, b) return {value = a.value * b.value} end,    -- a * b
  __div = function(a, b) return {value = a.value / b.value} end,    -- a / b
  __mod = function(a, b) return {value = a.value % b.value} end,    -- a % b
  __pow = function(a, b) return {value = a.value ^ b.value} end,    -- a ^ b
  __unm = function(a) return {value = -a.value} end,                -- -a
  __concat = function(a, b) return {value = a.value .. b.value} end,-- a .. b
  __len = function(a) return a.length end,                          -- #a
  __eq = function(a, b) return a.value == b.value end,              -- a == b
  __lt = function(a, b) return a.value < b.value end,               -- a < b
  __le = function(a, b) return a.value <= b.value end,              -- a <= b
  __index = function(tbl, key) return key.."_not_found" end,        -- tbl[key]
  __newindex = function(tbl, key, value) rawset(tbl, key, value*2) end -- tbl[key] = value
}
```

## Error Handling

Try-catch style error handling:
```lua
local success, result = pcall(function()
  -- Code that might error
  return 10 / 0  -- Will cause an error
end)

if success then
  print("Result:", result)
else
  print("Error:", result)  -- Error: attempt to divide by zero
end
```

Throwing errors:
```lua
function divide(a, b)
  if b == 0 then
    error("Cannot divide by zero", 2)  -- 2 is the level for stack trace
  end
  return a / b
end

local success, result = pcall(divide, 10, 0)
if not success then
  print("Error:", result)  -- Error: Cannot divide by zero
end
```

## Modules

Creating a module (in `mymodule.lua`):
```lua
local M = {}

M.name = "My Module"

function M.sayHello(name)
  return "Hello, " .. name .. "!"
end

function M.add(a, b)
  return a + b
end

return M
```

Using a module:
```lua
-- Assuming mymodule.lua is in the Lua path
local mymodule = require("mymodule")

print(mymodule.name)  -- My Module
print(mymodule.sayHello("Alice"))  -- Hello, Alice!
print(mymodule.add(2, 3))  -- 5
```

## File I/O

Reading a file:
```lua
-- Read the entire file
local file = io.open("example.txt", "r")
if file then
  local content = file:read("*a")  -- "*a" reads the entire file
  print(content)
  file:close()
end

-- Read line by line
file = io.open("example.txt", "r")
if file then
  for line in file:lines() do
    print(line)
  end
  file:close()
end
```

Writing to a file:
```lua
local file = io.open("output.txt", "w")
if file then
  file:write("Hello, World!\n")
  file:write("This is Lua file I/O.")
  file:close()
end
```

## Standard Libraries

Math library:
```lua
print(math.abs(-10))      -- 10
print(math.sin(math.pi/2))-- 1.0
print(math.random())      -- Random number between 0 and 1
print(math.random(1, 10)) -- Random integer between 1 and 10
print(math.floor(3.7))    -- 3
print(math.ceil(3.2))     -- 4
print(math.max(2, 8, 4))  -- 8
```

String library (see Strings section)

Table library:
```lua
local t = {10, 20, 30}
table.insert(t, 40)        -- Add to the end: {10, 20, 30, 40}
table.insert(t, 2, 15)     -- Insert at position 2: {10, 15, 20, 30, 40}
table.remove(t, 1)         -- Remove first element: {15, 20, 30, 40}
table.sort(t)              -- Sort table: {15, 20, 30, 40}
print(table.concat(t, ", "))  -- Join elements: "15, 20, 30, 40"
```

OS library:
```lua
print(os.time())          -- Current time as timestamp
print(os.date("%Y-%m-%d")) -- Format date: "2023-02-14"
os.execute("dir")         -- Run system command
print(os.clock())         -- CPU time used
```

Debug library:
```lua
function example()
  print(debug.traceback())  -- Print stack traceback
  local info = debug.getinfo(1)  -- Get info about current function
  print(info.name, info.linedefined)
end
example()
```
Lua.md
Displaying Lua.md.