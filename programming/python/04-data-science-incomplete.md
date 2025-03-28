# Python for Data Science: The Essential Path

## Table of Contents

1. [Introduction to Data Science with Python](#introduction-to-data-science-with-python)
2. [NumPy: Numerical Computing](#numpy-numerical-computing)
3. [Pandas: Data Manipulation](#pandas-data-manipulation)
4. [Data Visualization](#data-visualization)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Statistical Analysis](#statistical-analysis)

---
7. [Machine Learning with Scikit-learn](#machine-learning-with-scikit-learn)
8. [Deep Learning Introduction](#deep-learning-introduction)
9. [Data Science Workflow](#data-science-workflow)
10. [Next Steps](#next-steps)

## Introduction to Data Science with Python

### What is Data Science?
Data science combines multiple fields including statistics, scientific methods, and data analysis to extract value from data. Python has become the dominant programming language in data science due to its simplicity and powerful ecosystem of libraries.

### The Data Science Ecosystem
Key libraries that form the foundation of Python's data science stack:
- **NumPy**: Numerical computing with multi-dimensional arrays
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Scientific computing
- **StatsModels**: Statistical modeling
- **TensorFlow/PyTorch**: Deep learning frameworks

### Setting Up Your Environment
The Anaconda distribution is the most convenient way to set up a data science environment:

```bash
# Install Anaconda (or miniconda for a minimal installation)
# Create a new environment
conda create -n datasci python=3.10

# Activate environment
conda activate datasci

# Install core packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch Jupyter Notebook
jupyter notebook
```

## NumPy: Numerical Computing

### NumPy Arrays
NumPy's core is the `ndarray` (n-dimensional array) - a fast, flexible container for large datasets.

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 4))  # 3x4 array of zeros
arr3 = np.ones((2, 3, 4))  # 3D array of ones
arr4 = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
arr5 = np.linspace(0, 1, 5)  # 5 evenly spaced points between 0 and 1

# Array attributes
print(arr1.shape)  # (5,)
print(arr2.ndim)   # 2 (dimensions)
print(arr3.size)   # 24 (total elements)
print(arr1.dtype)  # int64 (data type)
```

### Array Operations
NumPy operations are vectorized, making them much faster than Python loops.

```python
# Element-wise operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)  # [5, 7, 9]
print(a * b)  # [4, 10, 18]
print(a ** 2)  # [1, 4, 9]

# Broadcasting (operating with different shapes)
c = np.array([[1, 2, 3], [4, 5, 6]])
print(c + 1)  # Add 1 to each element

# Aggregation functions
print(np.sum(c))  # 21
print(np.mean(c))  # 3.5
print(np.max(c, axis=0))  # [4, 5, 6] (max along columns)
print(np.min(c, axis=1))  # [1, 4] (min along rows)
```

### Array Indexing and Slicing
Access elements using indices and slices.

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Indexing
print(arr[0, 0])  # 1
print(arr[2, 3])  # 12

# Slicing [start:stop:step]
print(arr[:, 0])  # First column: [1, 5, 9]
print(arr[1, :])  # Second row: [5, 6, 7, 8]
print(arr[0:2, 1:3])  # Sub-array: [[2, 3], [6, 7]]

# Boolean indexing
mask = arr > 6
print(arr[mask])  # [7, 8, 9, 10, 11, 12]

# Fancy indexing
indices = np.array([0, 2])
print(arr[indices, :])  # Select rows 0 and 2
```

### Array Manipulation
Reshape, combine, and split arrays.

```python
# Reshape
arr = np.arange(12)
print(arr.reshape(3, 4))  # Reshape to 3x4
print(arr.reshape(2, -1))  # 2 rows, columns automatically calculated

# Transpose
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d.T)  # Transpose

# Concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.concatenate([a, b], axis=0))  # Vertical
print(np.concatenate([a, b], axis=1))  # Horizontal

# Split
arr = np.arange(8).reshape(2, 4)
print(np.split(arr, 2, axis=1))  # Split into 2 along columns
```

### Random Numbers
NumPy provides comprehensive random number capabilities.

```python
# Set seed for reproducibility
np.random.seed(42)

# Random numbers
print(np.random.random(5))  # 5 random floats between 0 and 1
print(np.random.randint(1, 10, 5))  # 5 random integers from 1 to 9
print(np.random.normal(0, 1, 5))  # 5 samples from standard normal distribution

# Random sampling
data = np.arange(100)
print(np.random.choice(data, 5, replace=False))  # 5 unique samples
```

## Pandas: Data Manipulation

### Series and DataFrames
The core data structures in Pandas.

```python
import pandas as pd

# Series (1D labeled array)
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
print(s['a'])  # 1
print(s[['a', 'c', 'e']])  # Select multiple items

# DataFrame (2D labeled data structure)
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 34, 29, 42],
    'City': ['New York', 'Paris', 'Berlin', 'London']
}
df = pd.DataFrame(data)
print(df.head())  # View first 5 rows
```

### Reading and Writing Data
Pandas supports many data formats.

```python
# CSV
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx', index=False)

# JSON
df = pd.read_json('data.json')
df.to_json('output.json')

# SQL
import sqlite3
connection = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', connection)
df.to_sql('new_table', connection, if_exists='replace')
```

### DataFrame Operations
Basic operations on DataFrames.

```python
# Basic information
print(df.info())  # Column info and data types
print(df.describe())  # Statistical summary
print(df.shape)  # (rows, columns)

# Accessing data
print(df.columns)  # Column names
print(df['Name'])  # Single column
print(df[['Name', 'Age']])  # Multiple columns
print(df.iloc[0])  # First row by position
print(df.iloc[0, 1])  # First row, second column
print(df.loc[0, 'Name'])  # First row, 'Name' column

# Adding/removing columns
df['Country'] = ['USA', 'France', 'Germany', 'UK']
df = df.drop('Country', axis=1)  # Remove column

# Basic calculations
print(df['Age'].mean())
print(df['Age'].max())
print(df['Age'].min())
```

### Filtering and Sorting
Select and organize data.

```python
# Filtering with conditions
print(df[df['Age'] > 30])  # People over 30
print(df[(df['Age'] > 30) & (df['City'] == 'London')])  # Multiple conditions

# Sorting
print(df.sort_values('Age'))  # Sort by age ascending
print(df.sort_values('Age', ascending=False))  # Sort by age descending
print(df.sort_values(['City', 'Age']))  # Sort by multiple columns
```

### Handling Missing Data
Methods to identify and handle missing values.

```python
# Check for missing values
print(df.isnull().sum())  # Count missing values per column

# Drop missing values
df_cleaned = df.dropna()  # Drop rows with any missing values
df_cleaned = df.dropna(subset=['Age'])  # Drop rows with missing Age

# Fill missing values
df_filled = df.fillna(0)  # Replace missing with 0
df_filled = df.fillna({'Age': df['Age'].mean(), 'City': 'Unknown'})  # Different values for different columns
```

### Grouping and Aggregating
Powerful data summarization.

```python
# Group by City and calculate statistics
city_stats = df.groupby('City').agg({
    'Age': ['mean', 'min', 'max', 'count']
})

# Multiple aggregations
result = df.groupby(['City']).agg({
    'Age': ['mean', 'std'],
    'Name': 'count'
}).reset_index()

# Custom aggregation
def age_range(x):
    return x.max() - x.min()

by_city = df.groupby('City').agg({
    'Age': [np.mean, age_range]
})
```

### Reshaping Data
Transform between wide and long formats.

```python
# Example wide data (one row per subject)
wide_data = pd.DataFrame({
    'subject': ['A', 'B', 'C'],
    'math_score': [90, 80, 70],
    'science_score': [95, 85, 75],
    'history_score': [88, 82, 72]
})

# Convert to long format
long_data = pd.melt(
    wide_data,
    id_vars=['subject'],
    value_vars=['math_score', 'science_score', 'history_score'],
    var_name='subject',
    value_name='score'
)

# Convert back to wide format
wide_again = long_data.pivot(
    index='subject',
    columns='subject',
    values='score'
)
```

### Merge and Join
Combine multiple DataFrames.

```python
# Sample data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [3, 2, 1, 3, 5],
    'amount': [50, 25, 10, 30, 15]
})

# Inner join (only matching rows)
inner_join = pd.merge(
    customers,
    orders,
    on='customer_id',
    how='inner'
)

# Left join (all rows from left, matching from right)
left_join = pd.merge(
    customers,
    orders,
    on='customer_id',
    how='left'
)

# Right join (all rows from right, matching from left)
right_join = pd.merge(
    customers,
    orders,
    on='customer_id',
    how='right'
)

# Outer join (all rows from both)
outer_join = pd.merge(
    customers,
    orders,
    on='customer_id',
    how='outer'
)
```

## Data Visualization

### Matplotlib Basics
The foundation for most Python visualization.

```python
import matplotlib.pyplot as plt

# Basic line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave', fontsize=18)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.grid(True)
plt.savefig('sine_wave.png', dpi=300)
plt.show()

# Multiple plots in one figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st plot
plt.plot(x, np.sin(x))
plt.title('Sine')

plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd plot
plt.plot(x, np.cos(x))
plt.title('Cosine')

plt.subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd plot
plt.plot(x, np.sin(2*x))
plt.title('Sine 2x')

plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th plot
plt.plot(x, np.cos(2*x))
plt.title('Cosine 2x')

plt.tight_layout()
plt.show()
```

### Pandas Visualization
Pandas has built-in plotting methods based on Matplotlib.

```python
# Create sample DataFrame
data = {
    'Year': range(2010, 2021),
    'Sales': [10, 15, 13, 17, 20, 22, 25, 30, 32, 35, 37],
    'Expenses': [8, 11, 11, 13, 15, 17, 20, 22, 25, 27, 30]
}
df = pd.DataFrame(data)

# Line plot
df.plot(x='Year', y=['Sales', 'Expenses'], figsize=(10, 6))
plt.title('Sales and Expenses Over Time')
plt.ylabel('Amount (millions)')
plt.grid(True)
plt.show()

# Bar plot
df.plot.bar(x='Year', y=['Sales', 'Expenses'], figsize=(12, 6))
plt.title('Sales and Expenses by Year')
plt.show()

# Scatter plot
df.plot.scatter(x='Sales', y='Expenses', s=df['Year']-2000, alpha=0.7)
plt.title('Sales vs Expenses')
plt.show()

# Histogram
df['Sales'].plot.hist(bins=5, alpha=0.5)
plt.title('Distribution of Sales')
plt.show()

# Pie chart
sales_2020 = pd.Series([15, 25, 40, 20], index=['Q1', 'Q2', 'Q3', 'Q4'])
sales_2020.plot.pie(figsize=(8, 8), autopct='%1.1f%%')
plt.title('2020 Quarterly Sales')
plt.show()

### Seaborn: Statistical Visualization
Seaborn is built on Matplotlib and provides a higher-level interface.

```python
import seaborn as sns

# Set the aesthetics for all plots
sns.set_style("whitegrid")
sns.set_context("talk")

# Load a sample dataset
tips = sns.load_dataset('tips')
print(tips.head())

# Distribution plots
plt.figure(figsize=(10, 6))
sns.histplot(tips['total_bill'], kde=True)
plt.title('Distribution of Total Bill')
plt.show()

# Categorical plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Total Bill Distribution by Day')
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='total_bill', y='tip', data=tips)
plt.title('Tip vs Total Bill')
plt.show()

# Visualize relationships between multiple variables
sns.pairplot(tips, hue='time', height=2.5)
plt.suptitle('Pairwise Relationships in Tips Data', y=1.02)
plt.show()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Tips Data')
plt.show()

# Categorical plot with multiple variables
sns.catplot(x='day', y='total_bill', hue='sex', kind='violin', data=tips, height=6, aspect=1.5)
plt.title('Total Bill by Day and Gender')
plt.show()
```

### Interactive Visualization with Plotly
Plotly creates interactive, publication-quality graphs.

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data
df = px.data.gapminder().query("continent=='Europe' and year==2007")

# Basic interactive scatter plot
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="country",
                hover_name="country", log_x=True, size_max=60)
fig.update_layout(title='GDP per Capita vs Life Expectancy in Europe (2007)')
fig.show()

# Line chart with multiple lines
gapminder = px.data.gapminder()
fig = px.line(gapminder, x="year", y="lifeExp", color="continent", line_group="country", 
              hover_name="country", line_shape="spline", render_mode="svg")
fig.update_layout(title='Life Expectancy Over Time by Continent')
fig.show()

# Interactive map
fig = px.choropleth(gapminder, locations="iso_alpha", color="lifeExp", 
                    hover_name="country", animation_frame="year", 
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(title='Global Life Expectancy Evolution')
fig.show()
```

## Exploratory Data Analysis

### The EDA Process
Exploratory Data Analysis (EDA) is a critical first step in data analysis to discover patterns, identify anomalies, test hypotheses, and validate assumptions.

```python
# Load a dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example with Titanic dataset
titanic = sns.load_dataset('titanic')

# 1. Understanding the data structure
print(titanic.info())  # Data types and missing values
print(titanic.head())  # First few rows
print(titanic.describe())  # Statistical summary

# 2. Checking for missing values
missing_values = titanic.isnull().sum()
print(missing_values[missing_values > 0])  # Columns with missing values

plt.figure(figsize=(10, 6))
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Titanic Dataset')
plt.show()

# 3. Analyzing distributions of numerical features
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(titanic['age'].dropna(), kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(titanic['fare'].dropna(), kde=True)
plt.title('Fare Distribution')

plt.subplot(2, 2, 3)
sns.countplot(x='class', data=titanic)
plt.title('Passenger Class')

plt.subplot(2, 2, 4)
sns.countplot(x='survived', data=titanic)
plt.title('Survival')

plt.tight_layout()
plt.show()

# 4. Exploring relationships between variables
sns.catplot(x='class', y='survived', hue='sex', kind='bar', data=titanic)
plt.title('Survival Rate by Class and Gender')
plt.show()

# 5. Correlation analysis
plt.figure(figsize=(10, 8))
numerical_features = titanic.select_dtypes(include=[np.number])
correlation = numerical_features.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Between Numerical Features')
plt.show()

# 6. Creating derived features
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1
titanic['is_alone'] = (titanic['family_size'] == 1).astype(int)

plt.figure(figsize=(10, 5))
sns.barplot(x='family_size', y='survived', data=titanic)
plt.title('Survival Rate by Family Size')
plt.show()
```

### Data Cleaning Techniques
Methods for handling missing values, outliers, and inconsistent data.

```python
# 1. Handling missing values
# Option 1: Drop missing values
df_dropna = titanic.dropna(subset=['age'])

# Option 2: Fill with a central tendency
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Option 3: Fill based on groups
titanic['age'].fillna(titanic.groupby(['sex', 'class'])['age'].transform('median'), inplace=True)

# 2. Detecting and handling outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='fare', data=titanic)
plt.title('Boxplot to Identify Outliers in Fare')
plt.show()

# Z-score method
from scipy import stats
z_scores = stats.zscore(titanic['fare'].dropna())
outliers = (abs(z_scores) > 3)
print(f"Number of outliers: {outliers.sum()}")

# IQR method
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ((titanic['fare'] < lower_bound) | (titanic['fare'] > upper_bound))
print(f"Number of outliers with IQR method: {outliers.sum()}")

# 3. Handling duplicates
duplicate_rows = titanic.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")
titanic_no_dupes = titanic.drop_duplicates()

# 4. Data type conversion
# Convert categorical variables to numeric
titanic['sex_encoded'] = titanic['sex'].map({'male': 0, 'female': 1})

# Create dummy variables from categorical columns
embarked_dummies = pd.get_dummies(titanic['embarked'], prefix='embarked')
titanic = pd.concat([titanic, embarked_dummies], axis=1)
```

## Statistical Analysis

### Descriptive Statistics
Summarizing and understanding the main characteristics of a dataset.

```python
import pandas as pd
import numpy as np
from scipy import stats

# Central Tendency
mean = titanic['age'].mean()
median = titanic['age'].median()
mode = titanic['age'].mode()[0]
print(f"Mean: {mean:.2f}, Median: {median:.2f}, Mode: {mode:.2f}")

# Dispersion
variance = titanic['age'].var()
std_dev = titanic['age'].std()
range_val = titanic['age'].max() - titanic['age'].min()
IQR = titanic['age'].quantile(0.75) - titanic['age'].quantile(0.25)
print(f"Variance: {variance:.2f}, Std Dev: {std_dev:.2f}, Range: {range_val:.2f}, IQR: {IQR:.2f}")

# Shape
skewness = titanic['age'].skew()
kurtosis = titanic['age'].kurtosis()
print(f"Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")

# Comprehensive summary
from scipy.stats import describe
desc = describe(titanic['age'].dropna())
print(f"N: {desc.nobs}, Min: {desc.minmax[0]:.2f}, Max: {desc.minmax[1]:.2f}")
print(f"Mean: {desc.mean:.2f}, Variance: {desc.variance:.2f}")
print(f"Skewness: {desc.skewness:.2f}, Kurtosis: {desc.kurtosis:.2f}")
```

### Hypothesis Testing
Testing statistical hypotheses to make inferences.

```python
import scipy.stats as stats

# Example: Is there a significant difference in age between survivors and non-survivors?
survivors = titanic[titanic['survived'] == 1]['age'].dropna()
non_survivors = titanic[titanic['survived'] == 0]['age'].dropna()

# 1. T-test (parametric test for comparing means of two groups)
t_stat, p_value = stats.ttest_ind(survivors, non_survivors, equal_var=False)
print(f"T-test: t={t_stat:.2f}, p={p_value:.4f}")
print(f"Significant difference at α=0.05: {p_value < 0.05}")

# 2. Mann-Whitney U test (non-parametric alternative to t-test)
u_stat, p_value = stats.mannwhitneyu(survivors, non_survivors)
print(f"Mann-Whitney U test: U={u_stat:.2f}, p={p_value:.4f}")
print(f"Significant difference at α=0.05: {p_value < 0.05}")

# 3. Chi-squared test (for categorical variables)
# Is survival independent of gender?
contingency_table = pd.crosstab(titanic['survived'], titanic['sex'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test: χ²={chi2:.2f}, p={p:.4f}, dof={dof}")
print(f"Significant relationship at α=0.05: {p < 0.05}")

# 4. ANOVA (comparing means across multiple groups)
# Is there a significant difference in age between passenger classes?
classes = [
    titanic[titanic['class'] == 'First']['age'].dropna(),
    titanic[titanic['class'] == 'Second']['age'].dropna(),
    titanic[titanic['class'] == 'Third']['age'].dropna()
]
f_stat, p_value = stats.f_oneway(*classes)
print(f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
print(f"Significant difference at α=0.05: {p_value < 0.05}")

# 5. Correlation test
corr, p_value = stats.pearsonr(titanic['age'].dropna(), titanic['fare'].dropna())
print(f"Pearson correlation: r={corr:.2f}, p={p_value:.4f}")
print(f"Significant correlation at α=0.05: {p_value < 0.05}")
```

### Confidence Intervals
Estimating parameters with uncertainty.

```python
import numpy as np
from scipy import stats

# Confidence interval for mean age
mean = titanic['age'].mean()
std_err = stats.sem(titanic['age'].dropna())
ci_95 = stats.t.interval(0.95, len(titanic['age'].dropna())-1, loc=mean, scale=std_err)
print(f"95% CI for mean age: {ci_95[0]:.2f} to {ci_95[1]:.2f}")

# Confidence interval for proportions (e.g., survival rate)
survival_rate = titanic['survived'].mean()
n = len(titanic)
ci_95_prop = stats.norm.interval(0.95, loc=survival_rate, scale=np.sqrt(survival_rate*(1-survival_rate)/n))
print(f"95% CI for survival rate: {ci_95_prop[0]:.2f} to {ci_95_prop[1]:.2f}")

# Bootstrap confidence intervals (non-parametric)
from sklearn.utils import resample

bootstrap_means = []
for _ in range(1000):
    sample = resample(titanic['age'].dropna())
    bootstrap_means.append(sample.mean())

bootstrap_ci = np.percentile(bootstrap_means, [2.5, 97.5])
print(f"95% Bootstrap CI for mean