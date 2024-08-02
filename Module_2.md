# Transforming Linear Algebra to Computational Language
## Introduction

In the first module, we established a solid foundation in matrix algebra by exploring pseudocode and implementing fundamental matrix operations using Python. We practiced key concepts such as matrix addition, subtraction, multiplication, and determinants through practical examples in image processing, leveraging the `SymPy` library for symbolic computation.

As we begin the second module, **“Transforming Linear Algebra to Computational Language,”** our focus will shift towards applying these concepts with greater depth and actionable insight. This module is designed to bridge the theoretical knowledge from matrix algebra with practical computational applications. You will learn to interpret and utilize matrix operations, solve systems of equations, and analyze the rank of matrices within a variety of real-world contexts.

A new concept we will introduce is the **Rank-Nullity Theorem**, which provides a fundamental relationship between the rank of a matrix and the dimensions of its null space. This theorem is crucial for understanding the solution spaces of linear systems and the properties of linear transformations. By applying this theorem, you will be able to gain deeper insights into the structure of solutions and the behavior of matrix transformations.

This transition will not only reinforce your understanding of linear algebra but also enhance your ability to apply these concepts effectively in computational settings. Through engaging examples and practical exercises, you will gain valuable experience in transforming abstract mathematical principles into tangible solutions, setting a strong groundwork for advanced computational techniques.

## Relearning of Terms and Operations in Linear Algebra
In this section, we will revisit fundamental matrix operations such as addition, subtraction, scaling, and more through practical examples. Our goal is to transform theoretical linear algebra into modern computational applications. We will demonstrate these concepts using Python, focusing on practical and industrial applications.

### Matrix Addition and Subtraction in Data Analysis

Matrix addition and subtraction are fundamental operations that help in combining datasets and analyzing differences.

**Simple Example: Combining Quarterly Sales Data**

We begin with quarterly sales data from different regions and combine them to get the total sales.

**Tabular Data:**

|Region	|Q1	|Q2	|Q3	|Q4|
|--- |--- |---|---|---|
|A	|2500	|2800	|3100	|2900|
|B	|1500	|1600	|1700	|1800|

**From Scratch Python Implementation:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Quarterly sales data
sales_region_a = np.array([2500, 2800, 3100, 2900])
sales_region_b = np.array([1500, 1600, 1700, 1800])

# Combine sales data
total_sales = sales_region_a + sales_region_b

# Visualization
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
plt.bar(quarters, total_sales, color='skyblue')
plt.xlabel('Quarter')
plt.ylabel('Total Sales')
plt.title('Combined Quarterly Sales Data for Regions A and B')
plt.show()
```

Using `pandas` to handle tabular data:

```python
import pandas as pd
import matplotlib.pyplot as plt

# DataFrames for quarterly sales data
df_a = pd.DataFrame({'Q1': [2500], 'Q2': [2800], 'Q3': [3100], 'Q4': [2900]}, index=['Region A'])
df_b = pd.DataFrame({'Q1': [1500], 'Q2': [1600], 'Q3': [1700], 'Q4': [1800]}, index=['Region B'])

# Combine data
df_total = df_a.add(df_b)

# Visualization
df_total.T.plot(kind='bar', color='skyblue')
plt.xlabel('Quarter')
plt.ylabel('Total Sales')
plt.title('Combined Quarterly Sales Data for Regions A and B')
plt.show()
```

We can extend the this in to more advanced examples. Irrespective to the size of the data, for representation aggregation tasks matrix models are best options and are used in industry as a standard. Let us consider an advanced example to analyse difference in stock prices. For this example we are using a simulated data. The python code for this simulation process is shown below:

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulated observed and predicted stock prices
observed_prices = np.random.uniform(100, 200, size=(100, 5))
predicted_prices = np.random.uniform(95, 210, size=(100, 5))

# Calculate the difference matrix
price_differences = observed_prices - predicted_prices

# Visualization
plt.imshow(price_differences, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Stock Price Differences')
plt.xlabel('Stock Index')
plt.ylabel('Day Index')
plt.show()
```

Another important matrix operation relevant to data analytics and Machine Learning application is scaling. This is considered as a statistical tool to make various features (attributes) in to same scale so as to avoid unnecessary misleading impact in data analysis and its intepretation. In Machine Learning context, this pre-processing stage is inevitable so as to make the model relevant and usable.

**Simple Example: Normalizing Employee Performance Data**

**Tabular Data:**

|Employee	|Metric A	|Metric B|
|--- |--- |---|
|X	|80	|700|
|Y	|90	|800|
|Z	|100	|900|
|A	|110	|1000|
|B	|120	|1100|

Using simple python code we can simulate the model for min-max scaling.

```python
import numpy as np
import matplotlib.pyplot as plt

# Employee performance data with varying scales
data = np.array([[80, 700], [90, 800], [100, 900], [110, 1000], [120, 1100]])

# Manual scaling
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
scaled_data = (data - min_vals) / (max_vals - min_vals)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='viridis')
plt.title('Original Data')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(scaled_data, cmap='viridis')
plt.title('Scaled Data')
plt.colorbar()

plt.show()
```

This method will confine the feature values (attributes) into the range $[0,1]$. So in effect all the features are scaled proportionally to the data spectrum.

Similarly we can use the standard scaling (transformation to normal distribution) using the transformation $\dfrac{x-\bar{x}}{S.D.}$. The python code for this operation is given below:

```python
# Standard scaling from scratch
def standard_scaling(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    scaled_data = (data - mean) / std
    return scaled_data

# Apply standard scaling
scaled_data_scratch = standard_scaling(data)

print("Standard Scaled Data (from scratch):\n", scaled_data_scratch)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='viridis')
plt.title('Original Data')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(scaled_data_scratch, cmap='viridis')
plt.title('Scaled Data')
plt.colorbar()

plt.show()
```

Here we will use one more type of visualization to demonstrate the distribution of data.

```python
import seaborn as sns
# Create plots
plt.figure(figsize=(14, 7))

# Plot for original data
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True, bins=10, palette="viridis")
plt.title('Original Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Plot for standard scaled data
plt.subplot(1, 2, 2)
sns.histplot(scaled_data_scratch, kde=True, bins=10, palette="viridis")
plt.title('Standard Scaled Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```
A scatter plot showing the impact of scaling is shown below.

```python
# Plot original and scaled data
plt.figure(figsize=(14, 7))

# Original Data
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.title('Original Data')
plt.xlabel('Metric A')
plt.ylabel('Metric B')

# Standard Scaled Data
plt.subplot(1, 3, 2)
plt.scatter(scaled_data_scratch[:, 0], scaled_data_scratch[:, 1], color='green')
plt.title('Standard Scaled Data')
plt.xlabel('Metric A (Standard Scaled)')
plt.ylabel('Metric B (Standard Scaled)')

# Min-Max Scaled Data
plt.subplot(1, 3, 3)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], color='red')
plt.title('Min-Max Scaled Data')
plt.xlabel('Metric A (Min-Max Scaled)')
plt.ylabel('Metric B (Min-Max Scaled)')

plt.tight_layout()
plt.show()
```

We can use the `scikit-learn` library for do the same thing in a very simple handy approach. The python code for this job is shown below.

```python
from sklearn.preprocessing import MinMaxScaler

# Min-max scaling using sklearn
scaler = MinMaxScaler()
min_max_scaled_data_sklearn = scaler.fit_transform(data)

print("Min-Max Scaled Data (using sklearn):\n", min_max_scaled_data_sklearn)
```

```python
from sklearn.preprocessing import StandardScaler

# Standard scaling using sklearn
scaler = StandardScaler()
scaled_data_sklearn = scaler.fit_transform(data)

print("Standard Scaled Data (using sklearn):\n", scaled_data_sklearn)
```

A scatter plot showing the impact on scaling is shown bellow.

```python
# Plot original and scaled data
plt.figure(figsize=(14, 7))

# Original Data
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.title('Original Data')
plt.xlabel('Metric A')
plt.ylabel('Metric B')

# Standard Scaled Data
plt.subplot(1, 3, 2)
plt.scatter(scaled_data_sklearn[:, 0], scaled_data_sklearn[:, 1], color='green')
plt.title('Standard Scaled Data')
plt.xlabel('Metric A (Standard Scaled)')
plt.ylabel('Metric B (Standard Scaled)')

# Min-Max Scaled Data
plt.subplot(1, 3, 3)
plt.scatter(min_max_scaled_data_sklearn[:, 0], min_max_scaled_data_sklearn[:, 1], color='red')
plt.title('Min-Max Scaled Data')
plt.xlabel('Metric A (Min-Max Scaled)')
plt.ylabel('Metric B (Min-Max Scaled)')

plt.tight_layout()
plt.show()
```

### More on Matrix Product and its Applications
In the first module of our course, we introduced matrix products as scalar projections, focusing on how matrices interact through basic operations. In this section, we will expand on this by exploring different types of matrix products that have practical importance in various fields. One such product is the Hadamard product, which is particularly useful in applications ranging from image processing to neural networks and statistical analysis. We will cover the definition, properties, and examples of the *Hadamard product*, and then delve into practical applications with simulated data.

#### Hadamard Product
The Hadamard product (or element-wise product) of two matrices is a binary operation that combines two matrices of the same dimensions to produce another matrix of the same dimensions, where each element is the product of corresponding elements in the original matrices.

> [!NOTE]
> ## Definition (Hadamard Product):
> For two matrices $A$ and $B$ of the same dimension $m\times n$, the Hadamard product $A\circ B$ is defined as: $`(A\circ B)_{ij} = A_{ij}\cdot B_{ij}`$ where $\cdot$ denotes element-wise multiplication.

>[!TIP]
> ## Properties of Hadamard Product
> 1. Commutativity: $A \circ B = B \circ A$
> 2. Associativity: $(A \circ B) \circ C = A \circ (B \circ C)$
> 3. Distributivity: $A \circ (B + C) = (A \circ B) + (A \circ C)$

Some simple examples to demonstrate the Hadamard product is given below.

Example 1: Basic Hadamard Product

Given matrices:

$$A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}, \quad
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}$$

The Hadamard product $A\circ B$ is:

$$A \circ B = \begin{pmatrix}
1 \cdot 5 & 2 \cdot 6 \\
3 \cdot 7 & 4 \cdot 8
\end{pmatrix} = \begin{pmatrix}
5 & 12 \\
21 & 32
\end{pmatrix}$$

Example 2: Hadamard Product with Larger Matrices

Given matrices:

$$A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}, \quad
B = \begin{pmatrix}
9 & 8 & 7 \\
6 & 5 & 4 \\
3 & 2 & 1
\end{pmatrix}$$

The Hadamard product $A\circ B$ is:

$$A \circ B = \begin{pmatrix}
1 \cdot 9 & 2 \cdot 8 & 3 \cdot 7 \\
4 \cdot 6 & 5 \cdot 5 & 6 \cdot 4 \\
7 \cdot 3 & 8 \cdot 2 & 9 \cdot 1
\end{pmatrix} = \begin{pmatrix}
9 & 16 & 21 \\
24 & 25 & 24 \\
21 & 16 & 9
\end{pmatrix}$$

In the following code chunks the computational process of Hadamard product is implemented in `Python`. Here both the from the scratch and use of external module versions are included.

**1. Compute Hadamard Product from Scratch (without Libraries)**

Here’s how you can compute the Hadamard product manually:

```python
# Define matrices A and B
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8, 9], [10, 11, 12]]

# Function to compute Hadamard product
def hadamard_product(A, B):
    # Get the number of rows and columns
    num_rows = len(A)
    num_cols = len(A[0])
    
    # Initialize the result matrix
    result = [[0]*num_cols for _ in range(num_rows)]
    
    # Compute the Hadamard product
    for i in range(num_rows):
        for j in range(num_cols):
            result[i][j] = A[i][j] * B[i][j]
    
    return result

# Compute Hadamard product
hadamard_product_result = hadamard_product(A, B)

# Display result
print("Hadamard Product (From Scratch):")
for row in hadamard_product_result:
    print(row)
```

**2. Compute Hadamard Product Using** `SymPy`

Here’s how to compute the Hadamard product using `SymPy`:

```python
import sympy as sp

# Define matrices A and B
A = sp.Matrix([[1, 2, 3], [4, 5, 6]])
B = sp.Matrix([[7, 8, 9], [10, 11, 12]])

# Compute Hadamard product using SymPy
Hadamard_product_sympy = A.multiply_elementwise(B)

# Display result
print("Hadamard Product (Using SymPy):")
print(Hadamard_product_sympy)
```

**Practical Applications**

*Application 1: Image Masking*

The Hadamard product can be used for image masking. Here’s how you can apply a mask to an image and visualize it:

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated large image (2D array) using NumPy
image = np.random.rand(100, 100)

# Simulated mask (binary matrix) using NumPy
mask = np.random.randint(0, 2, size=(100, 100))

# Compute Hadamard product
masked_image = image * mask

# Plot original image and masked image
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(masked_image, cmap='gray')
ax[1].set_title('Masked Image')
plt.show()
```

*Application 2: Element-wise Scaling in Neural Networks*

The Hadamard product can be used for dropout1 in neural networks. A simple simulated example is given below.

```python
# Simulated large activations (2D array) using NumPy
activations = np.random.rand(100, 100)

# Simulated dropout mask (binary matrix) using NumPy
dropout_mask = np.random.randint(0, 2, size=(100, 100))

# Apply dropout
dropped_activations = activations * dropout_mask

# Display results
print("Original Activations:")
print(activations)
print("\nDropout Mask:")
print(dropout_mask)
print("\nDropped Activations:")
print(dropped_activations)
```

*Application 3: Statistical Data Analysis*

In statistics, the Hadamard product can be applied to scale covariance matrices. Here’s how we can compute the covariance matrix using matrix operations and apply scaling.

```python
import sympy as sp
import numpy as np

# Simulated large dataset (2D array) using NumPy
data = np.random.rand(100, 10)

# Compute the mean of each column
mean = np.mean(data, axis=0)

# Center the data
centered_data = data - mean

# Compute the covariance matrix using matrix product operation
cov_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
cov_matrix_sympy = sp.Matrix(cov_matrix)

# Simulated scaling factors (2D array) using SymPy Matrix
scaling_factors = sp.Matrix(np.random.rand(10, 10))

# Compute Hadamard product
scaled_cov_matrix = cov_matrix_sympy.multiply(scaling_factors)

# Display results
print("Covariance Matrix:")
print(cov_matrix_sympy)
print("\nScaling Factors:")
print(scaling_factors)
print("\nScaled Covariance Matrix:")
print(scaled_cov_matrix)
```

#### Inner Product of Matrices

The inner product of two matrices is a generalized extension of the dot product, where each matrix is treated as a vector in a high-dimensional space. For two matrices $A$ and $B$ of the same dimension $m\times n$, the inner product is defined as the sum of the element-wise products of the matrices.

>[!NOTE]
>## Definition (Inner product)

For two matrices $A$ and $B$ of dimension $m\times n$, the inner product $\langle A, B \rangle$ is given by:

$$\langle A, B \rangle = \sum_{i=1}^{m} \sum_{j=1}^{n} A_{ij} \cdot B_{ij}$$

where $\cdot$ denotes element-wise multiplication.

>[!NOTE]
>## Properties

1. Commutativity: $\langle A, B \rangle = \langle B, A \rangle$

2. Linearity: $\langle A + C, B \rangle = \langle A, B \rangle + \langle C, B \rangle$

3. Positive Definiteness: $\langle A, A \rangle \geq 0$
with equality if and only if $A$ is a zero matrix.

Some simple examples showing the mathematical process of calculating the inner product is given bellow.

**Example 1: Basic Inner Product**

Given matrices:

$$A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}, \quad
B = \begin{pmatrix}
5 & 6 \\
7 & 8
\end{pmatrix}$$

The inner product $\langle A, B \rangle$ is:

$$\langle A, B \rangle = 1 \cdot 5 + 2 \cdot 6 + 3 \cdot 7 + 4 \cdot 8 = 5 + 12 + 21 + 32 = 70$$

**Example 2: Inner Product with Larger Matrices**

Given matrices:

$$A = \begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}, \quad
B = \begin{pmatrix}
9 & 8 & 7 \\
6 & 5 & 4 \\
3 & 2 & 1
\end{pmatrix}$$

The inner product $\langle A, B \rangle$ is calculated as:

$$\begin{align*}
\langle A, B \rangle &= 1 \cdot 9 + 2 \cdot 8 + 3 \cdot 7 + 4 \cdot 6 + 5 \cdot 5 + 6 \cdot 4 + 7 \cdot 3 + 8 \cdot 2 + 9 \cdot 1\\
&= 9 + 16 + 21 + 24 + 25 + 24 + 21 + 16 + 9\\
&= 175
\end{align*}$$

Now let’s look into the computational part of inner product.

*1.Compute Inner Product from Scratch (without Libraries)*
Here’s how you can compute the inner product from the scratch:

```python
# Define matrices A and B
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8, 9], [10, 11, 12]]

# Function to compute inner product
def inner_product(A, B):
    # Get the number of rows and columns
    num_rows = len(A)
    num_cols = len(A[0])
    
    # Initialize the result
    result = 0
    
    # Compute the inner product
    for i in range(num_rows):
        for j in range(num_cols):
            result += A[i][j] * B[i][j]
    
    return result

# Compute inner product
inner_product_result = inner_product(A, B)

# Display result
print("Inner Product (From Scratch):")
print(inner_product_result)
```

*2.Compute Inner Product Using NumPy*
Here’s how to compute the inner product using `Numpy`:

```python
import numpy as np
# Define matrices A and B
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])
# calculating innerproduct
inner_product = (A*B).sum() # calculate element-wise product, then column sum

print("Inner Product (Using numpy):")
print(inner_product)
```

The same operation can be done using `SymPy` functions as follows.

```python
import sympy as sp
import numpy as np  
# Define matrices A and B
A = sp.Matrix([[1, 2, 3], [4, 5, 6]])
B = sp.Matrix([[7, 8, 9], [10, 11, 12]])

# Compute element-wise product
elementwise_product = A.multiply_elementwise(B)

# Calculate sum of each column
inner_product_sympy = np.sum(elementwise_product)

# Display result
print("Inner Product (Using SymPy):")
print(inner_product_sympy)
```

A vector dot product (in Physics) can be calculated using `SymPy .dot()` function as shown below.

Let  $`A=\begin{pmatrix}1&2&3\end{pmatrix}`$ and  $`B=\begin{pmatrix}4&5&6\end{pmatrix}`$, then the dot product,$`A\cdot B`$ is computed as:

```python
import sympy as sp
A=sp.Matrix([1,2,3])
B=sp.Matrix([4,5,6])
display(A.dot(B)) # calculate fot product of A and B
```

>[!CAUTION]
>In SymPy , `sp.Matrix([1,2,3])` create a column vector. But `np.array([1,2,3])` creates a row vector. So be careful while applying matrix/ dot product operations on these objects.


The same dot product using `numpy` object can be done as follows:

```python
import numpy as np
A=np.array([1,2,3])
B=np.array([4,5,6])
display(A.dot(B.T))# dot() stands for dot product B.T represents the transpose of B
```

**Practical Applications**

*Application 1: Signal Processing*

In signal processing, the inner product can be used to measure the similarity between two signals. Here the most popular measure of similarity is the `cosine` similarity. This measure is defined as:

 $$\cos \theta=\dfrac{A\cdot B}{||A|| ||B||}$$

Now consider two digital signals are given. It’s cosine similarity measure can be calculated with a simulated data as shown below.

```python
import numpy as np

# Simulated large signals (1D array) using NumPy
signal1 = np.sin(np.random.rand(1000))
signal2 = np.cos(np.random.rand(1000))

# Compute inner product
inner_product_signal = np.dot(signal1, signal2)
#cosine_sim=np.dot(signal1,signal2)/(np.linalg.norm(signal1)*np.linalg.norm(signal2))
# Display result
cosine_sim=inner_product_signal/(np.sqrt(np.dot(signal1,signal1))*np.sqrt(np.dot(signal2,signal2)))
print("Inner Product (Using numpy):")
print(inner_product_signal)
print("Similarity of signals:")
print(cosine_sim)
```

*Application 2: Machine Learning - Feature Similarity*

In machine learning, the inner product is used to calculate the similarity between feature vectors.

```python
import numpy as np

# Simulated feature vectors (2D array) using NumPy
features1 = np.random.rand(100, 10)
features2 = np.random.rand(100, 10)

# Compute inner product for each feature vector
inner_products = np.einsum('ij,ij->i', features1, features2) # use Einstien's sum

# Display results
print("Inner Products of Feature Vectors:")
display(inner_products)
```

*Application 3: Covariance Matrix in Statistics*

The inner product can be used to compute covariance matrices for statistical data analysis. If $X$ is a given distribution and $x=X-\bar{X}$. Then the covariance of $X$ can be calculated as $cov(X)=\dfrac{1}{n-1}(x\cdot x^T)$. The python code a simulated data is shown below.

```python
import sympy as sp
import numpy as np

# Simulated large dataset (2D array) using NumPy
data = np.random.rand(100, 10)

# Compute the mean of each column
mean = np.mean(data, axis=0)

# Center the data
centered_data = data - mean

# Compute the covariance matrix using matrix product operation
cov_matrix = (centered_data.T @ centered_data) / (centered_data.shape[0] - 1)
cov_matrix_sympy = sp.Matrix(cov_matrix)

# Display results
print("Covariance Matrix:")
display(cov_matrix_sympy)
```
These examples demonstrate the use of inner product and dot product in various applications.
