# Python for Linear Algebra
## Pseudocode: the new language for algorithm design

Pseudocode is a way to describe algorithms in a structured but plain language. It helps in planning the logic without worrying about the syntax of a specific programming language. In this module, we’ll use Python-flavored pseudocode to describe various matrix operations.

> [!CAUTION]
> There are varities of approaches in writing pseudocode. Students can adopt any of the standard approach to write pseudocode.


### Matrix Sum
**Mathematical Procedure:**

To add two matrices $A$ and $B$, both matrices must have the same dimensions. The sum $C$ of two matrices $A$ and $B$ is calculated element-wise:

$$C[i][j]=A[i][j]+B[i][j]$$


**Example:**

Let $A$ and $B$ be two $2\times 2$ matrices:

$$A=\begin{bmatrix}
1 &2\\
3 &4
\end{bmatrix}
, B=\begin{bmatrix}
5 &6\\
7 &8
\end{bmatrix}$$

The sum $C$ is: 

$$C=A+B=\begin{bmatrix}
1+5 &2+6\\
3+7 &4+8
\end{bmatrix}=\begin{bmatrix}
6 &8\\
10 &12
\end{bmatrix}$$

**Pseudocode:**
```python
FUNCTION matrix_sum(A, B):
    Get the number of rows and columns in matrix A
    Create an empty matrix C with the same dimensions
    FOR each row i:
        FOR each column j:
            Set C[i][j] to the sum of A[i][j] and B[i][j]
    RETURN the matrix C
END FUNCTION
```
**Explanation:**

1. Determine the number of rows and columns in matrix $A$.
2. Create a new matrix $C$ with the same dimensions.
3. Loop through each element of the matrices and add corresponding elements.
4. Return the resulting matrix $C$.

### Matrix Difference
**Mathematical Procedure:**

To substact matrix $B$ from matrix $A$, both matrices must have the same dimensions. The difference $C$ of two matrices $A$ and $B$ is calculated element-wise:

$$C[i][j]=A[i][j]-B[i][j]$$

**Example:**

Let $A$ and $B$ be two $2\times 2$ matrices:

$$A=\begin{bmatrix}
9 &8\\
7 &6
\end{bmatrix}
, B=\begin{bmatrix}
1 &2\\
3 &4
\end{bmatrix}$$

The difference $C$ is: 

$$C=A-B=\begin{bmatrix}
9-1 &8-2\\
7-3 &6-4
\end{bmatrix}=\begin{bmatrix}
8 &6\\
4 &2
\end{bmatrix}$$

**Pseudocode:**

```python
FUNCTION matrix_difference(A, B):
    # Determine the number of rows and columns in matrix A
    rows = number_of_rows(A)
    cols = number_of_columns(A)
    
    # Create an empty matrix C with the same dimensions as A and B
    C = create_matrix(rows, cols)
    
    # Iterate through each row
    FOR i FROM 0 TO rows-1:
        # Iterate through each column
        FOR j FROM 0 TO cols-1:
            # Calculate the difference for each element and store it in C
            C[i][j] = A[i][j] - B[i][j]
    
    # Return the result matrix C
    RETURN C
END FUNCTION
```
In more human readable format the above pseudocode can be written as:

```python
FUNCTION matrix_difference(A, B):
    Get the number of rows and columns in matrix A
    Create an empty matrix C with the same dimensions
    FOR each row i:
        FOR each column j:
            Set C[i][j] to the difference of A[i][j] and B[i][j]
    RETURN the matrix C
END FUNCTION
```

**Explanation:**

1. Determine the number of rows and columns in matrix $A$.
2. Create a new matrix $C$ with the same dimensions.
3. Loop through each element of the matrices and subtract corresponding elements.
4. Return the resulting matrix $C$.

