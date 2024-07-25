# Python for Linear Algebra
## Pseudocode: the new language for algorithm design

Pseudocode is a way to describe algorithms in a structured but plain language. It helps in planning the logic without worrying about the syntax of a specific programming language. In this module, we’ll use Python-flavored pseudocode to describe various matrix operations.

::: {.callout_caution}
fdhsjGJ
:::

### Matrix Sum
**Mathematical Procedure:**

To add two matrices $A$ and $B$, both matrices must have the same dimensions. The sum $C$ of two matrices $A$ and $B$ is calculated element-wise:

$$C[i][j]=A[i][j]+B[i][j]$$

**Example:**

Let $A$ and $B$ be two $2$ matrices:

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

