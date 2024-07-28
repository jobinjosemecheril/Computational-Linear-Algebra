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

### Matrix Product
**Mathematical Procedure:**

To find the product of two matrices $A$ and $B$, the number of columns in $A$ must be equal to the number of rows in $B$. The element 
 in the product matrix $C$ is computed as:

 $$C[i][j]=\sum_k A[i][k].B[k][j]$$

**Example:**

Let $A$ be a $2\times 3$ matrix and $B$ be a $3\times 2$ matrix:

$$A=\begin{bmatrix}
1 &2 &3\\
4 &5 &6
\end{bmatrix}
, B=\begin{bmatrix}
7 &8\\
9 &10\\
11 &12
\end{bmatrix}$$

The product $C$ is: 

$$C=A.B=\begin{bmatrix}
58 &64\\
139 &154
\end{bmatrix}$$

**Pseudocode:**

```python
FUNCTION matrix_product(A, B):
    # Get the dimensions of A and B
    rows_A = number_of_rows(A)
    cols_A = number_of_columns(A)
    rows_B = number_of_rows(B)
    cols_B = number_of_columns(B)
    
    # Check if multiplication is possible
    IF cols_A != rows_B:
        RAISE Error("Incompatible matrix dimensions")
    
    # Initialize result matrix C
    C = create_matrix(rows_A, cols_B)
    
    # Calculate matrix product
    FOR each row i FROM 0 TO rows_A-1:
        FOR each column j FROM 0 TO cols_B-1:
            # Compute the sum for C[i][j]
            sum = 0
            FOR each k FROM 0 TO cols_A-1:
                sum = sum + A[i][k] * B[k][j]
            C[i][j] = sum
    
    RETURN C
END FUNCTION
```

A more human readable version of the `pseudocode` is shown below:

```python
FUNCTION matrix_product(A, B):
    Get the number of rows and columns in matrix A
    Get the number of columns in matrix B
    Create an empty matrix C with dimensions rows_A x cols_B
    FOR each row i in A:
        FOR each column j in B:
            Initialize C[i][j] to 0
            FOR each element k in the common dimension:
                Add the product of A[i][k] and B[k][j] to C[i][j]
    RETURN the matrix C
END FUNCTION
```

**Explanation:**

1. Determine the number of rows and columns in matrices $A$ and $B$.
2. Create a new matrix $C$ with dimensions $rows(A)\times columns(B)$.
3. Loop through each element of the resulting matrix $C[i][j]$ and calculate the dot
   product of the $i^{th}$ row of $A$ to $j^{th}$ column of $B$ for each element.
4. Return the resulting matrix $C$.


### Determinant
**Mathematical Procedure:**

To find the determinant of a square matrix  $A$, we can use the Laplace expansion, which involves breaking the matrix down into smaller submatrices. For a 
 $2\times 2$ matrix, the determinant is calculated as:

$$det(A)=A[0][0].A[1][1]-A[0][1].A[1][0]$$

For larger matrices, the determinant is calculated recursively.

**Example:**

Let $A$ be a $2\times 2$ matrix:

 $$A=\begin{bmatrix}
 4 &3\\
 6 &3
 \end{bmatrix}$$

The determinant of $A$ is:

$$det(A)=(4.3)-(6.3)=12-18=-6$$

**Pseudocode:**

```python
FUNCTION determinant(A):
    # Step 1: Get the size of the matrix
    n = number_of_rows(A)
    
    # Base case for a 2x2 matrix
    IF n == 2:
        RETURN A[0][0] * A[1][1] - A[0][1] * A[1][0]
    
    # Step 2: Initialize determinant to 0
    det = 0
    
    # Step 3: Loop through each column of the first row
    FOR each column j FROM 0 TO n-1:
        # Get the submatrix excluding the first row and current column
        submatrix = create_submatrix(A, 0, j)
        # Recursive call to determinant
        sub_det = determinant(submatrix)
        # Alternating sign and adding to the determinant
        det = det + ((-1) ^ j) * A[0][j] * sub_det
    
    RETURN det
END FUNCTION

FUNCTION create_sub_matrix(A, row, col):
    sub_matrix = create_matrix(number_of_rows(A)-1, number_of_columns(A)-1)
    sub_i = 0
    FOR i FROM 0 TO number_of_rows(A)-1:
        IF i == row:
            CONTINUE
        sub_j = 0
        FOR j FROM 0 TO number_of_columns(A)-1:
            IF j == col:
                CONTINUE
            sub_matrix[sub_i][sub_j] = A[i][j]
            sub_j = sub_j + 1
        sub_i = sub_i + 1
    RETURN sub_matrix
END FUNCTION
```

A human readable version of the same pseudocode is shown below:

```python
FUNCTION determinant(A):
    IF the size of A is 2x2:
        RETURN the difference between the product of the diagonals
    END IF
    Initialize det to 0
    FOR each column c in the first row:
        Create a sub_matrix by removing the first row and column c
        Add to det: the product of (-1)^c, the element A[0][c], and the determinant of the sub_matrix
    RETURN det
END FUNCTION

FUNCTION create_sub_matrix(A, row, col):
    Create an empty sub_matrix with dimensions one less than A
    Set sub_i to 0
    FOR each row i in A:
        IF i is the row to be removed:
            CONTINUE to the next row
        Set sub_j to 0
        FOR each column j in A:
            IF j is the column to be removed:
                CONTINUE to the next column
            Copy the element A[i][j] to sub_matrix[sub_i][sub_j]
            Increment sub_j
        Increment sub_i
    RETURN sub_matrix
END FUNCTION
```
**Explanation:**

1. If the matrix is $2\times 2$, calculate the determinant directly.
2. For larger matrices, use the Laplace expansion to recursively calculate the determinant.
3. Create submatrices by removing the current row and column.
4. Sum the determinants of the submatrices, adjusted for the sign and the current element.

### Rank of a Matrix
**Mathematical Procedure:**

The rank of a matrix $A$ is the maximum number of linearly independent rows or columns in $A$. This can be found using Gaussian elimination to transform the matrix into its row echelon form (REF) and then counting the number of non-zero rows.

**Example:**

Let $A$ be a $3\times 3$ matrix:

$$A=\begin{bmatrix}
1 &2 &3\\
4 &5 &6\\
7 &8 &9
\end{bmatrix}$$

After performing Gaussian elimination, we obtain:

$$REF(A)=\begin{bmatrix}
1 &2 &3\\
0 &-3 &-6\\
0 &0 &0
\end{bmatrix}$$

The rank of $A$ is the number of non-zero rows, which is 2.

**Pseudocode:**

```python
FUNCTION matrix_rank(A):
    # Step 1: Get the dimensions of the matrix
    rows = number_of_rows(A)
    cols = number_of_columns(A)
    
    # Step 2: Transform the matrix to row echelon form
    row_echelon_form(A, rows, cols)
    
    # Step 3: Count non-zero rows
    rank = 0
    FOR each row i FROM 0 TO rows-1:
        non_zero = FALSE
        FOR each column j FROM 0 TO cols-1:
            IF A[i][j] != 0:
                non_zero = TRUE
                BREAK
        IF non_zero:
            rank = rank + 1
    
    RETURN rank
END FUNCTION

FUNCTION row_echelon_form(A, rows, cols):
    # Perform Gaussian elimination
    lead = 0
    FOR r FROM 0 TO rows-1:
        IF lead >= cols:
            RETURN
        i = r
        WHILE A[i][lead] == 0:
            i = i + 1
            IF i == rows:
                i = r
                lead = lead + 1
                IF lead == cols:
                    RETURN
        # Swap rows i and r
        swap_rows(A, i, r)
        # Make A[r][lead] = 1
        lv = A[r][lead]
        A[r] = [m / float(lv) for m in A[r]]
        # Make all rows below r have 0 in column lead
        FOR i FROM r + 1 TO rows-1:
            lv = A[i][lead]
            A[i] = [iv - lv * rv for rv, iv in zip(A[r], A[i])]
        lead = lead + 1
END FUNCTION

FUNCTION swap_rows(A, row1, row2):
    temp = A[row1]
    A[row1] = A[row2]
    A[row2] = temp
END FUNCTION
```
A more human readable version of the above pseudocode is shown below:

```python
FUNCTION rank(A):
    Get the number of rows and columns in matrix A
    Initialize the rank to 0
    FOR each row i in A:
        IF the element in the current row and column is non-zero:
            Increment the rank
            FOR each row below the current row:
                Calculate the multiplier to zero out the element below the diagonal
                Subtract the appropriate multiple of the current row from each row below
        ELSE:
            Initialize a variable to track if a swap is needed
            FOR each row below the current row:
                IF a non-zero element is found in the current column:
                    Swap the current row with the row having the non-zero element
                    Set the swap variable to True
                    BREAK the loop
            IF no swap was made:
                Decrement the rank
    RETURN the rank
END FUNCTION
```
**Explanation:**

1. Initialize the rank to 0.
2. Loop through each row of the matrix.
3. If the diagonal element is non-zero, increment the rank and perform row operations to zero out the elements below the diagonal.
4. If the diagonal element is zero, try to swap with a lower row that has a non-zero element in the same column.
5. If no such row is found, decrement the rank.
6. Return the resulting rank of the matrix.

### Solving a System of Equations
**Mathematical Procedure:**

To solve a system of linear equations represented as $AX=b$, where $A$ is the coefficient matrix, $x$ is the vector of variables, and $b$
 is the constant vector, we can use Gaussian elimination to transform the augmented matrix $[A|b]$ into its row echelon form (REF) and then perform back substitution to find the solution vector $x$.

**Example:**

Consider the system of equations:

$$\begin{cases}
x + 2y + 3z &= 9 \\
4x + 5y + 6z& = 24 \\
7x + 8y + 9z& = 39
\end{cases}$$

The augmented matrix is:

$$[A|b]=\begin{bmatrix}
1 &2 &3 &| &9\\
4 &5 &6 &| &24\\
7 &8 &9 &| &39
\end{bmatrix}$$

After performing Gaussian elimination on the augmented matrix, we get:

$$REF(A)=\begin{bmatrix}
1 &2 &3 &| &9\\
0 &-3 &-6 &| &-12\\
0 &0 &0 &| &0
\end{bmatrix}$$

Performing back substitution, we solve for $z,y,$ and $x$:

$${\begin{cases}
z=1\\
y=0\\
x=3
\end{cases}$$

Therefore, the solution vector is $`\mathbf{x}=\begin{bmatrix}
3\\
0\\
1
\end{bmatrix}`$

**Pseudocode:**

```python
FUNCTION solve_system_of_equations(A, b):
    # Step 1: Get the dimensions of the matrix
    rows = number_of_rows(A)
    cols = number_of_columns(A)
    
    # Step 2: Create the augmented matrix
    augmented_matrix = create_augmented_matrix(A, b)
    
    # Step 3: Transform the augmented matrix to row echelon form
    row_echelon_form(augmented_matrix, rows, cols)
    
    # Step 4: Perform back substitution
    solution = back_substitution(augmented_matrix, rows, cols)
    
    RETURN solution
END FUNCTION

FUNCTION create_augmented_matrix(A, b):
    # Combine A and b into an augmented matrix
    augmented_matrix = []
    FOR i FROM 0 TO number_of_rows(A)-1:
        augmented_matrix.append(A[i] + [b[i]])
    RETURN augmented_matrix
END FUNCTION

FUNCTION row_echelon_form(augmented_matrix, rows, cols):
    # Perform Gaussian elimination
    lead = 0
    FOR r FROM 0 TO rows-1:
        IF lead >= cols:
            RETURN
        i = r
        WHILE augmented_matrix[i][lead] == 0:
            i = i + 1
            IF i == rows:
                i = r
                lead = lead + 1
                IF lead == cols:
                    RETURN
        # Swap rows i and r
        swap_rows(augmented_matrix, i, r)
        # Make augmented_matrix[r][lead] = 1
        lv = augmented_matrix[r][lead]
        augmented_matrix[r] = [m / float(lv) for m in augmented_matrix[r]]
        # Make all rows below r have 0 in column lead
        FOR i FROM r + 1 TO rows-1:
            lv = augmented_matrix[i][lead]
            augmented_matrix[i] = [iv - lv * rv for rv, iv in zip(augmented_matrix[r], augmented_matrix[i])]
        lead = lead + 1
END FUNCTION

FUNCTION back_substitution(augmented_matrix, rows, cols):
    # Initialize the solution vector
    solution = [0 for _ in range(rows)]
    # Perform back substitution
    FOR i FROM rows-1 DOWNTO 0:
        solution[i] = augmented_matrix[i][cols-1]
        FOR j FROM i+1 TO cols-2:
            solution[i] = solution[i] - augmented_matrix[i][j] * solution[j]
    RETURN solution
END FUNCTION

FUNCTION swap_rows(matrix, row1, row2):
    temp = matrix[row1]
    matrix[row1] = matrix[row2]
    matrix[row2] = temp
END FUNCTION
```
**Explanation:**

1. Augment the coefficient matrix $A$ with the constant matrix $b$.
2. Perform Gaussian elimination to reduce the augmented matrix to row echelon form.
3. Back-substitute to find the solution vector $X$.
4. Return the solution vector $X$.

### Review Problems
**Q1:** Fill in the missing parts of the pseudocode to yield a meaningful algebraic operation on of two matrices.

**Pseudocode:**
```python
FUNCTION matrix_op1(A, B):
    rows = number_of_rows(A)
    cols = number_of_columns(A)
    result = create_matrix(rows, cols, 0)
    
    FOR i FROM 0 TO rows-1:
        FOR j FROM 0 TO cols-1:
            result[i][j] = A[i][j] + ---
    
    RETURN result
END FUNCTION
```
**Q2:** Write the pseudocode to get a useful derivable from a given matrix by filling in the missing part.

**Pseudocode:**
```python
FUNCTION matrix_op2(A):
    rows = number_of_rows(A)
    cols = number_of_columns(A)
    result = create_matrix(cols, rows, 0)
    
    FOR i FROM 0 TO rows-1:
        FOR j FROM 0 TO cols-1:
            result[j][i] = A[i][--]
    
    RETURN result
END FUNCTION
```
## Transition from Pseudocode to Python Programming
In this course, our initial approach to understanding and solving linear algebra problems has been through pseudocode. Pseudocode allows us to focus on the logical steps and algorithms without getting bogged down by the syntax of a specific programming language. This method helps us build a strong foundation in the computational aspects of linear algebra.

However, to fully leverage the power of computational tools and prepare for real-world applications, it is essential to implement these algorithms in a practical programming language. Python is a highly versatile and widely-used language in the fields of data science, artificial intelligence, and engineering. By transitioning from pseudocode to Python, we align with the following course objectives:

1. **Practical Implementation:** Python provides numerous libraries and tools, such as NumPy and SciPy, which are specifically designed for numerical computations and linear algebra. Implementing our algorithms in Python allows us to perform complex calculations efficiently and accurately.

2. **Hands-On Experience:** Moving to Python programming gives students hands-on experience in coding, debugging, and optimizing algorithms. This practical experience is crucial for developing the skills required in modern computational tasks.

3. **Industry Relevance:** Python is extensively used in industry for data analysis, machine learning, and scientific research. Familiarity with Python and its libraries ensures that students are well-prepared for internships, research projects, and future careers in these fields.

4. **Integration with Other Tools:** Python’s compatibility with various tools and platforms allows for seamless integration into larger projects and workflows. This integration is vital for tackling real-world problems that often require multi-disciplinary approaches.

5. **Enhanced Learning:** Implementing algorithms in Python helps reinforce theoretical concepts by providing immediate feedback through code execution and results visualization. This iterative learning process deepens understanding and retention of the material.

By transitioning to Python programming, we not only achieve our course objectives but also equip students with valuable skills that are directly applicable to their academic and professional pursuits.

## Python Fundamentals
### Python Programming Overview
Python is a high-level, interpreted programming language that was created by Guido van Rossum and first released in 1991. Its design philosophy emphasizes code readability and simplicity, making it an excellent choice for both beginners and experienced developers. Over the years, Python has undergone significant development and improvement, with major releases adding new features and optimizations. The language’s versatility and ease of use have made it popular in various domains, including web development, data science, artificial intelligence, scientific computing, automation, and more. Python’s extensive standard library and active community contribute to its widespread adoption, making it one of the most popular programming languages in the world today.

### Variables
In Python, variables are used to store data that can be used and manipulated throughout a program. Variables do not need explicit declaration to reserve memory space. The declaration happens automatically when a value is assigned to a variable.

**Basic Input/Output Functions**

Python provides built-in functions for basic input and output operations. The `print()` function is used to display output, while the `input()` function is used to take input from the user.

Output with *`print()`* function

> Example 1

```python
# Printing text
print("Hello, World!")

# Printing multiple values
x = 5
y = 10
print("The value of x is:", x, "and the value of y is:", y)
```
> Example 2

```python
# Assigning values to variables
a = 10
b = 20.5
name = "Alice"

# Printing the values
print("Values Stored in the Variables:")
print(a)
print(b)
print(name)
```
Input with *`input()`* Function:

```python
# Taking input from the user
name = input("Enter usr name: ")
print("Hello, " + name + "!")

# Taking numerical input
age = int(input("Enter usr age: "))
print("us are", age, "years old.")
```
> [!Note]
> The `print()` function in Python, defined in the built-in `__builtin__` module, is used to display output on the screen, providing a simple way to output text and variable values to the console.


**Combining Variables and Input/Output**

We can combine variables and input/output functions to create interactive programs.

> Example

```python
# Program to calculate the sum of two numbers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Calculate sum
sum = num1 + num2

# Display the result
print("The sum of", num1, "and", num2, "is", sum)
```
## Python Programming Style
###### Indentation

Python uses indentation to define the blocks of code. Proper indentation is crucial as it affects the program’s flow. Use 4 spaces per indentation level.

```python
if a > b:
    print("a is greater than b")
else:
    print("b is greater than or equal to a")
```

###### Comments
Use comments to explain user code. Comments begin with the `#` symbol and extend to the end of the line. Write comments that are clear and concise. See the example:

```python
# This is a comment
a = 10  # This is an inline comment
```

###### Variable Naming
Use meaningful variable names to make user code more understandable. Variable names should be in lowercase with words separated by underscores.
```python
student_name = "John"
total_score = 95
```
###### Consistent Style
Follow the `PEP 8` style guide for Python code to maintain consistency and readability. Use blank lines to separate different sections of user code. See the following example of function definition:

```python

def calculate_sum(x, y):
    return x + y

result = calculate_sum(5, 3)
print(result)
```
## Basic Datatypes in Python
In Python, a datatype is a classification that specifies which type of value a variable can hold. Understanding datatypes is essential as it helps in performing appropriate operations on variables. Python supports various built-in datatypes, which can be categorized into several groups.

### Numeric Types
Numeric types represent data that consists of numbers. Python has three distinct numeric types:

1. **Integers** (`int`):
     - Whole numbers, positive or negative, without decimals.
     - Example: `a = 10`, `b = -5`.
2. **Floating Point Numbers** (`float`):
     - Numbers that contain a decimal point.
     - Example: `pi = 3.14`, `temperature = -7.5`.
3. **Complex Numbers** (`complex`):
     - Numbers with a real and an imaginary part.
     - Example: `z = 3 + 4j`.
```python
# Examples of numeric types
a = 10         # Integer
pi = 3.14      # Float
z = 3 + 4j     # Complex
```
### Sequence Types
Sequence types are used to store multiple items in a single variable. Python has several sequence types, including:

###### String Type
Strings in Python are sequences of characters enclosed in quotes. They are used to handle and manipulate textual data.

#### Characteristics of Strings

- *Ordered*: Characters in a string have a defined order.
- *Immutable*: Strings cannot be modified after they are created.
- *Heterogeneous*: Strings can include any combination of letters, numbers, and symbols.
    
#### Creating Strings

Strings can be created using single quotes, double quotes, or triple quotes for multiline strings.

> Example:

```python
# Creating strings with different types of quotes
single_quoted = 'Hello, World!'
double_quoted = "Hello, World!"
multiline_string = """This is a
multiline string"""
```
**Accessing String Characters**

Characters in a string are accessed using their index, with the first character having an index of 0. Negative indexing can be used to access characters from the end.

> Example

```python
# Accessing characters in a string
first_char = single_quoted[0]  # Output: 'H'
last_char = single_quoted[-1]  # Output: '!'
```

**Common String Methods**

Python provides various methods for string manipulation:

1. `upper()`: Converts all characters to uppercase.
2. `lower()`: Converts all characters to lowercase.
3. `strip()`: Removes leading and trailing whitespace.
4. `replace(old, new)`: Replaces occurrences of a substring with another substring.
5. `split(separator)`: Splits the string into a list based on a separator.

> Example

```python
# Using string methods
text = "   hello, world!   "
uppercase_text = text.upper()       # Result: "   HELLO, WORLD!   "
stripped_text = text.strip()        # Result: "hello, world!"
replaced_text = text.replace("world", "Python")  # Result: "   hello, Python!   "
words = text.split(",")             # Result: ['hello', ' world!   ']
```

###### List Type
Lists are one of the most versatile and commonly used sequence types in Python. They allow for the storage and manipulation of ordered collections of items.

> **Characteristics of Lists**

- *Ordered*: The items in a list have a defined order, which will not change unless explicitly modified.
- *Mutable*: The content of a list can be changed after its creation (i.e., items can be added, removed, or modified).
- *Dynamic*: Lists can grow or shrink in size as items are added or removed.
- *Heterogeneous*: Items in a list can be of different data types (e.g., integers, strings, floats).

**Creating Lists**

Lists are created by placing comma-separated values inside square brackets.

> Example


```pyhon
# Creating a list of fruits
fruits = ["apple", "banana", "cherry"]

# Creating a mixed list
mixed_list = [1, "Hello", 3.14]
```

**Accessing List Items**

List items are accessed using their index, with the first item having an index of 0.

> Example

```python

# Accessing the first item
first_fruit = fruits[0]  # Output: "apple"

# Accessing the last item
last_fruit = fruits[-1]  # Output: "cherry"
```
**Modifying Lists**

Lists can be modified by changing the value of specific items, adding new items, or removing existing items.

> Example

```python
# Changing the value of an item
fruits[1] = "blueberry"  # fruits is now ["apple", "blueberry", "cherry"]

# Adding a new item
fruits.append("orange")  # fruits is now ["apple", "blueberry", "cherry", "orange"]

# Removing an item
fruits.remove("blueberry")  # fruits is now ["apple", "cherry", "orange"]
```

**List Methods**

`Python` provides several built-in methods to work with lists:

1. `append(item)`: Adds an item to the end of the list.
2. `insert(index, item)`: Inserts an item at a specified index.
3. `remove(item)`: Removes the first occurrence of an item.
4. `pop(index)`: Removes and returns the item at the specified index.
5. `sort()`: Sorts the list in ascending order.
6. `reverse()`: Reverses the order of the list.

> Example

```python
# Using list methods
numbers = [5, 2, 9, 1]

numbers.append(4)     # numbers is now [5, 2, 9, 1, 4]
numbers.sort()        # numbers is now [1, 2, 4, 5, 9]
numbers.reverse()     # numbers is now [9, 5, 4, 2, 1]
first_number = numbers.pop(0)  # first_number is 9, numbers is now [5, 4, 2, 1]
```
###### Tuple Type

Tuples are a built-in sequence type in Python that is used to store an ordered collection of items. Unlike lists, tuples are immutable, which means their contents cannot be changed after creation.

**Characteristics of Tuples**

- `Ordered`: Tuples maintain the order of items, which is consistent throughout their lifetime.
- `Immutable`: Once a tuple is created, its contents cannot be modified. This includes adding, removing, or changing items.
- `Fixed Size`: The size of a tuple is fixed; it cannot grow or shrink after creation.
- `Heterogeneous`: Tuples can contain items of different data types, such as integers, strings, and floats.

**Creating Tuples**

Tuples are created by placing comma-separated values inside parentheses. Single-element tuples require a trailing comma.

> Example

```python
# Creating a tuple with multiple items
coordinates = (10, 20, 30)

# Creating a single-element tuple
single_element_tuple = (5,)

# Creating a tuple with mixed data types
mixed_tuple = (1, "Hello", 3.14)
```

**Accessing Tuple Items**

Tuple items are accessed using their index, with the first item having an index of 0. Negative indexing can be used to access items from the end.

> Example

```python
# Accessing the first item
x = coordinates[0]  # Output: 10

# Accessing the last item
z = coordinates[-1]  # Output: 30
```

**Modifying Tuples**

Since tuples are immutable, their contents cannot be modified. However, us can create new tuples by combining or slicing existing ones.

> Example

```python
# Combining tuples
new_coordinates = coordinates + (40, 50)  # Result: (10, 20, 30, 40, 50)

# Slicing tuples
sub_tuple = coordinates[1:3]  # Result: (20, 30)
```

**Tuple Methods**

Tuples have a limited set of built-in methods compared to lists:

1. `count(item)`: Returns the number of occurrences of the specified item.
2. `index(item)`: Returns the index of the first occurrence of the specified item.

> Example

```python
# Using tuple methods
numbers = (1, 2, 3, 1, 2, 1)

# Counting occurrences of an item
count_1 = numbers.count(1)  # Result: 3

# Finding the index of an item
index_2 = numbers.index(2)  # Result: 1
```

#### Mapping Types

Mapping types in Python are used to store data in key-value pairs. Unlike sequences, mappings do not maintain an order and are designed for quick lookups of data.

##### Dictionary (`dict`)
The primary mapping type in Python is the `dict`. Dictionaries store data as key-value pairs, where each key must be unique, and keys are used to access their corresponding values.

**Characteristics of Dictionaries**

- *Unordered*: The order of items is not guaranteed and may vary.
- *Mutable*: us can add, remove, and change items after creation.
- *Keys*: Must be unique and immutable (e.g., strings, numbers, tuples).
- *Values*: Can be of any data type and can be duplicated.

**Creating Dictionaries**

Dictionaries are created using curly braces `{}` with key-value pairs separated by colons `:`.

> Example

```python
# Creating a dictionary
student = {
    "name": "Alice",
    "age": 21,
    "major": "Computer Science"
}
```

**Accessing and Modifying Dictionary Items**

Items in a dictionary are accessed using their keys. us can also modify, add, or remove items.

> Example

```python
# Accessing a value
name = student["name"]  # Output: "Alice"

# Modifying a value
student["age"] = 22  # Updates the age to 22

# Adding a new key-value pair
student["graduation_year"] = 2024

# Removing a key-value pair
del student["major"]
```

**Dictionary Methods**

Python provides several built-in methods to work with dictionaries:

1. `keys()`: Returns a view object of all keys.
2. `values()`: Returns a view object of all values.
3. `items()`: Returns a view object of all key-value pairs.
4. `get(key, default)`: Returns the value for the specified key, or a default value if the key is not found.
5. `pop(key, default)`: Removes and returns the value for the specified key, or a default value if the key is not found.

> Example

```python
# Using dictionary methods
keys = student.keys()        # Result: dict_keys(['name', 'age', 'graduation_year'])
values = student.values()    # Result: dict_values(['Alice', 22, 2024])
items = student.items()      # Result: dict_items([('name', 'Alice'), ('age', 22), ('graduation_year', 2024)])
name = student.get("name")  # Result: "Alice"
age = student.pop("age")    # Result: 22
```

#### Set Types

Sets are a built-in data type in Python used to store unique, unordered collections of items. They are particularly useful for operations involving membership tests, set operations, and removing duplicates.

**Characteristics of Sets**

- *Unordered* : The items in a set do not have a specific order and may change.
- *Mutable* : us can add or remove items from a set after its creation.
- *Unique* : Sets do not allow duplicate items; all items must be unique.
- *Unindexed* : Sets do not support indexing or slicing.

**Creating Sets**

Sets are created using curly braces `{}` with comma-separated values, or using the `set()` function.

> Example

```python
# Creating a set using curly braces
fruits = {"apple", "banana", "cherry"}

# Creating a set using the set() function
numbers = set([1, 2, 3, 4, 5])
```

**Accessing and Modifying Set Items**

While us cannot access individual items by index, us can check for membership and perform operations like adding or removing items.

> Example

```python
# Checking membership
has_apple = "apple" in fruits  # Output: True

# Adding an item
fruits.add("orange")

# Removing an item
fruits.remove("banana")  # Raises KeyError if item is not present
```

**Set Operations**

Sets support various mathematical set operations, such as `union`, `intersection`, and `difference`.

> Example

```python
# Union of two sets
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union = set1 | set2  # Result: {1, 2, 3, 4, 5}

# Intersection of two sets
intersection = set1 & set2  # Result: {3}

# Difference between two sets
difference = set1 - set2  # Result: {1, 2}

# Symmetric difference (items in either set, but not in both)
symmetric_difference = set1 ^ set2  # Result: {1, 2, 4, 5}
```

**Set Methods**

Python provides several built-in methods for set operations:

1. `add(item)`: Adds an item to the set.
2. `remove(item)`: Removes an item from the set; raises KeyError if item is not present.
3. `discard(item)`: Removes an item from the set if present; does not raise an error if item is not found.
4. `pop()`: Removes and returns an arbitrary item from the set.
5. `clear()`: Removes all items from the set.


> Example

```python
# Using set methods
set1 = {1, 2, 3}

set1.add(4)        # set1 is now {1, 2, 3, 4}
set1.remove(2)     # set1 is now {1, 3, 4}
set1.discard(5)    # No error, set1 remains {1, 3, 4}
item = set1.pop()  # Removes and returns an arbitrary item, e.g., 1
set1.clear()      # set1 is now an empty set {}
```

#### Frozen Sets

Frozen sets are a built-in data type in Python that are similar to sets but are immutable. Once created, a frozen set cannot be modified, making it suitable for use as a key in dictionaries or as elements of other sets.

**Characteristics of Frozen Sets**

- *Unordered* : The items in a frozen set do not have a specific order and may change.
- *Immutable* : Unlike regular sets, frozen sets cannot be altered after creation. No items can be added or removed.
- *Unique* : Like sets, frozen sets do not allow duplicate items; all items must be unique.
- *Unindexed* : Frozen sets do not support indexing or slicing.

**Creating Frozen Sets**

Frozen sets are created using the `frozenset()` function, which takes an iterable as an argument.

> Example

```python
# Creating a frozen set
numbers = frozenset([1, 2, 3, 4, 5])

# Creating a frozen set from a set
fruits = frozenset({"apple", "banana", "cherry"})
```

**Accessing and Modifying Frozen Set Items**

Frozen sets do not support modification operations such as adding or removing items. However, us can perform membership tests and other set operations.

> Example

```python
# Checking membership
has_apple = "apple" in fruits  # Output: True

# Since frozenset is immutable, us cannot use add() or remove() methods
```

**Set Operations with Frozen Sets**

Frozen sets support various mathematical set operations similar to regular sets, such as union, intersection, and difference. These operations return new frozen sets and do not modify the original ones.

> Example

```python
# Union of two frozen sets
set1 = frozenset([1, 2, 3])
set2 = frozenset([3, 4, 5])
union = set1 | set2  # Result: frozenset({1, 2, 3, 4, 5})

# Intersection of two frozen sets
intersection = set1 & set2  # Result: frozenset({3})

# Difference between two frozen sets
difference = set1 - set2  # Result: frozenset({1, 2})

# Symmetric difference (items in either set, but not in both)
symmetric_difference = set1 ^ set2  # Result: frozenset({1, 2, 4, 5})
```

**Frozen Set Methods**

Frozen sets have a subset of the methods available to regular sets. The available methods include:

1. `copy()` : Returns a shallow copy of the frozen set.
2. `difference(other)` : Returns a new frozen set with elements in the original frozen set but not in other.
3. `intersection(other)` : Returns a new frozen set with elements common to both frozen sets.
4. `union(other)` : Returns a new frozen set with elements from both frozen sets.
5. `symmetric_difference(other)` : Returns a new frozen set with elements in either frozen set but not in both.

> Example

```python
# Using frozen set methods
set1 = frozenset([1, 2, 3])
set2 = frozenset([3, 4, 5])

# Getting the difference
difference = set1.difference(set2)  # Result: frozenset({1, 2})

# Getting the intersection
intersection = set1.intersection(set2)  # Result: frozenset({3})

# Getting the union
union = set1.union(set2)  # Result: frozenset({1, 2, 3, 4, 5})

# Getting the symmetric difference
symmetric_difference = set1.symmetric_difference(set2)  # Result: frozenset({1, 2, 4, 5})
```

## Control Structures in Python

Control structures in Python allow us to control the flow of execution in our programs. They help manage decision-making, looping, and the execution of code blocks based on certain conditions. Python provides several key control structures: `if` statements, `for` loops, `while` loops, and control flow statements like `break`, `continue`, and `pass`.

#### Conditional Statements

Conditional statements are used to execute code based on certain conditions. The primary conditional statement in Python is the `if` statement, which can be combined with `elif` and `else` to handle multiple conditions.

> Syntax

```python
if condition:
    # Code block to execute if condition is True
elif another_condition:
    # Code block to execute if another_condition is True
else:
    # Code block to execute if none of the above conditions are True
```

> Example: Program to classify a person based on his/her age.

```python
age = 20

if age < 18:
    print("us are a minor.")
elif age < 65:
    print("us are an adult.")
else:
    print("us are a senior citizen.")
```

#### Looping Statements

Looping statements are used to repeat a block of code multiple times. Python supports for loops and while loops.

**For Loop**

The `for` loop iterates over a sequence (like a list, tuple, or string) and executes a block of code for each item in the sequence.

> Syntax:

```python
for item in sequence:
    # Code block to execute for each item
```

> Example: Program to print names of fruits saved in a list.

```python
# Iterating over a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```

**While Loop**

The `while` loop repeatedly executes a block of code as long as a specified condition is True.

> Syntax:

```python
while condition:
    # Code block to execute while condition is True
```

> Example: Print all counting numbers less than 5.

```python
# Counting from 0 to 4
count = 0
while count < 5:
    print(count)
    count += 1
```

#### Control Flow Statements

Control flow statements alter the flow of execution within loops and conditionals.

**Break Statement**

The `break` statement exits the current loop, regardless of the loop’s condition.

> Example: Program to exit from the printing of whole numbers less than 10, while trigger 5.

```python
for i in range(10):
    if i == 5:
        break
    print(i)
# Output: 0 1 2 3 4
```

**Continue Statement**

The `continue` statement skips the rest of the code inside the current loop iteration and proceeds to the next iteration.

> Example: Program to print all the whole numbers in the range 5 except 2.

```python
for i in range(5):
    if i == 2:
        continue
    print(i) 
# Output: 0 1 3 4
```

**Pass Statement**

The `pass` statement is a placeholder that does nothing and is used when a statement is syntactically required but no action is needed.

> Example: Program to print all the whole numbers in the range 5 except 3.

```python
for i in range(5):
    if i == 3:
        pass  # Placeholder for future code
    else:
        print(i)
# Output: 0 1 2 4
```

> [!CAUTION]
> 
> Control flow structures are essential in Python programming for directing the flow of execution. However, improper use of these structures can lead to errors, inefficiencies, and unintended behaviors. Here are some cautions to keep in mind:
> 
> **Infinite Loops**
> - **Issue**: A while loop with a condition that never becomes False can lead to an infinite loop, which will cause the program to hang or become unresponsive.
> - **Caution**: Always ensure that the condition in a while loop will eventually become False, and include logic within the loop to modify the condition.
>   
> **Example:**
>   
>   ```python
>    # Infinite loop example
>   count = 0
>   while count < 5:
>       print(count)
>       # Missing count increment, causing an infinite loop
>   ```

## Functions in Python Programming

Functions are a fundamental concept in Python programming that enable code reuse, modularity, and organization. They allow us to encapsulate a block of code that performs a specific task, which can be executed whenever needed. Functions are essential for writing clean, maintainable, and scalable code, making them a cornerstone of effective programming practices.

**What is a Function?**

A function is a named block of code designed to perform a specific task. Functions can take inputs, called parameters or arguments, and can return outputs, which are the results of the computation or task performed by the function. By defining functions, us can write code once and reuse it multiple times, which enhances both efficiency and readability.

**Defining a Function**

In Python, functions are defined using the `def` keyword, followed by the function name, parentheses containing any parameters, and a colon. The function body, which contains the code to be executed, is indented below the function definition.

> Syntax

```python
def function_name(parameters):
    # Code block
    return result
```

> Example

```python
def greet(name):
    """
    Returns a greeting message for the given name.
    """
    return f"Hello, {name}!"
```

**Relevance of functions in Programming**

1. *Code Reusability* : Functions allow us to define a piece of code once and reuse it in multiple places. This reduces redundancy and helps maintain consistency across our codebase.

2. *Modularity* : Functions break down complex problems into smaller, manageable pieces. Each function can be focused on a specific task, making it easier to understand and maintain the code.

3. *Abstraction* : Functions enable us to abstract away the implementation details. We can use a function without needing to know its internal workings, which simplifies the code we write and enhances readability.

4. *Testing and Debugging* : Functions allow us to test individual components of our code separately. This isolation helps in identifying and fixing bugs more efficiently.

5. *Library Creation* : Functions are the building blocks of libraries and modules. By organizing related functions into libraries, we can create reusable components that can be shared and utilized across different projects.

> Example: Creating a Simple Library

**Stage 1**: Define Functions in a Module

```python
# my_library.py

def add(a, b):
    """
    Returns the sum of two numbers.
    """
    return a + b

def multiply(a, b):
    """
    Returns the product of two numbers.
    """
    return a * b
```
**Stage 2**: Use the Library in Another Program

```python
# main.py

import my_library

result_sum = my_library.add(5, 3)
result_product = my_library.multiply(5, 3)

print(f"Sum: {result_sum}")
print(f"Product: {result_product}")
```
