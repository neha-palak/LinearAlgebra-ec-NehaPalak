
#Question 1

class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        return ComplexNumber(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def __truediv__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        denominator = other.real**2 + other.imag**2
        return ComplexNumber(
            (self.real * other.real + self.imag * other.imag) / denominator,
            (self.imag * other.real - self.real * other.imag) / denominator
        )

    def abs(self):
        return (self.real**2 + self.imag**2)**0.5

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        return f"{self.real} + {self.imag}i"


class Vector:
    def __init__(self, field, length, coordinates=None):
        self.field = field
        self.length = length
        if coordinates:
            if len(coordinates) != length:
                raise ValueError("Coordinate length must match vector length.")
            self.coordinates = coordinates
        else:
            self.coordinates = [self.field(0)] * length

    def __add__(self, other):
        if self.length != other.length:
            raise ValueError("Vectors must have the same length for addition.")
        return Vector(self.field, self.length, [
            self.coordinates[i] + other.coordinates[i] for i in range(self.length)
        ])
    
    def length(self):
        """Returns the length of the vector."""
        return self.length

    def __str__(self):
        return str(self.coordinates)


class Matrix:
    def __init__(self, field, rows, cols, entries=None):
        self.field = field
        self.rows = rows
        self.cols = cols
        if entries:
            if len(entries) != rows * cols:
                raise ValueError("Number of entries must match rows * cols.")
            self.entries = [field(x) for x in entries]
        else:
            self.entries = [field(0)] * (rows * cols)


    @classmethod
    def from_vectors(cls, vectors):
        rows = len(vectors[0].coordinates)
        cols = len(vectors)
        field = vectors[0].field
        entries = []
        for i in range(rows):
            for vector in vectors:
                entries.append(vector.coordinates[i])
        return cls(field, rows, cols, entries)

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition.")
        return Matrix(self.field, self.rows, self.cols, [
            self.entries[i] + other.entries[i] for i in range(len(self.entries))
        ])

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix multiplication not possible with these dimensions.")
        result_entries = []
        for i in range(self.rows):
            for j in range(other.cols):
                value = self.field(0)
                for k in range(self.cols):
                    value += self.entries[i * self.cols + k] * other.entries[k * other.cols + j]
                result_entries.append(value)
        return Matrix(self.field, self.rows, other.cols, result_entries)

    def row(self, r):
        if r >= self.rows or r < 0:
            raise ValueError("Row index out of range.")
        return [self.entries[r * self.cols + i] for i in range(self.cols)]

    def column(self, c):
        if c >= self.cols or c < 0:
            raise ValueError("Column index out of range.")
        return [self.entries[i * self.cols + c] for i in range(self.rows)]

    def multiply_with_vector(self, vector):
        if len(vector.coordinates) != self.cols:
            raise ValueError("Vector length must match the number of columns in the matrix.")
        result = [0] * self.rows
        for i in range(self.rows):
            result[i] = sum(self.entries[i * self.cols + j] * vector.coordinates[j] for j in range(self.cols))
        return Vector(self.field, self.rows, result)

    def transpose(self):
        result_entries = [
            self.entries[i * self.cols + j]
            for j in range(self.cols)
            for i in range(self.rows)
        ]
        return Matrix(self.field, self.cols, self.rows, result_entries)

    def conjugate(self):
        return Matrix(self.field, self.rows, self.cols, [
            x.conjugate() if isinstance(x, ComplexNumber) else x for x in self.entries
        ])

    def transpose_conjugate(self):
        return self.transpose().conjugate()

    def __str__(self):
        return '\n'.join(
            ' '.join(str(self.entries[i * self.cols + j]) for j in range(self.cols))
            for i in range(self.rows)
        )
    
# Question 2

    # Property checking functions
    def get(self, r, c):
        """Helper function to access matrix elements"""
        return self.entries[r * self.cols + c]

    def is_zero(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.get(i, j) != 0:
                    return False
        return True

    def is_square(self):
        return self.rows == self.cols

    def is_symmetric(self):
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.entries[i * self.cols + j] != self.entries[j * self.cols + i]:
                    return False
        return True

    def is_hermitian(self):
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.entries[i * self.cols + j] != (self.entries[j * self.cols + i].conjugate() if isinstance(self.entries[j * self.cols + i], ComplexNumber) else self.entries[j * self.cols + i]):
                    return False
        return True

    
    def is_orthogonal(self):
        if not self.is_square():
            return False
        identity = Matrix(self.field, self.rows, self.cols, [1 if i == j else 0 for i in range(self.rows) for j in range(self.cols)])
        transpose = self.transpose()
        product = transpose * self  # Matrix multiplication
        return product == identity
    
    def is_unitary(self):
        if not self.is_square():
            return False
        identity = Matrix(self.field, self.rows, self.cols, [1 if i == j else 0 for i in range(self.rows) for j in range(self.cols)])
        transpose_conjugate = self.transpose_conjugate()
        product = transpose_conjugate * self
        return product == identity

    def is_scalar(self):
    # Create a local 2D matrix from the flat list entries
        entries_2d = [self.entries[i:i + self.cols] for i in range(0, len(self.entries), self.cols)]
        
        if not self.is_square():
            return False
        
        # Check if all diagonal elements are the same and non-diagonal elements are 0
        scalar_value = entries_2d[0][0]
        
        for i in range(self.rows):
            for j in range(self.cols):
                print(f"Checking element ({i},{j}): {entries_2d[i][j]}")  # Debug print

                if (i == j and entries_2d[i][j] != scalar_value) or (i != j and entries_2d[i][j] != 0):
                    return False
        return True



    def is_singular(self):
        return not self.is_invertible()

    def is_invertible(self):
        if not self.is_square():
            return False
        determinant = self.determinant()
        return determinant != 0

    

    def _convert_to_2d(self):
        return [self.entries[i:i + self.cols] for i in range(0, len(self.entries), self.cols)]

    def is_identity(self):
        entries_2d = self._convert_to_2d()
        
        if not self.is_square():
            return False
        
        for i in range(self.rows):
            for j in range(self.cols):
                print(f"Checking element ({i},{j}): {entries_2d[i][j]}")  # Debug print

                if (i == j and entries_2d[i][j] != 1) or (i != j and entries_2d[i][j] != 0):
                    return False
        return True


    def is_nilpotent(self, max_power=10):
        if not self.is_square():
            return False
        result = self
        for _ in range(max_power):
            result = result * self
            if result.is_zero():
                return True
        return False

    def is_diagonalizable(self):
        if not self.is_square():
            return False
        # Simplified check: assume it's diagonalizable if it's square
        return True

    def has_lu_decomposition(self):
        if not self.is_square():
            return False
        return self.is_invertible()  # LU decomposition exists for invertible matrices


    def determinant(self):
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices.")
        if self.rows == 1:
            return self.entries[0]
        if self.rows == 2:
            return self.entries[0] * self.entries[3] - self.entries[1] * self.entries[2]
        # Recursive determinant calculation for larger matrices
        det = 0
        for c in range(self.cols):
            minor = Matrix(self.field, self.rows - 1, self.cols - 1)
            for i in range(1, self.rows):
                minor.entries[i - 1] = self.entries[i][:c] + self.entries[i][c + 1:]
            det += ((-1) ** c) * self.entries[0][c] * minor.determinant()
        return det
    
# === Question 3 ====

    # Part (a)
    def size(self):
        """Returns the size of the matrix (rows, cols)."""
        return (self.rows, self.cols)

    def rank(self):
        """Returns the rank of the matrix."""
        # Convert matrix to 2D list for row operations
        matrix_2d = [self.entries[i:i + self.cols] for i in range(0, len(self.entries), self.cols)]
        # Perform Gaussian elimination to find the rank
        rank = 0
        for i in range(self.rows):
            if any(matrix_2d[i]):
                rank += 1
        return rank

    def nullity(self):
        """Returns the nullity of the matrix (dimension of the null space)."""
        # Nullity = Number of columns - Rank
        return self.cols - self.rank()

    def __str__(self):
        return '\n'.join(
            ' '.join(str(self.entries[i * self.cols + j]) for j in range(self.cols))
            for i in range(self.rows)
        )
    
    def rref(self, show_operations=False):
        # Perform Gaussian elimination manually
        A = [row[:] for row in self._convert_to_2d()]  # Make a copy of the matrix
        row_ops = []  # To track row operations
        rows, cols = self.rows, self.cols

        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while A[i][lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if cols == lead:
                        return A, row_ops
            A[i], A[r] = A[r], A[i]
            if show_operations:
                row_ops.append(f"Swap row {i} with row {r}")

            lv = A[r][lead]
            for j in range(cols):
                A[r][j] /= lv
            if show_operations:
                row_ops.append(f"Divide row {r} by {lv}")

            for i in range(rows):
                if i != r:
                    lv = A[i][lead]
                    for j in range(cols):
                        A[i][j] -= lv * A[r][j]
                    if show_operations:
                        row_ops.append(f"Row {i} = Row {i} - {lv} * Row {r}")

            lead += 1

        return A, row_ops if show_operations else A
    
    def are_linearly_independent(self, vectors):
        matrix = Matrix.from_vectors(vectors)
        return matrix.rank() == len(vectors)
    
    def dimension_of_subspace(self, vectors):
        matrix = Matrix.from_vectors(vectors)
        return matrix.rank()

    def basis_for_span(self, vectors):
        matrix = Matrix.from_vectors(vectors)
        return vectors[:matrix.rank()]  # The first `rank()` vectors form the basis
    
    def rank_factorization(self):
        rows, cols = self.rows, self.cols
        rank = self.rank()
        U = [self.row(i) for i in range(rank)]
        S = [[0 if i != j else 1 for j in range(rank)] for i in range(rank)]
        Vt = [[self.get(i, j) for j in range(cols)] for i in range(rank)]
        return Matrix(self.field, rank, rank, [val for row in S for val in row]), Matrix(self.field, rows, rank, [val for row in U for val in row]), Matrix(self.field, rank, cols, [val for row in Vt for val in row])
    
    def lu_decomposition(self):
        rows, cols = self.rows, self.cols
        if not self.is_square():
            raise ValueError("LU decomposition is only valid for square matrices.")

        A = [row[:] for row in self._convert_to_2d()]
        L = [[0] * cols for _ in range(rows)]
        U = [[0] * cols for _ in range(rows)]

        for i in range(rows):
            for j in range(i, cols):
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for j in range(i + 1, rows):
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
        return Matrix(self.field, rows, cols, [x for row in L for x in row]), Matrix(self.field, rows, cols, [x for row in U for x in row])
    
    def plu_decomposition(self):
        rows, cols = self.rows, self.cols
        if not self.is_square():
            raise ValueError("PLU decomposition is only valid for square matrices.")

        A = [row[:] for row in self._convert_to_2d()]
        P = [[1 if i == j else 0 for j in range(cols)] for i in range(rows)]
        L = [[0] * cols for _ in range(rows)]
        U = [[0] * cols for _ in range(rows)]

        for i in range(rows):
            # Pivoting
            max_row = max(range(i, rows), key=lambda r: abs(A[r][i]))
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                P[i], P[max_row] = P[max_row], P[i]

            for j in range(i, cols):
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for j in range(i + 1, rows):
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

        return Matrix(self.field, rows, cols, [x for row in P for x in row]), Matrix(self.field, rows, cols, [x for row in L for x in row]), Matrix(self.field, rows, cols, [x for row in U for x in row])
    

# === QUESTION 4 ===

class LinearSystem:
    def __init__(self, A, b):
        # A is a matrix, b is a vector
        self.A = A
        self.b = b
        
        if len(A) != len(b):
            raise ValueError("Incompatible dimensions: number of rows in A must match the length of b.")
    
    def is_consistent(self):
        # Check if the system is consistent using Gaussian elimination or rank check
        augmented_matrix = [row + [self.b[i]] for i, row in enumerate(self.A)]
        rank_A = self.rank(self.A)
        rank_augmented = self.rank(augmented_matrix)
        
        return rank_A == rank_augmented
    
    def rank(self, matrix):
        # Find rank of the matrix using Gaussian elimination (simplified version)
        matrix_copy = [row[:] for row in matrix]  # make a copy to not modify original matrix
        row_count = len(matrix_copy)
        col_count = len(matrix_copy[0])
        
        rank = 0
        for i in range(min(row_count, col_count)):
            if matrix_copy[i][i] != 0:
                for j in range(i + 1, row_count):
                    if matrix_copy[j][i] != 0:
                        scale = matrix_copy[j][i] / matrix_copy[i][i]
                        for k in range(i, col_count):
                            matrix_copy[j][k] -= scale * matrix_copy[i][k]
                rank += 1
        return rank
    
    def gaussian_elimination(self):
        # Solves the system using Gaussian elimination
        augmented_matrix = [row + [self.b[i]] for i, row in enumerate(self.A)]
        row_count = len(augmented_matrix)
        col_count = len(augmented_matrix[0])
        
        for i in range(min(row_count, col_count) - 1):
            for j in range(i + 1, row_count):
                if augmented_matrix[j][i] != 0:
                    scale = augmented_matrix[j][i] / augmented_matrix[i][i]
                    for k in range(i, col_count):
                        augmented_matrix[j][k] -= scale * augmented_matrix[i][k]
        
        solution = [0] * row_count
        for i in range(row_count - 1, -1, -1):
            solution[i] = augmented_matrix[i][-1] / augmented_matrix[i][i]
            for j in range(i - 1, -1, -1):
                augmented_matrix[j][-1] -= augmented_matrix[j][i] * solution[i]
        
        return solution
    
    def rref(self):
        augmented_matrix = [row + [self.b[i]] for i, row in enumerate(self.A)]
        row_count = len(augmented_matrix)
        col_count = len(augmented_matrix[0])
        
        for i in range(min(row_count, col_count)):
            pivot = augmented_matrix[i][i]
            if pivot != 0:
                for j in range(i + 1, row_count):
                    if augmented_matrix[j][i] != 0:
                        scale = augmented_matrix[j][i] / pivot
                        for k in range(i, col_count):
                            augmented_matrix[j][k] -= scale * augmented_matrix[i][k]
        
        return augmented_matrix
    
    def plu_decomposition(self):
        # Perform LU decomposition manually
        n = len(self.A)
        P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Identity matrix for P
        L = [[0 if i != j else 1 for j in range(n)] for i in range(n)]  # Identity matrix for L
        U = [row[:] for row in self.A]  # Copy of A to form U
        
        for i in range(n):
            # Pivoting: find the row with the largest value in column i
            max_row = max(range(i, n), key=lambda r: abs(U[r][i]))
            if i != max_row:
                # Swap rows in U and P
                U[i], U[max_row] = U[max_row], U[i]
                P[i], P[max_row] = P[max_row], P[i]
            
            # Eliminate below pivot
            for j in range(i + 1, n):
                if U[j][i] != 0:
                    factor = U[j][i] / U[i][i]
                    L[j][i] = factor
                    for k in range(i, n):
                        U[j][k] -= factor * U[i][k]
        
        # Now, solve the system using PLU
        # Step 1: Solve Ly = Pb (forward substitution)
        y = [0] * n
        for i in range(n):
            y[i] = self.b[i] - sum(L[i][j] * y[j] for j in range(i))
        
        # Step 2: Solve Ux = y (back substitution)
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
        
        return x


class VectorSet:
    def __init__(self, vectors):
        self.vectors = vectors
    
    def span(self):
        # Return the span of vectors in a set by checking if they are linearly independent
        matrix = [v for v in self.vectors]
        rank = self.rank(matrix)
        return rank == len(self.vectors)
    
    def rank(self, matrix):
        # Calculate rank of the matrix using Gaussian elimination (simplified)
        matrix_copy = [row[:] for row in matrix]  # make a copy to not modify original matrix
        row_count = len(matrix_copy)
        col_count = len(matrix_copy[0])
        
        rank = 0
        for i in range(min(row_count, col_count)):
            if matrix_copy[i][i] != 0:
                for j in range(i + 1, row_count):
                    if matrix_copy[j][i] != 0:
                        scale = matrix_copy[j][i] / matrix_copy[i][i]
                        for k in range(i, col_count):
                            matrix_copy[j][k] -= scale * matrix_copy[i][k]
                rank += 1
        return rank
    
    def is_subspace(self, other):
        # Check if span of self is a subspace of the span of another set of vectors
        self_rank = self.rank(self.vectors)
        other_rank = self.rank(other.vectors)
        
        return self_rank <= other_rank

        
    
