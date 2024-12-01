from Q1 import Matrix

class Matrix:
    def __init__(self, field, rows, cols, entries=None):
        self.field = field
        self.rows = rows
        self.cols = cols
        self.entries = entries or [[0 for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, index):
        return self.entries[index]

    def __setitem__(self, index, value):
        self.entries[index] = value

    def __str__(self):
        return "\n".join(" ".join(f"{val:.2f}" for val in row) for row in self.entries)

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions do not match for addition.")
        result = Matrix(self.field, self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.entries[i][j] = self.entries[i][j] + other.entries[i][j]
        return result

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions do not match for multiplication.")
        result = Matrix(self.field, self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                result.entries[i][j] = sum(
                    self.entries[i][k] * other.entries[k][j] for k in range(self.cols)
                )
        return result

    def transpose(self):
        return Matrix(self.field, self.cols, self.rows, [[self.entries[j][i] for j in range(self.rows)] for i in range(self.cols)])

    def conjugate(self):
        return Matrix(self.field, self.rows, self.cols, [[val.conjugate() if isinstance(val, ComplexNumber) else val for val in row] for row in self.entries])

    def transpose_conjugate(self):
        return self.conjugate().transpose()

    # --- Property-checking functions ---
    def is_zero(self):
        return all(all(val == 0 for val in row) for row in self.entries)

    def is_square(self):
        return self.rows == self.cols

    def is_symmetric(self):
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.entries[i][j] != self.entries[j][i]:
                    return False
        return True

    def is_hermitian(self):
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.entries[i][j] != (self.entries[j][i].conjugate() if isinstance(self.entries[j][i], ComplexNumber) else self.entries[j][i]):
                    return False
        return True

    def is_orthogonal(self):
        if not self.is_square():
            return False
        identity = Matrix(self.field, self.rows, self.cols, [[1 if i == j else 0 for j in range(self.cols)] for i in range(self.rows)])
        transpose = self.transpose()
        product = transpose * self
        return product == identity

    def is_unitary(self):
        if not self.is_square():
            return False
        identity = Matrix(self.field, self.rows, self.cols, [[1 if i == j else 0 for j in range(self.cols)] for i in range(self.rows)])
        transpose_conjugate = self.transpose_conjugate()
        product = transpose_conjugate * self
        return product == identity

    def is_scalar(self):
        if not self.is_square():
            return False
        scalar_value = self.entries[0][0]
        for i in range(self.rows):
            for j in range(self.cols):
                if i == j and self.entries[i][j] != scalar_value:
                    return False
                if i != j and self.entries[i][j] != 0:
                    return False
        return True

    def is_singular(self):
        return not self.is_invertible()

    def is_invertible(self):
        if not self.is_square():
            return False
        determinant = self.determinant()
        return determinant != 0

    def is_identity(self):
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if (i == j and self.entries[i][j] != 1) or (i != j and self.entries[i][j] != 0):
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
        # Simplified check: a matrix is diagonalizable if it has distinct eigenvalues.
        # In a full implementation, we'd calculate the eigenvalues and check their multiplicity.
        return True  # Placeholder

    def has_lu_decomposition(self):
        if not self.is_square():
            return False
        # Simplified check: LU decomposition exists if the matrix is non-singular.
        return self.is_invertible()

    def determinant(self):
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices.")
        if self.rows == 1:
            return self.entries[0][0]
        if self.rows == 2:
            return self.entries[0][0] * self.entries[1][1] - self.entries[0][1] * self.entries[1][0]
        # Recursive determinant calculation for larger matrices
        det = 0
        for c in range(self.cols):
            minor = Matrix(self.field, self.rows - 1, self.cols - 1)
            for i in range(1, self.rows):
                minor.entries[i - 1] = self.entries[i][:c] + self.entries[i][c + 1:]
            det += ((-1) ** c) * self.entries[0][c] * minor.determinant()
        return det
