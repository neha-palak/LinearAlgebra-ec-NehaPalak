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

