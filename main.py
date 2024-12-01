from ques import ComplexNumber
from ques import Vector
from ques import Matrix

def main():
    # (a) ComplexNumber Examples
    print("=== Complex Numbers ===")
    c1 = ComplexNumber(3, 4)
    c2 = ComplexNumber(1, -2)
    print("Complex Addition:", c1 + c2)
    print("Complex Multiplication:", c1 * c2)
    print("Complex Division:", c1 / c2)
    print("Complex Absolute Value:", c1.abs())
    print("Complex Conjugate:", c1.conjugate())

    # (b) Vector Examples
    print("\n=== Vectors ===")
    v1 = Vector(float, 3, [1, 2, 3])
    v2 = Vector(float, 3, [4, 5, 6])
    print("Vector 1:", v1)
    print("Vector 2:", v2)
    print("Vector Addition:", v1 + v2)

    # (c) Matrix Initialization Examples
    print("\n=== Matrices ===")
    m1 = Matrix(float, 2, 2, [1, 2, 3, 4])
    m2 = Matrix(float, 2, 2, [5, 6, 7, 8])
    print("Matrix 1:")
    print(m1)
    print("Matrix 2:")
    print(m2)

    # (e) Addition and Multiplication of Matrices
    print("\nMatrix Addition:")
    print(m1 + m2)

    print("Matrix Multiplication:")
    print(m1 * m2)

    # (f) Specific Rows and Columns
    print("\nRow 1 of Matrix 1:", m1.row(0))
    print("Column 1 of Matrix 1:", m1.column(0))

    # (g) Transpose, Conjugate, and Transpose-Conjugate
    print("\nMatrix Transpose:")
    print(m1.transpose())
    print("Matrix Conjugate (Real Matrix):")
    print(m1.conjugate())
    print("Matrix Transpose-Conjugate:")
    print(m1.transpose_conjugate())


    #QUESTION 2
    # (a)
    m1 = Matrix(float, 2, 2, [1, 0, 0, 1])  # Identity Matrix

    print("\n=== Matrix Properties ===")
    print("Matrix:")
    print(m1)
    print("Is Zero Matrix?", m1.is_zero())
    print("Is Symmetric?", m1.is_symmetric())
    print("Is Hermitian?", m1.is_hermitian())
    print("Is Orthogonal?", m1.is_orthogonal())
    print("Is Unitary?", m1.is_unitary())
    print("Is Singular?", m1.is_singular())
    print("Is Invertible?", m1.is_invertible())
    print("Is Nilpotent?", m1.is_nilpotent())
    print("Has LU Decomposition?", m1.has_lu_decomposition())
    print("Is Scalar?", m1.is_scalar())
    print("Is Identity?", m1.is_identity())


if __name__ == "__main__":
    main()
