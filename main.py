from ques import ComplexNumber
from ques import Vector
from ques import Matrix
from ques import LinearSystem
from ques import VectorSet

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
    print("Length of vector 1:", v1.length)

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

    #QUESTION 3
    # Elementary Matrix Operations

    print("\n=== Elementary Matrix Operations ===")
    
    # LU Decomposition
    try:
        L, U = m1.lu_decomposition()
        print("LU Decomposition - L:")
        print(L)
        print("LU Decomposition - U:")
        print(U)
    except ValueError as e:
        print("Error in LU Decomposition:", e)

    # PLU Decomposition
    try:
        P, L, U = m1.plu_decomposition()
        print("PLU Decomposition - P:")
        print(P)
        print("PLU Decomposition - L:")
        print(L)
        print("PLU Decomposition - U:")
        print(U)
    except ValueError as e:
        print("Error in PLU Decomposition:", e)

    # Reduced Row Echelon Form (RREF)
    rref, row_ops = m1.rref()
    print("Reduced Row Echelon Form (RREF):")
    print(rref)
    print("Row Operations to achieve RREF:")
    print(row_ops)

    # Rank and Nullity
    print("Size of m1:", m1.size())
    print("Rank of m1:", m1.rank())
    print("Nullity of m1:", m1.nullity())


# QUESTION 4
# System of Linear Equations

    # (a) System of Linear Equations Example
    A = [[2, 3, -1], [4, -1, 2], [1, 2, 3]]
    b = [5, 6, 7]
    system = LinearSystem(A, b)
    
    print("Is the system consistent?", system.is_consistent())
    if system.is_consistent():
        solution = system.gaussian_elimination()
        print("Solution using Gaussian Elimination:", solution)
    
    # (b) Subspace Example
    S1 = [[1, 0], [0, 1]]
    S2 = [[1, 1], [1, 0]]
    set1 = VectorSet(S1)
    set2 = VectorSet(S2)
    print("Is span of S1 a subspace of span of S2?", set1.is_subspace(set2))
    
    # (c) Augmented matrix and RREF solution
    print("Augmented matrix and RREF:")
    print(system.rref())
    
    # (d) PLU Decomposition for solving system
    print("Solution using PLU Decomposition:")
    solution_plu = system.plu_decomposition()
    print(solution_plu)


# QUESTION 5
# Invertible Matrices operations
    # (a) Check if the matrix is square and invertible
    print("=== Matrix Operations ===")
    m1 = Matrix(float, 3, 3, [2, 1, 1, 1, 3, 2, 1, 0, 3])
    print("Matrix 1:")
    print(m1)
    print("Is Matrix 1 square?", m1.is_square())
    print("Is Matrix 1 invertible?", m1.is_invertible())
    

    # (b) Inverse of Matrix by Row Reduction
    print("\nInverse of Matrix by Row Reduction:")
    inverse_row_reduction = m1.row_reduce()
    if inverse_row_reduction:
        print(inverse_row_reduction)
    else:
        print("Matrix is not invertible by row reduction.")

    # (c) Inverse of Matrix by Adjoint
    print("\nInverse of Matrix by Adjoint:")
    inverse_adjoint = m1.inverse_by_adjoint()
    if inverse_adjoint:
        print(inverse_adjoint)
    else:
        print("Matrix is not invertible by adjoint.")


 # QUESTION 6: Coordinates and Change of Basis
    print("\n=== Coordinates and Change of Basis ===")
    
    v1 = Vector(float, 2, [1, 0])
    v2 = Vector(float, 2, [0, 1])
    set_S = VectorSet([v1, v2])
    v3 = Vector(float, 2, [3, 4])
    print(f"Is vector {v3.coordinates} in the span of {[vec.coordinates for vec in set_S.vectors]}?")
    print(set_S.is_in_span(v3))

    print("Linear combination representation of v3:")
    print(set_S.linear_combination(v3))

    S1 = VectorSet([v1, v2])
    S2 = VectorSet([v1, Vector(float, 2, [1, 1])])
    print("Do S1 and S2 span the same subspace?")
    print(VectorSet.span_equal(S1, S2))

    basis = [v1, v2]
    print("Coordinates of v3 in the basis:", set_S.coordinates_in_basis(basis, v3))
    coords = [3, 4]
    reconstructed_vector = set_S.vector_from_coordinates(basis, coords)
    print("Vector reconstructed from coordinates:", reconstructed_vector.coordinates)

    B1 = [v1, v2]
    B2 = [Vector(float, 2, [1, 1]), Vector(float, 2, [-1, 1])]
    print("Change of basis matrix from B1 to B2:")
    print(VectorSet.change_of_basis_matrix(B1, B2))

    print("Coordinates of v3 in B2:")
    print(VectorSet.change_coordinates(B1, B2, set_S.coordinates_in_basis(B1, v3)))



if __name__ == "__main__":
    main()