class MatrixMultiplication:
    def matrix_multiplication(n, A, B):
        if n <= 0:
            return []
        if len(A) != n or len(B) != n:
            raise ValueError("Matrix dimensions do not match the expected n x n size.")
        if any(len(row) != n for row in A) or any(len(row) != n for row in B):
            raise ValueError("Both matrices must be n x n.")

        C = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                total = A[i][0] * B[0][j]
                for k in range(1, n):
                    total += A[i][k] * B[k][j]
                C[i][j] = total
        return C



class StrassensAlgorithm:
    def _matrix_add(A, B):
        return [
            [A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))
        ]


    def _matrix_subtract(A, B):
        return [
            [A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))
        ]


    def _pad(matrix, size):
        # zero of the entries' own type, so padding doesn't change the matrix's number type
        zero = type(matrix[0][0])(0)
        extra = size - len(matrix)
        return [row + [zero] * extra for row in matrix] + [[zero] * size for _ in range(extra)]


    def _trim(matrix, n):
        return [row[:n] for row in matrix[:n]]


    def strassens_algorithm(n, A, B):
        # handling unexpected cases
        if n <= 0:
            return []
        if n & (n - 1):
            raise ValueError("Strassen's algorithm requires n to be a power of 2.")
        if len(A) != n or len(B) != n:
            raise ValueError("Matrix dimensions do not match the expected n x n size.")
        if any(len(row) != n for row in A) or any(len(row) != n for row in B):
            raise ValueError("Both matrices must be n x n.")

        # base case
        if n == 1:
            return [[A[0][0] * B[0][0]]]

        return StrassensAlgorithm.strassen_step(n, A, B, StrassensAlgorithm.strassens_algorithm)


    def strassen_step(n, A, B, multiply):
        # one level of Strassen; multiply(half, X, Y) computes the 7 half-size products

        # local aliases for readability
        add = StrassensAlgorithm._matrix_add
        subtract = StrassensAlgorithm._matrix_subtract

        # splitting n x n matrix into n/2 x n/2 submatrices
        half = n // 2
        def split(matrix):
            top = [row[:half] for row in matrix[:half]]
            bottom = [row[:half] for row in matrix[half:]]
            left = [row[:half] for row in matrix[:half]]
            right = [row[half:] for row in matrix[:half]]
            return {
                "A11": [row[:half] for row in matrix[:half]],
                "A12": [row[half:] for row in matrix[:half]],
                "A21": [row[:half] for row in matrix[half:]],
                "A22": [row[half:] for row in matrix[half:]],
            }

        A_parts = split(A)
        B_parts = split(B)

        A11, A12, A21, A22 = (
            A_parts["A11"],
            A_parts["A12"],
            A_parts["A21"],
            A_parts["A22"],
        )
        B11, B12, B21, B22 = (
            B_parts["A11"],
            B_parts["A12"],
            B_parts["A21"],
            B_parts["A22"],
        )

        # A = [[a, b], [c, d]], B = [[e, g], [f, h]]
        # multiplications on submatrices
        # p1 = a(g - h)
        p1 = multiply(half, A11, subtract(B12, B22))
        # p2 = (a + b)h
        p2 = multiply(half, add(A11, A12), B22)
        # p3 = (c + d)e
        p3 = multiply(half, add(A21, A22), B11)
        # p4 = d(f - e)
        p4 = multiply(half, A22, subtract(B21, B11))
        # p5 = (a + d)(e + h)
        p5 = multiply(half, add(A11, A22), add(B11, B22))
        # p6 = (b - d)(f + h)
        p6 = multiply(half, subtract(A12, A22), add(B21, B22))
        # p7 = (a - c)(e + g)
        p7 = multiply(half, subtract(A11, A21), add(B11, B12))

        # addition/subtractions to compute final quadrants
        # r = p5 + p4 - p2 + p6
        r = add(subtract(add(p5, p4), p2), p6)
        # s = p1 + p2
        s = add(p1, p2)
        # t = p3 + p4
        t = add(p3, p4)
        # u = p5 + p1 - p3 - p7
        u = subtract(add(p5, p1), add(p3, p7))

        C = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(half):
            for j in range(half):
                C[i][j] = r[i][j]
                C[i][j + half] = s[i][j]
                C[i + half][j] = t[i][j]
                C[i + half][j + half] = u[i][j]
        return C



class HybridAlgorithm:
    def hybrid_algorithm(n, A, B, h):
        # direct method once the matrices are 2^h x 2^h or smaller, Strassen above that
        if n <= 2 ** h:
            return MatrixMultiplication.matrix_multiplication(n, A, B)

        def multiply(half, X, Y):
            return HybridAlgorithm.hybrid_algorithm(half, X, Y, h)

        return StrassensAlgorithm.strassen_step(n, A, B, multiply)



class StaticPaddingStrassen:
    def static_padding_strassen(n, A, B):
        # pad once to the smallest 2^k >= n, run Strassen, keep the leading n x n block
        if n <= 0:
            return []
        size = 1 << (n - 1).bit_length()
        if size == n:
            return StrassensAlgorithm.strassens_algorithm(n, A, B)
        C = StrassensAlgorithm.strassens_algorithm(
            size, StrassensAlgorithm._pad(A, size), StrassensAlgorithm._pad(B, size)
        )
        return StrassensAlgorithm._trim(C, n)



class DynamicPaddingStrassen:
    def dynamic_padding_strassen(n, A, B):
        # split normally when n is even; when n is odd, add one zero row and column first
        if n <= 0:
            return []
        if n == 1:
            return [[A[0][0] * B[0][0]]]
        if n % 2 == 1:
            C = DynamicPaddingStrassen.dynamic_padding_strassen(
                n + 1, StrassensAlgorithm._pad(A, n + 1), StrassensAlgorithm._pad(B, n + 1)
            )
            return StrassensAlgorithm._trim(C, n)
        return StrassensAlgorithm.strassen_step(
            n, A, B, DynamicPaddingStrassen.dynamic_padding_strassen
        )
