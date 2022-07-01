"""Arithmetic tests for the multiplications"""
import unittest

import numpy as np
from multiplications import matmul_diagonal, matmul_hybrid


class TestArithmetic(unittest.TestCase):
    """Test class"""

    def test_matmul(self, n=100):
        """Matmul testcase for matmul_diagonal and matmul_hybrid"""
        for _ in range(n):
            a = np.random.randint(1, 40)
            b = np.random.randint(a, 50)
            matrix = np.random.random((a, b))
            vector = np.random.random((b,))
            assert sum(matmul_diagonal(matrix, vector) - matrix @ vector) < 1e-8
            if b % a == 0:
                assert sum(matmul_hybrid(matrix, vector) - matrix @ vector) < 1e-8
