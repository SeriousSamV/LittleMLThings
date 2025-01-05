import math
from typing import List

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), 'vectors must be the same length'

    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), 'vectors must be the same length'

    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, 'no vectors provided!'
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), 'vectors must all be same length'

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


assert vector_sum([[1, 2], [3, 4], [5, 6]]) == [9, 12]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), 'vectors must be same length'

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32


def sum_of_squares(v: Vector) -> float:
    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14


def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v))


assert magnitude([3, 4]) == 5


def squared_distance(v: Vector, w: Vector) -> float:
    return sum_of_squares(subtract(v, w))


assert squared_distance([5, 7], [2, 11]) == 25


def distance(v: Vector, w: Vector) -> float:
    return math.sqrt(squared_distance(v, w))


assert distance([5, 7], [2, 11]) == 5.0

Matrix = List[Vector]


def shape(A: Matrix) -> tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)


def get_row(A: Matrix, i: int) -> Vector:
    return A[i]


assert get_row([[1, 2, 3], [4, 5, 6]], 0) == [1, 2, 3]


def get_column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]


assert get_column([[1, 2, 3], [4, 5, 6]], 0) == [1, 4]


def make_matrix(num_rows: int, num_cols: int, entry_fn: callable) -> Matrix:
    return [[entry_fn(i, j) for i in range(num_cols)] for j in range(num_rows)]


assert make_matrix(2, 3, lambda i, j: i + j) == [[0, 1, 2], [1, 2, 3]]


def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
