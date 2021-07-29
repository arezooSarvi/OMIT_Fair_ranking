import numpy as np


def example_stochastic_matrix(set_length, num_permutations, k=10):
    """This will produce a stochastic matrix, where the sum of each column sums to 1, and the sum of each
    row is bounded by 1. If l=set_length, this function will return a doubly stochastic matrix. Each column
    represents a position, the columns represent the different items."""
    coefficients = np.random.random(num_permutations)
    coefficients = coefficients / np.sum(coefficients)
    sm = np.zeros([set_length, k])
    for coefficient in coefficients:
        P = random_permutation_matrix(set_length, top_k=k)
        sm = sm + coefficient * P
    return sm


def add_single_column_with_rest_probabilities(potentially_stochastic_matrix):
    potentially_stochastic_matrix = np.array(potentially_stochastic_matrix)
    column_sum = np.sum(potentially_stochastic_matrix, axis=0)[0]
    ones = np.ones(
        shape=(
            len(potentially_stochastic_matrix),
            1,
        )
    )
    rest_prob_rows = np.sum(potentially_stochastic_matrix, axis=1)
    rest_prob_rows = (column_sum * ones.T - rest_prob_rows * ones.T).T
    potentially_stochastic_matrix = np.concatenate(
        (potentially_stochastic_matrix, rest_prob_rows), axis=1
    )
    return potentially_stochastic_matrix


def test_add_single_column_with_rest_probabilities():
    mat = [[1 / 4, 1 / 2], [1 / 4, 0], [1 / 4, 1 / 3], [1 / 4, 1 / 6]]
    rest_added = add_single_column_with_rest_probabilities(mat)
    assert np.allclose(
        [
            [1 / 4, 1 / 2, 1 / 4],
            [1 / 4, 0, 3 / 4],
            [1 / 4, 1 / 3, 5 / 12],
            [1 / 4, 1 / 6, 7 / 12],
        ],
        rest_added,
    )


def list_to_permutation_matrix(permut_as_list, k=None):
    set_length = len(permut_as_list)
    if k is None:
        k = set_length

    top_k = np.arange(k)
    E = np.identity(set_length)  # initialize an identity matrix
    permutation = np.array([top_k, permut_as_list])  # butterfly permutation example
    P = np.zeros([set_length, k])  # initialize the permutation matrix
    for j in range(0, set_length):
        P[j] = E[permutation[1][j]][:k]  # Only consider positions up to k
    return P


def random_permutation_matrix(n, top_k=None):
    per = np.random.permutation(n)
    return list_to_permutation_matrix(per, top_k)


def inverse_permutation(matrix):
    return np.array(matrix).T


def test_inverse_permutation(n):
    permute = random_permutation_matrix(n)
    inverse_permute = inverse_permutation(permute)
    assert np.allclose(np.identity(n), permute.dot(inverse_permute))


if __name__ == "__main__":
    test_inverse_permutation(5)
    test_add_single_column_with_rest_probabilities()
