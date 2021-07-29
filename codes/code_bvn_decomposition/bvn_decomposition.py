import itertools
from networkx import from_numpy_matrix
from matching import hopcroft_karp_matching
from random_stochastic_matrix import *
from functools import partial, reduce
from outlier_metrics import *
from sample_distributions import *
import numpy as np
#: Any number smaller than this will be rounded down to 0 when computing the
#: difference between NumPy arrays of floats.
TOLERANCE = np.finfo(np.float).eps * 10.0


def to_pattern_matrix(D):
    """Returns the Boolean matrix in the same shape as `D` with ones exactly
    where there are nonzero entries in `D`.
    `D` must be a NumPy array.
    """
    result = np.zeros_like(D)
    # This is a cleverer way of doing
    #
    #     for (u, v) in zip(*(D.nonzero())):
    #         result[u, v] = 1
    #
    result[D.nonzero()] = 1
    return result


def zeros(m, n):
    """Convenience function for ``numpy.zeros((m, n))``."""
    return np.zeros((m, n))


def hstack(left, right):
    """Convenience function for ``numpy.hstack((left, right))``."""
    return np.hstack((left, right))


def vstack(top, bottom):
    """Convenience function for ``numpy.vstack((top, bottom))``."""
    return np.vstack((top, bottom))


def four_blocks(topleft, topright, bottomleft, bottomright):
    """Convenience function that creates a block matrix with the specified
    blocks.
    Each argument must be a NumPy matrix. The two top matrices must have the
    same number of rows, as must the two bottom matrices. The two left matrices
    must have the same number of columns, as must the two right matrices.
    """
    return vstack(hstack(topleft, topright), hstack(bottomleft, bottomright))


def to_bipartite_matrix(A):
    """Returns the adjacency matrix of a bipartite graph whose biadjacency
    matrix is `A`.
    `A` must be a NumPy array.
    If `A` has **m** rows and **n** columns, then the returned matrix has **m +
    n** rows and columns.
    """
    m, n = A.shape
    return four_blocks(zeros(m, m), A, A.T, zeros(n, n))


def to_permutation_matrix(matches, m, n):
    """Converts a permutation into a permutation matrix.
    `matches` is a dictionary whose keys are vertices and whose values are
    partners. For each vertex ``u`` and ``v``, entry (``u``, ``v``) in the
    returned matrix will be a ``1`` if and only if ``matches[u] == v``.
    Pre-condition: `matches` must be a permutation on an initial subset of the
    natural numbers.
    Returns a permutation matrix as a square NumPy array.
    """
    P = np.zeros((m, n))
    # This is a cleverer way of doing
    #
    #     for (u, v) in matches.items():
    #         P[u, v] = 1
    #
    targets = tuple(zip(*(matches.items())))
    P[targets] = 1
    return P


def quality_fn(indices, scores, quality_measure=count_outliers):
    q = quality_measure([scores[i] for i in indices])
    return q


def birkhoff_von_neumann_decomposition(
    D,
    heuristic=None,
    re_search_outliers=False,
    quality_measure=None,
    scores=None,
    reverse_heuristic=False,
    sort_scores=False,
    top_k=None,
    local_search=False,
):
    """SOURCE: https://github.com/jfinkels/birkhoff/blob/master/birkhoff.py. This uses just no heuristic."""
    m, n = D.shape
    right_nodes_counter = None
    if m < n:
        raise ValueError(
            "The number of items to rank must be at least the number of ranks. "
            + "The input matrix should have the ranks as columns and the items as rows"
        )
    if scores is None and heuristic is not None:
        raise AssertionError(
            "Please provide the scores of each item as a list, when using a heuristic."
        )

    elif (
        m > n
    ):  # For a non-squared matrix our current algorithm extends the matrix to a squared matrix
        right_nodes_counter = {column: 1 for column in range(m, m + n)}
        D = add_single_column_with_rest_probabilities(D)
        right_nodes_counter[m + n] = m - n
    if m > n:
        indices = list(itertools.product(range(m), range(n + 1)))
    elif m == n:
        indices = list(itertools.product(range(m), range(n)))
    # If a heuristic is given the quality function is used to evaluate a set of items (leftnodes) with respect to their score
    quality = (
        None
        if heuristic is None
        else partial(quality_fn, quality_measure=heuristic, scores=scores)
    )

    if top_k is not None:
        n = top_k
    # the sort_scores approach is a baseline approach, where the rows of the input
    # matrix are scored wrt the scores before applying the BvN decomposition
    if sort_scores:
        L = [(scores[i], i) for i in range(len(scores))]
        if sort_scores == "reverse_sort":
            L.sort(reverse=True)
        else:
            L.sort()
        sorted_scores, sort_permutation = zip(*L)
        sort_permutation = list_to_permutation_matrix(sort_permutation)
        scores = sort_permutation.dot(scores)
        D = sort_permutation.dot(D)

    # These two lists store the coefficients and matrices that we iteratively split off with the BvN decomposition
    coefficients = []
    permutations = []

    # Create a copy of D so that we don't modify it directly. Cast the
    # entries of the matrix to floating point numbers, regardless of
    # whether they were integers.
    S = D.astype("float")
    outlier_permutations = []
    outlier_coefficients = []
    while not np.allclose(S, 0):
        # Create an undirected graph whose adjacency matrix contains a 1
        # exactly where the matrix S has a nonzero entry.
        W = to_pattern_matrix(S)

        # Construct the bipartite graph whose left and right vertices both
        # represent the vertex set of the pattern graph (whose adjacency matrix
        # is ``W``).
        X = to_bipartite_matrix(W)
        # Mathijs: X is the adjecency matrix of the bipartite graph as graph

        # Convert the matrix of a bipartite graph into a NetworkX graph object.
        G = from_numpy_matrix(X)

        # Compute a perfect matching for this graph. The dictionary `M` has one
        # entry for each matched vertex (in both the left and the right vertex
        # sets), and the corresponding value is its partner.
        #
        # The bipartite maximum matching algorithm requires specifying
        # the left set of nodes in the bipartite graph. By construction,
        # the left set of nodes is {0, ..., n - 1} and the right set is
        # {n, ..., 2n - 1}; see `to_bipartite_matrix()`.
        left_nodes = range(m)

        M = hopcroft_karp_matching(
            G,
            quality,
            left_nodes,
            reverse=reverse_heuristic,
            local_search=local_search,
            top_k=n,
            scores=scores,
            right_nodes_counter=right_nodes_counter,
        )

        # However, since we have both a left vertex set and a right vertex set,
        # each representing the original vertex set of the pattern graph
        # (``W``), we need to convert any vertex greater than ``n`` to its
        # original vertex number. To do this,
        #
        #   - ignore any keys greater than ``n``, since they are already
        #     covered by earlier key/value pairs,
        #   - ensure that all values are less than ``n``.
        #

        M = {u: v - m for u, v in M.items() if u < m}
        # Convert that perfect matching to a permutation matrix.

        P = to_permutation_matrix(M, m, m)
        P = P[:, : n + 1]
        # TODO check for the number of outliers of the ranking. If there are outliers do some local search
        # Get the smallest entry of S corresponding to the 1 entries in the
        # permutation matrix.

        q = min(S[i, j] for (i, j) in indices if P[i, j] == 1)
        # Determine the rows with a non-zero entry in the first n columns. These correspond to the items in the top-k
        i = [j for j, row in enumerate(P) if sum(row[:n]) > 0]
        # If the re_search_outliers argument is passed True we group the permutation matrices into outlier and
        # not outlier containing sets.

        quality_fn(indices=i, scores=scores, quality_measure=quality_measure)
        if (
            re_search_outliers
            and quality_fn(indices=i, scores=scores, quality_measure=quality_measure)
            >= 1
        ):
            # Store the coefficient and the permutation matrix for later.
            outlier_permutations.append(P[:, :n])
            outlier_coefficients.append(q)
        else:
            coefficients.append(q)
            permutations.append(P[:, :n])
        # Subtract P scaled by q. After this subtraction, S has a zero entry
        # where the value q used to live.
        S -= q * P
        # PRECISION ISSUE: There seems to be a problem with floating point
        # precision here, so we need to round down to 0 any entry that is very
        # small.
        S[np.abs(S) < TOLERANCE] = 0.0

    # If the re_search_outliers is passed as True we aggregate the previously grouped rankings with outliers
    # to a new matrix that we input into another iteration of the BvN decomposition algorithm.
    if re_search_outliers:
        # if the decomposition did not find any lists without outliers we do not retry splitting off matrices.
        if coefficients:
            # If the decomposition did not find any litst with outlier we return the decomposition
            if not outlier_permutations:
                decomp = list(zip(coefficients, permutations))
                decomp = [(c, mat[:, :n]) for c, mat in list(decomp)]
                return decomp
            # We recombine the outlier matrices.
            outliers = [
                q * P for q, P in zip(outlier_coefficients, outlier_permutations)
            ]
            # print(
            #     "amount of lists wih outliers len(outliers):",
            #     len(outliers),
            #     "without_outliers: ",
            #     len(coefficients),
            #     "remaining_weight: ",
            #     sum(outlier_coefficients),
            #     sum(coefficients),
            # )
            outliers_sum = reduce(lambda x, y: x + y, outliers)
            # Since our algorithm does not actually sample randomly but in order
            # we permute the matrix to get a different decomposition
            permut = (random_permutation_matrix(m), random_permutation_matrix(n))
            scores_perm = permut[0].dot(scores)
            outliers_sum = permut[0].dot(outliers_sum[:, :n]).dot(permut[1])
            # call the bvn algorithm on this matrix
            rest_decomposition = birkhoff_von_neumann_decomposition(
                outliers_sum,
                heuristic=heuristic,
                quality_measure=quality_measure,
                scores=scores_perm,
                re_search_outliers=True,
                reverse_heuristic=reverse_heuristic,
                top_k=n,
                local_search=local_search,
            )
            # Permute all matrices in the decomposition back
            rest_decomposition = [
                (c, permut[0].T.dot(mat).dot(permut[1].T))
                for c, mat in list(rest_decomposition)
            ]
            decomp = list(zip(coefficients, permutations)) + rest_decomposition
            # We are only interested in the first n (top-k) columns of the permutation matrices
            decomp = [(c, mat[:, :n]) for c, mat in list(decomp)]
            return decomp
        decomp = list(zip(outlier_coefficients, outlier_permutations))
        decomp = [(c, mat[:, :n]) for c, mat in list(decomp)]
        return decomp
    decomp = list(zip(coefficients, permutations)) + list(
        zip(outlier_coefficients, outlier_permutations)
    )
    if sort_scores:
        # permute back for evaluation:
        decomp = [(c, sort_permutation.T.dot(mat)) for c, mat in list(decomp)]
    decomp = [(c, mat[:, :n]) for c, mat in list(decomp)]
    return decomp


def assert_decomposed_correctly(matrix, decomposition):
    X = np.zeros(decomposition[0][1].shape)
    for coeff, pmatrix in decomposition:
        X += coeff * pmatrix
    assert np.allclose(matrix, X)


def assert_permutation_matrix(decomposition):
    for coef, mat in decomposition:
        # Check that each matrix in the decomposition is a permutation matrix
        assert np.allclose(
            [sum([row[j] for row in mat]) for j in range(len(mat[0]))],
            np.ones(len(mat[0])),
        )
    print("Columns of permutation matrices sum to 1")
    # TODO add check for only ones and 0 used in matrix


def assert_coefficients_sum_to_1(decomposition):
    assert np.isclose(sum([coeff for coeff, _ in decomposition]), 1)
    print("Coefficients sum to 1.")


def test_bvn(set_length=20, num_permutations=20, k=10):
    E = example_stochastic_matrix(
        set_length=set_length, num_permutations=num_permutations, k=k
    )
    scores = sample_normal_distribution(sample_size=set_length)

    # Vanilla BvN
    a = birkhoff_von_neumann_decomposition(E)
    assert_coefficients_sum_to_1(a)
    assert_permutation_matrix(a)
    assert_decomposed_correctly(E, a)
    print("Test Vanilla BvN successful")

    # Heuristic
    a = birkhoff_von_neumann_decomposition(E, heuristic=count_outliers, scores=scores)
    assert_coefficients_sum_to_1(a)
    assert_permutation_matrix(a)
    assert_decomposed_correctly(E, a)
    print("Test BvN with heuristic successful")

    # Re-Searching
    a = birkhoff_von_neumann_decomposition(
        E, scores=scores, re_search_outliers=True, quality_measure=count_outliers
    )
    assert_coefficients_sum_to_1(a)
    X = np.zeros(a[0][1].shape)
    for coeff, pmatrix in a:
        X += coeff * pmatrix
    assert_decomposed_correctly(E, a)
    assert_permutation_matrix(a)
    print("Test BvN with re-searching successful")

    # Local Search
    a = birkhoff_von_neumann_decomposition(
        E,
        scores=scores,
        heuristic=count_outliers,
        local_search=True,
        re_search_outliers=True,
        quality_measure=count_outliers,
    )
    assert_coefficients_sum_to_1(a)
    assert_permutation_matrix(a)
    assert_decomposed_correctly(E, a)
    print("Test BvN with local search successful")


if __name__ == "__main__":
    test_bvn(set_length=30, num_permutations=70, k=10)
