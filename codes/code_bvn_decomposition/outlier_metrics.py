import numpy as np


def count_outliers(list, alpha=2, method="std"):
    if method == "std":
        std = np.std(list)
        mean = np.mean(list)
        count = 0
        for i in list:
            if abs(i - mean) > alpha * std:
                count += 1
    elif method == "iqr":
        list = sorted(list)
        l = len(list)
        iqr = list[3 * (l // 4)] - list[l // 4]
        count = 0
        for i in list:
            if i > list[3 * (l // 4)] + alpha * iqr:
                count += 1
            elif i < list[(l // 4)] - alpha * iqr:
                count += 1
    return count


def test_count_outliers():
    list = [0, 0, 1000]
    print(count_outliers(list, method="iqr"))


def measure_outlierness(list, method="max"):
    std = np.std(list)
    mean = np.mean(list)
    list = [(abs((s - mean)) / std) for s in list]
    if method == "max":
        return max(list)
    if method == "min":
        return min(list)
    if method == "average":
        return np.mean(list)
    if method == "inverse_std":
        return 1 / std
    if method == "std":
        return std
    return 0


def probability_displayed_outlier_matrix(
    policy, observable_scores, outlier_definition="iqr"
):
    """Takes as an input a list of pairs of permutation matrix and probability coefficient
    and a list of observable scores and returns the probability that a matrix with an outlier is being displayed."""
    prob = 0
    for coefficient, matrix in policy:
        # get a list with all items in the top k ranking
        top_k = [i for i in range(len(matrix)) if sum(matrix[i]) == 1]
        # get score of all items in the list
        scores = [observable_scores[i] for i in top_k]
        # calculate whether the ranking has an outlier
        if count_outliers(scores, method=outlier_definition) > 0:
            # add the probability that this ranking is displayed to the total probability of getting an outlier list.
            prob += coefficient
    return prob


def test_probability_displayed_outlier_matrix_1():
    policy = [
        (0.4, [[1, 0], [0, 1], [0, 0]]),
        (0.35, [[0, 1], [0, 0], [1, 0]]),
        (0.25, [[0, 1], [1, 0], [0, 0]]),
    ]
    scores = [1, 2, 2]
    assert (
        probability_displayed_outlier_matrix(policy, scores, outlier_definition="std")
        == 0
    )


def test_probability_displayed_outlier_matrix_2():
    policy = [
        (0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        (0.35, [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]),
        (0.25, [[0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]),
    ]
    scores = [1, 1, 1, 1000000]
    assert probability_displayed_outlier_matrix(policy, scores) == 0.6


def expected_number_of_outliers(policy, observable_scores, outlier_definition="iqr"):
    """Takes as an input a list of pairs of permutation matrix and probability scalar and returns
    the expected number of outliers in a sampled ranking."""
    E = 0
    for coefficient, matrix in policy:
        # get a list with all items in the top k ranking
        top_k = [i for i in range(len(matrix)) if sum(matrix[i]) == 1]
        # get score of all items in the list
        scores = [observable_scores[i] for i in top_k]
        # calculate whether the ranking has an outlier
        E += coefficient * count_outliers(scores, method=outlier_definition)
    return E


def test_expected_number_of_outliers():
    policy = [
        (0.4, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
        (0.35, [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        (0.25, [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ]
    scores = [1, 1, 1, 1000000, 100]
    assert expected_number_of_outliers(policy, scores, outlier_definition="iqr") == 0.85


def approximate_expected_value_num_outliers(
    num_iterations, sample_size, sample_distribution_function, **kwargs
):
    outliers_count = []
    for _ in range(num_iterations):
        sample = sample_distribution_function(sample_size=sample_size, **kwargs)
        outliers_count.append(count_outliers(sample))
    return outliers_count


def get_average_std(policy, observable_scores):
    weighted_sum = 0
    for coefficient, ranking in policy:
        items_in_ranking = [i for i, item in enumerate(ranking) if sum(item) != 0]
        weighted_sum += coefficient * np.std(
            [observable_scores[i] for i in items_in_ranking]
        )
    return weighted_sum
