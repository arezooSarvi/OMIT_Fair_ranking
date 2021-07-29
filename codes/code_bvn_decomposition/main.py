from bvn_decomposition import *
from functools import partial
from sample_distributions import *
from outlier_metrics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AnalyzeDecomposition:
    def __init__(
        self, policy, observable_scores, name_method="not specified", print_metrics=True
    ):
        self.scores = observable_scores
        self.policy = policy
        self.name_method = name_method
        self.print_metrics = print_metrics

    def get_metrics(self):
        std = []
        max_outlieriness_element = []
        for coefficient, matrix in self.policy:
            # get a list with all items in the top k ranking
            top_k = [i for i in range(len(matrix)) if sum(matrix[i]) == 1]

            # get score of all items in the list
            scores = [self.scores[i] for i in top_k]

            # calculate outlieriness of element furthest away from mean
            max_outlieriness_element.append(measure_outlierness(scores))

            # add std of this ranking to list
            std.append(np.std(scores))
        # TODO: Mean and median should be weighted.
        output_dict = {
            "Number of permutationmatrices": len(std),
            "median_std": np.median(std),
            "mean_std": np.mean(std),
            "max_std": np.max(std),
            "min_std": np.min(std),
            "expected number of outliers": expected_number_of_outliers(
                self.policy, self.scores, outlier_definition="std"
            ),
            "probability of showing an outlier": probability_displayed_outlier_matrix(
                self.policy, self.scores, outlier_definition="std"
            ),
        }
        self.metrics = output_dict
        if self.print_metrics:
            print(output_dict)
        output_dict["name_method"] = self.name_method
        return output_dict


class MetricDataFrame:
    def __init__(self, observable_scores):
        self.observable_scores = observable_scores
        self.data_frame = None

    def update_data_frame(self, policy, name_method, print_metrics=True):
        analyze = AnalyzeDecomposition(
            observable_scores=self.observable_scores,
            policy=policy,
            name_method=name_method,
            print_metrics=print_metrics,
        )
        analyze.get_metrics()
        if self.data_frame is None:
            columns = [analyze.metrics.keys]
            self.data_frame = pd.DataFrame(columns=columns)
        self.data_frame = self.data_frame.append(analyze.metrics, ignore_index=True)

    def plot_data_frame(self, columns=None):
        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        df = self.data_frame
        if columns is not None:
            df = df[columns]
        df = df.round(decimals=4)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        fig.tight_layout()

        plt.show()


import time

start_time = time.time()

if __name__ == "__main__":
    m, n = 100, 10

    # For the simulated experiments we sample the scores from a distribution.
    # In the initial experiments we were using  the gaussian or powerlaw distributon.
    scores = sample_normal_distribution(sample_size=m, std=10, mean=100)

    # We also need to sample a stochastic matrix that simulates the output of a fair ranking algorithm.
    E = example_stochastic_matrix(set_length=m, num_permutations=400, k=n)

    # We will collect the results of each decomposition in the form of a variety of metrics in the MetricDataFrame.
    metrics_df = MetricDataFrame(observable_scores=scores)
    # Since we are also interested in the efficiency of our algorithm we also keep the runtimes.
    runtimes = []

    # For the experiments we will probably want for now compare the baseline model and the one with
    # heuristic=partial(measure_outlierness, method="max").

    start_time = time.time()

    #### Baseline ####
    print(
        "First we run the vanilla BvN decomposition without using a heuristic to chose the next item candidates:"
    )
    b = birkhoff_von_neumann_decomposition(E, scores=scores)
    metrics_df.update_data_frame(policy=b, name_method="vanilla", print_metrics=True)

    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    #### Baseline with sorted scores ####
    print("Vanilla BvN decomposition with sorted scores")
    s = birkhoff_von_neumann_decomposition(E, scores=scores, sort_scores=True)
    metrics_df.update_data_frame(
        policy=s, name_method="vanilla_sorted", print_metrics=True
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    #### Baseline with reversely sorted scores ####
    print("Vanilla BvN decomposition with reversly sorted scores:")
    s = birkhoff_von_neumann_decomposition(E, scores=scores, sort_scores="reverse_sort")
    metrics_df.update_data_frame(
        policy=s, name_method="vanilla_sorted_inverse", print_metrics=True
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    ### Maximize std heuristic ####
    print("We greedily add the items that maximize std when constructing the ranking: ")
    a = birkhoff_von_neumann_decomposition(
        E,
        heuristic=partial(measure_outlierness, method="inverse_std"),
        re_search_outliers=False,
        scores=scores,
    )
    metrics_df.update_data_frame(
        policy=a, name_method="maximize std heuristic", print_metrics=True
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    ### Minimize std heuristic ####
    print("We greedily add the items that minimize std when constructing the ranking: ")
    a = birkhoff_von_neumann_decomposition(
        E,
        heuristic=partial(measure_outlierness, method="std"),
        re_search_outliers=False,
        scores=scores,
    )
    metrics_df.update_data_frame(
        policy=a, name_method="minimize std heuristic", print_metrics=True
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    ### Count outliers heuristic ####
    print(
        "We use the count_outliers metric to sort the next item candidates by the amount of outliers the resulting list would have: "
    )
    a = birkhoff_von_neumann_decomposition(
        E, heuristic=count_outliers, re_search_outliers=False, scores=scores
    )
    metrics_df.update_data_frame(
        policy=a, name_method="count_outliers heuristic", print_metrics=True
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    #### Measure outlieriness by max outlier heuristic. Max method proved to be superior to average method.####
    print("We use the measure_outlierness metric to sort the next item candidates: ")
    c = birkhoff_von_neumann_decomposition(
        E,
        heuristic=partial(measure_outlierness, method="max"),
        re_search_outliers=False,
        scores=scores,
    )
    metrics_df.update_data_frame(
        policy=c, name_method="measure_outlierness heuristic", print_metrics=True
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    #### We run the same expreiment but add re-searching to remove even more outliers from the rankings. ####
    print("Add a re-search strategy to remove even more outliers from our rankings:")
    c_local = birkhoff_von_neumann_decomposition(
        E,
        heuristic=partial(measure_outlierness, method="max"),
        scores=scores,
        re_search_outliers=True,
        quality_measure=count_outliers,
    )
    metrics_df.update_data_frame(
        policy=c_local,
        name_method="measure_outlierness with researching",
        print_metrics=True,
    )
    runtime = time.time() - start_time
    print("--- %s seconds ---" % runtime)
    runtimes.append(runtime)
    start_time = time.time()

    # #### We run the same expreiment but instead of re-searching we use the local search strategy. ####
    # print("Add a local search strategy to remove even more outliers from our rankings:")
    # c_local = birkhoff_von_neumann_decomposition(
    #     E,
    #     heuristic=partial(measure_outlierness, method="max"),
    #     scores=scores,
    #     re_search_outliers=False,
    #     quality_measure=count_outliers,
    #     local_search=True,
    # )
    # metrics_df.update_data_frame(
    #     policy=c_local,
    #     name_method="measure_outlierness with local_search",
    #     print_metrics=True,
    # )
    # runtime = time.time() - start_time
    # print("--- %s seconds ---" % runtime)
    # runtimes.append(runtime)
    # start_time = time.time()
    #
    # #### We run the same expreiment but now with both re-searching and local search strategy. ####
    # print("Add a local search strategy and re-searching:")
    # c_local = birkhoff_von_neumann_decomposition(
    #     E,
    #     heuristic=partial(measure_outlierness, method="max"),
    #     scores=scores,
    #     re_search_outliers=True,
    #     quality_measure=count_outliers,
    #     local_search=True,
    # )
    # metrics_df.update_data_frame(
    #     policy=c_local,
    #     name_method="measure_outlierness with ls & rs",
    #     print_metrics=True,
    # )
    # runtime = time.time() - start_time
    # print("--- %s seconds ---" % runtime)
    # runtimes.append(runtime)
    # start_time = time.time()
    #
    # #### We use no heuristic but instead use the local search only. ####
    # print("Use only the re-search without using the heuristic:")
    # c_local = birkhoff_von_neumann_decomposition(
    #     E, scores=scores, re_search_outliers=True, quality_measure=count_outliers
    # )
    # metrics_df.update_data_frame(
    #     policy=c_local, name_method="vanilla with researching", print_metrics=True
    # )
    # runtime = time.time() - start_time
    # print("--- %s seconds ---" % runtime)
    # runtimes.append(runtime)

    metrics_df.data_frame["runtime"] = runtimes
    metrics_df.plot_data_frame(
        ["name_method", "median_std", "probability of showing an outlier", "runtime"]
    )
