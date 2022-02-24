

from codes.src.algorithms.FOEIR.od_methods import COPOD_od, MAD_od, MedKnn_od, quartile_od


def calculateOutlierMetrics(k, ranking, context_size, od_method):
    """
    Calculate outlierness metrics

    @param k: truncation point/length of the ranking
    @param ranking: list of candidates selected for the ranking
    @param context_size: length of the items list we consider to calculate outlierness and detect outliers
    @param od_method: outlier detection method

    return outliers_count, outlierness: No. of outliers/outlierness of the list
    """

    if len(ranking) < k:
        k = len(ranking)

    outlier_index_original_ranking, outlierness_original_ranking, scores = detectOutliers(ranking, context_size,
                                                                                          od_method)
    outliers_count = sum(outlier_index_original_ranking[0:k])
    outlierness = sum(outlierness_original_ranking[0:k])

    return outliers_count, outlierness


def detectOutliers(ranking, context_size, od_method):
    scores = [int(i.features[-1]) for i in ranking[0:context_size]]
    if od_method == 'copod':
        outlier_index_original_ranking, outlierness_original_ranking, _ = COPOD_od(scores, return_list=True)
    if od_method == 'medknn':
        outlier_index_original_ranking, outlierness_original_ranking, _ = MedKnn_od(scores, return_list=True)
    if od_method == 'mad':
        outlier_index_original_ranking, outlierness_original_ranking, _ = MAD_od(scores, return_list=True)
    if od_method == 'quartile':
        outlier_index_original_ranking, outlierness_original_ranking = quartile_od(scores)
    outlierness_original_ranking = [i if outlier_index_original_ranking[id] else 0 for id, i in
                                    enumerate(outlierness_original_ranking)]
    return outlier_index_original_ranking, outlierness_original_ranking, scores

