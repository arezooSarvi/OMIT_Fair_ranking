import os

import pandas as pd
from codes.utils.utils import Utils
#constants for algorithms

# H/S stands for hard/soft constraint for doubly stochastic matrix
ALGO_FOEIRDTC_H = 'FOEIR-DTC-H'
ALGO_FOEIRDTC_S = 'FOEIR-DTC-S'
ALGO_RO_H = 'RO-H'
ALGO_RO_S = 'RO-S'
ALGO_OMIT_H = 'OMIT-H'
ALGO_OMIT_S = 'OMIT-S'
ALGO_LISTNET = 'ListNet'

#constants for measures
M_NDCG1 = 'NDCG@1'
M_NDCG5 = 'NDCG@5'
M_NDCG10 = 'NDCG@10'
M_NO_OUTLIERS1 = 'NO-OUTLIERS@1'
M_NO_OUTLIERS5 = 'NO-OUTLIERS@5'
M_NO_OUTLIERS10 = 'NO-OUTLIERS@10'
M_OUTLIERNESS1 = 'OUTLIERNESS@1'
M_OUTLIERNESS5 = 'OUTLIERNESS@5'
M_OUTLIERNESS10 = 'OUTLIERNESS@10'

M_DTR = 'DTR'


def calculateFinalResults(results):
    results = pd.DataFrame(results, columns=['qid', 'algoName', 'metric', 'value'])
    results_listNet = results.loc[results['algoName']=='ListNet']
    results_fair = results.loc[results['algoName'] == 'FOEIR-DTC']

    results_listNet = results_listNet.groupby('metric').mean()
    results_fair = results_fair.groupby('metric').mean()

    return results_listNet, results_fair
