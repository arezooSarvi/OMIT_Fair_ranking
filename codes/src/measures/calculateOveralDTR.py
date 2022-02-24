import os

import numpy as np
import pandas as pd

listNet_path = '../../results/rankings'
foeir_path = '../../results/rankings/FOEIR-DTC'

seq = pd.read_csv('../../../data/TREC2020/features/fold/query_seq_10000.csv',
                  header=None)


def loadRankings(path):
    rankings = []
    files = os.listdir(path)
    counter = 0
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            q = f.split('_')[2]
            if 'feature' in f:
                counter += 1
                if divmod(counter, 1000)[1] == 0: print(counter, ' / ', len(files))
                r = pd.read_csv(os.path.join(path, f))
                if path.split('/')[-1] == 'FOEIR-DTC':
                    rankings.append(r)
                else:
                    for i in range(sum(seq[1] == int(q))):
                        rankings.append(r)

    return rankings


def calculateExposureAndUtility(ranking, k):
    proCount = 0
    proListX = []
    unproCount = 0
    unproListX = []
    proU = 0
    unproU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX = []
    utility = []

    if k > 40:
        k = 40

    counter = 0
    for i in range(k):
        counter += 1
        if divmod(counter, 1000)[1] == 0: print(counter, ' / ', k)
        if ranking.iloc[i].Sensitive_Attribute:
            proCount += 1
            proListX.append(i)
            proU += ranking.iloc[i].Original_Score
        else:
            unproCount += 1
            unproListX.append(i)
            unproU += ranking.iloc[i].Original_Score

    v = np.arange(1, (k + 1), 1)
    v = 1 / np.log2(1 + v + 1)
    v = np.reshape(v, (1, k))

    v = np.transpose(v)
    proExposure = np.sum(v[proListX])
    unproExposure = np.sum(v[unproListX])

    return proExposure, unproExposure, proU, unproU, proCount, unproCount


def calculatedTR(rankings_path, algoName, k=40):
    rankings = loadRankings(rankings_path)
    proExposure = []
    unproExposure = []
    proUtility = []
    unproUtility = []
    proCountList = []
    unproCountList = []
    results = []

    for ranking in rankings:

        proExp, unproExp, proU, unproU, proCount, unproCount = calculateExposureAndUtility(ranking, len(ranking))
        proExposure.append(proExp)
        unproExposure.append(unproExp)
        proUtility.append(proU)
        unproUtility.append(unproU)
        proCountList.append(proCount)
        unproCountList.append(unproCount)

    top = 0
    bottom = 0

    # calculate value for each group
    if sum(proCountList) != 0:
        proU = sum(proUtility) / sum(proCountList)
        proExposure = sum(proExposure) / sum(proCountList)
        top = (proExposure / proU)

    if sum(unproCountList) != 0:
        unproU = sum(unproUtility) / sum(unproCountList)
        unproExposure = sum(unproExposure) / sum(unproCountList)
        bottom = (unproExposure / unproU)

    # calculate DTR
    dTR_origin = top / bottom
    print('dTR for ', algoName, ' : ', dTR_origin)
    dTD = abs(top - bottom)
    print('dTD for ', algoName, ' : ', dTD)

    print(top, '            ', bottom)
    print('**********************')
    return dTR_origin


calculatedTR(foeir_path, 'FOEIR')
# calculatedTR(listNet_path, 'ListNet')

