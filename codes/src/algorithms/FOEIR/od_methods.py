import numpy as np
from pyod.models import mad, copod, knn


def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (
                bounds['desired']['upper'] - bounds['desired']['lower']) / (
                        bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]


def quartile_od(scores):
    Q1 = np.percentile(scores, 25, interpolation="midpoint")
    Q3 = np.percentile(scores, 75, interpolation="midpoint")
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    outliers = [0] * len(scores)
    outlyingness = [0] * len(scores)
    for ind, x in enumerate(scores):
        if (x > up_lim) or (x < low_lim):
            outliers[ind] = 1
            outlyingness[ind] = min(abs(x - low_lim), abs(up_lim - x))
    return outliers, outlyingness


def MAD_od(scores, return_list=False):
    if not scores: return 0
    scores = np.squeeze(scores).reshape(len(scores), 1)
    model = mad.MAD()
    model.fit(scores)
    outliers_indx = model.predict(scores)
    outlierness = model.decision_function(scores)
    max_ = max(outlierness)
    min_ = min(outlierness)
    outlierness = [(i - min_) / (max_ - min_) for i in outlierness]
    outliers = [i.tolist()[0] for id, i in enumerate(scores) if outliers_indx[id]]
    outlierness_avg = sum([i for id, i in enumerate(outlierness) if outliers_indx[id]])
    if return_list: return list(outliers_indx), outlierness, outlierness_avg
    return outlierness
    # return  outliers, outlierness


def COPOD_od(scores, return_list=False):
    if not scores: return 0
    scores = np.squeeze(scores).reshape(len(scores), 1)
    model = copod.COPOD()
    model.fit(scores)
    outliers_indx = model.predict(scores)
    outlierness = model.decision_function(scores)
    max_ = max(outlierness)
    min_ = min(outlierness)
    outlierness = [((i - min_) / (max_ - min_)) for i in outlierness]
    # outlierness = [(i-min_)/(max_-min_) for i in outlierness]
    outliers = [i.tolist()[0] for id, i in enumerate(scores) if outliers_indx[id]]
    outlierness_avg = sum([i for id, i in enumerate(outlierness) if outliers_indx[id]])
    if return_list: return list(outliers_indx), outlierness, outlierness_avg
    return outlierness_avg
    # return outliers, outlierness


def MedKnn_od(scores, return_list=False):
    scores = np.squeeze(scores).reshape(len(scores), 1)
    model = knn.KNN(method='median')
    try:
        model.fit(scores)
    except:
        return list(np.zeros(len(scores))), list(np.zeros(len(scores))), 0
    outliers_indx = model.predict(scores)
    outlierness = model.decision_function(scores)
    max_ = max(outlierness)
    min_ = min(outlierness)
    outlierness = [(i - min_) / (max_ - min_) for i in outlierness]
    outliers = [i.tolist()[0] for id, i in enumerate(scores) if outliers_indx[id]]
    outlierness_avg = sum([i for id, i in enumerate(outlierness) if outliers_indx[id]])
    if return_list: return list(outliers_indx), outlierness, outlierness_avg
    return outlierness_avg


def apply_outlier_detection(data, od='quartile'):
    qid = data.qid.iloc[0]
    data = data.citations.tzo_list()
    if not data: return 0
    if len(data) > 20: data = data[0:20]
    outliers = []
    outlyingness = 0
    if od == 'quartile':
        outliers, outlyingness = quartile_od(data)
    if od == 'MAD':
        outliers, outlyingness = MAD_od(data)
    if od == 'copod':
        outliers, outlyingness = COPOD_od(data)
    if od == 'knn':
        outliers, outlyingness = MedKnn_od(data)

    return (qid, outliers, outlyingness)
