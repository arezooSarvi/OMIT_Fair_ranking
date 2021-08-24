
import os

import numpy as np
from codes.src.csvProcessing.csvPrinting import createPCSV
import pandas as pd


path = '../../results/doublyStochasticPropMatrix/FOEIR-DTCRanking'

def loadMatrices(path):

    rankings = []
    files = os.listdir(path)
    counter = 0
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            counter += 1
            if divmod(counter, 1000)[1] == 0: print(counter, ' / ', len(files))
            r = pd.read_csv(os.path.join(path,f), header=None)
            r = np.array(r)
            rankings.append(r)

    return rankings

def checkDoublyStochasticity(matrix_path):

    mats = loadMatrices(matrix_path)

    not_valid_mats = []
    dist_from_const_rows = []
    dist_from_const_cols = []

    for mat in mats:
        if np.sum(mat, axis=0) != 1 or np.sum(mat, axis=1) != 1:
            not_valid_mats.append(mat)
            dist_from_const_rows.append(1 - np.sum(mat, axis=0))
            dist_from_const_cols.append(1 - np.sum(mat, axis=1))

    print(len(not_valid_mats), ' / ', len(mats), ' matrices are not doubly stochastic')
    print('avg distance of sum of rows and columns from 1 is: ', np.mean(dist_from_const_rows), ' and ', np.mean(dist_from_const_cols), ' respectively')





checkDoublyStochasticity(path)