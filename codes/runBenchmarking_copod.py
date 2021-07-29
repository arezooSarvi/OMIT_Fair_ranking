# -*- coding: utf-8 -*-
"""
Created on Thu May 17 14:43:30 2018

@author: Laura
"""

from codes.src.candidateCreator.createCandidate import createCandidate as cC
from codes.src.csvProcessing.csvPrinting import createRankingCSV
from codes.src.algorithms.FOEIR.runFOEIR import runFOEIR
from codes.src.algorithms.ListNet.runListNet import runListNet
from codes.src.measures.runMetrics import runMetrics
import os
import pandas as pd
import csv
import datetime
import multiprocessing
from joblib import Parallel, delayed
import numpy as np

from codes.utils.utils import Utils

data_path = "../data/TREC2020/features/fold"
result_path = "../results"

"""

This method runs the whole benchmarking process.
We currently create rankings using every algorithm on each data set available in the benchmarking
and generate evaluations with the help of the implemented measures afterwards
Since we create candidates objects only once for each data set, evaluations
should always be done on the originalQualification attribute and then saved in 
the qualification attribute. Furthermore, all evaluations on the same data set
(same set of candidates) need to be done directly subsequently, i.e.
each algorithm should be evaluated completely for one data set (run, CSV creation,
and measure evalution) before the candidate list is passed on to the next 
algorithm.

IMPORTANT: An underscore in a file name denotes a query in our framework, hence
the file will be treated and evaluated as if it belonged to one data set
"""


def print_final_results(results, outlier_results, decp_algo=''):
    final_results_outliers = []
    for row in outlier_results:
        temp = {}
        temp['algoName'] = row[1]
        temp['metric'] = row[2]
        temp['value'] = row[3]
        final_results_outliers.append(temp.copy())
    final_results = pd.DataFrame(final_results_outliers, columns=final_results_outliers[0].keys())
    foeir = final_results.loc[final_results['algoName'] == 'FOEIR-DTC']
    foeir_outlierness = foeir.groupby('metric').mean()
    foeir_outlierness = foeir_outlierness.to_dict()['value']
    desired_order_list = ['#outliers@1', '#outliers@3', '#outliers@5', '#outliers@10', 'outlierness@1', 'outlierness@3',
                          'outlierness@5', 'outlierness@10', '#outliers@1_origin', '#outliers@3_origin',
                          '#outliers@5_origin', '#outliers@10_origin', 'outlierness@1_origin', 'outlierness@3_origin',
                          'outlierness@5_origin', 'outlierness@10_origin', 'estimated_swap_movements']
    foeir_outlierness = {k: foeir_outlierness[k] for k in desired_order_list}

    final_results = []
    for row in results:
        temp = {}
        temp['algoName'] = row[1]
        temp['metric'] = row[2]
        temp['value'] = row[3]
        final_results.append(temp.copy())
    final_results = pd.DataFrame(final_results)
    listnet = final_results.loc[final_results['algoName'] == 'ListNet']
    foeir = final_results.loc[final_results['algoName'] == 'FOEIR-DTC']
    listnet = listnet.groupby('metric').mean()
    foeir = foeir.groupby('metric').mean()
    print(" **** ListNet ****")
    print(listnet)
    print("**** FOEIR_" + decp_algo + " ****")
    print(foeir)
    print("**** FOEIR Outlierness_" + decp_algo + " ****")
    print(foeir_outlierness)
    output_name = "evaluationResults_ListNet_" + decp_algo + ".csv"
    listnet.to_csv(os.path.join(result_path, output_name), index=(False))
    output_name = "evaluationResultts_FOEIR_" + decp_algo + ".csv"
    foeir.to_csv(os.path.join(result_path, output_name), index=(False))
    output_name = "evaluationResultts_FOEIR_outlierness" + decp_algo + ".csv"
    # foeir_outlierness.to_csv(os.path.join(result_path, output_name), index=(False))
    Utils().write_dict_to_csv_with_a_row_for_each_key(foeir_outlierness, os.path.join(result_path, output_name))


def main_1(dataSetName, fileNames, listNetRanking, outliers_results, queryNumbers, query_seq_file, results, wind):
    listResults, outlierResults, listFileNames = evaluateLearning('ListNet', listNetRanking, dataSetName,
                                                                  queryNumbers, True,
                                                                  query_seq=query_seq_file,
                                                                  od_method='copod',
                                                                  outlier_window_size=wind)
    results += listResults
    outliers_results += outlierResults
    fileNames += listFileNames
    print_final_results(results, outlierResults, '_outlierness_COPOD')


def main():
    """
    This method starts the whole benchmkaring process. It first reads all 
    raw data sets and creates CSV files from those.
    Then it calls the method benchmarkingProcess to run the benchmarks on
    the implemented algorithms and evalute those with provided measures.
    The method then
    
    """
    # initialize list for evaluation results
    results = []
    outliers_results = []
    finalResults = []
    fileNames = []

    startTime = datetime.datetime.now()

    # read all data sets in TREC including all folds
    for dirpath, dirnames, files in os.walk(data_path):

        if 'fold' in dirpath:
            # construct extractions for different folds
            getTrain = dirpath + '/train.csv'
            getValidation = dirpath + '/validation.csv'
            getTest = dirpath + '/test.csv'
            query_seq = dirpath + '/query_seq_10000.csv'
            query_seq_file = pd.read_csv(query_seq, header=None)
            # constructs a candidate list for the test data set
            ranking, queryNumbers = cC.createLearningCandidate(getTest)

            # run ListNet learning process
            listNetRanking, dataSetName = runListNet(ranking, getTrain, [], getTest, maxIter=1, val=0.3)
            # evaluate listNet learning process, print ranked queries and start scoreBasedEval
            #Parallel(n_jobs=1)(delayed(main_1)(dataSetName, fileNames, listNetRanking, outliers_results,
             #                                  queryNumbers, query_seq_file, results, wind) for wind in
              #                 range(10, 41, 5))
            main_1(dataSetName, fileNames, listNetRanking, outliers_results, queryNumbers, query_seq_file, results, None)

    # ### average results on all queries
    # print_final_results(results)

    endTime = datetime.datetime.now()

    print("Total time of execution: " + str(endTime - startTime))

    # plotData()


def evaluateLearning(algoName, ranking, dataSetName, queryNumbers, listNet=False, k=40, query_seq=None,
                     od_method='copod', decomposition=None, m=None, outlier_window_size=None):
    """
    Evaluates the learning algorithms per query, creates an output file for each ranked query,
    and start the scoreBasedEval method for each query
    
    @param algoName: Name of the algorithm which created the query rankings
    @param ranking: A list of candidates from different queries with new calculated scores for them
    @param dataSetName: Name of the data set without query numbers
    @param queryNumbers: List of query identifiers
    @param k: turncation point of the ranking
    
    return evalResults list with the evaluation results for the algorithms
           evalResults looks like this: [dataSetName, Optimization Algorithm, Measure, Value of Measure]
           fileNames list with file names for each query.
    """
    # initialize list for evaluation results
    evalResults = []
    outlierResults = []
    fileNames = []

    # initialize k for evaluation purposes. This k is also used for calculation of FOIER algorithms
    evalK = k

    # check if evalK is not larger than 40
    if evalK > 40:
        print('Evaluations only done for k = 40 due to comparability reasons. Rankings are still created for ' + str(
            k) + '. If changes to this are wished, please open runBenchmarking and change line 226 accordingly.')
        evalK = 8

    # how many times each query is repeated in the test data
    query_repetition_count = 0
    # loop for each query
    progress_monitor = 0
    for query in queryNumbers:
        # if query != 63397:
        #     continue
        progress_monitor += 1
        # if progress_monitor < 150: continue
        print('************ ', progress_monitor, ' / ', len(queryNumbers), " ************")
        queryRanking = []
        queryProtected = []
        queryNonprotected = []
        output = []
        if query_seq is not None:
            query_repetition_count = query_seq[query_seq[1] == query].shape[0]
        else: query_repetition_count = 1
        if query_repetition_count == 0: continue
        finalPrinting = [
            ['Original_Score', 'learned_Scores', 'Ranking_Score_from_Postprocessing', 'Sensitive_Attribute']]
        # loop over the candidate list to construct the output
        for i in range(len(ranking)):

            # check if the query numbers are equal
            if ranking[i].query == query:

                originQ = str(ranking[i].originalQualification)
                learned = str(ranking[i].learnedScores)
                quali = str(ranking[i].qualification)
                proAttr = str(ranking[i].isProtected)

                output.append([originQ, learned, quali, proAttr])

                # construct list with candiates for one query
                queryRanking.append(ranking[i])

                if proAttr == 'True':
                    queryProtected.append(ranking[i])
                else:
                    queryNonprotected.append(ranking[i])
        if not queryProtected or not queryNonprotected: continue
        finalName = dataSetName + '_' + str(query)

        fileNames.append(finalName)

        # sort candidates by credit scores
        queryProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        queryNonprotected.sort(key=lambda candidate: candidate.qualification, reverse=True)

        # sorting the ranking in accordance with is new scores
        queryRanking.sort(key=lambda candidate: candidate.qualification, reverse=True)

        # update index accoring to the ranking
        queryRanking = updateCurrentIndex(queryRanking)
        queryRanking = updateLearnedIndex(queryRanking)
        # evaluate listNet
        # evalK instead of m
        evalResults += (
            runMetrics(evalK, queryProtected, queryNonprotected, queryRanking[0:m], queryRanking, finalName, 'ListNet'))

        output.sort(key=lambda x: x[2], reverse=True)

        finalPrinting += output

        # only start scoreBasedEval if the algorithm is listNet (baseline)
        if listNet == True:
            # run the score based evaluation on the ranked candidate list
            allResults = scoreBasedEval(finalName, "", k, True, queryProtected, queryNonprotected, queryRanking,
                                        listNet, query_rep=query_repetition_count, decomposition=decomposition, m=m,
                                        od_method=od_method, outlier_window_size=outlier_window_size)
            evalResults += allResults[0]
            outlierResults += allResults[1]
        try:
            with open("/Users/fsarvi/PycharmProjects/Fair_ranking/results/" + algoName + finalName + 'ranking.csv', 'w',
                      newline='') as mf:
                writer = csv.writer(mf)
                writer.writerows(finalPrinting)
        except Exception:
            raise Exception("Some error occured during file creation. Double check specifics.")
            pass

    return evalResults, outlierResults, fileNames


def scoreBasedEval(dataSetName, dataSetPath, k=40, features=True, protected=[], nonProtected=[], originalRanking=[],
                   listNet=False, query_rep=1, od_method='copod', decomposition=None, m=None, outlier_window_size=None):
    """
    Evaluates the learning to rank algorithms and runs 
    the optimization and evaluation of the post-processing methods
    
    @param dataSetName: Name of the data set
    @param dataSetPath: Path of the data sets for score based evaluation. 
    @param k: Provides the length of the ranking
    @param features: True if the provided data set has features for LFRanking, otherwise
    false
    @param protected: If data comes from a learning to rank algorithm this param holds a 
    list of candidates with protected group membership
    @param protected: If data comes from a learning to rank algorithm this param holds a 
    list of candidates with non-protected group membership
    @param protected: If data comes from a learning to rank algorithm this param holds a 
    list of candidates from the new ranking
    @param scoreData: Is set false if the data does not come from an already scored data 
    set but from a learning to rank algorithm
    
    returns a list of evaluation results of the form:
        [dataSetName, Optimization Algorithm, Measure, Value of Measure]
    """

    evalResults = []
    outliersResults = []

    # initialize k for evaluation purposes. This k is also used for calculation of FOIER algorithms
    evalK = k

    # check if evalK is not larger than 40
    if evalK > 40:
        print('Evaluations only done for k = 40 due to comparability reasons. Rankings are still created for ' + str(
            k) + '. If changes to this are wished, please open runBenchmarking and change line 226 accordingly.')
        evalK = 40

    # check if the given data comes from the base line algorithm ListNet
    # if it does not, construct candidates from the data
    if listNet == False:
        # creates Candidates from the preprocessed CSV files in folder preprocessedDataSets
        protected, nonProtected, originalRanking = cC.createScoreBased(dataSetPath)

    # #creates a csv with candidates ranked with color-blind ranking
    # createRankingCSV(originalRanking, 'Color-Blind/' + dataSetName + 'ranking.csv',k )
    # #run the metrics ones for the color-blind ranking
    # evalResults += (runMetrics(evalK, protected, nonProtected, originalRanking, originalRanking, dataSetName, 'Color-Blind'))

    # run for FOEIR-DPC
    # dpcRanking, dpcPath, isDPC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DPC', evalK)
    # if isDPC == True:
    #     dpcRanking = updateCurrentIndex(dpcRanking)
    #     createRankingCSV(dpcRanking, dpcPath,evalK)
    #     evalResults += (runMetrics(evalK, protected, nonProtected, dpcRanking, originalRanking, dataSetName, 'FOEIR-DPC'))

    dtcRankings, dtcPath, isDTC, analytics = runFOEIR(originalRanking, dataSetName, 'FOEIR-DTC', evalK,
                                                      query_rep=query_rep, od_method=od_method, m=m,
                                                      decomposition=decomposition,
                                                      outlier_window_size=outlier_window_size)
    if isDTC == True:
        for dtcRanking in dtcRankings:
            dtcRanking = updateCurrentIndex(dtcRanking)
            createRankingCSV(dtcRanking, dtcPath, len(dtcRanking))
            # evalK instead of m
            evalResults += (
                runMetrics(evalK, protected, nonProtected, dtcRanking, originalRanking, dataSetName, 'FOEIR-DTC'))
        for key in analytics:
            outliersResults.append([dataSetName, 'FOEIR-DTC', key, analytics.get(key, -1)])
    # [, , 'FairnessAtK', eval_FairnessAtK]
    # dicRanking, dicPath, isDIC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DIC', evalK)
    # if isDIC == True:
    #     dicRanking = updateCurrentIndex(dicRanking)
    #     createRankingCSV(dicRanking, dicPath,evalK)
    #     evalResults += (runMetrics(evalK, protected, nonProtected, dicRanking, originalRanking, dataSetName, 'FOEIR-DIC'))

    return evalResults, outliersResults


def updateCurrentIndex(ranking):
    """
    Updates the currentIndex of a ranking according to the current order in the
    list
    @param ranking: list with candidates of a ranking
    
    return list of candidates with updated currentIndex according to their 
    position in the current ranking
    
    """

    index = 0

    for i in range(len(ranking)):
        index += 1

        ranking[i].currentIndex = index

    return ranking


def updateLearnedIndex(ranking):
    """
    Updates the learnedIndex of a ranking according to the current order in the
    list
    @param ranking: list with candidates of a ranking
    
    return list of candidates with updated learnedIndex according to their 
    position in the current ranking
    
    """

    index = 0

    for i in range(len(ranking)):
        index += 1

        ranking[i].learnedIndex = index

    return ranking


def getDataSetName(fileName):
    """
    Extracts name of file for score based eval
    
    @param fileName: Name of the file with .csv ending
    
    return fileName without .csv
    """

    name = fileName.split('.')[0]

    return name


main()
