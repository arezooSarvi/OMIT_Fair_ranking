# -*- coding: utf-8 -*-


from codes.src.candidateCreator.createCandidate import createCandidate as cC
from codes.src.csvProcessing.csvPrinting import createRankingCSV
from codes.src.algorithms.FOEIR.runFOEIR import runFOEIR
from codes.src.algorithms.ListNet.runListNet import runListNet
from codes.src.measures.runMetrics import runMetrics
from codes.src.measures.finalEval import calculateFinalResults
from codes.utils import clear_results_folder as cl
import os
import pandas as pd
import csv
import datetime

data_date = "2020"
data_path = "../data/TREC" + data_date + "/features/fold"
result_path = "./results"

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


def main_1(dataSetName, fileNames, listNetRanking, queryNumbers, query_seq_file, od_method='copod'):
    results = []
    listResults, listFileNames = evaluateLearning('ListNet', listNetRanking, dataSetName,
                                                  queryNumbers, True,
                                                  query_seq=query_seq_file,
                                                  od_method=od_method)
    results += listResults
    fileNames += listFileNames
    listNet_results, fair_results = calculateFinalResults(results)
    listNet_results.to_csv(os.path.join(result_path, 'ListNet_' + od_method + '.csv'))
    fair_results.to_csv(os.path.join(result_path, 'Fair_' + od_method + '.csv'))


def main():
    """
    This method starts the whole benchmkaring process. It first reads all
    raw data sets and creates CSV files from those.
    Then it calls the method benchmarkingProcess to run the benchmarks on
    the implemented algorithms and evalute those with provided measures.
    The method then

    """
    # initialize list for evaluation results

    outliers_results = []
    finalResults = []
    fileNames = []

    cl.clear_results_folder()
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
            listNetRanking, dataSetName = runListNet(ranking, getTrain, [], getTest, maxIter=70, val=0.3)
            main_1(dataSetName, fileNames, listNetRanking, queryNumbers, query_seq_file, od_method='copod')

    endTime = datetime.datetime.now()

    print("Total time of execution: " + str(endTime - startTime))

    # plotData()


def evaluateLearning(algoName, ranking, dataSetName, queryNumbers, listNet=False, k=40, query_seq=None,
                     od_method='copod'):
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
        progress_monitor += 1
        print('************ ', progress_monitor, ' / ', len(queryNumbers), " ************")
        queryRanking = []
        queryProtected = []
        queryNonprotected = []
        output = []
        if query_seq is not None:
            query_repetition_count = query_seq[query_seq[1] == query].shape[0]
        else:
            query_repetition_count = 1
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
            runMetrics(evalK, queryProtected, queryNonprotected, queryRanking[0:k], queryRanking, finalName, 'ListNet',
                       od_method))

        output.sort(key=lambda x: x[2], reverse=True)

        finalPrinting += output

        # only start scoreBasedEval if the algorithm is listNet (baseline)
        if listNet == True:
            # run the score based evaluation on the ranked candidate list
            allResults = scoreBasedEval(finalName, "", k, queryProtected, queryNonprotected, queryRanking,
                                        listNet, query_rep=query_repetition_count,
                                        od_method=od_method)
            evalResults += allResults[0]
        try:
            with open(result_path + "/" + algoName + finalName + 'ranking.csv', 'w',
                      newline='') as mf:
                writer = csv.writer(mf)
                writer.writerows(finalPrinting)
        except Exception:
            raise Exception("Some error occured during file creation. Double check specifics.")
            pass

    return evalResults, fileNames


def scoreBasedEval(dataSetName, dataSetPath, k=40, protected=[], nonProtected=[], originalRanking=[],
                   listNet=False, query_rep=1, od_method='copod'):
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

    dtcRankings, dtcPath, isDTC = runFOEIR(originalRanking, dataSetName, 'FOEIR-DTC', evalK,
                                           query_rep=query_rep, od_method=od_method)
    if isDTC == True:
        ranking_counter = 0
        temp_results = []
        for dtcRanking in dtcRankings:
            dtcRanking = updateCurrentIndex(dtcRanking)
            ranking_counter += 1
            dtcPath_counter = dtcPath.split('.')[0] + '_' + str(ranking_counter) + '.' + dtcPath.split('.')[1]
            createRankingCSV(dtcRanking, dtcPath_counter, len(dtcRanking))
            # evalResults += (
            #     runMetrics(evalK, protected, nonProtected, dtcRanking, originalRanking, dataSetName, 'FOEIR-DTC'))
            temp_results += (
                runMetrics(evalK, protected, nonProtected, dtcRanking, originalRanking, dataSetName, 'FOEIR-DTC',
                           od_method))

        _, temp_results = calculateFinalResults(temp_results)
        result = runMetrics(evalK, protected, nonProtected, dtcRanking, originalRanking, dataSetName, 'FOEIR-DTC',
                            od_method)
        for row in result:
            row[3] = temp_results.loc[temp_results.index == row[2]]['value'].to_list()[0]
        evalResults += (result)

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

