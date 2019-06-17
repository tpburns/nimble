"""
Unit tests for functionality of the SessionLogger
"""

from __future__ import absolute_import
import os
import shutil
import time
import ast
import sys
import sqlite3
import tempfile
import re

from nose import with_setup
from nose.tools import raises
import six
from six import StringIO
import numpy

import nimble
from nimble.helpers import generateClassificationData
from nimble.calculate import rootMeanSquareError as RMSE
from nimble.configuration import configSafetyWrapper
from nimble.exceptions import InvalidArgumentValue
from nimble.exceptions import InvalidArgumentValueCombination
from nimble.exceptions import InvalidArgumentType

#####################
# Helpers for tests #
#####################

def removeLogFile():
    nimble.logger.active.cleanup()
    location = nimble.settings.get("logger", "location")
    name = nimble.settings.get("logger", "name")
    pathToFile = os.path.join(location, name + ".mr")
    if os.path.exists(pathToFile):
        os.remove(pathToFile)

def getLastLogData():
    query = "SELECT logInfo FROM logger ORDER BY entry DESC LIMIT 1"
    valueList = nimble.logger.active.extractFromLog(query)
    lastLog = valueList[0][0]
    return lastLog

#############
### SETUP ###
#############

@configSafetyWrapper
def testLogDirectoryAndFileSetup():
    """assert a new directory and log file are created with first attempt to log"""
    location = nimble.settings.get("logger", "location")
    name = nimble.settings.get("logger", "name")
    pathToFile = os.path.join(location, name + ".mr")
    if os.path.exists(location):
        shutil.rmtree(location)

    nimble.settings.set('logger', 'enabledByDefault', 'True')

    X = nimble.createData("Matrix", [])

    assert os.path.exists(location)
    assert os.path.exists(pathToFile)

#############
### INPUT ###
#############

@configSafetyWrapper
def testTopLevelInputFunction():
    removeLogFile()
    """assert the nimble.log function correctly inserts data into the log"""
    header = "input"
    logInfo = {"test": "testInput"}
    nimble.log(header, logInfo)
    # select all columns from the last entry into the logger
    query = "SELECT * FROM logger"
    lastLog = nimble.logger.active.extractFromLog(query)
    lastLog = lastLog[0]

    assert lastLog[0] == 1
    assert lastLog[2] == 0
    assert lastLog[3] == "User - " + header
    assert lastLog[4] == str(logInfo)

@configSafetyWrapper
def testNewRunNumberEachSetup():
    """assert that a new, sequential runNumber is generated each time the log file is reopened"""
    removeLogFile()
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    data = [[],[]]
    for run in range(5):
        nimble.createData("Matrix", data)
        # cleanup will require setup before the next log entry
        nimble.logger.active.cleanup()
    query = "SELECT runNumber FROM logger"
    lastLogs = nimble.logger.active.extractFromLog(query)

    for entry, log in enumerate(lastLogs):
        assert log[0] == entry

@configSafetyWrapper
def testLoadTypeFunctionsUseLog():
    """tests that createData is being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    # data
    trainX = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1],
              [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]]
    trainY = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    testX = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
    testY = [[0], [1], [2], [1]]

    # createData
    trainXObj = nimble.createData("Matrix", trainX)
    logInfo = getLastLogData()
    assert trainXObj.getTypeString() in logInfo
    assert "'numPoints': 12" in logInfo
    assert "'numFeatures': 3" in logInfo

    trainYObj = nimble.createData("List", trainY)
    logInfo = getLastLogData()
    assert trainYObj.getTypeString() in logInfo
    assert "'numPoints': 12" in logInfo
    assert "'numFeatures': 1" in logInfo

    testXObj = nimble.createData("Sparse", testX)
    logInfo = getLastLogData()
    assert testXObj.getTypeString() in logInfo
    assert "'numPoints': 4" in logInfo
    assert "'numFeatures': 3" in logInfo

    testYObj = nimble.createData("DataFrame", testY)
    logInfo = getLastLogData()
    assert testYObj.getTypeString() in logInfo
    assert "'numPoints': 4" in logInfo
    assert "'numFeatures': 1" in logInfo

    # the sparsity and seed are also stored for random data
    randomObj = nimble.createRandomData("Matrix", 5, 5, 0)
    logInfo = getLastLogData()
    assert randomObj.getTypeString() in logInfo
    assert "'sparsity': 0" in logInfo
    assert "seed" in logInfo

    # loadTrainedLearner
    tl = nimble.train('custom.KNNClassifier', trainXObj, trainYObj, arguments={'k': 1})
    with tempfile.NamedTemporaryFile(suffix=".nimm") as tmpFile:
        tl.save(tmpFile.name)
        load = nimble.loadTrainedLearner(tmpFile.name)
    logInfo = getLastLogData()
    assert "TrainedLearner" in logInfo
    assert "'learnerName': 'KNNClassifier'" in logInfo
    assert "'learnerArgs': {'k': 1}" in logInfo

    # loadData
    with tempfile.NamedTemporaryFile(suffix=".nimd") as tmpFile:
        randomObj.save(tmpFile.name)
        load = nimble.loadData(tmpFile.name)
    logInfo = getLastLogData()
    assert load.getTypeString() in logInfo
    assert "'numPoints': 5" in logInfo
    assert "'numFeatures': 5" in logInfo

@configSafetyWrapper
def test_setRandomSeed():
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    nimble.randomness.startAlternateControl()
    nimble.setRandomSeed(1337)
    nimble.randomness.endAlternateControl()
    logInfo = getLastLogData()
    assert "{'seed': 1337}" in logInfo

@configSafetyWrapper
def testRunTypeFunctionsUseLog():
    """tests that top level and TrainedLearner functions are being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    # data
    trainX = [[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1],
              [1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0], [0,0,1]]
    trainY = [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1], [2]]
    testX = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]
    testY = [[0], [1], [2], [1]]

    trainXObj = nimble.createData("Matrix", trainX, useLog=False)
    trainYObj = nimble.createData("Matrix", trainY, useLog=False)
    testXObj = nimble.createData("Matrix", testX, useLog=False)
    testYObj = nimble.createData("Matrix", testY, useLog=False)

    timePattern = re.compile(r"'time': [0-9]+\.[0-9]+")

    # train
    tl = nimble.train("sciKitLearn.SVC", trainXObj, trainYObj, performanceFunction=RMSE)
    logInfo = getLastLogData()
    assert "'function': 'train'" in logInfo
    assert re.search(timePattern, logInfo)

    # trainAndApply
    predictions = nimble.trainAndApply("sciKitLearn.SVC", trainXObj, trainYObj,
                                    testXObj, performanceFunction=RMSE)
    logInfo = getLastLogData()
    assert "'function': 'trainAndApply'" in logInfo
    assert re.search(timePattern, logInfo)

    # trainAndTest
    performance = nimble.trainAndTest("sciKitLearn.SVC", trainXObj, trainYObj,
                                   testXObj, testYObj,
                                   performanceFunction=RMSE)
    logInfo = getLastLogData()
    assert "'function': 'trainAndTest'" in logInfo
    # ensure that metrics is storing performanceFunction and result
    assert "'metrics': {'rootMeanSquareError': 0.0}" in logInfo
    assert re.search(timePattern, logInfo)

    # normalizeData
    # copy to avoid modifying original data
    trainXNormalize = trainXObj.copy()
    testXNormalize = testXObj.copy()
    nimble.normalizeData('sciKitLearn.PCA', trainXNormalize, testX=testXNormalize)
    logInfo = getLastLogData()
    assert "'function': 'normalizeData'" in logInfo
    assert re.search(timePattern, logInfo)

    # trainAndTestOnTrainingData
    results = nimble.trainAndTestOnTrainingData("sciKitLearn.SVC", trainXObj,
                                             trainYObj,
                                             performanceFunction=RMSE)
    logInfo = getLastLogData()
    assert "'function': 'trainAndTestOnTrainingData'" in logInfo
    # ensure that metrics is storing performanceFunction and result
    assert "'metrics': {'rootMeanSquareError': 0.0}" in logInfo
    assert re.search(timePattern, logInfo)

    # TrainedLearner.apply
    predictions = tl.apply(testXObj)
    logInfo = getLastLogData()
    assert "'function': 'TrainedLearner.apply'" in logInfo
    assert re.search(timePattern, logInfo)

    # TrainedLearner.test
    performance = tl.test(testXObj, testYObj, performanceFunction=RMSE)
    logInfo = getLastLogData()
    assert "'function': 'TrainedLearner.test'" in logInfo
    # ensure that metrics is storing performanceFunction and result
    assert "'metrics': {'rootMeanSquareError': 0.0}" in logInfo
    assert re.search(timePattern, logInfo)

    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')

    # crossValidate
    top = nimble.crossValidate('custom.KNNClassifier', trainXObj, trainYObj,
                            performanceFunction=RMSE)
    logInfo = getLastLogData()
    assert "'learner': 'custom.KNNClassifier'" in logInfo


@configSafetyWrapper
def testPrepTypeFunctionsUseLog():
    """Test that the functions in base using useLog are being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')

    data = [["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1], ["a", 1, 1],
            ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2], ["b", 2, 2],
            ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3], ["c", 3, 3]]

    def checkLogContents(funcName, objectID, arguments=None):
        lastLog = getLastLogData()
        expFunc = "'function': '{0}'".format(funcName)
        expID = "'object': '{0}'".format(objectID)
        assert expFunc in lastLog
        assert expID in lastLog

        if arguments:
            assert 'arguments' in lastLog
            for argName, argVal in arguments:
                expArgs1 = "'{0}': '{1}'".format(argName, argVal)
                # double quotations may wrap the second arg if it contains quotations
                expArgs2 = """'{0}': "{1}" """.format(argName, argVal).strip()
                assert expArgs1 in lastLog or expArgs2 in lastLog

    ########
    # Base #
    ########

    # replaceFeatureWithBinaryFeatures
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.replaceFeatureWithBinaryFeatures(0)
    checkLogContents('replaceFeatureWithBinaryFeatures', 'Matrix', [('featureToReplace', 0)])

    # transformFeatureToIntegers
    dataObj = nimble.createData("List", data, useLog=False)
    dataObj.transformFeatureToIntegers(0)
    checkLogContents('transformFeatureToIntegers', 'List', [('featureToConvert', 0)])

    # trainAndTestSets
    dataObj = nimble.createData("DataFrame", data, useLog=False)
    train, test = dataObj.trainAndTestSets(testFraction=0.5)
    checkLogContents('trainAndTestSets', 'DataFrame', [('testFraction', 0.5)])

    # groupByFeature
    dataObj = nimble.createData("Sparse", data, useLog=False)
    calculated = dataObj.groupByFeature(by=0)
    checkLogContents('groupByFeature', 'Sparse', [('by', 0)])

    # referenceDataFrom
    dataObj = nimble.createData("Matrix", data, useLog=False, name='refData')
    dataObj.referenceDataFrom(dataObj)
    checkLogContents('referenceDataFrom', 'Matrix', [('other', 'refData')])

    # transpose
    dataObj = nimble.createData("List", data, useLog=False)
    dataObj.transpose()
    checkLogContents('transpose', 'List')

    # fillWith
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.fillWith(1, 2, 0, 4, 0)
    checkLogContents('fillWith', "Matrix",
        [('values', 1), ('pointStart', 2), ('pointEnd', 4), ('featureStart', 0),
         ('featureEnd', 0)])

    # fillUsingAllData
    dataObj = nimble.createData("Matrix", data, useLog=False)
    def simpleFiller(obj, match, **kwargs):
        return nimble.createData('Matrix', numpy.zeros_like(dataObj.data))
    dataObj.fillUsingAllData('a', fill=simpleFiller, a=1, b=3)
    checkLogContents('fillUsingAllData', 'Matrix', [('match', 'a'), ('fill', 'simpleFiller'),
                                                    ('a', 1), ('b', 3)])

    # flattenToOnePoint
    dataObj = nimble.createData("DataFrame", data, useLog=False)
    dataObj.flattenToOnePoint()

    checkLogContents('flattenToOnePoint', "DataFrame")
    # unflattenFromOnePoint; using same dataObj from flattenToOnePoint
    dataObj.unflattenFromOnePoint(18)
    checkLogContents('unflattenFromOnePoint', "DataFrame")

    # flattenToOneFeature
    dataObj = nimble.createData("Sparse", data, useLog=False)
    dataObj.flattenToOneFeature()
    checkLogContents('flattenToOneFeature', "Sparse")

    # unflattenFromOnePoint; using same dataObj from flattenToOneFeature
    dataObj.unflattenFromOneFeature(3)
    checkLogContents('unflattenFromOneFeature', "Sparse")

    # merge
    dPtNames = ['p' + str(i) for i in range(18)]
    dFtNames = ['f0', 'f1', 'f2']
    dataObj = nimble.createData("Matrix", data, pointNames=dPtNames,
                             featureNames=dFtNames, useLog=False)
    mData = [[1, 4], [2, 5], [3, 6]]
    mPtNames = ['p0', 'p6', 'p12']
    mFtNames = ['f2', 'f3']
    mergeObj = nimble.createData('Matrix', mData, pointNames=mPtNames,
                              featureNames=mFtNames, useLog=False)
    dataObj.merge(mergeObj, point='intersection', feature='union')
    checkLogContents('merge', "Matrix", [("other", mergeObj.name),
                                         ("point", 'intersection')])

    ############################
    # Points/Features/Elements #
    ############################

    def simpleMapper(vector):
        vID = vector[0]
        intList = []
        for i in range(1, len(vector)):
            intList.append(vector[i])
        ret = []
        for value in intList:
            ret.append((vID, value))
        return ret

    def simpleReducer(identifier, valuesList):
        total = 0
        for value in valuesList:
            total += value
        return (identifier, total)

    # points.mapReduce
    dataObj = nimble.createData("Matrix", data, useLog=False)
    calculated = dataObj.points.mapReduce(simpleMapper,simpleReducer)
    checkLogContents('points.mapReduce', "Matrix", [("mapper", "simpleMapper"),
                                                    ("reducer", "simpleReducer")])

    # features.mapReduce
    dataObj = nimble.createData("Matrix", numpy.array(data, dtype=object).T,
                             featureNames=False, useLog=False)
    calculated = dataObj.features.mapReduce(simpleMapper,simpleReducer)
    checkLogContents('features.mapReduce', "Matrix", [("mapper", "simpleMapper"),
                                                    ("reducer", "simpleReducer")])

    # elements.calculate
    dataObj = nimble.createData("Matrix", data, useLog=False)
    calculated = dataObj.elements.calculate(lambda x: len(x), features=0)
    checkLogContents('elements.calculate', "Matrix", [('function', "lambda x: len(x)"),
                                                      ('features', [0])])

    # points.calculate
    dataObj = nimble.createData("Matrix", data, useLog=False)
    calculated = dataObj.points.calculate(lambda x: len(x))
    checkLogContents('points.calculate', "Matrix", [('function', "lambda x: len(x)")])

    # features.calculate
    dataObj = nimble.createData("Matrix", data, useLog=False)
    calculated = dataObj.features.calculate(lambda x: len(x), features=0)
    checkLogContents('features.calculate', "Matrix", [('function', "lambda x: len(x)"),
                                                      ('features', [0])])

    # points.shuffle
    dataObj = nimble.createData("List", data, useLog=False)
    dataObj.points.shuffle()
    checkLogContents('points.shuffle', "List")

    # features.shuffle
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.features.shuffle()
    checkLogContents('features.shuffle', "Matrix")

    # points.normalize
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.points.normalize(subtract=0, divide=1)
    checkLogContents('points.normalize', "Matrix", [('subtract', 0), ('divide', 1)])

    # features.normalize
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.features.normalize(subtract=0, divide=1)
    checkLogContents('features.normalize', "Matrix", [('subtract', 0), ('divide', 1)])

    # points.sort
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.points.sort(sortBy=dataObj.features.getName(0))
    checkLogContents('points.sort', "Matrix", [('sortBy', dataObj.features.getName(0))])

    # features.sort
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.features.sort(sortBy=[2, 1, 0])
    checkLogContents('features.sort', "Matrix", [('sortBy', [2, 1, 0])])

    # points.copy
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.points.copy(0)
    checkLogContents('points.copy', "Matrix", [('toCopy', 0)])

    # features.copy
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.features.copy(number=1)
    checkLogContents('features.copy', "Matrix", [('number', 1)])

    # points.extract
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.points.extract(toExtract=0)
    checkLogContents('points.extract', "Matrix", [('toExtract', 0)])

    # features.extract
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.features.extract(number=1)
    checkLogContents('features.extract', "Matrix", [('number', 1)])

    # points.delete
    dataObj = nimble.createData("Matrix", data, useLog=False,
                             pointNames=['p' + str(i) for i in range(18)])
    extracted = dataObj.points.delete(start='p0', end='p3')
    checkLogContents('points.delete', "Matrix", [('start', 'p0'), ('end', 'p3')])

    # features.delete
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.features.delete(number=2, randomize=True)
    checkLogContents('features.delete', "Matrix", [('number', 2), ('randomize', True)])

    def retainer(vector):
        return True

    # points.retain
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.points.retain(toRetain=retainer)
    checkLogContents('points.retain', "Matrix", [('toRetain', 'retainer')])

    # features.retain
    dataObj = nimble.createData("Matrix", data, useLog=False)
    extracted = dataObj.features.retain(toRetain=lambda ft: True)
    checkLogContents('features.retain', "Matrix", [('toRetain', 'lambda ft: True')])

    # points.transform
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataCopy = dataObj.copy()
    calculated = dataCopy.points.transform(lambda x: [val for val in x])
    checkLogContents('points.transform', "Matrix",
                     [('function', 'lambda x: [val for val in x]')])

    # features.transform
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataCopy = dataObj.copy()
    calculated = dataCopy.features.transform(lambda x: [val for val in x], features=0)
    checkLogContents('features.transform', "Matrix",
                     [('function', 'lambda x: [val for val in x]'), ('features', [0])])

    # elements.transform
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataCopy = dataObj.copy()
    calculated = dataCopy.elements.transform(lambda x: [val for val in x], features=0)
    checkLogContents('elements.transform', "Matrix",
                     [('toTransform', 'lambda x: [val for val in x]'), ('features', [0])])

    # points.add
    dataObj = nimble.createData("Matrix", data, useLog=False)
    appendData = [["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4], ["d", 4, 4]]
    toAppend = nimble.createData("Matrix", appendData, useLog=False)
    dataObj.points.add(toAppend)
    checkLogContents('points.add', "Matrix", [('toAdd', toAppend.name)])

    # features.add
    dataObj = nimble.createData("Matrix", data, useLog=False)
    appendData = numpy.zeros((18,1))
    toAppend = nimble.createData("Matrix", appendData, useLog=False)
    dataObj.features.add(toAppend)
    checkLogContents('features.add', "Matrix", [('toAdd', toAppend.name)])

    # points.fill
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.points.fill(nimble.match.nonNumeric, 0)
    checkLogContents('points.fill', "Matrix", [('match', 'nonNumeric'), ('fill', 0)])

    # features.fill
    dataObj = nimble.createData("Matrix", data, useLog=False)
    dataObj.features.fill(1, nimble.fill.mean, features=[1,2])
    checkLogContents('features.fill', "Matrix", [('match', 1), ('fill', 'mean')])

    # elements.multiply/power
    dataObj = nimble.ones('Matrix', 5, 5)
    fives = nimble.ones('Matrix', 5, 5) * 5
    fives.name = 'fives'
    dataObj.elements.multiply(fives)
    checkLogContents('elements.multiply', 'Matrix', [('other', 'fives')])
    zeros = nimble.zeros('Matrix', 5, 5, name='zeros')
    dataObj.elements.power(zeros)
    checkLogContents('elements.power', 'Matrix', [('other', 'zeros')])

    # features.splitByParsing
    toSplit = [[1, 'a0', 2], [1, 'a1', 2], [3, 'b0', 4], [5, 'c0', 6]]
    fNames = ['keep1', 'split', 'keep2']
    dataObj = nimble.createData('List', toSplit, featureNames=fNames, useLog=False)
    dataObj.features.splitByParsing('split', 1, ['str', 'int'])
    checkLogContents('features.splitByParsing', 'List',
                     [('feature', 'split'), ('rule', 1), ('resultingNames', ['str', 'int'])])

    # points.splitByCollapsingFeatures
    toSplit = [['NYC', 4, 5, 10], ['LA', 20, 21, 21], ['CHI', 0, 2, 7]]
    fNames = ['city', 'jan', 'feb', 'mar']
    dataObj = nimble.createData('DataFrame', toSplit, featureNames=fNames, useLog=False)
    dataObj.points.splitByCollapsingFeatures(['jan', 'feb', 'mar'],
                                              'month', 'temp')
    checkLogContents('points.splitByCollapsingFeatures', 'DataFrame',
                     [('featuresToCollapse', ['jan', 'feb', 'mar']),
                      ('featureForNames', 'month'), ('featureForValues', 'temp')])

    # points.combineByExpandingFeatures
    toCombine = [['Bolt', '100m', 9.81],
                 ['Bolt', '200m', 19.78],
                 ['Gatlin', '100m', 9.89],
                 ['de Grasse', '200m', 20.02],
                 ['de Grasse', '100m', 9.91]]
    fNames = ['athlete', 'dist', 'time']
    dataObj = nimble.createData('Matrix', toCombine, featureNames=fNames, useLog=False)
    dataObj.points.combineByExpandingFeatures('dist', 'time')
    checkLogContents('points.combineByExpandingFeatures', 'Matrix',
                     [('featureWithFeatureNames', 'dist'), ('featureWithValues', 'time')])

    # points.setName
    dataObj = nimble.createData('Matrix', data, useLog=False)
    dataObj.points.setName(0, 'newPtName')
    checkLogContents('points.setName', 'Matrix', [('oldIdentifier', 0),
                                                  ('newName', 'newPtName')])

    # features.setName
    dataObj = nimble.createData('Matrix', data, useLog=False)
    dataObj.features.setName(0, 'newFtName')
    checkLogContents('features.setName', 'Matrix', [('oldIdentifier', 0),
                                                  ('newName', 'newFtName')])

    # points.setNames
    dataObj = nimble.createData('Matrix', data, useLog=False)
    newPtNames = ['point' + str(i) for i in range(18)]
    dataObj.points.setNames(newPtNames)
    checkLogContents('points.setNames', 'Matrix', [('assignments', newPtNames)])

    # features.setNames
    dataObj = nimble.createData('Matrix', data, useLog=False)
    newFtNames = ['feature' + str(i) for i in range(3)]
    dataObj.features.setNames(newFtNames)
    checkLogContents('features.setNames', 'Matrix', [('assignments', newFtNames)])


@configSafetyWrapper
def testDataTypeFunctionsUseLog():
    """Test that the data type functions are being logged"""
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    data = [["a", 1], ["a", 1], ["a", 1], ["a", 1], ["a", 1], ["a", 1],
            ["b", 2], ["b", 2], ["b", 2], ["b", 2], ["b", 2], ["b", 2],
            ["c", 3], ["c", 3], ["c", 3], ["c", 3], ["c", 3], ["c", 3]]

    # featureReport
    dataObj = nimble.createData("Matrix", data, useLog=False)
    fReport = dataObj[:,1].featureReport()

    logInfo = getLastLogData()
    assert "'reportType': 'feature'" in logInfo

    # summaryReport
    dataObj = nimble.createData("Matrix", data, useLog=False)
    sReport = dataObj.summaryReport()

    logInfo = getLastLogData()
    assert "'reportType': 'summary'" in logInfo


@configSafetyWrapper
def testHandmadeLogEntriesInput():
    typeQuery = "SELECT logType FROM logger ORDER BY entry DESC LIMIT 1"
    # custom string
    customString = "enter this string into the log"
    nimble.log("customString", customString)

    logType = nimble.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - customString"
    assert customString in logInfo

    # custom list
    customList = ["this", "custom", "list", 1, 2, 3, {"list":"tested"}]
    nimble.log("customList", customList)

    logType = nimble.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - customList"
    for value in customList:
        assert str(value) in logInfo

    # custom dict
    customDict = {"custom":"dict", "log":"testing", 1:2, 3:"four"}
    nimble.log("customDict", customDict)

    logType = nimble.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - customDict"
    for key in customDict.keys():
        assert str(key) in logInfo
    for value in customDict.values():
        assert str(value) in logInfo

    # heading matches nimble logType
    nimble.log('run', "User log with heading that matches a logType")
    logType = nimble.logger.active.extractFromLog(typeQuery)[0][0]
    logInfo = getLastLogData()
    assert logType == "User - run"
    assert "User log with heading that matches a logType" in logInfo

@raises(InvalidArgumentType)
def testLogUnacceptedlogType():
    nimble.log(["unacceptable"], "you can't do this")

@raises(InvalidArgumentType)
def testLogUnacceptedlogInfo():
    dataObj = nimble.createData("Matrix", [[1]], useLog=False)
    nimble.log("acceptable", dataObj)

@raises(InvalidArgumentValue)
def testLogHeadingTooLong():
    heading = "#" * 51
    nimble.log(heading, 'foo')

##############
### OUTPUT ###
##############

@configSafetyWrapper
def testShowLogToFile():
    removeLogFile()
    nimble.createData("Matrix", [[1], [2], [3]], useLog = True)
    nimble.createData("Matrix", [[4], [5], [6]], useLog = True)
    #write to log
    location = nimble.settings.get("logger", "location")
    name = "showLogTestFile.txt"
    pathToFile = os.path.join(location,name)
    nimble.showLog(saveToFileName=pathToFile)
    assert os.path.exists(pathToFile)

    originalSize = os.path.getsize(pathToFile)
    removeLogFile()

    #overwrite
    nimble.createData("Matrix", [[1], [2], [3]], useLog = True)
    nimble.showLog(saveToFileName=pathToFile)
    overwriteSize = os.path.getsize(pathToFile)
    assert overwriteSize < originalSize

    #append
    nimble.createData("Matrix", [[4], [5], [6]], useLog = True)
    nimble.showLog(saveToFileName=pathToFile, append=True)
    appendSize = os.path.getsize(pathToFile)
    assert appendSize > originalSize


@configSafetyWrapper
def testShowLogToStdOut():
    saved_stdout = sys.stdout
    try:
        location = nimble.settings.get("logger", "location")
        name = "showLogTestFile.txt"
        pathToFile = os.path.join(location,name)
        # create showLog file with default arguments
        nimble.showLog(saveToFileName=pathToFile)

        # get content of file as a string
        with open(pathToFile) as log:
            lines = log.readlines()
        fileContent = "".join(lines)
        fileContent = fileContent.strip()

        # redirect stdout
        out = StringIO()
        sys.stdout = out

        # showLog to stdout with default arguments
        nimble.showLog()
        stdoutContent = out.getvalue().strip()

        assert stdoutContent == fileContent

    finally:
        sys.stdout = saved_stdout

@configSafetyWrapper
def testShowLogSearchFilters():
    """test the level of detail, runNumber, date, text, maxEntries search filters"""
    removeLogFile()
    nimble.settings.set('logger', 'enabledByDefault', 'True')
    nimble.settings.set('logger', 'enableCrossValidationDeepLogging', 'True')
    # create an example log file
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3],
             [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    data2 = [[1, 0, 0, 1],
             [0, 1, 0, 2],
             [0, 0, 1, 3]]
    # add data to log
    for i in range(5):
        # load
        trainObj = nimble.createData('Matrix', data=data1, featureNames=variables)
        testObj = nimble.createData('Matrix', data=data2, featureNames=variables)
        # data
        report = trainObj.summaryReport()
        # prep
        trainYObj = trainObj.features.extract(3)
        testYObj = testObj.features.extract(3)
        # run and crossVal
        results = nimble.trainAndTest('Custom.KNNClassifier', trainX=trainObj,
                                   trainY=trainYObj, testX=testObj, testY=testYObj,
                                   performanceFunction=RMSE,
                                   arguments={"k": nimble.CV([3, 5])})
    # edit log runNumbers and timestamps
    location = nimble.settings.get("logger", "location")
    name = nimble.settings.get("logger", "name")
    pathToFile = os.path.join(location, name + ".mr")
    conn = sqlite3.connect(pathToFile)
    c = conn.cursor()
    c.execute("UPDATE logger SET timestamp = '2018-03-22 12:00:00' WHERE entry <= 7")
    conn.commit()
    c.execute("UPDATE logger SET runNumber = 1, timestamp = '2018-03-23 12:00:00' WHERE entry > 7 AND entry <= 14")
    conn.commit()
    c.execute("UPDATE logger SET runNumber = 2, timestamp = '2018-03-23 18:00:00' WHERE entry > 14 AND entry <= 21")
    conn.commit()
    c.execute("UPDATE logger SET runNumber = 3, timestamp = '2018-03-25 12:00:00' WHERE entry > 21 AND entry <= 28")
    conn.commit()
    c.execute("UPDATE logger SET runNumber = 4, timestamp = '2018-04-24 12:00:00' WHERE entry > 28")
    conn.commit()

    location = nimble.settings.get("logger", "location")
    name = "showLogTestFile.txt"
    pathToFile = os.path.join(location,name)
    nimble.showLog(levelOfDetail=3, leastRunsAgo=0, mostRunsAgo=5, maximumEntries=100, saveToFileName=pathToFile)
    fullShowLogSize = os.path.getsize(pathToFile)
    nimble.showLog(levelOfDetail=3, leastRunsAgo=0, mostRunsAgo=5, maximumEntries=100)
    # level of detail
    nimble.showLog(levelOfDetail=3, saveToFileName=pathToFile)
    mostDetailedSize = os.path.getsize(pathToFile)

    nimble.showLog(levelOfDetail=2, saveToFileName=pathToFile)
    lessDetailedSize = os.path.getsize(pathToFile)
    assert lessDetailedSize < mostDetailedSize

    nimble.showLog(levelOfDetail=1, saveToFileName=pathToFile)
    leastDetailedSize = os.path.getsize(pathToFile)
    assert leastDetailedSize < lessDetailedSize

    # runNumber
    nimble.showLog(levelOfDetail=3, mostRunsAgo=4, saveToFileName=pathToFile)
    fewerRunsAgoSize = os.path.getsize(pathToFile)
    assert fewerRunsAgoSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, leastRunsAgo=1, mostRunsAgo=5, saveToFileName=pathToFile)
    moreRunsAgoSize = os.path.getsize(pathToFile)
    assert moreRunsAgoSize < fullShowLogSize

    assert moreRunsAgoSize == fewerRunsAgoSize

    nimble.showLog(levelOfDetail=3, leastRunsAgo=2, mostRunsAgo=4, saveToFileName=pathToFile)
    runSelectionSize = os.path.getsize(pathToFile)
    assert runSelectionSize < moreRunsAgoSize

    # startDate
    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, startDate="2018-03-23", saveToFileName=pathToFile)
    startLaterSize = os.path.getsize(pathToFile)
    assert startLaterSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, startDate="2018-04-24", saveToFileName=pathToFile)
    startLastSize = os.path.getsize(pathToFile)
    assert startLastSize < startLaterSize

    # endDate
    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, endDate="2018-03-25", saveToFileName=pathToFile)
    endEarlierSize = os.path.getsize(pathToFile)
    assert endEarlierSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, endDate="2018-03-22", saveToFileName=pathToFile)
    endEarliestSize = os.path.getsize(pathToFile)
    assert endEarliestSize < endEarlierSize

    # startDate and endDate
    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, startDate="2018-03-23", endDate="2018-03-25", saveToFileName=pathToFile)
    dateSelectionSize = os.path.getsize(pathToFile)
    assert dateSelectionSize < startLaterSize
    assert dateSelectionSize < endEarlierSize

    # startDate and endDate with time
    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, startDate="2018-03-23 11:00", endDate="2018-03-23 17:00:00", saveToFileName=pathToFile)
    timeSelectionSize = os.path.getsize(pathToFile)
    assert timeSelectionSize < dateSelectionSize

    #text
    nimble.showLog(levelOfDetail=3, mostRunsAgo=1, searchForText=None, saveToFileName=pathToFile)
    oneRunSize = os.path.getsize(pathToFile)

    nimble.showLog(levelOfDetail=3, mostRunsAgo=1, searchForText="trainAndTest", saveToFileName=pathToFile)
    trainSearchSize = os.path.getsize(pathToFile)
    assert trainSearchSize < oneRunSize

    nimble.showLog(levelOfDetail=3, mostRunsAgo=1, searchForText="Matrix", saveToFileName=pathToFile)
    loadSearchSize = os.path.getsize(pathToFile)
    assert loadSearchSize < oneRunSize

    # regex
    nimble.showLog(levelOfDetail=3, mostRunsAgo=1, searchForText="Mat.+x", regex=True, saveToFileName=pathToFile)
    loadRegexSize = os.path.getsize(pathToFile)
    assert loadSearchSize == loadRegexSize

    # maximumEntries
    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, maximumEntries=34, saveToFileName=pathToFile)
    oneLessSize = os.path.getsize(pathToFile)
    assert oneLessSize < fullShowLogSize

    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, maximumEntries=33, saveToFileName=pathToFile)
    twoLessSize = os.path.getsize(pathToFile)
    assert twoLessSize < oneLessSize

    nimble.showLog(levelOfDetail=3, mostRunsAgo=5, maximumEntries=7, saveToFileName=pathToFile)
    maxEntriesOneRun = os.path.getsize(pathToFile)
    assert maxEntriesOneRun == oneRunSize

    # showLog returns None, file still created with only header
    nimble.showLog(levelOfDetail=3, maximumEntries=1, saveToFileName=pathToFile)
    oneEntrySize = os.path.getsize(pathToFile)
    # pick startDate after final date in log
    nimble.showLog(levelOfDetail=3, startDate="2018-05-24", saveToFileName=pathToFile)
    noDataSize = os.path.getsize(pathToFile)
    assert noDataSize < oneEntrySize

@raises(InvalidArgumentValue)
def testLevelofDetailNotInRange():
    nimble.showLog(levelOfDetail=6)

@raises(InvalidArgumentValueCombination)
def testStartGreaterThanEndDate():
    nimble.showLog(startDate="2018-03-24", endDate="2018-03-22")

def testInvalidDateTimeFormats():
    # year invalid format
    try:
        nimble.showLog(startDate="18-03-24")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # month invalid format
    try:
        nimble.showLog(startDate="2018-3-24")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # day invalid format
    try:
        nimble.showLog(startDate="2018-04-1")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # date format ok but invalid date
    try:
        nimble.showLog(startDate="2018-02-31")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # hour invalid format
    try:
        nimble.showLog(startDate="2018-03-24 1:00")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # minute invalid format
    try:
        nimble.showLog(startDate="2018-03-24 01:19.22")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # second invalid
    try:
        nimble.showLog(startDate="2018-03-24 01:19:0.2")
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

@raises(InvalidArgumentValue)
def testLeastRunsAgoNegative():
    nimble.showLog(leastRunsAgo=-2)

@raises(InvalidArgumentValueCombination)
def testMostRunsLessThanLeastRuns():
    nimble.showLog(leastRunsAgo=2, mostRunsAgo=1)

def testShowLogSuccessWithUserLog():
    """ Test user headings that match defined logTypes are successfully rendered """
    for lType in nimble.logger.active.logTypes:
        nimble.log(lType, "foo")
    # defined logTypes require specific input to render correctly, if these
    # headings are stored as a defined logType, the log would not render
    # nimble.log should prepend "User - " to the heading to avoid this conflict.
    nimble.showLog()
