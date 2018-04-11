from __future__ import absolute_import
from __future__ import print_function
import os
import time
import six
import inspect
from datetime import datetime
from dateutil.parser import parse
from unqlite import UnQLite

import UML
from UML.exceptions import ArgumentException


"""
    Handle logging of creating and testing learners.
    Currently stores data in a SQLite database file and generates
    a human readable log by querying the tables within the database.
    There is a hierarchical structure to the log, allowing the user
    to specify the level of detail in the log:

    Current Hierarchy
    Level ?: Data creation and preprocessing logs
    Level 1: Outputs basic information about the run (timestamp, run number,
             learner name, train and test object details) and boolean values
             for the availability of additional information
    Level 2: Parameter, metric, and timer data if available
    Level 3: Cross validation
    Level 4: Epoch data
"""


class UmlLogger(object):
    def __init__(self, logLocation, logName):
        fullLogDesignator = os.path.join(logLocation, logName)
        self.logLocation = logLocation
        self.logName = logName
        self.logFileName = fullLogDesignator + ".mr"
        self.isAvailable = False
        self.keepData = True
        self.runNumber = None


    def setup(self, newFileName=None):
        """
            Try to open the file that will be used for logging.  If newFileName
            is present, will reset the log file to use the new file name and Attempt
            to open it.  Otherwise, will use the file name provided when this was
            instantiated.  If successfully opens the file, set isAvailable to true.
        """
        if newFileName is not None and isinstance(newFileName, (str, six.text_type)):
            self.logFileName = newFileName

        dirPath = os.path.dirname(self.logFileName)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        self.db = UnQLite(self.logFileName)
        self.log = self.db.collection('log')
        self.log.create()
        try:
            lastLog = self.log.last_record_id()
            lastRun = self.log.fetch(lastLog)["runNumber"]
            self.runNumber = lastRun + 1
        except TypeError:
            self.runNumber = 0
        self.isAvailable = self.log.exists()

    def cleanup(self):
        # only need to call if we have previously called setup
        if self.isAvailable:
            self.db.close()


    def insertIntoLog(self, logMessage):
        """ Inserts a json style message into the log"""
        if self.isAvailable:
            self.log.store(logMessage)
        else:
            self.setup(self.logFileName)
            logMessage["runNumber"] = self.runNumber
            self.log.store(logMessage)


    def logLoad(self, returnType, name=None, path=None):
        """
        Send pertinent information about the loading of some data set to the log file
        """
        #TODO only log if name or path is present?
        logMessage = {"type": "load"}
        timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))
        logMessage["timestamp"] = timestamp
        logMessage["runNumber"] = self.runNumber
        logMessage["name"] = name
        logMessage["path"] = path

        self.insertIntoLog(logMessage)


    # def logData(self): #TODO
    #     """
    #     Send pertinent information about a data object that has been loaded/created to the log file
    #     """
    #     timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))
    #     logMessage = {"type" : "data",
                  # "timestamp": timestamp,
                  # "function": function, } #TODO
    #     self.insertIntoLog(logMessage)
    #
    #
    # def logPrep(self): #TODO
    #     """
    #     Send pertinent information about the data preparation step performed to the log file
    #     """
    #     timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))
    #     logMessage = {"type" : "prep",
                  # "timestamp": timestamp,
                  # "function": function, } #TODO
    #     self.insertIntoLog(logMessage)


    def logRun(self, umlFunction, trainData, trainLabels, testData, testLabels,
                               learnerFunction, parameters, metrics, timer,
                               extraInfo=None, numFolds=None):
        """
        Send the pertinent information about the run to the log file
        """
        logMessage = {"type" : "run"}
        logMessage["function"] = umlFunction

        timestamp = (time.strftime('%Y-%m-%d %H:%M:%S'))
        logMessage["timestamp"] = timestamp

        logMessage["runNumber"] = self.runNumber
        if isinstance(learnerFunction, (str, six.text_type)):
            functionCall = learnerFunction
        else:
            #we get the source code of the function as a list of strings and glue them together
            funcLines = inspect.getsourcelines(learnerFunction)
            funcString = ""
            for i in range(len(funcLines) - 1):
                funcString += str(funcLines[i])
            if funcLines is None:
                funcLines = "N/A"
            functionCall = funcString
        logMessage["learner"] = functionCall

        trainDataName = None
        trainDataPath = None
        numTrainPoints = None
        numTrainFeatures = None
        testDataName = None
        testDataPath = None
        numTestPoints = None
        numTestFeatures = None

        #log info about training data, if present
        if trainData is not None:
            #If present, add name and path of source files for train and test data
            if trainData.name is not None:
                trainDataName = trainData.name
            if trainData.path is not None:
                trainDataPath = trainData.path
            numTrainPoints = trainData.points
            numTrainFeatures = trainData.features

        #log info about testing data, if present
        if testData is not None:
            if testData.name is not None:
                testDataName = testData.name
            if testData.path is not None:
                testDataPath = testData.path
            numTestPoints = testData.points
            numTestFeatures = testData.features

        logMessage["trainDataName"] = trainDataName
        logMessage["trainDataPath"] = trainDataPath
        logMessage["numTrainPoints"] = numTrainPoints
        logMessage["numTrainFeatures"] = numTrainFeatures
        logMessage["testDataName"] = testDataName
        logMessage["testDataPath"] = testDataPath
        logMessage["numTestPoints"] = numTestPoints
        logMessage["numTestFeatures"] = numTestFeatures

        if parameters is not None and parameters != {}:
            logMessage['parameters'] = parameters

        if metrics is not None and metrics is not {}:
            logMessage["metrics"] = metrics

        if timer is not None and timer.cumulativeTimes is not {}:
            logMessage["timer"] = sum(timer.cumulativeTimes.values())

        self.insertIntoLog(logMessage)


    def logCrossValidation(self, trainData, trainLabels, learnerName, metric, performance,
                           timer, learnerArgs, folds=None):
        """
        Send information about selection of a set of parameters using cross validation
        """
        pass
        #print("metric", metric, "performance", performance, "learnerArgs", learnerArgs)


    def _showLogImplementation(self, levelOfDetail, leastRunsAgo, mostRunsAgo, startDate,
                               endDate, saveToFileName, maximumEntries, searchForText):
        """ Implementation of showLog function for UML"""

        # search for text
        if searchForText is not None:
            runLogs = self.log.filter(lambda log: (searchForText in log.keys() or searchForText in log.values()))

        # search by runsAgo
        elif startDate is None and endDate is None:
            lastLog = self.log.last_record_id()
            try:
                nextRun = self.log.fetch(lastLog)["runNumber"] + 1
            except TypeError:
                nextRun = 1
            startRun = nextRun - mostRunsAgo
            if startRun < 0:
                msg = "mostRunsAgo is greater than the number of runs. "
                msg += "This number cannot exceed {}".format(nextRun)
                raise ArgumentException(msg)
            endRun = nextRun - leastRunsAgo
            if endRun < startRun:
                raise ArgumentException("leastRunsAgo must be less than mostRunsAgo")
            runNumbers = range(startRun, endRun)
            runLogs = self.log.filter(lambda log: log["runNumber"] in runNumbers)

        # search by date
        elif startDate is not None and endDate is not None:
            startDate = parse(startDate)
            endDate = parse(endDate)
            strip = datetime.strptime
            runLogs = self.log.filter(lambda log: strip(log["timestamp"], "%Y-%m-%d %H:%M:%S") >= startDate
                                              and strip(log["timestamp"], "%Y-%m-%d %H:%M:%S") <= endDate)
        elif startDate is not None:
            startDate = parse(startDate)
            runLogs = self.log.filter(lambda log: datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S") >= startDate)
        elif endDate is not None:
            endDate = parse(endDate)
            runLogs = self.log.filter(lambda log: datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S") <= endDate)

        # filter logs for specified run numbers
        if len(runLogs) > maximumEntries:
            entryCutoff = len(runLogs) - maximumEntries
            runLogs = runLogs[entryCutoff:]

        # adjust for level of detail
        if levelOfDetail == 1:
            pass
        elif levelOfDetail == 2:
            fullLog = '*' * 35 + " UML LOGS " + '*' * 35
            for log in runLogs:
                if log["type"] == 'load':
                    fullLog += self.buildLoadLogString(log)
                    fullLog += '*' * 80
                elif log["type"] == 'prep':
                    pass
                    # fullLog += "\n"
                    # fullLog +=  # TODO
                    # fullLog += '*' * 80
                elif log["type"] == 'run':
                    fullLog += "\n"
                    fullLog += self.buildRunLogString(log)
                    fullLog += '*' * 80
                else:
                    pass
                    # fullLog += "\n"
                    # fullLog += # TODO
                    # fullLog += '*' * 80
            if saveToFileName is not None:
                # TODO check if file exists and append if already exists?
                filePath = os.path.join(self.logLocation, saveToFileName)
                with open(filePath, mode='w') as f:
                    f.write(fullLog)
            else:
                print(fullLog)


    def buildRunLogString(self, log, maximumEntries=100, searchForText=None):
        """ Extracts and formats information from the 'runs' table for printable output """
        # header data
        fullLog = "\n"
        fullLog += "Run Number: {}\n".format(log['runNumber'])
        fullLog += "Timestamp: {}\n".format(log['timestamp'])
        fullLog += "UML Function: {}\n".format(log['function'])
        fullLog += "Learner Function: {}\n".format(log['learner'])
        timer = log.get("timer", False)
        if timer:
            fullLog += "Completed in {:.3f} seconds\n".format(log['timer'])
        fullLog += "\n"
        # training data
        trainDataKeys = ["trainDataName", "trainDataPath", "numTrainPoints", "numTrainFeatures"]
        trainDataValues = [str(log[key]) for key in trainDataKeys] # must be strings
        fullLog += _formatRunLine(trainDataKeys, trainDataValues)
        # testing data
        testDataKeys = ["testDataName", "testDataPath", "numTestPoints", "numTestFeatures"]
        testDataValues = [str(log[key]) for key in testDataKeys] # must be strings
        fullLog += _formatRunLine(testDataKeys, testDataValues)
        # parameter data
        parameters = log.get("parameters", None)
        if parameters is not None:
            fullLog += _logDictionary("parameters", parameters)
        # metric data
        metrics = log.get("metrics", None)
        if metrics is not None:
            fullLog += _logDictionary("metrics", metrics)

        return fullLog


    def buildLoadLogString(self, log, maximumEntries=100, searchForText=None):
        fullLog = "\n"
        fullLog += "Run Number: {}\n".format(log['runNumber'])
        fullLog += "Data loaded at: {}\n".format(log['timestamp'])
        if log['path'] is not None:
            fullLog += "Path: {}\n".format(log['path'])
        if log['name'] is not None:
            fullLog += "Name: {}\n".format(log['name'])
        return fullLog


#######################
### Generic Helpers ###
#######################

def _logDictionary(key, dictionary):
    dictionaryKeys = dictionary.keys()
    dictionaryValues = [dictionary[key] for key in dictionaryKeys]
    # values must be strings
    dictionaryValues = map(str, dictionaryValues)
    return _formatRunLine(dictionaryKeys, dictionaryValues)

def _formatRunLine(columnNames, rowValues):
    """ Formats """
    columnNames, rowValues = _removeItemsWithoutData(columnNames, rowValues)
    if columnNames == []:
        return ""
    lineLog = ("{:20s}" * len(columnNames)).format(*columnNames)
    lineLog += "\n"
    lineLog += ("{:20s}" * len(rowValues)).format(*rowValues)
    lineLog += "\n\n"

    return lineLog


def _removeItemsWithoutData(columnNames, rowValues):
    """ Prevents the Log from displaying columns that do not have a data"""
    keepIndexes = []
    for index, item in enumerate(rowValues):
        if item !=  "None":
            keepIndexes.append(index)
    keepColumnName = []
    keepRowValue = []
    for index in keepIndexes:
        keepColumnName.append(columnNames[index])
        keepRowValue.append(rowValues[index])
    return keepColumnName, keepRowValue


#######################
### Initialization  ###
#######################

def initLoggerAndLogConfig():
    """Sets up or reads configuration options associated with logging,
    and initializes the currently active logger object using those
    options.

    """
    try:
        location = UML.settings.get("logger", "location")
        if location == "":
            location = './logs-UML'
            UML.settings.set("logger", "location", location)
            UML.settings.saveChanges("logger", "location")
    except:
        location = './logs-UML'
        UML.settings.set("logger", "location", location)
        UML.settings.saveChanges("logger", "location")
    finally:
        def cleanThenReInit_Loc(newLocation):
            UML.logger.active.cleanup()
            currName = UML.settings.get("logger", 'name')
            UML.logger.active = UmlLogger(newLocation, currName)
        UML.settings.hook("logger", "location", cleanThenReInit_Loc)

    try:
        name = UML.settings.get("logger", "name")
    except:
        name = "log-UML"
        UML.settings.set("logger", "name", name)
        UML.settings.saveChanges("logger", "name")
    finally:
        def cleanThenReInit_Name(newName):
            UML.logger.active.cleanup()
            currLoc = UML.settings.get("logger", 'location')
            UML.logger.active = UmlLogger(currLoc, newName)

        UML.settings.hook("logger", "name", cleanThenReInit_Name)

    try:
        loggingEnabled = UML.settings.get("logger", "enabledByDefault")
    except:
        loggingEnabled = 'True'
        UML.settings.set("logger", "enabledByDefault", loggingEnabled)
        UML.settings.saveChanges("logger", "enabledByDefault")

    try:
        mirror = UML.settings.get("logger", "mirrorToStandardOut")
    except:
        mirror = 'False'
        UML.settings.set("logger", "mirrorToStandardOut", mirror)
        UML.settings.saveChanges("logger", "mirrorToStandardOut")

    try:
        deepCV = UML.settings.get("logger", 'enableCrossValidationDeepLogging')
    except:
        deepCV = 'False'
        UML.settings.set("logger", 'enableCrossValidationDeepLogging', deepCV)
        UML.settings.saveChanges("logger", 'enableCrossValidationDeepLogging')

    try:
        deepMulti = UML.settings.get("logger", 'enableMultiClassStrategyDeepLogging')
    except:
        deepMulti = 'False'
        UML.settings.set("logger", 'enableMultiClassStrategyDeepLogging', deepMulti)
        UML.settings.saveChanges("logger", 'enableMultiClassStrategyDeepLogging')
    UML.logger.active = UmlLogger(location, name)
