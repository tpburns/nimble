"""
Helper functions for any functions defined in uml.py

They are separated here so that that (most) top level
user facing functions are contained in uml.py without
the distraction of helpers

"""

import inspect
import numpy
import scipy.io
import os.path
import re 
import datetime

from UML.exceptions import ArgumentException
from UML.data import CooSparseData
from UML.data import DenseMatrixData
from UML.data import RowListData
from UML.data import Base

def _loadCoo(data, featureNames, fileType):
	if fileType is None:
		return CooSparseData(data, featureNames)

	# since file type is not None, that is an indicator that we must read from a file
	path = data
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) = _loadCSVtoMatrix(path)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(path)
	else:
		raise ArgumentException("Unrecognized file type")

	if tempFeatureNames is not None:
			featureNames = tempFeatureNames
	return CooSparseData(data, featureNames, os.path.basename(path), path)


def _loadDMD(data, featureNames, fileType):
	if fileType is None:
		return DenseMatrixData(data, featureNames)

	# since file type is not None, that is an indicator that we must read from a file
	path = data
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) = _loadCSVtoMatrix(path)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(path)
	else:
		raise ArgumentException("Unrecognized file type")

	if tempFeatureNames is not None:
			featureNames = tempFeatureNames
	return DenseMatrixData(data, featureNames, os.path.basename(path), path)


def _loadRLD(data, featureNames, fileType):
	if fileType is None:
		return RowListData(data, featureNames)

	# since file type is not None, that is an indicator that we must read from a file
	path = data
	tempFeatureNames = None
	if fileType == 'csv':
		(data, tempFeatureNames) =_loadCSVtoList(data)
	elif fileType == 'mtx':
		(data, tempFeatureNames) = _loadMTXtoAuto(data)
	else:
		raise ArgumentException("Unrecognized file type")

	# if we load from file, we assume that data will be read; feature names may not
	# be, thus we check
	if tempFeatureNames is not None:
			featureNames = tempFeatureNames

	return RowListData(data, featureNames, os.path.basename(path), path)

def _loadCSVtoMatrix(path):
	inFile = open(path, 'rU')
	firstLine = inFile.readline()
	featureNames = None
	skip_header = 0

	# test if this is a line defining featureNames
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		featureNames = scrubbedLine.split(',')
		skip_header = 1

	# check the types in the first data containing line.
	line = firstLine
	while (line == "") or (line[0] == '#'):
		line = inFile.readline()
	lineList = line.split(',')
	for datum in lineList:
		try:
			num = numpy.float(datum)
		except ValueError:
			raise ValueError("Cannot load a file with non numerical typed columns")

	inFile.close()

	data = numpy.genfromtxt(path, delimiter=',', skip_header=skip_header)
	return (data, featureNames)

def _loadMTXtoAuto(path):
	"""
	Uses scipy helpers to read a matrix market file; returning whatever is most
	appropriate for the file. If it is a matrix market array type, a numpy
	dense matrix is returned as data, if it is a matrix market coordinate type, a
	sparse scipy coo_matrix is returned as data. If featureNames are present,
	they are also read.

	"""
	inFile = open(path, 'rU')
	featureNames = None

	# read through the comment lines
	while True:
		currLine = inFile.readline()
		if currLine[0] != '%':
			break
		if len(currLine) > 1 and currLine[1] == "#":
			# strip '%#' from the begining of the line
			scrubbedLine = currLine[2:]
			# strip newline from end of line
			scrubbedLine = scrubbedLine.rstrip()
			featureNames = scrubbedLine.split(',')

	inFile.close()

	data = scipy.io.mmread(path)
	return (data, featureNames)

def _intFloatOrString(inString):
	ret = inString
	try:
		ret = int(inString)
	except ValueError:
		ret = float(inString)
	# this will return an int or float if either of the above two are successful
	finally:
		if ret == "":
			return None
		return ret

def _defaultParser(line):
	"""
	When given a comma separated value line, it will attempt to convert
	the values first to int, then to float, and if all else fails will
	keep values as strings. Returns list of values.

	"""
	ret = []
	lineList = line.split(',')
	for entry in lineList:
		ret.append(_intFloatOrString(entry))
	return ret


def _loadCSVtoList(path):
	inFile = open(path, 'rU')
	firstLine = inFile.readline()
	featureNameList = None

	# test if this is a line defining featureNames
	if firstLine[0] == "#":
		# strip '#' from the begining of the line
		scrubbedLine = firstLine[1:]
		# strip newline from end of line
		scrubbedLine = scrubbedLine.rstrip()
		featureNameList = scrubbedLine.split(',')
		featureNameMap = {}
		for name in featureNameList:
			featureNameMap[name] = featureNameList.index(name)
	#if not, get the iterator pointed back at the first line again	
	else:
		inFile.close()
		inFile = open(path, 'rU')

	#list of datapoints in the file, where each data point is a list
	data = []
	for currLine in inFile:
		currLine = currLine.rstrip()
		#ignore empty lines
		if len(currLine) == 0:
			continue

		data.append(_defaultParser(currLine))

	if featureNameList == None:
		return (data, None)

	inFile.close()

	return (data, featureNameMap)



def countWins(predictions):
	"""
	Count how many contests were won by each label in the set.  If a class label doesn't
	win any predictions, it will not be included in the results.  Return a dictionary:
	{classLabel: # of contests won}
	"""
	predictionCounts = {}
	for prediction in predictions:
		if prediction in predictionCounts:
			predictionCounts[prediction] += 1
		else:
			predictionCounts[prediction] = 1

	return predictionCounts

def extractWinningPredictionLabel(predictions):
	"""
	Provided a list of tournament winners (class labels) for one point/row in a test set,
	choose the label that wins the most tournaments.  Returns the winning label.
	"""
	#Count how many times each class won
	predictionCounts = countWins(predictions)

	#get the class that won the most tournaments
	#TODO: what if there are ties?
	return max(predictionCounts.iterkeys(), key=(lambda key: predictionCounts[key]))


def extractWinningPredictionIndex(predictionScores):
	"""
	Provided a list of confidence scores for one point/row in a test set,
	return the index of the column (i.e. label) of the highest score.  If
	no score in the list of predictionScores is a number greater than negative
	infinity, returns None.  
	"""

	maxScore = float("-inf")
	maxScoreIndex = -1
	for i in range(len(predictionScores)):
		score = predictionScores[i]
		if score > maxScore:
			maxScore = score
			maxScoreIndex = i

	if maxScoreIndex == -1:
		return None
	else:
		return maxScoreIndex

def extractWinningPredictionIndexAndScore(predictionScores, featureNamesInverse):
	"""
	Provided a list of confidence scores for one point/row in a test set,
	return the index of the column (i.e. label) of the highest score.  If
	no score in the list of predictionScores is a number greater than negative
	infinity, returns None.  
	"""
	allScores = extractConfidenceScores(predictionScores, featureNamesInverse)

	if allScores is None:
		return None
	else:
		return allScores[0]


def extractConfidenceScores(predictionScores, featureNamesInverse):
	"""
	Provided a list of confidence scores for one point/row in a test set,
	return an ordered list of (label, score) tuples.  List is ordered
	by score in descending order.
	"""

	if predictionScores is None or len(predictionScores) == 0:
		return None

	scoreMap = {}
	for i in range(len(predictionScores)):
		score = predictionScores[i]
		label = featureNamesInverse[i]
		scoreMap[label] = score

	return scoreMap


#TODO this is a helper, move to utilities package?
def copyLabels(dataSet, dependentVar):
	"""
		A helper function to simplify the process of obtaining a 1-dimensional matrix of class
		labels from a data matrix.  Useful in functions which have an argument that may be
		a column index or a 1-dimensional matrix.  If 'dependentVar' is an index, this function
		will return a copy of the column in 'dataSet' indexed by 'dependentVar'.  If 'dependentVar'
		is itself a column (1-dimensional matrix w/shape (nx1)), dependentVar will be returned.

		dataSet:  matrix containing labels and, possibly, features.  May be empty if 'dependentVar'
		is a 1-column matrix containing labels.

		dependentVar: Either a column index indicating which column in dataSet contains class labels,
		or a matrix containing 1 column of class labels.

		returns A 1-column matrix of class labels
	"""
	if isinstance(dependentVar, Base):
		#The known Indicator argument already contains all known
		#labels, so we do not need to do any further processing
		labels = dependentVar
	elif isinstance(dependentVar, (str, unicode, int)):
		#known Indicator is an index; we extract the column it indicates
		#from knownValues
		labels = dataSet.copyFeatures([dependentVar])
	else:
		raise ArgumentException("Missing or improperly formatted indicator for known labels in computeMetrics")

	return labels




def applyCodeVersions(functionTextList, inputHash):
	"""applies all the different various versions of code that can be generated from functionText to each of the variables specified in inputHash, where
	data is plugged into the variable with name inputVariableName. Returns the result of each application as a list.
	functionTextList is a list of text objects, each of which defines a python function
	inputHash is of the form {variable1Name:variable1Value, variable2Name:variable2Value, ...}
	"""
	results = []
	for codeText in functionTextList:
		results.append(executeCode(codeText, inputHash))
	return results



def executeCode(code, inputHash):
	"""Execute the given code stored as text in codeText, starting with the variable values specified in inputHash
	This function assumes the code consists of EITHER an entire function definition OR a single line of code with
	statements seperated by semi-colons OR as a python function object.
	"""
	#inputHash = inputHash.copy() #make a copy so we don't modify it... but it doesn't seem necessary
	if isSingleLineOfCode(code): return executeOneLinerCode(code, inputHash) #it's one line of text (with ;'s to seperate statemetns')
	elif isinstance(code, (str, unicode)): return executeFunctionCode(code, inputHash) #it's the text of a function definition
	else: return code(**inputHash)	#assume it's a function itself


def executeOneLinerCode(codeText, inputHash):
	"""Execute the given code stored as text in codeText, starting with the variable values specified in inputHash
	This function assumes the code consists of just one line (with multiple statements seperated by semi-colans.
	Note: if the last statement in the line starts X=... then the X= gets stripped off (to prevent it from getting broken by A=(X=...)).
	"""
	if not isSingleLineOfCode(codeText): raise Exception("The code text was not just one line of code.")
	codeText = codeText.strip()
	localVariables = inputHash.copy()
	pieces = codeText.split(";")
	lastPiece = pieces[-1].strip()
	lastPiece = re.sub("\A([\w])+[\s]*=", "",  lastPiece) #if the last statement begins with something like X = ... this removes the X = part.
	lastPiece = lastPiece.strip()
	pieces[-1] = "RESULTING_VALUE_ZX7_ = (" + lastPiece + ")"
	codeText = ";".join(pieces)
	#oneLiner = True

	print "Code text: "+str(codeText)
	exec(codeText, globals(), localVariables)	#apply the code
	return localVariables["RESULTING_VALUE_ZX7_"]



def executeFunctionCode(codeText, inputHash):
	"""Execute the given code stored as text in codeText, starting with the variable values specified in inputHash
	This function assumes the code consists of an entire function definition.
	"""
	if not "def" in codeText: raise Exception("No function definition was found in this code!")
	localVariables = {}
	exec(codeText, globals(), localVariables)	#apply the code, which declares the function definition
	#foundFunc = False
	#result = None
	for varName, varValue in localVariables.iteritems():
		if "function" in str(type(varValue)):
			return varValue(**inputHash)
	

def isSingleLineOfCode(codeText):
	if not isinstance(codeText, (str, unicode)): return False
	codeText = codeText.strip()
	try:
		codeText.strip().index("\n")
		return False
	except ValueError:
		return True



def _incrementTrialWindows(allData, orderedFeature, currEndTrain, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize):
	"""
	Helper which will calculate the start and end of the training and testing sizes given the current
	position in the full data set. 

	"""
#	set_trace()
	# determine the location of endTrain.
	if currEndTrain is None:
		# points are zero indexed, thus -1 for the num of points case
#		set_trace()
		endTrain = _jumpForward(allData, orderedFeature, 0, minTrainSize, -1)
	else:
		endTrain =_jumpForward(allData, orderedFeature, currEndTrain, stepSize)

	# the value we don't want to split from the training set
	nonSplit = allData[endTrain,orderedFeature]
	# we're doing a lookahead here, thus -1 from the last possible index, and  +1 to our lookup
	while (endTrain < allData.points() - 1 and allData[endTrain+1,orderedFeature] == nonSplit):
		endTrain += 1

	if endTrain == allData.points() -1:
		return None

	# we get the start for training by counting back from endTrain
	startTrain = _jumpBack(allData, orderedFeature, endTrain, maxTrainSize, -1)
	if startTrain < 0:
		startTrain = 0
#	if _diffLessThan(allData, orderedFeature, startTrain, endTrain, minTrainSize):
#		return _incrementTrialWindows(allData, orderedFeature, currEndTrain+1, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize)
#		return None

	# we get the start and end of the test set by counting forward from endTrain
	# speciffically, we go forward by one, and as much more forward as specified by gap
	startTest = _jumpForward(allData, orderedFeature, endTrain+1, gap)
	if startTest >= allData.points():
		return None

	endTest = _jumpForward(allData, orderedFeature, startTest, maxTestSize, -1)
	if endTest >= allData.points():
		endTest = allData.points() - 1
	if _diffLessThan(allData, orderedFeature, startTest, endTest, minTestSize):
#		return _incrementTrialWindows(allData, orderedFeature, currEndTrain+1, minTrainSize, maxTrainSize, stepSize, gap, minTestSize, maxTestSize)
		return None

	return (startTrain, endTrain, startTest, endTest)


def _jumpBack(allData, orderedFeature, start, delta, intCaseOffset=0):
	if isinstance(delta,datetime.timedelta):
		endPoint = start
		startVal = datetime.timedelta(float(allData[start,orderedFeature]))
		# loop as long as we don't run off the end of the data
		while (endPoint > 0):
			if (startVal - datetime.timedelta(float(allData[endPoint-1,orderedFeature])) > delta):
				break
			endPoint = endPoint - 1 
	else:
		endPoint = start - (delta + intCaseOffset)

	return endPoint


def _jumpForward(allData, orderedFeature, start, delta, intCaseOffset=0):
	if isinstance(delta,datetime.timedelta):
		endPoint = start
		startVal = datetime.timedelta(float(allData[start,orderedFeature]))
		# loop as long as we don't run off the end of the data
		while (endPoint < allData.points() - 1):
			if (datetime.timedelta(float(allData[endPoint+1,orderedFeature])) - startVal > delta):
				break
			endPoint = endPoint + 1 
	else:
		endPoint = start + (delta + intCaseOffset)

	return endPoint



def _diffLessThan(allData, orderedFeature, startPoint, endPoint, delta):
	if isinstance(delta,datetime.timedelta):
		startVal = datetime.timedelta(float(allData[startPoint,orderedFeature]))
		endVal = datetime.timedelta(float(allData[endPoint,orderedFeature]))
		return (endVal - startVal) < delta
	else:
		return (endPoint - startPoint + 1) < delta





def computeMetrics(dependentVar, knownData, predictedData, performanceFunctions, negativeLabel=None):
	"""
		Calculate one or more error metrics, given a list of known labels and a list of
		predicted labels.  Return as a dictionary associating the performance metric with
		its numerical result.

		dependentVar: either an int/string representing a column index in knownData
		containing the known labels, or an n x 1 matrix that contains the known labels

		knownData: matrix containing the known labels of the test set, as well as the
		features of the test set. Can be None if 'knownIndicator' contains the labels,
		and none of the performance functions needs features as input.

		predictedData: Matrix containing predicted class labels for a testing set.
		Assumes that the predicted label in the nth row of predictedLabels is associated
		with the same data point/instance as the label in the nth row of knownLabels.

		performanceFunctions: list of functions that compute some kind of error metric.
		Functions must be either a string that defines a proper function, a one-liner
		function (see Combinations.py), or function code.  Also, they are expected to
		take at least 2 arguments:  a vector or known labels and a vector of predicted
		labels.  Optionally, they may take the features of the test set as a third argument,
		as a matrix.

		negativeLabel: Label of the 'negative' class in the testing set.  This parameter is
		only relevant for binary class problems; and only needed for some error metrics
		(proportionPerentNegative50/90).

		Returns: a dictionary associating each performance metric with the (presumably)
		numerical value computed by running the function over the known labels & predicted labels
	"""
	from UML.metrics import proportionPercentNegative90
	from UML.metrics import proportionPercentNegative50
	from UML.metrics import bottomProportionPercentNegative10

	if isinstance(dependentVar, (list, Base)):
		#The known Indicator argument already contains all known
		#labels, so we do not need to do any further processing
		knownLabels = dependentVar
	elif dependentVar is not None:
		#known Indicator is an index; we extract the column it indicates
		#from knownValues
		knownLabels = knownData.copyColumns([dependentVar])
	else:
		raise ArgumentException("Missing indicator for known labels in computeMetrics")

	results = {}
	parameterHash = {"knownValues":knownLabels, "predictedValues":predictedData}
	for func in performanceFunctions:
		#some functions need negativeLabel as an argument.
		if func == proportionPercentNegative90 or func == proportionPercentNegative50 or func == bottomProportionPercentNegative10:
			parameterHash["negativeLabel"] = negativeLabel
			results[inspect.getsource(func)] = executeCode(func, parameterHash)
			del parameterHash["negativeLabel"]
		elif len(inspect.getargspec(func).args) == 2:
			#the metric function only takes two arguments: we assume they
			#are the known class labels and the predicted class labels
			if func.__name__ != "<lambda>":
				results[func.__name__] = executeCode(func, parameterHash)
			else:
				results[inspect.getsource(func)] = executeCode(func, parameterHash)
		elif len(inspect.getargspec(func).args) == 3:
			#the metric function takes three arguments:  known class labels,
			#features, and predicted class labels. add features to the parameter hash
			#divide X into labels and features
			#TODO correctly separate known labels and features in all cases
			parameterHash["features"] = knownData
			if func.__name__ != "<lambda>":
				results[func.__name__] = executeCode(func, parameterHash)
			else:
				results[inspect.getsource(func)] = executeCode(func, parameterHash)
		else:
			raise Exception("One of the functions passed to computeMetrics has an invalid signature: "+func.__name__)
	return results

def confusion_matrix_generator(knownY, predictedY):
	""" Given two vectors, one of known class labels (as strings) and one of predicted labels,
	compute the confusion matrix.  Returns a 2-dimensional dictionary in which outer label is
	keyed by known label, inner label is keyed by predicted label, and the value stored is the count
	of instances for each combination.  Works for an indefinite number of class labels.
	"""
	confusionCounts = {}
	for known, predicted in zip(knownY, predictedY):
		if confusionCounts[known] is None:
			confusionCounts[known] = {predicted:1}
		elif confusionCounts[known][predicted] is None:
			confusionCounts[known][predicted] = 1
		else:
			confusionCounts[known][predicted] += 1

	#if there are any entries in the square matrix confusionCounts,
	#then there value must be 0.  Go through and fill them in.
	for knownY in confusionCounts:
		if confusionCounts[knownY][knownY] is None:
			confusionCounts[knownY][knownY] = 0

	return confusionCounts

def print_confusion_matrix(confusionMatrix):
	""" Print a confusion matrix in human readable form, with
	rows indexed by known labels, and columns indexed by predictedlabels.
	confusionMatrix is a 2-dimensional dictionary, that is also primarily
	indexed by known labels, and secondarily indexed by predicted labels,
	with the value at confusionMatrix[knownLabel][predictedLabel] being the
	count of posts that fell into that slot.  Does not need to be sorted.
	"""
	#print heading
	print "*"*30 + "Confusion Matrix"+"*"*30
	print "\n\n"

	#print top line - just the column headings for
	#predicted labels
	spacer = " "*15
	sortedLabels = sorted(confusionMatrix.iterKeys())
	for knownLabel in sortedLabels:
		spacer += " "*(6 - len(knownLabel)) + knownLabel

	print spacer
	totalPostCount = 0
	for knownLabel in sortedLabels:
		outputBuffer = knownLabel+" "*(15 - len(knownLabel))
		for predictedLabel in sortedLabels:
			count = confusionMatrix[knownLabel][predictedLabel]
			totalPostCount += count
			outputBuffer += " "*(6 - len(count)) + count
		print outputBuffer

	print "Total post count: " + totalPostCount

def checkPrintConfusionMatrix():
	X = {"classLabel": ["A", "B", "C", "C", "B", "C", "A", "B", "C", "C", "B", "C", "A", "B", "C", "C", "B", "C"]}
	Y = ["A", "C", "C", "A", "B", "C", "A", "C", "C", "A", "B", "C", "A", "C", "C", "A", "B", "C"]
	functions = [confusion_matrix_generator]
	classLabelIndex = "classLabel"
	confusionMatrixResults = computeMetrics(classLabelIndex, X, Y, functions)
	confusionMatrix = confusionMatrixResults["confusion_matrix_generator"]
	print_confusion_matrix(confusionMatrix)



def computeError(knownValues, predictedValues, loopFunction, compressionFunction):
	"""
		A generic function to compute different kinds of error metrics.  knownValues
		is a 1d Base object with one known label (or number) per row. predictedValues is a 1d Base
		object with one predictedLabel (or score) per row.  The ith row in knownValues should refer
		to the same point as the ith row in predictedValues. loopFunction is a function to be applied
		to each row in knownValues/predictedValues, that takes 3 arguments: a known class label,
		a predicted label, and runningTotal, which contains the successive output of loopFunction.
		compressionFunction is a function that should take two arguments: runningTotal, the final
		output of loopFunction, and n, the number of values in knownValues/predictedValues.
	"""
	if knownValues is None or not isinstance(knownValues, Base) or knownValues.points == 0:
		raise ArgumentException("Empty 'knownValues' argument in error calculator")
	elif predictedValues is None or not isinstance(predictedValues, Base) or predictedValues.points == 0:
		raise ArgumentException("Empty 'predictedValues' argument in error calculator")

	if not isinstance(knownValues, DenseMatrixData):
		knownValues = knownValues.toDenseMatrixData()

	if not isinstance(predictedValues, DenseMatrixData):
		predictedValues = predictedValues.toDenseMatrixData()

	n=0.0
	runningTotal=0.0
	#Go through all values in known and predicted values, and pass those values to loopFunction
	for i in xrange(predictedValues.points()):
		pV = predictedValues.data[i][0]
		aV = knownValues.data[i][0]
		runningTotal = loopFunction(aV, pV, runningTotal)
		n += 1
	if n > 0:
		try:
			#provide the final value from loopFunction to compressionFunction, along with the
			#number of values looped over
			runningTotal = compressionFunction(runningTotal, n)
		except ZeroDivisionError:
			raise ZeroDivisionError('Tried to divide by zero when calculating performance metric')
			return
	else:
		raise ArgumentException("Empty argument(s) in error calculator")

	return runningTotal



def generateAllPairs(items):
	"""
		Given a list of items, generate a list of all possible pairs 
		(2-combinations) of items from the list, and return as a list
		of tuples.  Assumes that no two items in the list refer to the same
		object or number.  If there are duplicates in the input list, there
		will be duplicates in the output list.
	"""
	if items is None or len(items) == 0:
		return None

	pairs = []
	for i in range(len(items)):
		firstItem = items[i]
		for j in range(i+1, len(items)):
			secondItem = items[j]
			pair = (firstItem, secondItem)
			pairs.append(pair)

	return pairs

