"""
Module containing most of the user facing functions for the top level uml import.

"""

import numpy
import scipy.sparse
import inspect
import operator
import re 
import datetime
import os
import copy
import ConfigParser

import UML
from UML.exceptions import ArgumentException

from UML.logger import UmlLogger
from UML.logger import Stopwatch

from UML.helpers import findBestInterface
from UML.helpers import _learnerQuery
from UML.helpers import _validScoreMode
from UML.helpers import _validMultiClassStrategy
from UML.helpers import _unpackLearnerName
from UML.helpers import _validArguments
from UML.helpers import _validData
from UML.helpers import _2dOutputFlagCheck
from UML.helpers import LearnerInspector
from UML.helpers import copyLabels
from UML.helpers import ArgumentIterator
from UML.helpers import trainAndApplyOneVsAll
from UML.helpers import trainAndApplyOneVsOne
from UML.helpers import _mergeArguments
from UML.helpers import crossValidateBackend
from UML.helpers import isAllowedRaw
from UML.helpers import initDataObject
from UML.helpers import createDataFromFile

from UML.randomness import numpyRandom

from UML.interfaces.interface_helpers import checkClassificationStrategy

from UML.calculate import detectBestResult



UMLPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def createRandomData(returnType, numPoints, numFeatures, sparsity, numericType="float", featureNames=None, name=None):
	"""
	Generates a data object with random contents and numPoints points and numFeatures features. 

	If numericType is 'float' (default) then the value of (point, feature) pairs are sampled from a normal
	distribution (location 0, scale 1).

	If numericType is 'int' then value of (point, feature) pairs are sampled from uniform integer distribution [1 100].

	The sparsity is the likelihood that the value of a (point,feature) pair is zero.

	Zeros are not counted in/do not affect the aforementioned sampling distribution.
	"""

	if numPoints < 1:
		raise ArgumentException("must specify a positive nonzero number of points")
	if numFeatures < 1:
		raise ArgumentException("must specify a positive nonzero number of features")
	if sparsity < 0 or sparsity >=1:
		raise ArgumentException("sparsity must be greater than zero and less than one")
	if numericType != "int" and numericType != "float":
		raise ArgumentException("numericType may only be 'int' or 'float'")


	#note: sparse is not stochastic sparsity, it uses rigid density measures
	if returnType.lower() == 'sparse':

		density = 1.0 - float(sparsity)
		numNonZeroValues = int(numPoints * numFeatures * density)

		pointIndices = numpyRandom.randint(low=0, high=numPoints, size=numNonZeroValues)
		featureIndices = numpyRandom.randint(low=0, high=numFeatures, size=numNonZeroValues)

		if numericType == 'int':
			dataVector = numpyRandom.randint(low=1, high=100, size=numNonZeroValues)
		#numeric type is float; distribution is normal
		else: 
			dataVector = numpyRandom.normal(0, 1, size=numNonZeroValues) 

		#pointIndices and featureIndices are 
		randData = scipy.sparse.coo.coo_matrix((dataVector, (pointIndices, featureIndices)), (numPoints, numFeatures))
			
	#for non-sparse matrices, use numpy to generate matrices with sparsity characterics
	else:
		if numericType == 'int':
			filledIntMatrix = numpyRandom.randint(1, 100, (numPoints, numFeatures))
		else:
			filledFloatMatrix = numpyRandom.normal(loc=0.0, scale=1.0, size=(numPoints,numFeatures))

		#if sparsity is zero
		if abs(float(sparsity) - 0.0) < 0.0000000001:
			if numericType == 'int':
				randData = filledIntMatrix
			else:
				randData = filledFloatMatrix
		else:
			binarySparsityMatrix = numpyRandom.binomial(1, 1.0-sparsity, (numPoints, numFeatures))

			if numericType == 'int':
				randData = binarySparsityMatrix * filledIntMatrix
			else:
				randData = binarySparsityMatrix * filledFloatMatrix

	return createData(returnType, data=randData, featureNames=featureNames, name=name)


def normalizeData(learnerName, trainX, trainY=None, testX=None, arguments={}, **kwarguments):
	"""
	Calls on the functionality of a package to train on some data and then modify both
	the training data and a set of test data according to the produced model.

	Parameters:

	learnerName : String name of the learner to be called, in the form 'package.learner'

	trainX: data to be used for training (as some form of UML data Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an identifier (numerical
	index or string name) that defines their placement in the trainX object as a
	feature ID.

	testX: data set to be used for testing (as some form of Base object)

	arguments : dictionary mapping argument names (strings) to their values,
	to be used during training and application. example: {'dimensions':5, 'k':5}

	**kwarguments : kwargs specified variables that are passed to the learner. Same
	format as the arguments parameter.

	"""
	(packName, trueLearnerName) = _unpackLearnerName(learnerName)

	tl = UML.train(learnerName, trainX, trainY, arguments, **kwarguments)
	normalizedTrain = tl.apply(trainX, arguments=arguments, **kwarguments)
	if normalizedTrain.getTypeString() != trainX.getTypeString():
		normalizedTrain = normalizedTrain.copyAs(trainX.getTypeString())

	if testX is not None:
		normalizedTest = tl.apply(testX, arguments=arguments, **kwarguments)
		if normalizedTest.getTypeString() != testX.getTypeString():
			normalizedTest = normalizedTest.copyAs(testX.getTypeString())

	# modify references and names for trainX and testX
	trainX.referenceDataFrom(normalizedTrain)
	trainX.name = trainX.name + " " + trueLearnerName

	if testX is not None:
		testX.referenceDataFrom(normalizedTest)
		testX.name = testX.name + " " + trueLearnerName

def registerCustomLearnerAsDefault(customPackageName, learnerClassObject):
	""" Register the given customLearner class so that it is callable by the
	top level UML functions through the interface of the specified custom
	package. This operation modifies the saved configuration file so that
	this change will be reflected during future sesssions.

	customPackageName : The string name of the package preface you want to use when calling
	the learner. If there is already an interface for a custom package with this name, the
	learner will be accessible through that interface. If there is no interface to a custom
	package of that name, then one will be created. You cannot register a custom learner to
	be callable through the interface for a non-custom package (such as ScikitLearn or MLPY).
	Therefore, customPackageName cannot be a value which is the accepted alias of another
	package's interface.

	learnerClassObject : The class object implementing the learner you want registered. It
	will be checked using UML.interfaces.CustomLearner.validateSubclass to ensure that all
	details of the provided implementation are acceptable.

	"""
	UML.helpers.registerCustomLearnerBackend(customPackageName, learnerClassObject, True)

def registerCustomLearner(customPackageName, learnerClassObject):
	"""
	Register the given customLearner class so that it is callable by the
	top level UML functions through the interface of the specified custom
	package. Though this operation by itself is temporary, it has effects
	in UML.settings, so subsequent saveChanges operations may cause it to
	be reflected in future sessions.

	customPackageName : The string name of the package preface you want to use when calling
	the learner. If there is already an interface for a custom package with this name, the
	learner will be accessible through that interface. If there is no interface to a custom
	package of that name, then one will be created. You cannot register a custom learner to
	be callable through the interface for a non-custom package (such as ScikitLearn or MLPY).
	Therefore, customPackageName cannot be a value which is the accepted alias of another
	package's interface.

	learnerClassObject : The class object implementing the learner you want registered. It
	will be checked using UML.interfaces.CustomLearner.validateSubclass to ensure that all
	details of the provided implementation are acceptable.

	"""
	UML.helpers.registerCustomLearnerBackend(customPackageName, learnerClassObject, False)


def deregisterCustomLearnerAsDefault(customPackageName, learnerName):
	"""
	Remove accessibility of the learner with the given name from the
	interface of the package with the given name permenantly. This
	operation modifies the saved configuration file so that this
	change will be reflected during future sesssions.

	customPackageName : the name of the interface / custom package from which the learner
	named 'learnerName' is to be removed from. If that learner was the last one grouped in
	that custom package, then the interface is removed from the UML.interfaces.available list.

	learnerName : the name of the learner to be removed from the interface / custom package with
	the name 'customPackageName'

	"""
	UML.helpers._deregisterCustomLearnerBackend(customPackageName, learnerName, True)

def deregisterCustomLearner(customPackageName, learnerName):
	"""
	Remove accessibility of the learner with the given name from the
	interface of the package with the given name temporarily in this
	session. This has effects in UML.settings, so subsequent saveChanges
	operations may cause it to be reflected in future sessions.

	customPackageName : the name of the interface / custom package from which the learner
	named 'learnerName' is to be removed from. If that learner was the last one grouped in
	that custom package, then the interface is removed from the UML.interfaces.available list.

	learnerName : the name of the learner to be removed from the interface / custom package with
	the name 'customPackageName'

	"""
	UML.helpers.deregisterCustomLearnerBacked(customPackageName, learnerName, False)


def learnerParameters(name):
	"""
	Takes a string of the form 'package.learnerName' and returns a list of
	strings which are the names of the parameters when calling package.learnerName

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty list. 

	"""
	return _learnerQuery(name, 'parameters')

def learnerDefaultValues(name):
	"""
	Takes a string of the form 'package.learnerName' and returns a returns a
	dict mapping of parameter names to their default values when calling
	package.learnerName

	If the name cannot be found within the package, then an exception will be thrown.
	If the name is found, be for some reason we cannot determine what the parameters
	are, then we return None. Note that if we have determined that there are no
	parameters, we return an empty dict. 

	"""
	return _learnerQuery(name, 'defaults')


def listLearners(package=None):
	"""
	Takes the name of a package, and returns a list of learners that are callable through that
	package's trainAndApply() interface.

	"""
	results = []
	if package is None:
		for interface in UML.interfaces.available:
			packageName = interface.getCanonicalName()
			currResults = interface.listLearners()
			for learnerName in currResults:
				results.append(packageName + "." + learnerName)
	else:
		interface = findBestInterface(package)
		currResults = interface.listLearners()
		for learnerName in currResults:
			results.append(learnerName)

	return results


def listDataFunctions():
	methodList = dir(UML.data.Base)
	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = getattr(UML.data.Base, methodName)
		try:
			(args, varargs, keywords, defaults) = inspect.getargspec(currMethod)
		except TypeError:
			continue

		retString = methodName + "("
		for i in xrange(0, len(args)):
			if i != 0:
				retString += ", "
			retString += args[i]
			if defaults is not None and i >= (len(args) - len(defaults)):
				retString += "=" + str(defaults[i - (len(args) - len(defaults))])
			
		# obliterate the last comma
		retString += ")"
		ret.append(retString)

	return ret


def listUMLFunctions():
	methodList = dir(UML)

	visibleMethodList = []
	for methodName in methodList:
		if not methodName.startswith('_'):
			visibleMethodList.append(methodName)

	ret = []
	for methodName in visibleMethodList:
		currMethod = eval("UML." + methodName)
		if "__call__" not in dir(currMethod):
			continue
		(args, varargs, keywords, defaults) = inspect.getargspec(currMethod)

		retString = methodName + "("
		for i in xrange(0, len(args)):
			if i != 0:
				retString += ", "
			retString += args[i]
			if defaults is not None and i >= (len(args) - len(defaults)):
				retString += "=" + str(defaults[i - (len(args) - len(defaults))])
			
		# obliterate the last comma
		retString += ")"
		ret.append(retString)

	return ret


def createData(returnType, data, pointNames=None, featureNames=None, fileType=None,
				name=None, ignoreNonNumericalFeatures=False, useLog=None):
	"""Function to instantiate one of the UML data container types.

	returnType: string (or None) indicating which kind of UML data type you want
	returned. If None is given, UML will attempt to detect the type most
	appropriate for the data. Currently accepted are the strings "List",
	"Matrix", and "Sparse" -- which are case sensitive.

	data: the source of the data to be loaded into the returned object. The
	source may be any number of in-python objects (lists, numpy arrays, numpy
	matrices, scipy sparse objects) as long as they specify a 2d matrix of
	data. Alternatively, the data may be read from a file, specified either
	as a string path, or a currently open file-like object.

	pointNames: the source for point names in the returned object. They may
	be specified explictly by some list-like or dict-like object, so long
	as all points in the data are assigned a name and the names for each
	point are unique.. If the point names are imbedded in the data, then a
	valid point index may be passed to this argument, and that point will be
	extracted and assigned (note: this works regardless of whether the data
	is sourced from an in-python object or a file). Finally, if this argument
	is None, and the data is being loaded from a file, then it is possible
	the names will be automatically assigned.

	featureNames: the source for feature names in the returned object. They may
	be specified explictly by some list-like or dict-like object, so long
	as all points in the data are assigned a name and the names for each
	feature are unique. If the feature names are imbedded in the data, then a
	valid feature index may be passed to this argument, and that feature will
	be extracted and assigned as the names (note: this works regardless of
	whether the data is sourced from an in-python object or	a file). Finally,
	if this argument is None, and the data is being loaded from a file, then it
	is possible the names will be automatically assigned.

	fileType: allows the user to explictly specify the format expected when
	loading from a file. Normally, if a file is being loaded, the extension
	of the file name is used to indicate the format. However, if fileType is
	specified, it will override the file extension. Also, when loading from a
	file with no extension, the user is requred to specify a format via
	fileType. This argument is ignored if loading from a python object.
	Currently accepted values are "csv" and "mtx", with a default value of None

	name: When not None, this value is set as the name attribute of the
	returned object

	ignoreNonNumericalFeatures: True or False (default False) value indicating
	whether, when loading from a file, features containing non numercal data
	shouldn't be loaded into the final object. For example, you may be loading
	a file which has a column of strings; setting this flag to true will allow
	you to load that file into a Matrix object (which may contain floats only).
	Currently only has an effect on csv files, as the matrix market format
	does not support non numerical values.

	useLog: True, False, or None (default) valued flag indicating whether this
	call should be logged by the UML logger. If None, the configurable	global
	default is used.

	"""

	#retAllowed = ['List', 'Matrix', 'Sparse', None]
	retAllowed = copy.copy(UML.data.available)
	retAllowed.append(None)
	if returnType not in retAllowed:
		raise ArgumentException("returnType must be a value in " + str(retAllowed))

	def looksFileLike(toCheck):
		hasRead = hasattr(toCheck, 'read')
		hasWrite = hasattr(toCheck, 'write')
		return (hasRead and hasWrite)

	# input is raw data
	if isAllowedRaw(data):
		return initDataObject(returnType, data, pointNames, featureNames, name, None)
	# input is an open file or a path to a file
	elif isinstance(data, basestring) or looksFileLike(data):
		return createDataFromFile(returnType, data, pointNames, featureNames, fileType, name, ignoreNonNumericalFeatures)
	# no other allowed inputs
	else:
		raise ArgumentException("data must contain either raw data or the path to a file to be loaded")


def crossValidate(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', useLog=None, **kwarguments):
	"""
	K-fold cross validation.
	Returns mean performance (float) across numFolds folds on a X Y.

	Parameters:

	learnerName (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass or int index for X) - labels/data about points in X

	performanceFunction (function) - Look in UML.calculate for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form: def func(knownValues, predictedValues).

	arguments (dict) - dictionary mapping argument names (strings)
	to their values. The parameter is sent to trainAndApply() through its arguments
	parameter. example: {'dimensions':5, 'k':5}

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments - kwargs specified variables that are passed to the learner.
	To make use of multiple permutations, specify different values for a
	parameter as a tuple. eg. a=(1,2,3) will generate an error score for 
	the learner when the learner was passed all three values of a, separately.

	"""
	bestResult = crossValidateReturnBest(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, useLog, **kwarguments)
	return bestResult[1]
	#return crossValidateBackend(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, useLog, **kwarguments)

def crossValidateReturnAll(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', useLog=None, **kwarguments):
	"""
	Calculates the cross validated error for each argument permutation that can 
	be generated by the merge of arguments and kwarguments.

	example **kwarguments: {'a':(1,2,3), 'b':(4,5)}
	generates permutations of dict in the format:
	{'a':1, 'b':4}, {'a':2, 'b':4}, {'a':3, 'b':4}, {'a':1, 'b':5}, 
	{'a':2, 'b':5}, {'a':3, 'b':5}

	For each permutation of 'arguments', crossValidateReturnAll uses cross 
	validation to generate a performance score for the algorithm, given the 
	particular argument permutation.

	Returns a list of tuples, where every tuple contains a dict representing 
	the argument sent to trainAndApply, and a float represennting the cross 
	validated error associated with that argument dict.
	example list element: ({'arg1':2, 'arg2':'max'}, 89.0000123)

	Arguments:

	learnerName (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass or int index for X) - labels/data about points in X

	performanceFunction (function) - Look in UML.calculate for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form: def func(knownValues, predictedValues).

	arguments (dict) - dictionary mapping argument names (strings)
	to their values, to be merged with kwargs. To make use of multiple
	permutations, specify different values for a parameter as a tuple. eg.
	a=(1,2,3) will generate an error score for  the learner when the learner
	was passed all three values of a, separately.

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments - kwargs specified variables that are passed to the learner,
	after being merged with arguments. To make use of multiple permutations,
	specify different values for a parameter as a tuple. eg. a=(1,2,3) will
	generate an error score for the learner when the learner was passed all
	three values of a, separately.

	"""	
	return crossValidateBackend(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, useLog, **kwarguments)


def crossValidateReturnBest(learnerName, X, Y, performanceFunction, arguments={},
			numFolds=10, scoreMode='label', useLog=None,
			maximumIsBest='Automatic', **kwarguments):
	"""
	For each possible argument permutation generated by arguments, 
	crossValidateReturnBest runs crossValidate to compute a mean error for the 
	argument combination. 

	crossValidateReturnBest then RETURNS the best argument and error as a tuple:
	(argument_as_dict, cross_validated_performance_float)

	Arguments:
	learnerName (string) - UML compliant algorithm name in the form 
	'package.algorithm' e.g. 'sciKitLearn.KNeighborsClassifier'

	X (UML.Base subclass) - points/features data

	Y (UML.Base subclass or int index for X) - labels/data about points in X

	performanceFunction (function) - Look in UML.calculate for premade options.
	Function used by computeMetrics to generate a performance score for the run.
	function is of the form: def func(knownValues, predictedValues).

	arguments (dict) - dictionary mapping argument names (strings)
	to their values, to be merged with kwargs. To make use of multiple
	permutations, specify different values for a parameter as a tuple. eg.
	a=(1,2,3) will generate an error score for  the learner when the learner
	was passed all three values of a, separately.

	numFolds (int) - the number of folds used in the cross validation. Can't
	exceed the number of points in X, Y

	scoreMode - used by computeMetrics

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	maximumIsBest - Controls the definition of optimality for choosing
	the best score and parameter set. If True, then the the highest score
	will be the best score. If False, the lowerest score will be best.
	By default, this is set to 'Automatic' which attempts to determine
	the best score using a few simple experiments.

	kwarguments - kwargs specified variables that are passed to the learner,
	after being merged with arguments. To make use of multiple permutations,
	specify different values for a parameter as a tuple. eg. a=(1,2,3) will
	generate an error score for the learner when the learner was passed all
	three values of a, separately.

	"""
	resultsAll = crossValidateReturnAll(learnerName, X, Y, performanceFunction, arguments, numFolds, scoreMode, useLog, **kwarguments)

	bestArgumentAndScoreTuple = None

	if isinstance(maximumIsBest, basestring) and maximumIsBest.lower() == 'automatic':
		detected = detectBestResult(performanceFunction)
		if detected == 'max':
			maximumIsBest = True
		elif detected == 'min':
			maximumIsBest = False
		else:
			msg = "Unable to automatically determine whether maximal or "
			msg += "minimal scores are considered optimal for the the "
			msg += "given performanceFunction. This must therefore be "
			msg += "specified directly using the 'maximumIsBest' argument, "
			msg += "using either True or False." 

	for curResultTuple in resultsAll:
		curArgument, curScore = curResultTuple
		#if curArgument is the first or best we've seen: 
		#store its details in bestArgumentAndScoreTuple
		if bestArgumentAndScoreTuple is None:
			bestArgumentAndScoreTuple = curResultTuple
		else:
			if (maximumIsBest and curScore > bestArgumentAndScoreTuple[1]):
				bestArgumentAndScoreTuple = curResultTuple
			if ((not maximumIsBest) and curScore < bestArgumentAndScoreTuple[1]):
				bestArgumentAndScoreTuple = curResultTuple

	return bestArgumentAndScoreTuple


def learnerType(learnerNames):
	"""
	Returns the string or list of strings representation of a best guess for 
	the type of learner(s) specified by the learner name(s) in learnerNames.

	If learnerNames is a single string (not a list of strings), then only a single 
	result is returned, instead of a list.
	
	LearnerType first queries the appropriate interface object for a definitive return
	value. If the interface doesn't provide a satisfactory answer, then this method
	calls a backend which generates a series of artificial data sets with particular
	traits to look for heuristic evidence of a classifier, regressor, etc.
	"""
	#argument checking
	if not isinstance(learnerNames, list):
		learnerNames = [learnerNames]

	resultsList = []
	secondPassLearnerNames = []
	for name in learnerNames:
		if not isinstance(name, str):
			raise ArgumentException("learnerNames must be a string or a list of strings.")

		splitTuple = _unpackLearnerName(name)
		currInterface = findBestInterface(splitTuple[0])
		allValidLearnerNames = currInterface.listLearners()
		if not splitTuple[1] in allValidLearnerNames:
			raise ArgumentException(name + " is not a valid learner on your machine.")
		result = currInterface.learnerType(splitTuple[1])
		if result == 'UNKNOWN' or result == 'other' or result is None:
			resultsList.append(None)
			secondPassLearnerNames.append(name)
		else:
			resultsList.append(result)
			secondPassLearnerNames.append(None)
		
	#have valid arguments - a list of learner names
	learnerInspectorObj = LearnerInspector()

	for index in range(len(secondPassLearnerNames)):
		curLearnerName = secondPassLearnerNames[index]
		if curLearnerName is None:
			continue
		resultsList[index] = learnerInspectorObj.learnerType(curLearnerName)

	#if only one algo was requested, remove type from list an return as single string
	if len(resultsList) == 1:
		resultsList = resultsList[0]

	return resultsList


def train(learnerName, trainX, trainY=None, arguments={},  multiClassStrategy='default', useLog=None, **kwarguments):
	"""
	Trains and returns the specified learner using the provided data. The return value is a
	UniversalInterface.trainedLearner object.

	ARGUMENTS:
	
	learnerName: algorithm to be called, in the form 'package.learnerName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the traing data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on.

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'
	
	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: The collection of extra key:value argument pairs included in this call to
	train(). They will be merged with the arguments dict, and passed on through to the
	learner.

	"""
	(package, learnerName) = _unpackLearnerName(learnerName)
	_validData(trainX, trainY, None, None, [False, False])
	_validArguments(arguments)
	_validArguments(kwarguments)
	_2dOutputFlagCheck(trainX, trainY, None, multiClassStrategy)
	merged = _mergeArguments(arguments, kwarguments)

	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	if useLog:
		timer = Stopwatch()
	else:
		timer = None

	interface = findBestInterface(package)

	# TODO how do we do multiclassStrategy?

	trainedLearner = interface.train(learnerName, trainX, trainY, merged, timer)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, None, None, funcString, None, None, None, timer, extraInfo=merged)

	return trainedLearner

def trainAndApply(learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', useLog=None, **kwarguments):
	"""
	Trains and returns the results of applying the learner to the test data (i.e.
	performing prediction, transformation, etc. as appropriate to the learner).

	ARGUMENTS:
	
	learnerName: algorithm to be called, in the form 'package.learnerName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	testX: data set on which the trained learner will be applied (i.e. performing
	prediction, transformation, etc. as appropriate to the learner). Must be
	some form of UML data Base object. 

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on.

	output: The kind of UML data object that the output of this function should be
	in. Any of the normal string inputs to the createData 'returnType' parameter are
	accepted here. Alternatively, the value 'match' will indicate to use the type
	of the 'trainX' parameter.

	scoreMode: In the case of a classifying learner, this specifies the type of output
	wanted: 'label' if we class labels are desired, 'bestScore' if both the class
	label and the score associated with that class are desired, or 'allScores' if
	a matrix containing the scores for every class label are desired.

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'
	
	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: The collection of extra key:value argument pairs included in this call to
	train(). They will be merged with the arguments dict, and passed on through to the
	learner.

	"""
	(package, learnerName) = _unpackLearnerName(learnerName)
	fullName = package + '.' + learnerName
	_validData(trainX, trainY, testX, None, [False, False])
	_validScoreMode(scoreMode)
	_validMultiClassStrategy(multiClassStrategy)
	_validArguments(arguments)
	_validArguments(kwarguments)
	_2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)
	merged = _mergeArguments(arguments, kwarguments)

	if testX is None:
		testX = trainX

	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	if useLog:
		timer = Stopwatch()
	else:
		timer = None

	interface = findBestInterface(package)

	results = None
	if multiClassStrategy != 'default':
		trialResult = checkClassificationStrategy(interface, learnerName, merged)
		# We only use our own version of the strategy if the internal method is different than
		# what we want.
		if multiClassStrategy == 'OneVsAll' and trialResult != 'OneVsAll':
			results = trainAndApplyOneVsAll(fullName, trainX, trainY, testX, arguments=merged, scoreMode=scoreMode, useLog=useLog, timer=timer)
		if multiClassStrategy == 'OneVsOne' and trialResult != 'OneVsOne':
			results = trainAndApplyOneVsOne(fullName, trainX, trainY, testX, arguments=merged, scoreMode=scoreMode, useLog=useLog, timer=timer)

	if results is None:
		results = interface.trainAndApply(learnerName, trainX, trainY, testX, merged, output, scoreMode, timer)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, testX, None, funcString, None, results, None, timer, extraInfo=merged)

	return results


def trainAndTest(learnerName, trainX, trainY, testX, testY, performanceFunction, arguments={}, output=None, scoreMode='label', useLog=None, **kwarguments):
	"""
	For each permutation of the merge of 'arguments' and 'kwarguments' (more below),
	trainAndTest uses cross validation to generate a performance score for the algorithm,
	given the particular argument permutation. The argument permutation that performed
	best cross validating over the training data is then used as the lone argument for
	training on the whole training data set. Finally, the learned model generates
	predictions for the testing set, an the performance of those predictions is
	calculated and returned.

	If no additional arguments are supplied via arguments or **kwarguments, then
	trainAndTest just returns the performance of the algorithm with default arguments
	on the testing data.

	ARGUMENTS:

	learnerName: training algorithm to be called, in the form 'package.algorithmName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	testX: data set to be used for testing (as some form of Base object)

	testY: used to retrieve the known class labels of the test data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their location in the testX object.

	performanceFunction: Function used by computeMetrics to generate a performance score
	for the run. function is of the form: def func(knownValues, predictedValues).
	Look in UML.calculate for pre-made options.

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on. The syntax for prescribing
	different arguments for algorithm: arguments of the form {arg1=(1,2,3), arg2=(4,5,6)}
	correspond to permutations/argument states with one element from arg1 and one element 
	from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: optional arguments to be passed to the specified learner. Will be merged
	with the arguments parameter before being passed on to the learner.
	The syntax for prescribing different arguments for algorithm:
	**kwarguments of the form arg1=(1,2,3), arg2=(4,5,6)
	correspond to permutations/argument states with one element from arg1 and one element 
	from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

	"""
	(package, trueLearnerName) = _unpackLearnerName(learnerName)
	_validData(trainX, trainY, testX, testY, [True, True])
	_validArguments(arguments)
	_validArguments(kwarguments)
	#_2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)

	_2dOutputFlagCheck(trainX, trainY, scoreMode, None)
	merged = _mergeArguments(arguments, kwarguments)

	trainY = copyLabels(trainX, trainY)
	testY = copyLabels(testX, testY)

	interface = findBestInterface(package)
	
	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	#if we are logging this run, we need to start the timer
	if useLog:
		timer = Stopwatch()
		timer.start('crossValidateReturnBest')
	else:
		timer = None

	# perform CV (if needed)
	argCheck = ArgumentIterator(merged)
	if argCheck.numPermutations != 1:
		#modify numFolds if needed
		numFolds = trainX.pointCount if trainX.pointCount < 10 else 10
		#sig (learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', useLog=None, maximize=False, **kwarguments):
		bestArgument, bestScore = UML.crossValidateReturnBest(learnerName, trainX, trainY, performanceFunction, merged, numFolds=numFolds, scoreMode=scoreMode, useLog=useLog)
	else:
		bestArgument = merged

	if useLog:
		timer.stop('crossValidateReturnBest')

	predictions = interface.trainAndApply(trueLearnerName, trainX, trainY, testX, arguments=bestArgument, output=output, scoreMode=scoreMode, timer=timer)
	performance = UML.helpers.computeMetrics(testY, None, predictions, performanceFunction)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, testX, testY, funcString, [performanceFunction], predictions, [performance], timer, bestArgument)

	return performance


def trainAndTestOnTrainingData(learnerName, trainX, trainY, performanceFunction,
			crossValidationError=False, numFolds=10, arguments={}, output=None,
			scoreMode='label', useLog=None, **kwarguments):
	"""
	trainAndTestOnTrainingData is the function for doing learner creation
	and evaluation in a single step with only a single data set (no withheld
	testing set). By default, this will calculate training error for the
	learner trained on that data set. However, cross validation error can
	instead be calculated by setting the parameter crossVadiationError to be
	true. In that case, we will partition the training set into a parameter
	controlled number of folds, and iteratively withhold each single fold to be
	used as the testing set of the learner trained on the rest of the data.

	ARGUMENTS:

	learnerName: training algorithm to be called, in the form 'package.algorithmName'.

	trainX: data set to be used for training (as some form of Base object)

	trainY: used to retrieve the known class labels of the training data. Either
	contains the labels themselves (as a Base object) or an index (numerical or string) 
	that defines their locale in the trainX object

	performanceFunction: Function used by computeMetrics to generate a performance score
	for the run. function is of the form: def func(knownValues, predictedValues).
	Look in UML.calculate for pre-made options.

	crossValidationError: True or False, according to whether we will calculate
	cross validation error or training error. In True case, the training data
	is split in the numFolds number of partitions. Each of those is iteratively
	withheld and used as the testing set for a learner trained on the combination
	of all of the non-withheld data. The performance results for each of those
	tests are then averaged together to act as the return value. In the False
	case, we train on the training data, and then use the same data as the
	withheld testing data. By default, this flag is set to False.

	numFolds: the (int) number of folds used in the cross validation. Can't
	exceed the number of points in X, Y. Default is 10.

	arguments: dict containing the parameters to be passed to the learner, in the
	form of a mapping between (string) parameter names, and values. Will be merged
	with the contents of **kwarguments before being passed on. The syntax for prescribing
	different arguments for algorithm: arguments of the form {arg1=(1,2,3), arg2=(4,5,6)}
	correspond to permutations/argument states with one element from arg1 and one element 
	from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

	output: Can be used to force a format on the data object resulting from applying
	the learner to whatever is considered the testing data. This object is not seen
	by the user, but will be used by the input performanceFunction, and if the user
	has prior knowledge of requirements of that function, then they can enforced
	by UML instead of manually. By default, this parameter is set to None, indicating
	to match the formatting of the training data objects.

	multiClassStrategy: may only be 'default' 'OneVsAll' or 'OneVsOne'

	useLog - local control for whether to send results/timing to the logger.
	If None (default), use the value as specified in the "logger"
	"enabledByDefault" configuration option. If True, send to the logger
	regardless of the global option. If False, do NOT send to the logger,
	regardless of the global option.

	kwarguments: optional arguments to be passed to the specified learner. Will be merged
	with the arguments parameter before being passed on to the learner.
	The syntax for prescribing different arguments for algorithm:
	**kwarguments of the form arg1=(1,2,3), arg2=(4,5,6)
	correspond to permutations/argument states with one element from arg1 and one element 
	from arg2, such that an example generated permutation/argument state would be "arg1=2, arg2=4"

	"""
	(package, trueLearnerName) = _unpackLearnerName(learnerName)
	_validData(trainX, trainY, None, None, [False, False])
	_validArguments(arguments)
	_validArguments(kwarguments)
	#_2dOutputFlagCheck(trainX, trainY, scoreMode, multiClassStrategy)

	_2dOutputFlagCheck(trainX, trainY, scoreMode, None)
	merged = _mergeArguments(arguments, kwarguments)

	trainY = copyLabels(trainX, trainY)

	interface = findBestInterface(package)
	
	if useLog is None:
		useLog = UML.settings.get("logger", "enabledByDefault")
		useLog = True if useLog.lower() == 'true' else False

	# check that there are no free parameters?
	argCheck = ArgumentIterator(merged)
	if argCheck.numPermutations != 1:
		wrong = []
		for k in merged:
			v = merged[k]
			if isinstance(v, tuple):
				wrong.append((k,v))

		def kvFormater(kv):
			return str(k) + ":" + str(v)

		msg = "trainAndTestOnTrainingData() requires that there be no free parameters "
		msg += "in the input. "
		#msg += "When calling UML.trainAndTest(), passing a tuple as the "
		#msg += "value for any learner parameters indicates to UML to be select a "
		#msg += "a value using cross validation. However, in this function, cross "
		#msg += "validation is reserved for measuring prediction error.
		msg += "Therefore, the following inputed learner parameters (given as "
		msg += "key:value pairs) must be changed to have non-tuple values: "
		msg += UML.exceptions.prettyListString(wrong, useAnd=True, itemStr=kvFormater)

		raise ArgumentException(msg)

	# if we are logging this run, we need to setup the timer, to be passed
	# downward
	if useLog:
		timer = Stopwatch()
	else:
		timer = None

	if crossValidationError:
		if useLog:
			timer.start('crossValidationError')
		predictions = None
		#sig: crossValidate(learnerName, X, Y, performanceFunction, arguments={}, numFolds=10, scoreMode='label', useLog=None, **kwarguments):
		performance = crossValidate(learnerName, trainX, trainY, performanceFunction, arguments=merged, numFolds=numFolds, scoreMode=scoreMode, useLog=useLog)
		if useLog:
			timer.stop('crossValidationError')
	else:
		predictions = interface.trainAndApply(trueLearnerName, trainX, trainY, trainX, arguments=merged, output=output, scoreMode=scoreMode, timer=timer)
		performance = UML.helpers.computeMetrics(trainY, None, predictions, performanceFunction)

	if useLog:
		funcString = interface.getCanonicalName() + '.' + learnerName
		UML.logger.active.logRun(trainX, trainY, None, None, funcString, [performanceFunction], predictions, [performance], timer, merged)

	return performance
