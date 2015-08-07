

import UML
from UML.data import Base
from UML.data import Matrix
from math import sqrt

from UML.exceptions import ArgumentException

def _validatePredictedAsLabels(predictedValues):
	if not isinstance(predictedValues, UML.data.Base):
		raise ArgumentException("predictedValues must be derived class of UML.data.Base")
	if predictedValues.featureCount > 1:
		raise ArgumentException("predictedValues must be labels only; this has more than one feature")

def _computeError(knownValues, predictedValues, loopFunction, compressionFunction):
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
	if knownValues is None or not isinstance(knownValues, Base) or knownValues.pointCount == 0:
		raise ArgumentException("Empty 'knownValues' argument in error calculator")
	elif predictedValues is None or not isinstance(predictedValues, Base) or predictedValues.pointCount == 0:
		raise ArgumentException("Empty 'predictedValues' argument in error calculator")

	if not isinstance(knownValues, Matrix):
		knownValues = knownValues.copyAs(format="Matrix")

	if not isinstance(predictedValues, Matrix):
		predictedValues = predictedValues.copyAs(format="Matrix")

	n = 0.0
	runningTotal = 0.0
	#Go through all values in known and predicted values, and pass those values to loopFunction
	for i in xrange(predictedValues.pointCount):
		pV = predictedValues[i,0]
		aV = knownValues[i,0]
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


def rootMeanSquareError(knownValues, predictedValues):
	"""
		Compute the root mean square error.  Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	_validatePredictedAsLabels(predictedValues)
	return _computeError(knownValues, predictedValues, lambda x,y,z: z + (y - x)**2, lambda x,y: sqrt(x/y))

def meanFeaturewiseRootMeanSquareError(knownValues, predictedValues):
	"""For 2d prediction data, compute the RMSE of each feature, then average
	the results.
	"""
	if knownValues.featureCount != predictedValues.featureCount:
		raise ArgumentException("The known and predicted data must have the same number of features")
	if knownValues.pointCount != predictedValues.pointCount:
		raise ArgumentException("The known and predicted data must have the same number of points")

	results = []
	for i in xrange(knownValues.featureCount):
		currKnown = knownValues.copyFeatures(i)
		currPred = predictedValues.copyFeatures(i)
		results.append(rootMeanSquareError(currKnown, currPred))

	return float(sum(results)) / knownValues.featureCount


def meanAbsoluteError(knownValues, predictedValues):
	"""
		Compute mean absolute error. Assumes that knownValues and predictedValues contain
		numerical values, rather than categorical data.
	"""
	_validatePredictedAsLabels(predictedValues)
	return _computeError(knownValues, predictedValues, lambda x,y,z: z + abs(y - x), lambda x,y: x/y)

def fractionIncorrect(knownValues, predictedValues):
	"""
		Compute the proportion of incorrect predictions within a set of
		instances.  Assumes that values in knownValues and predictedValues are categorical.
	"""
	_validatePredictedAsLabels(predictedValues)
	return _computeError(knownValues, predictedValues, lambda x,y,z: z if x == y else z + 1, lambda x,y: x/y)
