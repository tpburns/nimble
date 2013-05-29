import numpy
from nose.tools import *
from ..metric_functions import rmse, classificationError, computeError, meanAbsoluteError, proportionPercentNegative50, proportionPercentNegative90, bottomProportionPercentNegative10
from ..performance_interface import computeMetrics
from ...processing.dense_matrix_data import DenseMatrixData
from UML import data
from UML.utility import ArgumentException

def testProportionPercentNegative():
	"""
	Unit test for proportionPercentNegative50/90
	"""
	knownLabelsOne = [[1], [2], [2], [2], [1], [1], [1], [2], [2], [2], [1], [2], [2], [2], [1], [1], [1], [2], [2], [2]]
	knownLabelsTwo = [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
	knownLabelsThree = [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]]
	knownLabelsFour = [[2], [1], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2], [2]]

	knownLabelsOneBaseData = data('dense', knownLabelsOne, sendToLog=False)
	knownLabelsTwoBaseData = data('dense', knownLabelsTwo, sendToLog=False)
	knownLabelsThreeBaseData = data('dense', knownLabelsThree, sendToLog=False)
	knownLabelsFourBaseData = data('dense', knownLabelsFour, sendToLog=False)

	predictedScoreList = []
	for i in range (20):
		oneScore = i * 0.05
		twoScore = 1.0 - i * 0.05
		predictedScoreList.append([oneScore, twoScore])

	predictedScoreListBaseData = data('dense', predictedScoreList, ['1', '2'])

	topHalfProportionNegativeOne = proportionPercentNegative50(knownLabelsOneBaseData, predictedScoreListBaseData, negativeLabel='1')
	topNinetyProportionNegativeOne = proportionPercentNegative90(knownLabelsOneBaseData, predictedScoreListBaseData, negativeLabel='1')
	topHalfProportionNegativeTwo = proportionPercentNegative50(knownLabelsTwoBaseData, predictedScoreListBaseData, negativeLabel='1')
	topNinetyProportionNegativeTwo = proportionPercentNegative90(knownLabelsTwoBaseData, predictedScoreListBaseData, negativeLabel='1')
	topHalfProportionNegativeThree = proportionPercentNegative50(knownLabelsThreeBaseData, predictedScoreListBaseData, negativeLabel='1')
	topNinetyProportionNegativeThree = proportionPercentNegative90(knownLabelsThreeBaseData, predictedScoreListBaseData, negativeLabel='1')
	topHalfProportionNegativeFour = proportionPercentNegative50(knownLabelsFourBaseData, predictedScoreListBaseData, negativeLabel='1')
	topNinetyProportionNegativeFour = proportionPercentNegative90(knownLabelsFourBaseData, predictedScoreListBaseData, negativeLabel='1')
	
	assert topHalfProportionNegativeOne == 0.4
	assert topNinetyProportionNegativeOne >= 0.443 and topNinetyProportionNegativeOne <= 0.445
	assert topHalfProportionNegativeTwo == 0.0
	assert topNinetyProportionNegativeTwo >= 0.443 and topNinetyProportionNegativeTwo <= 0.445
	assert topHalfProportionNegativeThree == 0.0
	assert topNinetyProportionNegativeThree == 0.0
	assert topHalfProportionNegativeFour == 0.10
	assert topNinetyProportionNegativeFour >= 0.0554 and topNinetyProportionNegativeFour <= 0.0556


#####################################
# performance combinations function #
#####################################
def testPerfCombinations():
	knownLabels = numpy.array([1.0,2.0,3.0])
	predictedLabels = numpy.array([1.0,2.0,3.0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	metricFunctions = [rmse, meanAbsoluteError, classificationError]
	results = computeMetrics(knownLabelsDense, None, predictedLabelsDense, metricFunctions)
	assert results[rmse] == 0.0
	assert results[meanAbsoluteError] == 0.0
	assert results[classificationError] == 0.0

	knownLabels = numpy.array([1.5,2.5,3.5])
	predictedLabels = numpy.array([1.0,2.0,3.0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	metricFunctions = [rmse, meanAbsoluteError, classificationError]
	results = computeMetrics(knownLabelsDense, None, predictedLabelsDense, metricFunctions)
	assert results[rmse] > 0.49
	assert results[rmse] < 0.51
	assert results[meanAbsoluteError] > 0.49
	assert results[meanAbsoluteError] < 0.51

############################
# generic error calculator #
############################
@raises(ArgumentException)
def testGenericErrorCalculatorEmptyKnownInput():
	"""
		Test that computeError raises an exception if knownLabels is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1,2,3])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	computeError(knownLabelsDense, predictedLabelsDense, lambda x,y,z: z, lambda x,y: x)

@raises(ArgumentException)
def testGenericErrorCalculatorEmptyPredictedInput():
	"""
		Test that computeError raises an exception if predictedLabels is empty
	"""
	knownLabels = numpy.array([1,2,3])
	predictedLabels = numpy.array([])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	computeError(knownLabelsDense, predictedLabelsDense, lambda x,y,z: z, lambda x,y: x)

@raises(ZeroDivisionError)
def testGenericErrorCalculatorDivideByZero():
	"""
		Test that computeError raises a divide by zero exception if the outerFunction argument
		would lead to division by zero.
	"""
	knownLabels = numpy.array([1,2,3])
	predictedLabels = numpy.array([1,2,3])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	computeError(knownLabelsDense, predictedLabelsDense, lambda x,y,z: z, lambda x,y: y/x)

def testGenericErrorCalculator():
	knownLabels = numpy.array([1.0, 2.0, 3.0])
	predictedLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	sameRate = computeError(knownLabelsDense, predictedLabelsDense, lambda x,y,z: z, lambda x,y: x)
	assert sameRate == 0.0

###########################
# Root mean squared error #
###########################


@raises(ArgumentException)
def testRmseEmptyKnownValues():
	"""
		Check that the rmse calculator correctly throws an
		exception if knownLabels vector is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1, 2, 3])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	rmseRate = rmse(knownLabelsDense, predictedLabelsDense)

@raises(ArgumentException)
def testRmseEmptyPredictedValues():
	"""
		Check that the rmse calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = numpy.array([])
	knownLabels = numpy.array([1, 2, 3])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	rmseRate = rmse(knownLabelsDense, predictedLabelsDense)


def testRmse():
	"""
		Check that the rmse calculator works correctly when
		all inputs are zero, and when all known values are
		the same as predicted values.
	"""
	predictedLabels = numpy.array([0,0,0])
	knownLabels = numpy.array([0,0,0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	rmseRate = rmse(knownLabelsDense, predictedLabelsDense)
	assert rmseRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	rmseRate = rmse(knownLabelsDense, predictedLabelsDense)
	assert rmseRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.5, 2.5, 3.5])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	rmseRate = rmse(knownLabelsDense, predictedLabelsDense)
	assert rmseRate > 0.49
	assert rmseRate < 0.51

#######################
# Mean Absolute Error #
#######################
@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyKnownValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if knownLabels vector is empty
	"""
	knownLabels = numpy.array([])
	predictedLabels = numpy.array([1, 2, 3])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsDense, predictedLabelsDense)

@raises(ArgumentException)
def testMeanAbsoluteErrorEmptyPredictedValues():
	"""
		Check that the mean absolute error calculator correctly throws an
		exception if predictedLabels vector is empty
	"""
	predictedLabels = numpy.array([])
	knownLabels = numpy.array([1, 2, 3])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsDense, predictedLabelsDense)

def testMeanAbsoluteError():
	"""
		Check that the mean absolute error calculator works correctly when
		all inputs are zero, or predictions are exactly the same as all known
		values, and are non-zero
	"""
	predictedLabels = numpy.array([0,0,0])
	knownLabels = numpy.array([0,0,0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsDense, predictedLabelsDense)
	assert maeRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.0, 2.0, 3.0])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsDense, predictedLabelsDense)
	assert maeRate == 0.0

	predictedLabels = numpy.array([1.0, 2.0, 3.0])
	knownLabels = numpy.array([1.5, 2.5, 3.5])

	knownLabelsDense = DenseMatrixData(knownLabels)
	predictedLabelsDense = DenseMatrixData(predictedLabels)

	maeRate = meanAbsoluteError(knownLabelsDense, predictedLabelsDense)
	assert maeRate > 0.49
	assert maeRate < 0.51


########################
# Classification Error #
########################

#@raises(ArgumentException)
#def testClassificationErrorEmptyKnownValues():

