from __future__ import absolute_import

import numpy
from nose.tools import *

import UML
from UML import createData
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination
from UML.calculate import meanAbsoluteError
from UML.calculate import rootMeanSquareError
from UML.calculate import meanFeaturewiseRootMeanSquareError
from UML.calculate import fractionCorrect
from UML.calculate import fractionIncorrect
from UML.calculate import rSquared
from UML.calculate import varianceFractionRemaining
from ..assertionHelpers import noLogEntryExpected

#################
# _computeError #
#################

@raises(InvalidArgumentValue)
def testGenericErrorCalculatorEmptyKnownInput():
    """
    Test that _computeError raises an exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: x)


@raises(InvalidArgumentValue)
def testGenericErrorCalculatorEmptyPredictedInput():
    """
    Test that _computeError raises an exception if predictedLabels is empty
    """
    knownLabels = numpy.array([1, 2, 3])
    predictedLabels = numpy.array([])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: x)


@raises(ZeroDivisionError)
def testGenericErrorCalculatorDivideByZero():
    """
    Test that _computeError raises a divide by zero exception if the
    outerFunction argument would lead to division by zero.
    """
    knownLabels = numpy.array([1, 2, 3])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z, lambda x, y: y / x)


def testGenericErrorCalculator():
    knownLabels = numpy.array([1.0, 2.0, 3.0])
    predictedLabels = numpy.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    sameRate = UML.calculate.loss._computeError(knownLabelsMatrix, predictedLabelsMatrix, lambda x, y, z: z,
                                                lambda x, y: x)
    assert sameRate == 0.0


#######################
# Mean Absolute Error #
#######################

@raises(InvalidArgumentValue)
def testMeanAbsoluteErrorEmptyKnownValues():
    """
    Check that the mean absolute error calculator correctly throws an
    exception if knownLabels vector is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testMeanAbsoluteErrorEmptyPredictedValues():
    """
    Check that the mean absolute error calculator correctly throws an
    exception if predictedLabels vector is empty
    """
    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testMeanAbsoluteError():
    """
    Check that the mean absolute error calculator works correctly when
    all inputs are zero, or predictions are exactly the same as all known
    values, and are non-zero
    """
    predictedLabels = numpy.array([0, 0, 0])
    knownLabels = numpy.array([0, 0, 0])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = numpy.array([1.0, 2.0, 3.0])
    knownLabels = numpy.array([1.0, 2.0, 3.0])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate == 0.0

    predictedLabels = numpy.array([1.0, 2.0, 3.0])
    knownLabels = numpy.array([1.5, 2.5, 3.5])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    knownLabelsMatrix.transpose(useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels, useLog=False)
    predictedLabelsMatrix.transpose(useLog=False)

    maeRate = meanAbsoluteError(knownLabelsMatrix, predictedLabelsMatrix)
    assert maeRate > 0.49
    assert maeRate < 0.51


###########################
# Root mean squared error #
###########################

@raises(InvalidArgumentValue)
def testRmseEmptyKnownValues():
    """
    rootMeanSquareError calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    rootMeanSquareErrorRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testRmseEmptyPredictedValues():
    """
    rootMeanSquareError calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testRmse():
    """
    Check that the rootMeanSquareError calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    predictedLabels = numpy.array([[0], [0], [0]])
    knownLabels = numpy.array([[0], [0], [0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.5], [2.5], [3.5]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    rmseRate = rootMeanSquareError(knownLabelsMatrix, predictedLabelsMatrix)
    assert rmseRate > 0.49
    assert rmseRate < 0.51


######################################
# meanFeaturewiseRootMeanSquareError #
######################################

@raises(InvalidArgumentValueCombination)
def testMFRMSE_badshapePoints():
    predictedLabels = numpy.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = numpy.array([[0, 0], [0, 0], [0, 0]])

    knowns = createData('Matrix', data=knownLabels)
    predicted = createData('Matrix', data=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)


@raises(InvalidArgumentValueCombination)
def testMFRMSE_badshapeFeatures():
    predictedLabels = numpy.array([[0], [0], [0], [0]])
    knownLabels = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = createData('Matrix', data=knownLabels)
    predicted = createData('Matrix', data=predictedLabels)

    meanFeaturewiseRootMeanSquareError(knowns, predicted)

@noLogEntryExpected
def testMFRMSE_simpleSuccess():
    predictedLabels = numpy.array([[0, 2], [0, 2], [0, 2], [0, 2]])
    knownLabels = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0]])

    knowns = createData('Matrix', data=knownLabels, useLog=False)
    predicted = createData('Matrix', data=predictedLabels, useLog=False)

    mfrmseRate = meanFeaturewiseRootMeanSquareError(knowns, predicted)
    assert mfrmseRate == 1.0

###################
# fractionCorrect #
###################

@raises(InvalidArgumentValue)
def testFractionCorrectEmptyKnownValues():
    """
    fractionCorrect calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testFractionCorrectEmptyPredictedValues():
    """
    fractionCorrect calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testFractionCorrect():
    """
    Check that the fractionCorrect calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    predictedLabels = numpy.array([[0], [0], [0]])
    knownLabels = numpy.array([[0], [0], [0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 1.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    fc = fractionCorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fc == 0.5

#####################
# fractionIncorrect #
#####################

@raises(InvalidArgumentValue)
def testFractionIncorrectEmptyKnownValues():
    """
    fractionIncorrect calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValue)
def testFractionIncorrectEmptyPredictedValues():
    """
    fractionIncorrect calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testFractionIncorrect():
    """
    Check that the fractionIncorrect calculator works correctly when
    all inputs are zero, and when all known values are
    the same as predicted values.
    """
    predictedLabels = numpy.array([[0], [0], [0]])
    knownLabels = numpy.array([[0], [0], [0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [1.0], [2.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    fi = fractionIncorrect(knownLabelsMatrix, predictedLabelsMatrix)
    assert fi == 0.5

#############################
# varianceFractionRemaining #
#############################

@raises(InvalidArgumentValueCombination)
def testVarianceFractionRemainingEmptyKnownValues():
    """
    varianceFractionRemaining calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValueCombination)
def testVarianceFractionRemainingEmptyPredictedValues():
    """
    varianceFractionRemaining calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testVarianceFractionRemaining():
    """
    Check that the varianceFractionRemaining calculator works correctly.
    """
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    vfr = varianceFractionRemaining(knownLabelsMatrix, predictedLabelsMatrix)
    assert vfr == 0.4

############
# rSquared #
############

@raises(InvalidArgumentValueCombination)
def testRSquaredEmptyKnownValues():
    """
    rSquared calculator throws exception if knownLabels is empty
    """
    knownLabels = numpy.array([])
    predictedLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)


@raises(InvalidArgumentValueCombination)
def testRSquaredEmptyPredictedValues():
    """
    rSquared calculator throws exception if predictedLabels is empty
    """

    predictedLabels = numpy.array([])
    knownLabels = numpy.array([1, 2, 3])

    knownLabelsMatrix = createData('Matrix', data=knownLabels)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)

@noLogEntryExpected
def testRSquared():
    """
    Check that the rSquared calculator works correctly.
    """
    predictedLabels = numpy.array([[1.0], [2.0], [3.0]])
    knownLabels = numpy.array([[1.0], [2.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 1.0

    predictedLabels = numpy.array([[1.0], [2.0], [3.0], [4.0]])
    knownLabels = numpy.array([[1.0], [2.0], [4.0], [3.0]])

    knownLabelsMatrix = createData('Matrix', data=knownLabels, useLog=False)
    predictedLabelsMatrix = createData('Matrix', data=predictedLabels,
                                       useLog=False)

    rsq = rSquared(knownLabelsMatrix, predictedLabelsMatrix)
    assert rsq == 0.6
