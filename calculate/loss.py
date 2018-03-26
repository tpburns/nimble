"""
Definitions for functions that can be used as performance functions by
UML. Specifically, this only contains those functions that measure
loss; or in other words, those where smaller values indicate a higher
level of correctness in the predicted values.

"""

from __future__ import absolute_import
import numpy

import UML
from UML.data import Base
from UML.data import Matrix
from math import sqrt, log, exp

from UML.exceptions import ArgumentException
from six.moves import range

# TODO check that min optimal value actually makes sense for all computations


def _validateValues(values):
    if not isinstance(values, Base):
        msg = "knownValues and predictedValues must be derived class of UML.data.Base"
        raise ArgumentException(msg)

    if values.features > 1 and values.points > 1:
        msg = ('knownValues and predictedValues must be of the form 1xn or nx1. '
               'Only objects with either one feature for multiple points '
               'or one point for multiple features.')
        raise ArgumentException(msg)

    if values.points == 0 or values.features == 0:
        msg = 'knownValues and predictedValues cannot be empty.'
        raise ArgumentException(msg)


def _validateMatchShapes(knownValues, predictedValues):
    if knownValues.features != predictedValues.features:
        msg = "knownValues and predictedValues must have the same number of features"
        raise ArgumentException(msg)

    if knownValues.points != predictedValues.points:
        msg = "knownValues and predictedValues must have the same number of points"
        raise ArgumentException(msg)


def _validate(knownValues, predictedValues):
    _validateValues(knownValues)
    _validateValues(predictedValues)
    _validateMatchShapes(knownValues, predictedValues)


def _toMatrix(values):
    if values.features > 1 and values.points == 1:
        return values.copyAs('Matrix', rowsArePoints=False)
    else:
        return values.copyAs('Matrix', rowsArePoints=True)


def _computeError(knownValues, predictedValues, loopFunction, compressionFunction):
    """
    Generic function to compute different kinds of error metrics.

    knownValues - 1D Base object with one known label (or number) per row.

    predictedValues - 1D Base object with one predictedLabel (or score) per row.

    loopFunction - is a function to be applied to each row in knownValues/predictedValues, 
    that takes 3 arguments: a known class label, a predicted label, and runningTotal, 
    which contains the successive output of loopFunction.

    compressionFunction - is a function that should take two arguments: runningTotal, the final
    output of loopFunction, and n, the number of values in knownValues/predictedValues.

    The ith row in knownValues are assume refer to the same point as the ith row in predictedValues.
    """
    knownValues = _toMatrix(knownValues)
    predictedValues = _toMatrix(predictedValues)

    n = 0.0
    runningTotal = 0.0
    for aV, pV in zip(knownValues, predictedValues):
        runningTotal = loopFunction(aV, pV, runningTotal)
        n += 1
    if n > 0:
        try:
            # provide the final value from loopFunction to compressionFunction, along with the
            # number of values looped over
            runningTotal = compressionFunction(runningTotal, n)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                'Tried to divide by zero when calculating performance metric')
    else:
        raise ArgumentException("Empty argument(s) in error calculator")

    return runningTotal


def meanSquareLoss(knownValues, predictedValues):
    """
    Compute the mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (y - x) ** 2,
                         lambda x, y: x / y)


meanSquareLoss.optimal = 'min'


def sumSquareLoss(knownValues, predictedValues):
    """
    Compute the sum square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (y - x) ** 2,
                         lambda x, y: x)


sumSquareLoss.optimal = 'min'


def rootMeanSquareLoss(knownValues, predictedValues):
    """
    Compute the root mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    return sqrt(meanSquareLoss(knownValues, predictedValues))


rootMeanSquareLoss.optimal = 'min'


def crossEntropyLoss(knownValues, predictedValues):
    """
    Compute the cross-entropy loss.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    TODO Values should be probabilities? check if so
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (x * log(y, 10)),
                         lambda x, y: -x / y)


crossEntropyLoss.optimal = 'min'


def exponentialLoss(knownValues, predictedValues, tau):
    """
    Compute exponetialLoss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    # TODO check valid values for tau exception when 0
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (y - x) ** 2,
                         lambda x, y: tau * exp(1 / tau * x))


exponentialLoss.optimal = 'min'


def quadraticLoss(knownValues, predictedValues):
    """
    Compute exponetialLoss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y, z: z + (x - y)**2,
                         lambda x, y: 0.5 * x)


quadraticLoss.optimal = 'min'


def meanFeaturewiseRootMeanSquareError(knownValues, predictedValues):
    """For 2d prediction data, compute the RMSE of each feature, then average
    the results.
    """
    if knownValues.features != predictedValues.features:
        raise ArgumentException(
            "The known and predicted data must have the same number of features")
    if knownValues.points != predictedValues.points:
        raise ArgumentException(
            "The known and predicted data must have the same number of points")

    results = []
    for i in range(knownValues.features):
        currKnown = knownValues.copyFeatures(i)
        currPred = predictedValues.copyFeatures(i)
        results.append(rootMeanSquareLoss(currKnown, currPred))

    return float(sum(results)) / knownValues.features


meanFeaturewiseRootMeanSquareError.optimal = 'min'


def meanAbsoluteError(knownValues, predictedValues):
    """
        Compute mean absolute error. Assumes that knownValues and predictedValues contain
        numerical values, rather than categorical data.
    """
    # _validatePredictedAsLabels(predictedValues)
    return _computeError(knownValues, predictedValues, lambda x, y, z: z + abs(y - x), lambda x, y: x / y)


meanAbsoluteError.optimal = 'min'


def fractionIncorrect(knownValues, predictedValues):
    """
        Compute the proportion of incorrect predictions within a set of
        instances.  Assumes that values in knownValues and predictedValues are categorical.
    """
    # _validatePredictedAsLabels(predictedValues)
    return _computeError(knownValues, predictedValues, lambda x, y, z: z if x == y else z + 1, lambda x, y: x / y)


fractionIncorrect.optimal = 'min'


def varianceFractionRemaining(knownValues, predictedValues):
    """
    Calculate the how much variance is has not been correctly predicted in the
    predicted values. This will be equal to 1 - UML.calculate.rsquared() of
    the same inputs.
    """
    if knownValues.points != predictedValues.points:
        raise Exception("Objects had different numbers of points")
    if knownValues.features != predictedValues.features:
        raise Exception(
            "Objects had different numbers of features. Known values had " + str(
                knownValues.features) + " and predicted values had " + str(predictedValues.features))
    diffObject = predictedValues - knownValues
    rawDiff = diffObject.copyAs("numpy array")
    rawKnowns = knownValues.copyAs("numpy array")
    assert rawDiff.shape[1] == 1
    avgSqDif = numpy.dot(rawDiff.T, rawDiff)[0, 0] / float(len(rawDiff))
    return avgSqDif / float(numpy.var(rawKnowns))


varianceFractionRemaining.optimal = 'min'
