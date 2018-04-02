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
from math import sqrt, log, exp

from UML.exceptions import ArgumentException
from six.moves import range
try:
    import itertools.izip as zip
except ImportError:
    pass

from sklearn.metrics import mean_squared_error, log_loss, hinge_loss

# TODO check that min optimal value actually makes sense for all computations


def _validateValues(values):
    if not isinstance(values, Base):
        msg = 'knownValues and predictedValues must be derived class of UML.data.Base'
        raise ArgumentException(msg)

    if values.features > 1 and values.points > 1:
        msg = ('knownValues and predictedValues must be of the form 1xn or nx1. '
               'Only objects with either one feature for multiple points '
               'or one point for multiple features.')
        raise ArgumentException(msg)

    if values.points == 0 or values.features == 0:
        msg = 'knownValues and predictedValues cannot be empty.'
        raise ArgumentException(msg)


def _validateNumOfElements(knownValues, predictedValues):
    if len(knownValues) != len(predictedValues):
        msg = "knownValues and predictedValues must have the same number of elements"
        raise ArgumentException(msg)


def _validate(knownValues, predictedValues):
    _validateValues(knownValues)
    _validateValues(predictedValues)
    _validateNumOfElements(knownValues, predictedValues)


def _toNumpyArray(values):
    if values.points > 1:
        return values.copyAs('numpy array', rowsArePoints=False)
    else:
        return values.copyAs('numpy array', rowsArePoints=True)


def _computeError(knownValues, predictedValues, loopFunction, compressionFunction):
    """
    Generic function to compute different kinds of error metrics.

    knownValues - 1D Base object with one known label (or number) per row/feature.

    predictedValues - 1D Base object with one predictedLabel (or score) per row/feature.

    loopFunction - is a function to be applied to each row/feature in knownValues/predictedValues, 
    that takes two arguments: a known class label and a predicted label.

    compressionFunction - is a function that should take two arguments: summatory and n,
    the number of values in knownValues/predictedValues.

    The ith row/features in knownValues are assume refer to the same point as the ith row in predictedValues.
    """
    # loop_function_map = map(loopFunction, knownValues, predictedValues)
    # summatory = sum(loop_function_map)
    n = len(predictedValues)
    summatory = 0.0
    for aV, pV in zip(knownValues, predictedValues):
        summatory += loopFunction(aV, pV)

    # print ('summatory: {}'.format(summatory))
    try:
        # provide the final value from loopFunction to compressionFunction, along with the
        # number of values looped over
        summatory = compressionFunction(summatory, n)
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Tried to divide by zero when calculating performance metric')

    return summatory


##################
# meanSquareLoss #
##################

def meanSquareLoss(knownValues, predictedValues):
    """
    Compute the mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: (y - x)**2,
                         lambda x, y: x / y)


meanSquareLoss.optimal = 'min'


def meanSquareLoss_alt(knownValues, predictedValues):
    """
    Compute the mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    knownValues = knownValues.copyAs('numpy array')
    predictedValues = predictedValues.copyAs('numpy array')
    return (mean_squared_error(knownValues, predictedValues))


######################
# rootMeanSquareLoss #
######################

def rootMeanSquareLoss(knownValues, predictedValues):
    """
    Compute the root mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    return sqrt(meanSquareLoss(knownValues, predictedValues))


rootMeanSquareLoss.optimal = 'min'


def rootMeanSquareLoss_alt(knownValues, predictedValues):
    """
    Compute the root mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    return sqrt(meanSquareLoss_alt(knownValues, predictedValues))


#################
# sumSquareLoss #
#################

def sumSquareLoss(knownValues, predictedValues):
    """
    Compute the sum square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: (y - x) ** 2,
                         lambda x, y: x)


sumSquareLoss.optimal = 'min'


def sumSquareLoss_alt(knownValues, predictedValues):
    """
    Compute the sum square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    return meanSquareLoss_alt(knownValues, predictedValues) * len(predictedValues)


####################
# crossEntropyLoss #
####################

def crossEntropyLoss(knownValues, predictedValues, eps=1e-15):
    """
    Compute the cross-entropy loss.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.

    knownValues should labels 0 or 1. 

    predictedValues should be probabilities.
    TODO Should we just support more labels?
    """
    for k, p in zip(knownValues, predictedValues):
        # print(k, p)
        if k != 1 and k != 0:
            msg = 'knownValues classes should be labels 0 or 1.'
            raise ArgumentException(msg)

        if p > 1 or p < 0:
            msg = ('predictedValues for cross entropy loss involved values from a logistic'
                   'regression model correspond to probabilities between 0 and 1.')
            raise ArgumentException(msg)

        def loopFuction(x, y):
            if y == 0 or y == 1:
                numpy.clip(y, eps, 1 - eps)
            return (x * log(y)) + ((1 - x) * log(1 - y))

    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: ((x * log(y)) + ((1 - x) * log(1 - y))),
                         lambda x, y: -x / y)


crossEntropyLoss.optimal = 'min'


def crossEntropyLoss_alt(knownValues, predictedValues):
    """
    Compute the cross-entropy loss.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    _validate(knownValues, predictedValues)
    knownValues = _toNumpyArray(knownValues)
    predictedValues = _toNumpyArray(predictedValues)
    return log_loss(knownValues, predictedValues) / predictedValues.shape[1]


###################
# exponentialLoss #
###################


def exponentialLoss(knownValues, predictedValues, tau):
    """
    Compute exponetialLoss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """

    _validate(knownValues, predictedValues)
    if tau == 0:
        msg = 'tau needs to be different than zero'
        raise ZeroDivisionError(msg)

    try:
        return _computeError(knownValues, predictedValues,
                             lambda x, y: (y - x) ** 2,
                             lambda x, y: (tau * exp(x / tau) / y))
    except OverflowError:
        msg = ('Exponent computation not possible. The function works better when values '
               'between 0 and 1 are given. Also, changing tau value could help')
        raise OverflowError(msg)


exponentialLoss.optimal = 'min'


#################
# quadraticLoss #
#################


def quadraticLoss(knownValues, predictedValues):
    """
    Compute hingeLoss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    return _computeError(knownValues, predictedValues,
                         lambda x, y: (x - y)**2,
                         lambda x, y: 0.5 * x / y)


quadraticLoss.optimal = 'min'


#############
# hingeLoss #
#############


def hingeLoss(knownValues, predictedValues):
    """
    Compute exponetialLoss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    TODO: check how to do it without relying 
    """
    _validate(knownValues, predictedValues)
    knownValues = _toNumpyArray(knownValues)
    predictedValues = _toNumpyArray(predictedValues)
    return hinge_loss(knownValues, predictedValues)


hingeLoss.optimal = 'min'


#########################
# fractionIncorrectLoss #
#########################


def fractionIncorrectLoss(knownValues, predictedValues):
    """
    Compute the proportion of incorrect predictions within a set of instances.
    Assumes that values in knownValues and predictedValues are categorical.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: 0 if x == y else 1,
                         lambda x, y: x / y)


fractionIncorrectLoss.optimal = 'min'


##########
# l2Loss #
##########


def l2Loss(knownValues, predictedValues):
    """
    Compute sum of squares differences.
    Assumes that values in knownValues and predictedValues are categorical.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: (y - x)**2,
                         lambda x, y: x)


l2Loss.optimal = 'min'


##########
# l1Loss #
##########


def l1Loss(knownValues, predictedValues):
    """
    Compute sum of absolute value of differences.
    Assumes that values in knownValues and predictedValues are categorical.
    """
    _validate(knownValues, predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: abs(y - x),
                         lambda x, y: x)


l1Loss.optimal = 'min'


def meanAbsoluteError(knownValues, predictedValues):
    """
    Compute mean absolute error. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    # _validatePredictedAsLabels(predictedValues)
    return _computeError(knownValues, predictedValues,
                         lambda x, y: abs(y - x),
                         lambda x, y: x / y)


meanAbsoluteError.optimal = 'min'


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
