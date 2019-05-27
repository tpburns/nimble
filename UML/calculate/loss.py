"""
Definitions for functions that can be used as performance functions by
UML. Specifically, this only contains those functions that measure
loss; or in other words, those where smaller values indicate a higher
level of correctness in the predicted values.

"""

from __future__ import absolute_import, division
import numpy

import UML
from UML.data import Base
from math import sqrt, exp

from UML.exceptions import InvalidArgumentType
from UML.exceptions import InvalidArgumentValue
from UML.exceptions import InvalidArgumentTypeCombination
from UML.exceptions import InvalidArgumentValueCombination
from UML.calculate import  elementwisePower


from six.moves import range

from sklearn.metrics import mean_squared_error, log_loss, hinge_loss

# TODO check that min optimal value actually makes sense for all computations


def _validateValues(values):
    if not isinstance(values, Base):
        msg = 'knownValues and predictedValues must be derived class of UML.data.Base'
        raise InvalidArgumentType(msg)

    if len(values.features) > 1 and len(values.points) > 1:
        msg = ('knownValues and predictedValues must be of the form 1xn or nx1. '
               'Only objects with either one feature for multiple points '
               'or one point for multiple features.')
        raise InvalidArgumentValue(msg)

    if len(values.points) == 0 or len(values.features) == 0:
        msg = 'knownValues and predictedValues cannot be empty.'
        raise InvalidArgumentValue(msg)


def _validateNumOfElements(knownValues, predictedValues):
    if len(knownValues) != len(predictedValues):
        msg = "knownValues and predictedValues must have the same number of elements"
        raise InvalidArgumentValueCombination(msg)


def _validate(knownValues, predictedValues):
    _validateValues(knownValues)
    _validateValues(predictedValues)
    _validateNumOfElements(knownValues, predictedValues)


def _toNumpyArray(values):
    if len(values.points) > 1:
        return values.copy(to='numpy array', rowsArePoints=False).flatten()
    else:
        return values.copy(to='numpy array', rowsArePoints=True).flatten()


def _formatInputs(knownValues, predictedValues):
    knownValues = _toNumpyArray(knownValues)
    predictedValues = _toNumpyArray(predictedValues)
    return (knownValues, predictedValues)


##################
# meanSquareLoss #
##################

def meanSquareLoss(knownValues, predictedValues):
    """
    Compute the mean square error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    try:
        return sumSquareLoss(knownValues, predictedValues) / len(predictedValues)
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Tried to divide by zero when calculating metric. predictedValues size is 0')


meanSquareLoss.optimal = 'min'


def meanSquareLoss_alt(knownValues, predictedValues):
    """
    Compute the mean square error based on sklearn-learn mean_squared_error.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
    return mean_squared_error(knownValues, predictedValues)


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

    diffObj = knownValues - predictedValues
    return sum(elementwisePower(diffObj, 2))


sumSquareLoss.optimal = 'min'


####################
# crossEntropyLoss #
####################

def crossEntropyLoss(knownValues, predictedValues, eps=1e-15):
    """
    Compute the cross-entropy loss (log-loss).  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.

    knownValues should be label as 0 or 1.

    predictedValues should be probabilities.

    ((x * log(y)) + ((1 - x) * log(1 - y) / n

    TODO Should we just support two classes? (meaning logistic regression case)
    TODO Should we support user passing labels?
    """
    _validate(knownValues, predictedValues)
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)

    if knownValues.all() != 1 and knownValues.all() != 0:
        msg = 'knownValues classes should be labels 0 or 1.'
        raise InvalidArgumentValue(msg)

    if predictedValues.all() > 1 or predictedValues.all() < 0:
        msg = ('predictedValues for cross entropy loss involved values from a logistic'
               'regression model correspond to probabilities between 0 and 1.')
        raise InvalidArgumentValue(msg)

    numpy.clip(predictedValues, eps, 1 - eps)

    label_1 = knownValues.dot(numpy.log(predictedValues).T)
    label_0 = (1 - knownValues).dot(numpy.log(1 - predictedValues).T)

    try:
        return - numpy.asarray(label_1 + label_0) / predictedValues.size
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Tried to divide by zero when calculating metric. predictedValues size is 0')


crossEntropyLoss.optimal = 'min'


def crossEntropyLoss_alt(knownValues, predictedValues):
    """
    Compute the cross-entropy loss based on sklearn-learn log_loss function.  Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
    return log_loss(knownValues, predictedValues)


###################
# exponentialLoss #
###################


def exponentialLoss(knownValues, predictedValues, tau):
    """
    Compute exponetial loss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """

    if tau == 0:
        msg = 'tau needs to be different than zero'
        raise ZeroDivisionError(msg)

    summatory = sumSquareLoss(knownValues, predictedValues)

    try:
        return (tau * exp(summatory / tau)) / len(predictedValues)
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
    Compute quadratic loss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    summatory = sumSquareLoss(knownValues, predictedValues)
    try:
        return 0.5 * summatory / len(predictedValues)
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Tried to divide by zero when calculating metric. predictedValues size is 0')


quadraticLoss.optimal = 'min'


#############
# hingeLoss #
#############

def hingeLoss(knownValues, predictedValues):
    """
    Compute hinge loss. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    TODO: review it covers all cases we want
    """
    _validate(knownValues, predictedValues)
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
    return numpy.max(0, 1 - knownValues * predictedValues)


def hingeLoss_alt(knownValues, predictedValues):
    """
    Compute hinge loss based on sklearn-learn hinge_loss function. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
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
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
    summatory = (knownValues == predictedValues).sum()
    try:
        return summatory / predictedValues.size
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Tried to divide by zero when calculating metric. predictedValues size is 0')


fractionIncorrectLoss.optimal = 'min'


##########
# l2Loss #
##########


def l2Loss(knownValues, predictedValues):
    """
    Compute sum of squares differences.
    Assumes that values in knownValues and predictedValues are categorical.
    """
    return sumSquareLoss(knownValues, predictedValues)


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
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
    return numpy.absolute(knownValues - predictedValues).sum()


l1Loss.optimal = 'min'


def meanAbsoluteError(knownValues, predictedValues):
    """
    Compute mean absolute error. Assumes that knownValues and predictedValues contain
    numerical values, rather than categorical data.
    """
    try:
        return l1Loss(knownValues, predictedValues) / len(predictedValues)
    except ZeroDivisionError:
        raise ZeroDivisionError(
            'Tried to divide by zero when calculating metric. predictedValues size is 0')


meanAbsoluteError.optimal = 'min'


def meanFeaturewiseRootMeanSquareError(knownValues, predictedValues):
    """For 2d prediction data, compute the RMSE of each feature, then average
    the results.
    """
    if len(knownValues.features) != len(predictedValues.features):
        raise InvalidArgumentValueCombination(
            "The known and predicted data must have the same number of features")
    if knownValues.points != predictedValues.points:
        raise InvalidArgumentValueCombination(
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
    if len(knownValues.points )!= len(predictedValues.points):
        raise Exception("Objects had different numbers of points")
    if len(knownValues.features )!= len(predictedValues.features):
        raise Exception(
            "Objects had different numbers of features. Known values had " + str(
                len(knownValues.features)) + " and predicted values had " + str(len(predictedValues.features)))
    diffObject = predictedValues - knownValues
    rawDiff = diffObject.copyAs("numpy array")
    rawKnowns = knownValues.copyAs("numpy array")
    assert rawDiff.shape[1] == 1
    avgSqDif = numpy.dot(rawDiff.T, rawDiff)[0, 0] / float(len(rawDiff))
    return avgSqDif / float(numpy.var(rawKnowns))


varianceFractionRemaining.optimal = 'min'

