from __future__ import absolute_import , division
import numpy

import UML
from UML.data import Base
from math import sqrt, exp

from UML.exceptions import ArgumentException
from UML.calculate import fractionIncorrectLoss
from UML.calculate.loss import _validate, _formatInputs


def _checkIntegersVector(vector):
    vector_int = vector.astype(numpy.int)
    if not numpy.all((vector - vector_int) == 0):
        msg = 'Only integers supported'
        raise ValueError(msg)


###########################
# fractionCorrectAccuracy #
###########################

def fractionCorrectAccuracy(knownValues, predictedValues):
    return 1 - fractionIncorrectLoss(knownValues, predictedValues)


##############
# F1Accuracy #
##############

def F1Accuracy(knownValues, predictedValues):
    pass


#####################
# precisionAccuracy #
#####################

def precisionAccuracy(knownValues, predictedValues):
    _validate(knownValues, predictedValues)
    knownValues, predictedValues = _formatInputs(knownValues,
                                                 predictedValues)
    # print(knownValues.dtype)
    # if not issubclass(knownValues.dtype.type, numpy.integer):
    #     msg = 'Only integers are supported'
    #     raise ValueError(msg)

    _checkIntegersVector(knownValues)
    _checkIntegersVector(predictedValues)

    k_labels = numpy.unique(knownValues)
    p_labels = numpy.unique(predictedValues)
    
    k_lab_in_p = numpy.in1d(k_labels, p_labels)
    k_p_inersect = numpy.intersect1d(k_labels, p_labels)

    if k_labels.size <= 2:
        t_label = max(k_labels)
        t_k = (knownValues == t_label)
        t_p = (predictedValues == t_label)
        tp = numpy.logical_and(t_k, t_p).sum()
        fp = numpy.logical_and(t_k, numpy.logical_not(t_p)).sum()
        print('tp: {}, fp: {}'.format(tp, fp))
        return tp / (tp + fp)

    if k_lab_in_p.any():
        if k_labels.size < 1:
            pass
        elif k_labels.size == 2:
            pass
        else:
            pass
    else:
        return 0.0


##################
# recallAccuracy #
##################

def recallAccuracy(knownValues, predictedValues):
    pass


###############
# AUCAccuracy #
###############

def AUCAccuracy(knownValues, predictedValues):
    pass


#######################
# correlationAccuracy #
#######################

def correlationAccuracy(knownValues, predictedValues):
    pass

