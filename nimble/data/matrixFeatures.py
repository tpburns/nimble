"""
Method implementations and helpers acting specifically on features in a
Matrix object.
"""

from __future__ import absolute_import

import numpy

from nimble.exceptions import InvalidArgumentValue
from nimble.utility import numpy2DArray
from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .features import Features
from .features_view import FeaturesView

class MatrixFeatures(MatrixAxis, Features):
    """
    Matrix method implementations performed on the feature axis.

    Parameters
    ----------
    base : Matrix
        The Matrix instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self._base.data[:, :insertBefore]
        endData = self._base.data[:, insertBefore:]
        self._base.data = numpy.concatenate((startData, toAdd.data, endData),
                                              1)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            if len(currRet) != len(self._base.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise InvalidArgumentValue(msg)
            try:
                currRet = numpy.array(currRet, dtype=numpy.float)
            except ValueError:
                currRet = numpy.array(currRet, dtype=numpy.object_)
                # need self.data to be object dtype if inserting object dtype
                if numpy.issubdtype(self._base.data.dtype, numpy.number):
                    self._base.data = self._base.data.astype(numpy.object_)

            self._base.data[:, j] = numpy.array(currRet)

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._base.points) * len(self._base.features)
    #     self._base.data = self._base.data.reshape((numElements, 1),
    #                                                 order='F')
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numFeatures = divideInto
    #     numPoints = len(self._base.points) // numFeatures
    #     self._base.data = self._base.data.reshape((numPoints,
    #                                                    numFeatures),
    #                                                 order='F')

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = self._base.data[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._base.data[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base.data = numpy2DArray(tmpData)

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._base)

class MatrixFeaturesView(FeaturesView, AxisView, MatrixFeatures):
    """
    Limit functionality of MatrixFeatures to read-only.

    Parameters
    ----------
    base : MatrixView
        The MatrixView instance that will be queried.
    """
    pass

class nzIt(object):
    """
    Non-zero iterator to return when iterating through each feature.
    """
    def __init__(self, source):
        self._source = source
        self._pIndex = 0
        self._pStop = len(source.points)
        self._fIndex = 0
        self._fStop = len(source.features)

    def __iter__(self):
        return self

    def next(self):
        """
        Get next non zero value.
        """
        while self._fIndex < self._fStop:
            value = self._source.data[self._pIndex, self._fIndex]

            self._pIndex += 1
            if self._pIndex >= self._pStop:
                self._pIndex = 0
                self._fIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
