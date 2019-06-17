"""
Method implementations and helpers acting specifically on points in a
Matrix object.
"""

from __future__ import absolute_import

import numpy

from nimble.exceptions import InvalidArgumentValue

from .axis_view import AxisView
from .matrixAxis import MatrixAxis
from .points import Points
from .points_view import PointsView
from .dataHelpers import fillArrayWithCollapsedFeatures
from .dataHelpers import fillArrayWithExpandedFeatures

class MatrixPoints(MatrixAxis, Points):
    """
    Matrix method implementations performed on the points axis.

    Parameters
    ----------
    source : nimble data object
        The object containing point and feature data.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points from the toAdd object below the provided index
        in this object, the remaining points from this object will
        continue below the inserted points.
        """
        startData = self._source.data[:insertBefore, :]
        endData = self._source.data[insertBefore:, :]
        self._source.data = numpy.concatenate((startData, toAdd.data, endData),
                                              0)

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            if len(currRet) != len(self._source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise InvalidArgumentValue(msg)
            try:
                currRet = numpy.array(currRet, dtype=numpy.float)
            except ValueError:
                currRet = numpy.array(currRet, dtype=numpy.object_)
                # need self.data to be object dtype if inserting object dtype
                if numpy.issubdtype(self._source.data.dtype, numpy.number):
                    self._source.data = self._source.data.astype(numpy.object_)
            reshape = (1, len(self._source.features))
            self._source.data[i, :] = numpy.array(currRet).reshape(reshape)

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._source.points) * len(self._source.features)
    #     self._source.data = self._source.data.reshape((1, numElements),
    #                                                 order='C')
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numPoints = divideInto
    #     numFeatures = len(self._source.features) // numPoints
    #     self._source.data = self._source.data.reshape((numPoints,
    #                                                    numFeatures),
    #                                                 order='C')

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._source.data[:, collapseIndices]
        retainData = self._source.data[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, collapseData, currNumPoints,
            currFtNames, numRetPoints, numRetFeatures)

        self._source.data = numpy.matrix(tmpData)

    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures)

        self._source.data = numpy.matrix(tmpData)

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class MatrixPointsView(PointsView, AxisView, MatrixPoints):
    """
    Limit functionality of MatrixPoints to read-only
    """
    pass

class nzIt(object):
    """
    Non-zero iterator to return when iterating through each point.
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
        while self._pIndex < self._pStop:
            value = self._source.data[self._pIndex, self._fIndex]

            self._fIndex += 1
            if self._fIndex >= self._fStop:
                self._fIndex = 0
                self._pIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
