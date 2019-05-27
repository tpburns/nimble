"""
Method implementations and helpers acting specifically on points in a
DataFrame object.
"""

from __future__ import absolute_import
from __future__ import division

import numpy

import UML
from UML.exceptions import InvalidArgumentValue
from .axis_view import AxisView
from .dataframeAxis import DataFrameAxis
from .dataHelpers import fillArrayWithCollapsedFeatures
from .dataHelpers import fillArrayWithExpandedFeatures
from .points import Points
from .points_view import PointsView

pd = UML.importModule('pandas')
if pd:
    import pandas as pd

class DataFramePoints(DataFrameAxis, Points):
    """
    DataFrame method implementations performed on the points axis.

    Parameters
    ----------
    source : UML data object
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
        startData = self._source.data.iloc[:insertBefore, :]
        endData = self._source.data.iloc[insertBefore:, :]
        self._source.data = pd.concat((startData, toAdd.data, endData), axis=0)
        self._source._updateName(axis='point')

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            if len(currRet) != len(self._source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise InvalidArgumentValue(msg)

            self._source.data.iloc[i, :] = currRet

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._source.points) * len(self._source.features)
    #     self._source.data = pd.DataFrame(
    #         self._source.data.values.reshape((1, numElements), order='C'))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numPoints = divideInto
    #     numFeatures = len(self._source.features) // numPoints
    #     self._source.data = pd.DataFrame(
    #         self._source.data.values.reshape((numPoints, numFeatures),
    #                                         order='C'))

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = self._source.data.values[:, collapseIndices]
        retainData = self._source.data.values[:, retainIndices]

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, numpy.array(collapseData),
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._source.data = pd.DataFrame(tmpData)

    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures)

        self._source.data = pd.DataFrame(tmpData)

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class DataFramePointsView(PointsView, AxisView, DataFramePoints):
    """
    Limit functionality of DataFramePoints to read-only
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
            value = self._source.data.iloc[self._pIndex, self._fIndex]

            self._fIndex += 1
            if self._fIndex >= self._fStop:
                self._fIndex = 0
                self._pIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
