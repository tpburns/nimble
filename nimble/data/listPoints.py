"""
Method implementations and helpers acting specifically on points in a
List object.
"""

from __future__ import absolute_import

import numpy

from nimble.exceptions import InvalidArgumentValue

from .axis_view import AxisView
from .listAxis import ListAxis
from .points import Points
from .points_view import PointsView
from .dataHelpers import fillArrayWithCollapsedFeatures
from .dataHelpers import fillArrayWithExpandedFeatures

class ListPoints(ListAxis, Points):
    """
    List method implementations performed on the points axis.

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
        insert = toAdd.copy('pythonlist')
        if insertBefore != 0 and insertBefore != len(self):
            breakIdx = insertBefore - 1
            restartIdx = insertBefore
            start = self._source.view(pointEnd=breakIdx).copy('pythonlist')
            end = self._source.view(pointStart=restartIdx).copy('pythonlist')
            allData = start + insert + end
        elif insertBefore == 0:
            allData = insert + self._source.copy('pythonlist')
        else:
            allData = self._source.copy('pythonlist') + insert
        
        self._source.data = allData

    def _transform_implementation(self, function, limitTo):
        for i, p in enumerate(self):
            if limitTo is not None and i not in limitTo:
                continue
            currRet = function(p)
            if len(currRet) != len(self._source.features):
                msg = "function must return an iterable with as many elements "
                msg += "as features in this object"
                raise InvalidArgumentValue(msg)

            self._source.data[i] = currRet

    # def _flattenToOne_implementation(self):
    #     onto = self._source.data[0]
    #     for _ in range(1, len(self._source.points)):
    #         onto += self._source.data[1]
    #         del self._source.data[1]
    #
    #     self._source._numFeatures = len(onto)
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     result = []
    #     numPoints = divideInto
    #     numFeatures = len(self._source.features) // numPoints
    #     for i in range(numPoints):
    #         temp = self._source.data[0][(i*numFeatures):((i+1)*numFeatures)]
    #         result.append(temp)
    #
    #     self._source.data = result
    #     self._source._numFeatures = numFeatures

    ################################
    # Higher Order implementations #
    ################################

    def _splitByCollapsingFeatures_implementation(
            self, featuresToCollapse, collapseIndices, retainIndices,
            currNumPoints, currFtNames, numRetPoints, numRetFeatures):
        collapseData = []
        retainData = []
        for pt in self._source.data:
            collapseFeatures = []
            retainFeatures = []
            for idx in collapseIndices:
                collapseFeatures.append(pt[idx])
            for idx in retainIndices:
                retainFeatures.append(pt[idx])
            collapseData.append(collapseFeatures)
            retainData.append(retainFeatures)

        tmpData = fillArrayWithCollapsedFeatures(
            featuresToCollapse, retainData, numpy.array(collapseData),
            currNumPoints, currFtNames, numRetPoints, numRetFeatures)

        self._source.data = tmpData.tolist()
        self._source._numFeatures = numRetFeatures

    def _combineByExpandingFeatures_implementation(
            self, uniqueDict, namesIdx, uniqueNames, numRetFeatures):
        tmpData = fillArrayWithExpandedFeatures(uniqueDict, namesIdx,
                                                uniqueNames, numRetFeatures)

        self._source.data = tmpData.tolist()
        self._source._numFeatures = numRetFeatures

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        return nzIt(self._source)

class ListPointsView(PointsView, AxisView, ListPoints):
    """
    Limit functionality of ListPoints to read-only
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
            value = self._source.data[self._pIndex][self._fIndex]

            self._fIndex += 1
            if self._fIndex >= self._fStop:
                self._fIndex = 0
                self._pIndex += 1

            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
