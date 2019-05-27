"""
Implementations and helpers specific to performing axis-generic
operations on a UML Sparse object.
"""

from __future__ import absolute_import
from abc import abstractmethod

import numpy

import UML
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from .axis import Axis
from .points import Points
from .dataHelpers import sortIndexPosition

scipy = UML.importModule('scipy')
if scipy is not None:
    from scipy.sparse import coo_matrix

class SparseAxis(Axis):
    """
    Differentiate how Sparse methods act dependent on the axis.

    Also includes abstract methods which will be required to perform
    axis-specific operations.

    Parameters
    ----------
    source : UML data object
        The object containing point and feature data.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _structuralBackend_implementation(self, structure, targetList):
        """
        Backend for points/features.extract points/features.delete,
        points/features.retain, and points/features.copy. Returns a new
        object containing only the points or features in targetList and
        performs some modifications to the original object if necessary.
        This function does not perform all of the modification or
        process how each function handles the returned value, these are
        managed separately by each frontend function.
        """
        pointNames, featureNames = self._getStructuralNames(targetList)
        # SparseView or object dtype
        if (self._source.data.data is None
                or self._source.data.data.dtype == numpy.object_):
            return self._structuralIterative_implementation(
                structure, targetList, pointNames, featureNames)
        # nonview numeric objects
        return self._structuralVectorized_implementation(
            structure, targetList, pointNames, featureNames)

    def _sort_implementation(self, indexPosition):
        source = self._source
        # since we want to access with with positions in the original
        # data, we reverse the 'map'
        reverseIdxPosition = numpy.empty(len(indexPosition))
        for i, idxPos in enumerate(indexPosition):
            reverseIdxPosition[idxPos] = i

        if isinstance(self, Points):
            source.data.row[:] = reverseIdxPosition[source.data.row]
        else:
            source.data.col[:] = reverseIdxPosition[source.data.col]
        source._sorted = None

    def _transform_implementation(self, function, limitTo):
        modData = []
        modRow = []
        modCol = []

        if isinstance(self, Points):
            modTarget = modRow
            modOther = modCol
        else:
            modTarget = modCol
            modOther = modRow

        for viewID, view in enumerate(self):
            if limitTo is not None and viewID not in limitTo:
                currOut = list(view)
            else:
                currOut = function(view)

            # easy way to reuse code if we have a singular return
            if not hasattr(currOut, '__iter__'):
                currOut = [currOut]

            # if there are multiple values, they must be random accessible
            if not hasattr(currOut, '__getitem__'):
                msg = "function must return random accessible data "
                msg += "(ie has a __getitem__ attribute)"
                raise InvalidArgumentType(msg)

            for i, retVal in enumerate(currOut):
                if retVal != 0:
                    modData.append(retVal)
                    modTarget.append(viewID)
                    modOther.append(i)

        if len(modData) != 0:
            try:
                modData = numpy.array(modData, dtype=numpy.float)
            except Exception:
                modData = numpy.array(modData, dtype=numpy.object_)
            shape = (len(self._source.points), len(self._source.features))
            self._source.data = coo_matrix((modData, (modRow, modCol)),
                                           shape=shape)
            self._source._sorted = None

        ret = None
        return ret

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the points/features from the toAdd object below the
        provided index in this object, the remaining points/features
        from this object will continue below the inserted
        points/features.
        """
        selfData = self._source.data.data
        addData = toAdd.data.data
        newData = numpy.concatenate((selfData, addData))
        if isinstance(self, Points):
            selfAxis = self._source.data.row.copy()
            selfOffAxis = self._source.data.col
            addAxis = toAdd.data.row.copy()
            addOffAxis = toAdd.data.col
            addLength = len(toAdd.points)
            shape = (len(self) + addLength, len(self._source.features))
        else:
            selfAxis = self._source.data.col.copy()
            selfOffAxis = self._source.data.row
            addAxis = toAdd.data.col.copy()
            addOffAxis = toAdd.data.row
            addLength = len(toAdd.features)
            shape = (len(self._source.points), len(self) + addLength)

        selfAxis[selfAxis >= insertBefore] += addLength
        addAxis += insertBefore

        newAxis = numpy.concatenate((selfAxis, addAxis))
        newOffAxis = numpy.concatenate((selfOffAxis, addOffAxis))

        if isinstance(self, Points):
            rowColTuple = (newAxis, newOffAxis)
        else:
            rowColTuple = (newOffAxis, newAxis)

        self._source.data = coo_matrix((newData, rowColTuple), shape=shape)
        self._source._sorted = None

    #########################
    # Query implementations #
    #########################

    def _nonZeroIterator_implementation(self):
        if isinstance(self, Points):
            self._source._sortInternal('point')
        else:
            self._source._sortInternal('feature')

        return nzIt(self._source)

    ######################
    # Structural Helpers #
    ######################

    def _structuralVectorized_implementation(self, structure, targetList,
                                             pointNames, featureNames):
        """
        Use scipy csr or csc matrices for indexing targeted values
        """
        if structure != 'copy':
            notTarget = []
            for idx in range(len(self)):
                if idx not in targetList:
                    notTarget.append(idx)

        if isinstance(self, Points):
            data = self._source.data.tocsr()
            targeted = data[targetList, :]
            if structure != 'copy':
                notTargeted = data[notTarget, :]
        else:
            data = self._source.data.tocsc()
            targeted = data[:, targetList]
            if structure != 'copy':
                notTargeted = data[:, notTarget]

        if structure != 'copy':
            self._source.data = notTargeted.tocoo()
            self._source._sortInternal(self._axis)

        ret = targeted.tocoo()

        return UML.data.Sparse(ret, pointNames=pointNames,
                               featureNames=featureNames,
                               reuseData=True)

    def _structuralIterative_implementation(self, structure, targetList,
                                            pointNames, featureNames):
        """
        Iterate through each member to index targeted values
        """
        dtype = numpy.object_

        targetLength = len(targetList)
        targetData = []
        targetRows = []
        targetCols = []
        keepData = []
        keepRows = []
        keepCols = []
        keepIndex = 0

        # iterate through self._axis data
        for targetID, view in enumerate(self):
            # coo_matrix data for return object
            if targetID in targetList:
                for otherID, value in enumerate(view.data.data):
                    targetData.append(value)
                    if isinstance(self, Points):
                        targetRows.append(targetList.index(targetID))
                        targetCols.append(view.data.col[otherID])
                    else:
                        targetRows.append(view.data.row[otherID])
                        targetCols.append(targetList.index(targetID))
            # coo_matrix data for modified self._source
            elif structure != 'copy':
                for otherID, value in enumerate(view.data.data):
                    keepData.append(value)
                    if isinstance(self, Points):
                        keepRows.append(keepIndex)
                        keepCols.append(view.data.col[otherID])
                    else:
                        keepRows.append(view.data.row[otherID])
                        keepCols.append(keepIndex)
                keepIndex += 1

        # instantiate return data
        selfShape, targetShape = _calcShapes(self._source.data.shape,
                                             targetLength, self._axis)
        if structure != 'copy':
            keepData = numpy.array(keepData, dtype=dtype)
            self._source.data = coo_matrix((keepData, (keepRows, keepCols)),
                                           shape=selfShape)
        # need to manually set dtype or coo_matrix will force to simplest dtype
        targetData = numpy.array(targetData, dtype=dtype)
        ret = coo_matrix((targetData, (targetRows, targetCols)),
                         shape=targetShape)

        return UML.data.Sparse(ret, pointNames=pointNames,
                               featureNames=featureNames, reuseData=True)

    def _unique_implementation(self):
        if self._source._sorted is None:
            self._source._sortInternal("feature")
        count = len(self)
        hasAxisNames = self._namesCreated()
        getAxisName = self._getName
        getAxisNames = self._getNames
        data = self._source.data.data
        row = self._source.data.row
        col = self._source.data.col
        if isinstance(self, Points):
            axisLocator = row
            offAxisLocator = col
            hasOffAxisNames = self._source._featureNamesCreated()
            getOffAxisNames = self._source.features.getNames
        else:
            axisLocator = col
            offAxisLocator = row
            hasOffAxisNames = self._source._pointNamesCreated()
            getOffAxisNames = self._source.points.getNames

        unique = set()
        uniqueData = []
        uniqueAxis = []
        uniqueOffAxis = []
        keepNames = []
        axisCount = 0
        for i in range(count):
            axisLoc = axisLocator == i
            # data values can look the same but have zeros in different places;
            # zip with offAxis to ensure the locations are the same as well
            key = tuple(zip(data[axisLoc], offAxisLocator[axisLoc]))
            if key not in unique:
                unique.add(key)
                uniqueData.extend(data[axisLoc])
                uniqueAxis.extend([axisCount for _ in range(sum(axisLoc))])
                uniqueOffAxis.extend(offAxisLocator[axisLoc])
                if hasAxisNames:
                    keepNames.append(getAxisName(i))
                axisCount += 1

        if hasAxisNames and keepNames == getAxisNames():
            return self._source.copy()

        axisNames = False
        offAxisNames = False
        if len(keepNames) > 0:
            axisNames = keepNames
        if hasOffAxisNames:
            offAxisNames = getOffAxisNames()
        self._source._sorted = None

        uniqueData = numpy.array(uniqueData, dtype=numpy.object_)
        if isinstance(self, Points):
            shape = (axisCount, len(self._source.features))
            uniqueCoo = coo_matrix((uniqueData, (uniqueAxis, uniqueOffAxis)),
                                   shape=shape)
            return UML.createData('Sparse', uniqueCoo, pointNames=axisNames,
                                  featureNames=offAxisNames, useLog=False)
        else:
            shape = (len(self._source.points), axisCount)
            uniqueCoo = coo_matrix((uniqueData, (uniqueOffAxis, uniqueAxis)),
                                   shape=shape)
            return UML.createData('Sparse', uniqueCoo, pointNames=offAxisNames,
                                  featureNames=axisNames, useLog=False)

    ####################
    # Abstract Methods #
    ####################

    # @abstractmethod
    # def _flattenToOne_implementation(self):
    #     pass
    #
    # @abstractmethod
    # def _unflattenFromOne_implementation(self, divideInto):
    #     pass

###################
# Generic Helpers #
###################

def _calcShapes(currShape, numExtracted, axisType):
    (rowShape, colShape) = currShape
    if axisType == "feature":
        selfRowShape = rowShape
        selfColShape = colShape - numExtracted
        extRowShape = rowShape
        extColShape = numExtracted
    elif axisType == "point":
        selfRowShape = rowShape - numExtracted
        selfColShape = colShape
        extRowShape = numExtracted
        extColShape = colShape

    return ((selfRowShape, selfColShape), (extRowShape, extColShape))

class nzIt(object):
    """
    Non-zero iterator to return when iterating through points or
    features. The iteration axis is dependent on how the internal data
    is sorted before instantiation.
    """
    def __init__(self, source):
        self._source = source
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        """
        Get next non zero value.
        """
        while self._index < len(self._source.data.data):
            value = self._source.data.data[self._index]

            self._index += 1
            if value != 0:
                return value

        raise StopIteration

    def __next__(self):
        return self.next()
