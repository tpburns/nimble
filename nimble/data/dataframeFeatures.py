"""
Method implementations and helpers acting specifically on features in a
DataFrame object.
"""

import numpy

import nimble
from nimble.exceptions import InvalidArgumentValue
from nimble.utility import ImportModule
from .axis_view import AxisView
from .dataframeAxis import DataFrameAxis
from .features import Features
from .features_view import FeaturesView

pd = ImportModule('pandas')

class DataFrameFeatures(DataFrameAxis, Features):
    """
    DataFrame method implementations performed on the feature axis.

    Parameters
    ----------
    base : DataFrame
        The DataFrame instance that will be queried and modified.
    """

    ##############################
    # Structural implementations #
    ##############################

    def _insert_implementation(self, insertBefore, toInsert):
        """
        Insert the features from the toInsert object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self._base.data.iloc[:, :insertBefore]
        endData = self._base.data.iloc[:, insertBefore:]
        self._base.data = pd.concat((startData, toInsert.data, endData),
                                    axis=1, ignore_index=True)

    def _transform_implementation(self, function, limitTo):
        for j, f in enumerate(self):
            if limitTo is not None and j not in limitTo:
                continue
            currRet = function(f)
            if len(currRet) != len(self._base.points):
                msg = "function must return an iterable with as many elements "
                msg += "as points in this object"
                raise InvalidArgumentValue(msg)

            self._base.data.iloc[:, j] = currRet

    # def _flattenToOne_implementation(self):
    #     numElements = len(self._base.points) * len(self._base.features)
    #     self._base.data = pd.DataFrame(
    #         self._base.data.values.reshape((numElements, 1), order='F'))
    #
    # def _unflattenFromOne_implementation(self, divideInto):
    #     numFeatures = divideInto
    #     numPoints = len(self._base.points) // numFeatures
    #     self._base.data = pd.DataFrame(
    #         self._base.data.values.reshape((numPoints, numFeatures),
    #                                         order='F'))

    ################################
    # Higher Order implementations #
    ################################

    def _splitByParsing_implementation(self, featureIndex, splitList,
                                       numRetFeatures, numResultingFts):
        tmpData = numpy.empty(shape=(len(self._base.points), numRetFeatures),
                              dtype=numpy.object_)

        tmpData[:, :featureIndex] = self._base.data.values[:, :featureIndex]
        for i in range(numResultingFts):
            newFeat = []
            for lst in splitList:
                newFeat.append(lst[i])
            tmpData[:, featureIndex + i] = newFeat
        existingData = self._base.data.values[:, featureIndex + 1:]
        tmpData[:, featureIndex + numResultingFts:] = existingData

        self._base.data = pd.DataFrame(tmpData)

class DataFrameFeaturesView(FeaturesView, AxisView, DataFrameFeatures):
    """
    Limit functionality of DataFrameFeatures to read-only.

    Parameters
    ----------
    base : DataFrameView
        The DataFrameView instance that will be queried.
    """
    pass
