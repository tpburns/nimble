"""
Backend for unit tests of the high level functions defined by the base representation class.

Since these functions rely on implementations provided by a derived class, these
functions are to be called by unit tests of the derived class, with the appropiate
objects provided.

Methods tested in this file:

In object HighLevelDataSafe:
points.calculate, features.calculate, elements.calculate, points.count,
features.count, elements.count, elements.countUnique, points.unique,
features.unique, points.mapReduce, features.mapReduce,
isApproximatelyEqual, trainAndTestSets

In object HighLevelModifying:
replaceFeatureWithBinaryFeatures, points.shuffle, features.shuffle,
points.normalize, features.normalize, points.fill, features.fill,
fillUsingAllData, points.splitByCollapsingFeatures,
points.combineByExpandingFeatures, features.splitByParsing
"""

from __future__ import absolute_import
from copy import deepcopy
import os.path
import tempfile
import inspect

import numpy
import six
from six.moves import range
from nose.tools import *
from nose.plugins.attrib import attr
try:
    from unittest import mock #python >=3.3
except:
    import mock

import UML
from UML import match
from UML import fill
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.exceptions import InvalidArgumentValueCombination, ImproperObjectAction
from UML.data.dataHelpers import DEFAULT_PREFIX
from UML.randomness import numpyRandom
from .baseObject import DataTestObject
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected
from ..assertionHelpers import assertNoNamesGenerated


preserveName = "PreserveTestName"
preserveAPath = os.path.join(os.getcwd(), "correct", "looking", "path")
preserveRPath = os.path.relpath(preserveAPath)
preservePair = (preserveAPath, preserveRPath)

### Helpers used by tests in the test class ###

def simpleMapper(point):
    idInt = point[0]
    intList = []
    for i in range(1, len(point)):
        intList.append(point[i])
    ret = []
    for value in intList:
        ret.append((idInt, value))
    return ret


def simpleReducer(identifier, valuesList):
    total = 0
    for value in valuesList:
        total += value
    return (identifier, total)


def oddOnlyReducer(identifier, valuesList):
    if identifier % 2 == 0:
        return None
    return simpleReducer(identifier, valuesList)


def passThrough(value):
    return value


def plusOne(value):
    return (value + 1)


def plusOneOnlyEven(value):
    if value % 2 == 0:
        return (value + 1)
    else:
        return None

class CalledFunctionException(Exception):
    def __init__(self):
        pass

def calledException(*args, **kwargs):
    raise CalledFunctionException()

def noChange(value):
    return value


class HighLevelDataSafe(DataTestObject):
    #######################
    # .points.calculate() #
    #######################

    @raises(InvalidArgumentType)
    def test_points_calculate_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.points.calculate(None)

    @raises(ImproperObjectAction)
    def test_points_calculate_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.calculate(emitLower)

    @raises(ImproperObjectAction)
    def test_points_calculate_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        origObj.points.calculate(emitLower)

    @raises(InvalidArgumentValue)
    def test_points_calculate_functionReturns2D(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def return2D(point):
            return [[val for val in point]]

        calc = toTest.points.calculate(return2D)

    @raises(InvalidArgumentValue)
    def test_points_calculate_functionReturnsInvalidObj(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnInvalidObj(point):
            return int

        calc = toTest.points.calculate(returnInvalidObj)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.axis.constructIndicesList', side_effect=calledException)
    def test_points_calculate_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2,],[3,4]], pointNames=['a', 'b'])

        ret = toTest.points.calculate(noChange, points=['a', 'b'])

    @oneLogEntryExpected
    def test_points_calculate_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        lowerCounts = origObj.points.calculate(emitLower)

        expectedOut = [[0.1], [0.1], [0.1], [0.2]]
        exp = self.constructor(expectedOut, pointNames=pointNames)

        assert lowerCounts.isIdentical(exp)

    @oneLogEntryExpected
    def test_points_calculate_Handmade_lazyNameGeneration(self):
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData))

        def emitLower(point):
            return point[1]

        lowerCounts = origObj.points.calculate(emitLower)

        assertNoNamesGenerated(origObj)
        assertNoNamesGenerated(lowerCounts)

    @oneLogEntryExpected
    def test_points_calculate_functionReturnsUMLObject(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnUMLObj(point):
            ret = point * 2
            assert isinstance(ret, UML.data.Base)
            return ret

        calc = toTest.points.calculate(returnUMLObj)

        expData = [[2, 0.2, 0.02], [2, 0.2, 0.04], [2, 0.2, 0.06], [2, 0.4, 0.04]]
        exp = self.constructor(expData, pointNames=pointNames)

        assert calc.isIdentical(exp)

    def test_points_calculate_NamePathPreservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, path=preservePair)

        def emitLower(point):
            return point[toTest.features.getIndex('deci')]

        ret = toTest.points.calculate(emitLower)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

        assert ret.points.getNames() == toTest.points.getNames()

    def test_points_calculate_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitLower(point):
            return point[origObj.features.getIndex('deci')]

        lowerCounts = origObj.points.calculate(emitLower, points=['three', 2])

        expectedOut = [[0.2], [0.1]]
        expPnames = ['three', 'two']
        exp = self.constructor(expectedOut, pointNames=expPnames)

        assert lowerCounts.isIdentical(exp)

    def test_points_calculate_functionReturnsUMLObject_limited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnUMLObj(point):
            ret = point * 2
            assert isinstance(ret, UML.data.Base)
            return ret

        calc = toTest.points.calculate(returnUMLObj, points=['two', 'zero'])

        expData = [[2, 0.2, 0.06], [2, 0.2, 0.02]]
        exp = self.constructor(expData, pointNames=['two', 'zero'])

        assert calc.isIdentical(exp)

    def test_points_calculate_nonZeroItAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(point):
            ret = 0
            assert len(point) == 3
            for value in point.points.nonZeroIterator():
                ret += 1
            return ret

        counts = origObj.points.calculate(emitNumNZ)

        expectedOut = [[3], [2], [2], [1]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)

    ##########################
    # .features.calculate() #
    #########################

    @raises(ImproperObjectAction)
    def test_features_calculate_exceptionPEmpty(self):
        data = [[], []]
        data = numpy.array(data).T
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.calculate(emitAllEqual)

    @raises(ImproperObjectAction)
    def test_features_calculate_exceptionFEmpty(self):
        data = [[], []]
        data = numpy.array(data)
        origObj = self.constructor(data)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        origObj.features.calculate(emitAllEqual)

    @raises(InvalidArgumentType)
    def test_features_calculate_exceptionInputNone(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), featureNames=featureNames)
        origObj.features.calculate(None)

    @raises(InvalidArgumentValue)
    def test_features_calculate_functionReturns2D(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def return2D(feature):
            return [[val for val in feature]]

        calc = toTest.features.calculate(return2D)

    @raises(InvalidArgumentValue)
    def test_features_calculate_functionReturnsInvalidObj(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnInvalidObj(feature):
            return int

        calc = toTest.features.calculate(returnInvalidObj)

    @raises(CalledFunctionException)
    @mock.patch('UML.data.axis.constructIndicesList', side_effect=calledException)
    def test_features_calculate_calls_constructIndicesList(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        ret = toTest.features.calculate(noChange, features=['a', 'b'])

    @oneLogEntryExpected
    def test_features_calculate_Handmade(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature['zero']
            for value in feature:
                if value != first:
                    return 0
            return 1

        lowerCounts = origObj.features.calculate(emitAllEqual)
        expectedOut = [[1, 0, 0]]
        exp = self.constructor(expectedOut, featureNames=featureNames)
        assert lowerCounts.isIdentical(exp)

    @oneLogEntryExpected
    def test_features_calculate_Handmade_lazyNameGeneration(self):
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData))

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        lowerCounts = origObj.features.calculate(emitAllEqual)

        assertNoNamesGenerated(origObj)
        assertNoNamesGenerated(lowerCounts)

    @oneLogEntryExpected
    def test_features_calculate_functionReturnsUMLObject(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnUMLObj(feature):
            ret = feature * 2
            assert isinstance(ret, UML.data.Base)
            return ret

        calc = toTest.features.calculate(returnUMLObj)

        expData = [[2, 0.2, 0.02], [2, 0.2, 0.04], [2, 0.2, 0.06], [2, 0.4, 0.04]]
        exp = self.constructor(expData, featureNames=featureNames)

        assert calc.isIdentical(exp)

    def test_features_calculate_NamePath_preservation(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(origData, pointNames=pointNames,
                                  featureNames=featureNames, name=preserveName, path=preservePair)

        def emitAllEqual(feature):
            first = feature['zero']
            for value in feature:
                if value != first:
                    return 0
            return 1

        ret = toTest.features.calculate(emitAllEqual)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

        assert toTest.features.getNames() == ret.features.getNames()

    def test_features_calculate_HandmadeLimited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        origObj = self.constructor(deepcopy(origData), pointNames=pointNames, featureNames=featureNames)

        def emitAllEqual(feature):
            first = feature[0]
            for value in feature:
                if value != first:
                    return 0
            return 1

        lowerCounts = origObj.features.calculate(emitAllEqual, features=[0, 'centi'])
        expectedOut = [[1, 0]]
        expFNames = ['number', 'centi']
        exp = self.constructor(expectedOut, featureNames=expFNames)
        assert lowerCounts.isIdentical(exp)

    def test_features_calculate_functionReturnsUMLObject_limited(self):
        featureNames = {'number': 0, 'centi': 2, 'deci': 1}
        pointNames = {'zero': 0, 'one': 1, 'two': 2, 'three': 3}
        origData = [[1, 0.1, 0.01], [1, 0.1, 0.02], [1, 0.1, 0.03], [1, 0.2, 0.02]]
        toTest = self.constructor(deepcopy(origData), pointNames=pointNames,
                                  featureNames=featureNames)

        def returnUMLObj(feature):
            ret = feature * 2
            assert isinstance(ret, UML.data.Base)
            return ret

        calc = toTest.features.calculate(returnUMLObj, features=['deci', 'number'])

        expData = [[0.2, 2], [0.2, 2], [0.2, 2], [0.4, 2]]
        exp = self.constructor(expData, featureNames=['deci', 'number'])

        assert calc.isIdentical(exp)

    def test_features_calculate_nonZeroIterAndLen(self):
        origData = [[1, 1, 1], [1, 0, 2], [1, 1, 0], [0, 2, 0]]
        origObj = self.constructor(deepcopy(origData))

        def emitNumNZ(feature):
            ret = 0
            assert len(feature) == 4
            for value in feature.features.nonZeroIterator():
                ret += 1
            return ret

        counts = origObj.features.calculate(emitNumNZ)

        expectedOut = [[3, 3, 2]]
        exp = self.constructor(expectedOut)

        assert counts.isIdentical(exp)

    #######################
    # .elements.calculate #
    #######################

    @raises(CalledFunctionException)
    @mock.patch('UML.data.elements.constructIndicesList', side_effect=calledException)
    def test_elements_calculate_calls_constructIndicesList1(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], pointNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.elements.calculate(noChange, points=['a', 'b'])

    @raises(CalledFunctionException)
    @mock.patch('UML.data.elements.constructIndicesList', side_effect=calledException)
    def test_elements_calculate_calls_constructIndicesList2(self, mockFunc):
        toTest = self.constructor([[1,2],[3,4]], featureNames=['a', 'b'])

        def noChange(point):
            return point

        ret = toTest.elements.calculate(noChange, features=['a', 'b'])

    def test_elements_calculate_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, name=preserveName, path=preservePair)

        ret = toTest.elements.calculate(passThrough)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    @oneLogEntryExpected
    def test_elements_calculate_passthrough(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.calculate(passThrough)
        retRaw = ret.copy(to="python list")

        assert [1, 2, 3] in retRaw
        assert [4, 5, 6] in retRaw
        assert [7, 8, 9] in retRaw
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_elements_calculate_plusOnePreserve(self):
        data = [[1, 0, 3], [0, 5, 6], [7, 0, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.calculate(plusOne, preserveZeros=True)
        retRaw = ret.copy(to="python list")

        assert [2, 0, 4] in retRaw
        assert [0, 6, 7] in retRaw
        assert [8, 0, 10] in retRaw
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_elements_calculate_plusOneExclude(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.calculate(plusOneOnlyEven, skipNoneReturnValues=True)
        retRaw = ret.copy(to="python list")

        assert [1, 3, 3] in retRaw
        assert [5, 5, 7] in retRaw
        assert [7, 9, 9] in retRaw

    def test_elements_calculate_plusOneLimited(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        names = ['one', 'two', 'three']
        pnames = ['1', '4', '7']
        toTest = self.constructor(data, pointNames=pnames, featureNames=names)

        ret = toTest.elements.calculate(plusOneOnlyEven, points='4', features=[1, 'three'],
                                             skipNoneReturnValues=True)
        retRaw = ret.copy(to="python list")

        assert [5, 7] in retRaw

    def test_elements_calculate_All_zero(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret1 = toTest.elements.calculate(lambda x: 0)
        ret2 = toTest.elements.calculate(lambda x: 0, preserveZeros=True)

        expData = [[0,0,0],[0,0,0],[0,0,0]]
        expObj = self.constructor(expData)
        assert ret1 == expObj
        assert ret2 == expObj

    def test_elements_calculate_String_conversion_manipulations(self):
        def allString(val):
            return str(val)

        toSMap = {2:'two', 4:'four', 6:'six', 8:'eight'}
        def f1(val):
            return toSMap[val] if val in toSMap else val

        toIMap = {'two':2, 'four':4, 'six':6, 'eight':8}
        def f2(val):
            return toIMap[val] if val in toIMap else val

        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, elementType=object)
        ret0A = toTest.elements.calculate(allString)
        ret0B = toTest.elements.calculate(allString, preserveZeros=True)

        exp0Data = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
        exp0Obj = self.constructor(exp0Data, elementType=object)
        assert ret0A == exp0Obj
        assert ret0B == exp0Obj

        ret1 = toTest.elements.calculate(f1)

        exp1Data = [[1, 'two', 3], ['four', 5, 'six'], [7, 'eight', 9]]
        exp1Obj = self.constructor(exp1Data)

        assert ret1 == exp1Obj

        ret2 = ret1.elements.calculate(f2)

        exp2Obj = self.constructor(data)

        assert ret2 == exp2Obj

    ######################
    # points.mapReduce() #
    ######################

    @raises(ImproperObjectAction)
    def test_points_mapReduce_ExceptionNoFeatures(self):
        """ Test points.mapReduce() for ImproperObjectAction when there are no features  """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.points.mapReduce(simpleMapper, simpleReducer)

    def test_points_mapReduce_emptyResultNoPoints(self):
        """ Test points.mapReduce() when given point empty data """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        data = numpy.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionNoneMap(self):
        """ Test points.mapReduce() for InvalidArgumentType when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(None, simpleReducer)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionNoneReduce(self):
        """ Test points.mapReduce() for InvalidArgumentType when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(simpleMapper, None)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionUncallableMap(self):
        """ Test points.mapReduce() for InvalidArgumentType when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce("hello", simpleReducer)

    @raises(InvalidArgumentType)
    def test_points_mapReduce_ExceptionUncallableReduce(self):
        """ Test points.mapReduce() for InvalidArgumentType when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.points.mapReduce(simpleMapper, 5)

    @oneLogEntryExpected
    def test_points_mapReduce_handmade(self):
        """ Test points.mapReduce() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 5], [4, 11], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_points_mapReduce_handmade_lazyNameGeneration(self):
        """ Test points.mapReduce() against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_points_mapReduce_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, path=preservePair)

        ret = toTest.points.mapReduce(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    def test_points_mapReduce_handmadeNoneReturningReducer(self):
        """ Test points.mapReduce() against handmade output with a None returning Reducer """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.points.mapReduce(simpleMapper, oddOnlyReducer)

        exp = self.constructor([[1, 5], [7, 17]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))


    ########################
    # features.mapReduce() #
    ########################

    @raises(ImproperObjectAction)
    def test_features_mapReduce_ExceptionNoPoints(self):
        """ Test features.mapReduce() for ImproperObjectAction when there are no points  """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.features.mapReduce(simpleMapper, simpleReducer)

    def test_features_mapReduce_emptyResultNoFeatures(self):
        """ Test features.mapReduce() when given feature empty data """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        data = numpy.empty(shape=(0, 0))
        exp = self.constructor(data)
        assert ret.isIdentical(exp)

    @raises(InvalidArgumentType)
    def test_features_mapReduce_ExceptionNoneMap(self):
        """ Test features.mapReduce() for InvalidArgumentType when mapper is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce(None, simpleReducer)

    @raises(InvalidArgumentType)
    def test_features_mapReduce_ExceptionNoneReduce(self):
        """ Test features.mapReduce() for InvalidArgumentType when reducer is None """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce(simpleMapper, None)

    @raises(InvalidArgumentType)
    def test_features_mapReduce_ExceptionUncallableMap(self):
        """ Test features.mapReduce() for InvalidArgumentType when mapper is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce("hello", simpleReducer)

    @raises(InvalidArgumentType)
    def test_features_mapReduce_ExceptionUncallableReduce(self):
        """ Test features.mapReduce() for InvalidArgumentType when reducer is not callable """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        toTest.features.mapReduce(simpleMapper, 5)

    @oneLogEntryExpected
    def test_features_mapReduce_handmade(self):
        """ Test features.mapReduce() against handmade output """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        exp = self.constructor([[1, 11], [2, 13], [3, 15]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    def test_features_mapReduce_handmade_lazyNameGeneration(self):
        """ Test features.mapReduce() against handmade output """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(ret)

    def test_features_mapReduce_NamePath_preservation(self):
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames,
                                  name=preserveName, path=preservePair)

        ret = toTest.features.mapReduce(simpleMapper, simpleReducer)

        assert toTest.name == preserveName
        assert toTest.absolutePath == preserveAPath
        assert toTest.relativePath == preserveRPath

        assert ret.nameIsDefault()
        assert ret.absolutePath == preserveAPath
        assert ret.relativePath == preserveRPath

    def test_features_mapReduce_handmadeNoneReturningReducer(self):
        """ Test features.mapReduce() against handmade output with a None returning Reducer """
        featureNames = ["one", "two", "three"]
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.features.mapReduce(simpleMapper, oddOnlyReducer)

        exp = self.constructor([[1, 11], [3, 15]])

        assert (ret.isIdentical(exp))
        assert (toTest.isIdentical(self.constructor(data, featureNames=featureNames)))

    ####################
    # elements.count() #
    ####################
    @noLogEntryExpected
    def test_elements_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data)
        ret = toTest.elements.count('>=5')
        assert ret == 5

        ret = toTest.elements.count(lambda x: x % 2 == 1)
        assert ret == 5

    ##################
    # points.count() #
    ##################
    @noLogEntryExpected
    def test_points_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'], featureNames=['a', 'b', 'c'])
        ret = toTest.points.count('b>=5')
        assert ret == 2

        ret = toTest.points.count(lambda x: x['b'] >= 5)
        assert ret == 2

    ####################
    # features.count() #
    ####################
    @noLogEntryExpected
    def test_features_count(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        toTest = self.constructor(data, pointNames=['one', 'two', 'three'], featureNames=['a', 'b', 'c'])
        ret = toTest.features.count('two>=5')
        assert ret == 2

        ret = toTest.features.count(lambda x: x['two'] >= 5)
        assert ret == 2

    ##########################
    # isApproximatelyEqual() #
    ##########################

    @attr('slow')
    @noLogEntryExpected
    def test_isApproximatelyEqual_randomTest(self):
        """ Test isApproximatelyEqual() using randomly generated data """

        for x in range(2):
            points = 100
            features = 40
            data = numpy.zeros((points, features))

            for i in range(points):
                for j in range(features):
                    data[i, j] = numpyRandom.rand() * numpyRandom.randint(0, 5)

            toTest = self.constructor(data)

            for retType in UML.data.available:
                currObj = UML.createData(retType, data, useLog=False)
                assert toTest.isApproximatelyEqual(currObj)
                assert toTest.hashCode() == currObj.hashCode()
                assertNoNamesGenerated(toTest)
                assertNoNamesGenerated(currObj)


    ######################
    # trainAndTestSets() #
    ######################

    # simple sucess - no labels
    @logCountAssertionFactory(2)
    def test_trainAndTestSets_simple_nolabels(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, teX = toTest.trainAndTestSets(.5)

        assert len(trX.points) == 2
        assert len(trX.features) == 5
        assert len(teX.points) == 2
        assert len(teX.features) == 5

        # try the same test with a default named object
        toTest = self.constructor(data)

        trX, teX = toTest.trainAndTestSets(.5)
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(teX)

    # simple sucess - single label
    @logCountAssertionFactory(2)
    def test_trainAndTestSets_simple_singlelabel(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0)

        assert len(trX.points) == 2
        assert len(trX.features) == 4
        assert len(trY.points) == 2
        assert len(trY.features) == 1
        assert len(teX.points) == 2
        assert len(teX.features) == 4
        assert len(teY.points) == 2
        assert len(teY.features) == 1

        # try the same test with a default named object
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0)
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(trY)
        assertNoNamesGenerated(teX)
        assertNoNamesGenerated(teY)

    # simple sucess - multi label
    @logCountAssertionFactory(2)
    def test_trainAndTestSets_simple_multilabel(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        featureNames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, featureNames=featureNames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=[0, 'labs2'])

        assert len(trX.points) == 2
        assert len(trX.features) == 3
        assert len(trY.points) == 2
        assert len(trY.features) == 2
        assert len(teX.points) == 2
        assert len(teX.features) == 3
        assert len(teY.points) == 2
        assert len(teY.features) == 2

        # try the same test with a default named object
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=[0, 2])
        assertNoNamesGenerated(toTest)
        assertNoNamesGenerated(trX)
        assertNoNamesGenerated(trY)
        assertNoNamesGenerated(teX)
        assertNoNamesGenerated(teY)

    # edge cases 0/1 test portions
    def test_trainAndTestSets_0or1_testFraction(self):
        data = [[1, 2, 3, 33], [2, 5, 6, 66], [3, 8, 9, 99], [4, 11, 12, 111]]
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(0, 0)

        assert len(trX.points) == 4
        assert len(trY.points) == 4
        assert len(teX.points) == 0
        assert len(teY.points) == 0

        trX, trY, teX, teY = toTest.trainAndTestSets(1, 0)

        assert len(trX.points) == 0
        assert len(trY.points) == 0
        assert len(teX.points) == 4
        assert len(teY.points) == 4

    # each returned set independant of calling set
    def test_trainAndTestSets_unconnectedReturn(self):
        data = [[1, 1], [2, 2], [3, 3], [4, 4]]
        toTest = self.constructor(data)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0, randomOrder=False)

        assert trX == trY
        assert teX == teY

        def changeFirst(point):
            ret = []
            first = True
            for val in point:
                if first:
                    ret.append(-val)
                    first = False
                else:
                    ret.append(val)
            return ret

        # change the returned data
        trX.points.transform(changeFirst)
        trY.points.transform(changeFirst)
        assert trX[0, 0] == -1
        assert trY[0, 0] == -1

        # check all other data is unchanged
        assert toTest[0, 0] == 1
        assert teX[0, 0] == 3
        assert teY[0, 0] == 3

        # check that our multiple references are still the same

        assert trX == trY
        assert teX == teY

    def test_trainAndTestSets_nameAppend_PathPreserve(self):
        data = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
        toTest = self.constructor(data, )
        tmpFile = tempfile.NamedTemporaryFile(suffix='.csv')
        toTest.writeFile(tmpFile.name, fileFormat='csv')

        toTest = self.constructor(tmpFile.name, name='toTest')

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

        assert trX.name == 'toTest trainX'
        assert trX.path == tmpFile.name
        assert trX.absolutePath == tmpFile.name
        assert trX.relativePath == os.path.relpath(tmpFile.name)

        assert trY.name == 'toTest trainY'
        assert trY.path == tmpFile.name
        assert trY.path == tmpFile.name
        assert trY.absolutePath == tmpFile.name
        assert trY.relativePath == os.path.relpath(tmpFile.name)

        assert teX.name == 'toTest testX'
        assert teX.path == tmpFile.name
        assert teX.path == tmpFile.name
        assert teX.absolutePath == tmpFile.name
        assert teX.relativePath == os.path.relpath(tmpFile.name)

        assert teY.name == 'toTest testY'
        assert teY.path == tmpFile.name
        assert teY.path == tmpFile.name
        assert teY.absolutePath == tmpFile.name
        assert teY.relativePath == os.path.relpath(tmpFile.name)


    def test_trainAndTestSets_PandFnamesPerserved(self):
        data = [[1, 5, -1, 3, 33], [2, 5, -2, 6, 66], [3, 5, -2, 9, 99], [4, 5, -4, 12, 111]]
        pnames = ['one', 'two', 'three', 'four']
        fnames = ['labs1', 'fives', 'labs2', 'bozo', 'long']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)

        trX, trY, teX, teY = toTest.trainAndTestSets(.5, labels=0, randomOrder=False)

        assert trX.points.getNames() == ['one', 'two']
        assert trX.features.getNames() == ['fives', 'labs2', 'bozo', 'long']

        assert trY.points.getNames() == ['one', 'two']
        assert trY.features.getNames() == ['labs1']

        assert teX.points.getNames() == ['three', 'four']
        assert teX.features.getNames() == ['fives', 'labs2', 'bozo', 'long']

        assert teY.points.getNames() == ['three', 'four']
        assert teY.features.getNames() == ['labs1']


    def test_trainAndTestSets_randomOrder(self):
        data = [[1, 1], [2, 2], [3, 3], [4, 4]]
        toTest = self.constructor(data)

        for i in range(100):
            trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0, randomOrder=False)

            assert trX == trY
            assert trX[0] == 1
            assert trX[1] == 2

            assert teX == teY
            assert teX[0] == 3
            assert teX[1] == 4

        for i in range(100):
            trX, trY, teX, teY = toTest.trainAndTestSets(.5, 0)

            # just to make sure everything looks right
            assert trX == trY
            assert teX == teY
            # this means point ordering was randomized, so we return successfully
            if trX[0] != 1:
                return

        assert False  # implausible number of checks for random order were unsucessful

    #################
    # points.unique #
    #################

    def test_points_unique_allNames_string(self):
        data = [['George', 'Washington'], ['George', 'Washington'],
                ['John', 'Adams'], ['John', 'Adams'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'],  ['Thomas', 'Jefferson'],
                ['James', 'Madison']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 'Washington'], ['John', 'Adams'],
                   ['Thomas', 'Jefferson'], ['James', 'Madison']]
        exp = self.constructor(expData, pointNames=["p0", "p2", "p5", "p7"], featureNames=ftNames)

        ret = test.points.unique()

        assert ret == exp

    def test_points_unique_allNames_numeric(self):
        data = [[0, 0], [0, 0],
                [99, 99], [99, 99],
                [5.5, 11], [5.5, 11],  [5.5, 11],
                [-1, -2]]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0, 0], [99, 99], [5.5, 11], [-1, -2]]
        exp = self.constructor(expData, pointNames=["p0", "p2", "p4", "p7"], featureNames=ftNames)

        ret = test.points.unique()

        assert ret == exp

    @noLogEntryExpected
    def test_points_unique_allNames_mixed(self):
        data = [['George', 0], ['George', 0],
                ['John', 1], ['John', 1], ['John', 1],
                ['Thomas', 2],  ['Thomas', 2],
                ['James', 3]]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 0], ['John', 1],
                   ['Thomas', 2], ['James', 3]]
        exp = self.constructor(expData, pointNames=["p0", "p2", "p5", "p7"], featureNames=ftNames)

        ret = test.points.unique()

        assert ret == exp

    def test_points_unique_allNames_allUnique(self):
        data = [['George', 'Washington'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'], ['James', 'Madison'],
                ['James', 'Monroe'], ['John Quincy', 'Adams'],
                ['Andrew', 'Jackson'], ['Martin', 'Van Buren']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        exp = test.copy()

        ret = test.points.unique()

        assert ret == exp

    def test_points_unique_allDefaultNames(self):
        data = [['George', 'Washington'], ['George', 'Washington'],
                ['John', 'Adams'], ['John', 'Adams'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'],  ['Thomas', 'Jefferson'],
                ['James', 'Madison']]
        test = self.constructor(data)
        expData = [['George', 'Washington'], ['John', 'Adams'],
                   ['Thomas', 'Jefferson'], ['James', 'Madison']]
        exp = self.constructor(expData)

        ret = test.points.unique()

        assert ret == exp
        assertNoNamesGenerated(test)
        assertNoNamesGenerated(ret)

    def test_points_unique_subsetFeature0(self):
        data = [['George', 'Washington'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'], ['James', 'Madison'],
                ['James', 'Monroe'], ['John Quincy', 'Adams'],
                ['Andrew', 'Jackson'], ['Martin', 'Van Buren']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George'], ['John'], ['Thomas'], ['James'],
                   ['John Quincy'], ['Andrew'], ['Martin']]
        expPtNames = ["p0", "p1", "p2", "p3", "p5", "p6", "p7"]
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=["firstName"])

        ret = test[:, 0].points.unique()

        assert ret == exp

    def test_points_unique_subsetFeature1(self):
        data = [['George', 'Washington'], ['John', 'Adams'],
                ['Thomas', 'Jefferson'], ['James', 'Madison'],
                ['James', 'Monroe'], ['John Quincy', 'Adams'],
                ['Andrew', 'Jackson'], ['Martin', 'Van Buren']]
        ptNames = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        ftNames = ["firstName", "lastName"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['Washington'], ['Adams'], ['Jefferson'], ['Madison'],
                   ['Monroe'], ['Jackson'], ['Van Buren']]
        expPtNames = ["p0", "p1", "p2", "p3", "p4", "p6", "p7"]
        exp = self.constructor(expData, pointNames=expPtNames, featureNames=["lastName"])

        ret = test[:, 1].points.unique()

        assert ret == exp

    ###################
    # features.unique #
    ###################

    def test_features_unique_allNames_string(self):
        data = [['a','b','c','a','b','c'],
                ['1','2','3','1','2','4'],
                ['0','0','0','0','0','0']]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['a','b','c','c'], ['1','2','3','4'], ['0','0','0','0']]
        exp = self.constructor(expData, pointNames=ptNames, featureNames=["f0", "f1", "f2", "f5"])

        ret = test.features.unique()

        assert ret == exp

    def test_features_unique_allNames_numeric(self):
        data = [[0, 1, 2, 0, 1, 2],
                [0, 0, 0, 0, 0, 0],
                [-1, -2, -3, -1, -1, -3]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0, 1, 2, 1], [0, 0, 0, 0], [-1, -2, -3, -1]]
        exp = self.constructor(expData, pointNames=ptNames, featureNames=["f0", "f1", "f2", "f4"])

        ret = test.features.unique()

        assert ret == exp

    @noLogEntryExpected
    def test_features_unique_allNames_mixed(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 0, 'George', 0],
                   ['John', 1, 'James', 3],
                   ['Thomas', 2, 'Thomas', 2]]
        exp = self.constructor(expData, pointNames=ptNames, featureNames=["f0", "f1", "f2", "f3"])

        ret = test.features.unique()

        assert ret == exp

    def test_features_unique_allNames_allUnique(self):
        data = [['George', 0, 'George', 1, 'James', 2],
                ['John', 1, 'James', 0, 'John', 1],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 0]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        exp = test.copy()

        ret = test.features.unique()

        assert ret == exp

    def test_features_unique_allDefaultNames(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        test = self.constructor(data)

        expData = [['George', 0, 'George', 0],
                   ['John', 1, 'James', 3],
                   ['Thomas', 2, 'Thomas', 2]]
        exp = self.constructor(expData)

        ret = test.features.unique()

        assert ret == exp
        assertNoNamesGenerated(test)
        assertNoNamesGenerated(ret)

    def test_features_unique_subsetPoint0(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['George', 0]]
        exp = self.constructor(expData, pointNames=["p0"], featureNames=["f0", "f1"])

        ret = test[0, :].features.unique()

        assert ret == exp

    def test_features_unique_subsetPoint1(self):
        data = [['George', 0, 'George', 0, 'George', 0],
                ['John', 1, 'James', 3, 'John', 3],
                ['Thomas', 2, 'Thomas', 2, 'Thomas', 2]]
        ptNames = ["p0", "p1", "p2"]
        ftNames = ["f0", "f1", "f2", "f3", "f4", "f5"]
        test = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [['John', 1, 'James', 3]]
        exp = self.constructor(expData, pointNames=["p1"], featureNames=["f0", "f1", "f2", "f3"])

        ret = test[1, :].features.unique()

        assert ret == exp

    ########################
    # elements.countUnique #
    ########################
    @noLogEntryExpected
    def test_elements_countUnique_allPtsAndFtrs(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        toTest = self.constructor(data)
        unique = toTest.elements.countUnique()

        assert len(unique) == 6
        assert unique[1] == 2
        assert unique[2] == 2
        assert unique[3] == 2
        assert unique['a'] == 1
        assert unique['b'] == 1
        assert unique['c'] == 1
        # for Sparse, 0 is added to returned dictionary manually
        # want to test 0 is not added if the data doesn't contain zeros
        assert 0 not in unique
        assertNoNamesGenerated(toTest)

    def test_elements_countUnique_limitPoints(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        pNames = ['p1', 'p2', 'p3']
        toTest = self.constructor(data, pointNames=pNames)
        unique = toTest.elements.countUnique(points=0)

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[2] == 1
        assert unique[3] == 1

        unique = toTest.elements.countUnique(points='p1')

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[2] == 1
        assert unique[3] == 1

        unique = toTest.elements.countUnique(points=[0,'p3'])

        assert len(unique) == 3
        assert unique[1] == 2
        assert unique[2] == 2
        assert unique[3] == 2

    def test_elements_countUnique_limitFeatures(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        fNames = ['f1', 'f2', 'f3']
        toTest = self.constructor(data, featureNames=fNames)
        unique = toTest.elements.countUnique(features=0)

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[3] == 1
        assert unique['a'] == 1

        unique = toTest.elements.countUnique(features='f1')

        assert len(unique) == 3
        assert unique[1] == 1
        assert unique[3] == 1
        assert unique['a'] == 1

        unique = toTest.elements.countUnique(features=[0,'f3'])

        assert len(unique) == 4
        assert unique[1] == 2
        assert unique[3] == 2
        assert unique['a'] == 1
        assert unique['c'] == 1

    @noLogEntryExpected
    def test_elements_countUnique_limitPointsAndFeatures_cornercase(self):
        data = [[1, 2, 3], ['a', 'b', 'c'], [3, 2, 1]]
        fNames = ['f1', 'f2', 'f3']
        pNames = ['p1', 'p2', 'p3']
        toTest = self.constructor(data, featureNames=fNames, pointNames=pNames)

        unique = toTest.elements.countUnique(features=[0,'f3'], points=[0,'p3'])

        assert len(unique) == 2
        assert unique[1] == 2
        assert unique[3] == 2

    def test_elements_countUnique_zeroCount(self):
        data = [[0, 0, 0, 0, 1], [2, 0, 0, 0, 0], [0, 0, 3, 0, 0]]
        toTest = self.constructor(data)
        unique = toTest.elements.countUnique()

        assert 0 in unique
        assert unique[0] == 12


class HighLevelModifying(DataTestObject):

    ####################################
    # replaceFeatureWithBinaryFeatures #
    ####################################

    @raises(ImproperObjectAction)
    def test_replaceFeatureWithBinaryFeatures_PemptyException(self):
        """ Test replaceFeatureWithBinaryFeatures() with a point empty object """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.replaceFeatureWithBinaryFeatures(0)

    @raises(ImproperObjectAction)
    def test_replaceFeatureWithBinaryFeatures_FemptyException(self):
        """ Test replaceFeatureWithBinaryFeatures() with a feature empty object """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.replaceFeatureWithBinaryFeatures(0)

    @oneLogEntryExpected
    def test_replaceFeatureWithBinaryFeatures_handmade(self):
        """ Test replaceFeatureWithBinaryFeatures() against handmade output """
        data = [[1], [2], [3]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)
        getNames = self.constructor(data, featureNames=featureNames)
        ret = toTest.replaceFeatureWithBinaryFeatures(0)

        expData = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        expFeatureNames = []
        for point in getNames.points:
            expFeatureNames.append('col=' + str(point[0]))
        exp = self.constructor(expData, featureNames=expFeatureNames)

        assert toTest.isIdentical(exp)
        assert ret == expFeatureNames

    @oneLogEntryExpected
    def test_replaceFeatureWithBinaryFeatures_insertLocation(self):
        """ Test replaceFeatureWithBinaryFeatures() replaces at same index """
        data = [['a', 1, 'a'], ['b', 2, 'b'], ['c', 3, 'c']]
        featureNames = ['stay1', 'replace', 'stay2']
        toTest = self.constructor(data, featureNames=featureNames)
        getNames = self.constructor(data, featureNames=featureNames)
        ret = toTest.replaceFeatureWithBinaryFeatures(1)

        expData = [['a', 1, 0, 0, 'a'], ['b', 0, 1, 0, 'b'], ['c', 0, 0, 1, 'c']]
        expFeatureNames = []
        for point in getNames.points:
            expFeatureNames.append('replace=' + str(point[1]))
        expFeatureNames.insert(0, 'stay1')
        expFeatureNames.append('stay2')
        exp = self.constructor(expData, featureNames=expFeatureNames)

        assert toTest.isIdentical(exp)
        assert ret == expFeatureNames[1: -1]

    def test_replaceFeatureWithBinaryFeatures_NamePath_preservation(self):
        data = [[1], [2], [3]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.replaceFeatureWithBinaryFeatures(0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ##############################
    # transformFeatureToIntegers #
    ##############################

    @raises(ImproperObjectAction)
    def test_transformFeatureToIntegers_PemptyException(self):
        """ Test transformFeatureToIntegers() with an point empty object """
        data = [[], []]
        data = numpy.array(data).T
        toTest = self.constructor(data)
        toTest.transformFeatureToIntegers(0)

    @raises(ImproperObjectAction)
    def test_transformFeatureToIntegers_FemptyException(self):
        """ Test transformFeatureToIntegers() with an feature empty object """
        data = [[], []]
        data = numpy.array(data)
        toTest = self.constructor(data)
        toTest.transformFeatureToIntegers(0)

    @oneLogEntryExpected
    def test_transformFeatureToIntegers_handmade(self):
        """ Test transformFeatureToIntegers() against handmade output """
        data = [['a'], ['b'], ['c'], ['b'], ['a']]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)
        ret = toTest.transformFeatureToIntegers(0)

        assert toTest[0, 0] == toTest[4, 0]
        assert toTest[1, 0] == toTest[3, 0]
        assert toTest[0, 0] != toTest[1, 0]
        assert toTest[0, 0] != toTest[2, 0]

        # ensure data was transformed to a numeric type.
        for i in range(len(toTest.points)):
            # Matrix and Sparse might store values as floats or numpy types
            assert isinstance(toTest[i, 0], (int, float, numpy.number))

        # check ret
        assert len(ret) == 3
        assert all(isinstance(key, int) for key in ret.keys())
        for value in ret.values():
            assert value in ['a', 'b', 'c']

    def test_transformFeatureToIntegers_handmade_lazyNameGeneration(self):
        """ Test transformFeatureToIntegers() against handmade output """
        data = [['a'], ['b'], ['c'], ['b'], ['a']]
        toTest = self.constructor(data)
        ret = toTest.transformFeatureToIntegers(0)

        assertNoNamesGenerated(toTest)

    def test_transformFeatureToIntegers_pointNames(self):
        """ Test transformFeatureToIntegers preserves pointNames """
        data = [['a'], ['b'], ['c'], ['b'], ['a']]
        pnames = ['1a', '2a', '3', '2b', '1b']
        fnames = ['col']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = toTest.transformFeatureToIntegers(0)

        assert toTest.points.getName(0) == '1a'
        assert toTest.points.getName(1) == '2a'
        assert toTest.points.getName(2) == '3'
        assert toTest.points.getName(3) == '2b'
        assert toTest.points.getName(4) == '1b'

        # ensure data was transformed to a numeric type.
        for i in range(len(toTest.points)):
            # Matrix and Sparse might store values as floats or numpy types
            assert isinstance(toTest[i, 0], (int, float, numpy.number))

        # check ret
        assert len(ret) == 3
        assert all(isinstance(key, int) for key in ret.keys())
        for value in ret.values():
            assert value in ['a', 'b', 'c']

    def test_transformFeatureToIntegers_positioning(self):
        """ Test transformFeatureToIntegers preserves featurename mapping """
        data = [['a', 0], ['b', 1], ['c', 2], ['b', 3], ['a', 4]]
        pnames = ['1a', '2a', '3', '2b', '1b']
        fnames = ['col', 'pos']
        toTest = self.constructor(data, pointNames=pnames, featureNames=fnames)
        ret = toTest.transformFeatureToIntegers(0)

        assert toTest[0, 0] == toTest[4, 0]
        assert toTest[1, 0] == toTest[3, 0]
        assert toTest[0, 0] != toTest[1, 0]
        assert toTest[0, 0] != toTest[2, 0]

        assert toTest[0, 1] == 0
        assert toTest[1, 1] == 1
        assert toTest[2, 1] == 2
        assert toTest[3, 1] == 3
        assert toTest[4, 1] == 4

        # ensure data was transformed to a numeric type.
        for i in range(len(toTest.points)):
            # Matrix and Sparse might store values as floats or numpy types
            assert isinstance(toTest[i, 0], (int, float, numpy.number))

        # check ret
        assert len(ret) == 3
        assert all(isinstance(key, int) for key in ret.keys())
        for value in ret.values():
            assert value in ['a', 'b', 'c']

    def test_transformFeatureToIntegers_ZerosInFeatureValuesPreserved(self):
        data = [['a'], [52], [0], [0], [0], [52], ['a']]

        toTest = self.constructor(data, featureNames=False)
        ret = toTest.transformFeatureToIntegers(0)

        assert ret[0] == 0
        assert toTest[0, 0] == toTest[6, 0]
        assert toTest[1, 0] == toTest[5, 0]
        assert toTest[2, 0] == 0
        assert toTest[3, 0] == 0
        assert toTest[4, 0] == 0

    def test_transformFeatureToIntegers_NamePath_preservation(self):
        data = [[10], [20], [30.5], [20], [10]]
        featureNames = ['col']
        toTest = self.constructor(data, featureNames=featureNames)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.transformFeatureToIntegers(0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    ####################
    # points.shuffle() #
    ####################

    def testpoints_shuffle_noLongerEqual(self):
        """ Tests points.shuffle() results in a changed object """
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it shuffles it into the same configuration.
        # the odds are vanishingly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for i in range(5):
            ret = toTest.points.shuffle() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        assert all(ret is None for ret in returns)
        assertNoNamesGenerated(toTest)


    def testpoints_shuffle_NamePath_preservation(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # it is possible that it shuffles it into the same configuration.
        # we only test after we're sure we've done something
        while True:
            toTest.points.shuffle()  # RET CHECK
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ######################
    # features.shuffle() #
    ######################

    def test_features_shuffle_noLongerEqual(self):
        """ Tests features.shuffle() results in a changed object """
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        # it is possible that it shuffles it into the same configuration.
        # the odds are vanishly low that it will do so over consecutive calls
        # however. We will pass as long as it changes once
        returns = []
        for i in range(5):
            ret = toTest.features.shuffle() # RET CHECK
            returns.append(ret)
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert not toTest.isApproximatelyEqual(toCompare)

        assert all(ret is None for ret in returns)
        assertNoNamesGenerated(toTest)


    def test_features_shuffle_NamePath_preservation(self):
        data = [[1, 2, 3, 33], [4, 5, 6, 66], [7, 8, 9, 99], [10, 11, 12, 1111111]]
        toTest = self.constructor(deepcopy(data))
        toCompare = self.constructor(deepcopy(data))

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        # it is possible that it shuffles it into the same configuration.
        # we only test after we're sure we've done something
        while True:
            toTest.features.shuffle()  # RET CHECK
            if not toTest.isApproximatelyEqual(toCompare):
                break

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    ###########################################
    # points.normalize / features.normalize() #
    ###########################################
    @oneLogEntryExpected
    def normalizeHelper(self, caller, axis, subtract=None, divide=None, also=None):
        if axis == 'point':
            func = caller.points.normalize
        else:
            func = caller.features.normalize
        if 'cython' in str(func.__func__.__class__):#if it is a cython function
            d = func.__func__.__defaults__
            assert (d is None) or (d == (None, None, None))
        else:#if it is a normal python function
            a, va, vk, d = UML.helpers.inspectArguments(func)
            assert d == (None, None, None, None)

        if axis == 'point':
            return caller.points.normalize(subtract=subtract, divide=divide, applyResultTo=also)
        else:
            caller.transpose(useLog=False)
            if also is not None:
                also.transpose(useLog=False)
            ret = caller.features.normalize(subtract=subtract, divide=divide, applyResultTo=also)
            caller.transpose(useLog=False)
            if also is not None:
                also.transpose(useLog=False)
            return ret

    #exception different type from expected inputs
    def test_points_normalize_exception_unexpected_input_type(self):
        self.back_normalize_exception_unexpected_input_type("point")

    def test_features_normalize_exception_unexpected_input_type(self):
        self.back_normalize_exception_unexpected_input_type("feature")

    def back_normalize_exception_unexpected_input_type(self, axis):
        obj = self.constructor([[1, 2], [3, 4]])

        try:
            self.normalizeHelper(obj, axis, subtract={})
            assert False  # Expected InvalidArgumentType
        except InvalidArgumentType:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=set([1]))
            assert False  # Expected InvalidArgumentType
        except InvalidArgumentType:
            pass


    # exception non stats string
    def test_points_normalize_exception_unexpected_string_value(self):
        self.back_normalize_exception_unexpected_string_value('point')

    def test_features_normalize_exception_unexpected_string_value(self):
        self.back_normalize_exception_unexpected_string_value('feature')

    def back_normalize_exception_unexpected_string_value(self, axis):
        obj = self.constructor([[1, 2], [3, 4]])

        try:
            self.normalizeHelper(obj, axis, subtract="Hello")
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

        try:
            self.normalizeHelper(obj, axis, divide="enumerate")
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass


    # exception wrong length vector shaped UML object
    def test_points_normalize_exception_wrong_vector_length(self):
        self.back_normalize_exception_wrong_vector_length('point')

    def test_features_normalize_exception_wrong_vector_length(self):
        self.back_normalize_exception_wrong_vector_length('feature')

    def back_normalize_exception_wrong_vector_length(self, axis):
        obj = self.constructor([[1, 2], [3, 4]])
        vectorLong = self.constructor([[1, 2, 3, 4]])
        vectorShort = self.constructor([[11]])

        try:
            self.normalizeHelper(obj, axis, subtract=vectorLong)
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=vectorShort)
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass


    # exception wrong size of UML object
    def test_points_normalize_exception_wrong_size_object(self):
        self.back_normalize_exception_wrong_size_object('point')

    def test_features_normalize_exception_wrong_size_object(self):
        self.back_normalize_exception_wrong_size_object('feature')

    def back_normalize_exception_wrong_size_object(self, axis):
        obj = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5]])
        objBig = self.constructor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
        objSmall = self.constructor([[1, 1], [2, 2]])

        try:
            self.normalizeHelper(obj, axis, subtract=objBig)
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=objSmall)
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

    # applyResultTo is wrong shape in the normalized axis
    def test_points_normalize_exception_applyResultTo_wrong_shape(self):
        self.back_normalize_exception_applyResultTo_wrong_shape('point')

    def test_features_normalize_exception_applyResultTo_wrong_shape(self):
        self.back_normalize_exception_applyResultTo_wrong_shape('feature')

    def back_normalize_exception_applyResultTo_wrong_shape(self, axis):
        obj = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5]])
        alsoShort = self.constructor([[1, 2, 2], [3, 4, 4]])
        alsoLong = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5], [1, 23, 4]])

        try:
            self.normalizeHelper(obj, axis, subtract=1, also=alsoShort)
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=2, also=alsoLong)
            assert False  # Expected InvalidArgumentValue
        except InvalidArgumentValue:
            pass

    # applyResultTo is wrong shape when given obj subtract and divide
    def test_points_normalize_exception_applyResultTo_wrong_shape_obj_input(self):
        self.back_normalize_exception_applyResultTo_wrong_shape_obj_input('point')

    def test_features_normalize_exception_applyResultTo_wrong_shape_obj_input(self):
        self.back_normalize_exception_applyResultTo_wrong_shape_obj_input('feature')

    def back_normalize_exception_applyResultTo_wrong_shape_obj_input(self, axis):
        obj = self.constructor([[1, 2, 2], [3, 4, 4], [5, 5, 5]])
        alsoShort = self.constructor([[1, 2], [3, 4], [5, 5]])
        alsoLong = self.constructor([[1, 2, 2, 2], [3, 4, 4, 4], [5, 5, 5, 6]])

        sub_div = self.constructor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        try:
            self.normalizeHelper(obj, axis, subtract=sub_div, also=alsoShort)
            assert False  # Expected InvalidArgumentValueCombination
        except InvalidArgumentValueCombination:
            pass

        try:
            self.normalizeHelper(obj, axis, divide=sub_div, also=alsoLong)
            assert False  # Expected InvalidArgumentValueCombination
        except InvalidArgumentValueCombination:
            pass

    # successful float valued inputs
    def test_points_normalize_success_float_int_inputs_NoAlso(self):
        self.back_normalize_success_float_int_inputs_NoAlso("point")

    def test_features_normalize_success_float_int_inputs_NoAlso(self):
        self.back_normalize_success_float_int_inputs_NoAlso("feature")

    def back_normalize_success_float_int_inputs_NoAlso(self, axis):
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        expObj = self.constructor([[0, 0, 0], [4, 4, 4], [12, 12, 12]])

        ret = self.normalizeHelper(obj, axis, subtract=1, divide=0.5)

        assert ret is None
        assert expObj == obj
        assertNoNamesGenerated(obj)

        # vector versions
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        expObj = self.constructor([[0, 0, 0], [4, 4, 4], [12, 12, 12]])

        for retType in UML.data.available:
            currObj = obj.copy()
            sub = UML.createData(retType, [1] * 3)
            div = UML.createData(retType, [0.5] * 3)
            ret = self.normalizeHelper(currObj, axis, subtract=sub, divide=div)

            assert ret is None
            assert expObj == currObj
            assertNoNamesGenerated(currObj)


    # successful float valued inputs
    def test_points_normalize_success_float_int_inputs(self):
        self.back_normalize_success_float_int_inputs("point")

    def test_features_normalize_success_float_int_inputs(self):
        self.back_normalize_success_float_int_inputs("feature")

    def back_normalize_success_float_int_inputs(self, axis):
        obj = self.constructor([[1, 1, 1], [3, 3, 3], [7, 7, 7]])
        also = self.constructor([[-1, -1, -1], [.5, .5, .5], [2, 2, 2]])
        expObj = self.constructor([[0, 0, 0], [4, 4, 4], [12, 12, 12]])
        expAlso = self.constructor([[-4, -4, -4], [-1, -1, -1], [2, 2, 2]])

        ret = self.normalizeHelper(obj, axis, subtract=1, divide=0.5, also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # successful stats-string valued inputs
    def test_points_normalize_success_stat_string_inputs(self):
        self.back_normalize_success_stat_string_inputs("point")

    def test_features_normalize_success_stat_string_inputs(self):
        self.back_normalize_success_stat_string_inputs("feature")

    def back_normalize_success_stat_string_inputs(self, axis):
        obj = self.constructor([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
        also = self.constructor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        expObj = self.constructor([[0, 0, 0], [.5, .5, .5], [2, 2, 2]])
        expAlso = self.constructor([[0, 1, 2], [0, .5, 1], [0, -1, -2]])

        ret = self.normalizeHelper(obj, axis, subtract="unique count", divide="median", also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # successful vector object valued inputs
    def test_points_normalize_success_vector_object_inputs(self):
        self.back_normalize_success_vector_object_inputs("point")

    def test_features_normalize_success_vector_object_inputs(self):
        self.back_normalize_success_vector_object_inputs("feature")

    def back_normalize_success_vector_object_inputs(self, axis):
        obj = self.constructor([[1, 3, 7], [10, 30, 70], [100, 300, 700]])
        also = self.constructor([[2, 6, 14], [10, 30, 70], [100, 300, 700]])
        subVec = self.constructor([[1, 10, 100]])
        divVec = self.constructor([[.5], [5], [50]])
        expObj = self.constructor([[0, 4, 12], [0, 4, 12], [0, 4, 12]])
        expAlso = self.constructor([[2, 10, 26], [0, 4, 12], [0, 4, 12]])

        ret = self.normalizeHelper(obj, axis, subtract=subVec, divide=divVec, also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # successful matrix valued inputs
    def test_points_normalize_success_full_object_inputs(self):
        self.back_normalize_success_full_object_inputs("point")

    def test_features_normalize_success_full_object_inputs(self):
        self.back_normalize_success_full_object_inputs("feature")

    def back_normalize_success_full_object_inputs(self, axis):
        obj = self.constructor([[2, 10, 100], [3, 30, 300], [7, 70, 700]])
        also = self.constructor([[1, 5, 100], [3, 30, 300], [7, 70, 700]])

        subObj = self.constructor([[0, 5, 20], [3, 10, 60], [4, -30, 100]])
        divObj = self.constructor([[2, .5, 4], [2, 2, 4], [.25, 2, 6]])

        expObj = self.constructor([[1, 10, 20], [0, 10, 60], [12, 50, 100]])
        expAlso = self.constructor([[.5, 0, 20], [0, 10, 60], [12, 50, 100]])

        if axis == 'point':
            ret = self.normalizeHelper(obj, axis, subtract=subObj, divide=divObj, also=also)
        else:
            subObj.transpose()
            divObj.transpose()
            ret = self.normalizeHelper(obj, axis, subtract=subObj, divide=divObj, also=also)

        assert ret is None
        assert expObj == obj
        assert expAlso == also


    # string valued inputs and also values that are different in shape.
    def test_features_normalize_success_statString_diffSizeAlso(self):
        self.back_normalize_success_statString_diffSizeAlso("point")

    def test_points_normalize_success_statString_diffSizeAlso(self):
        self.back_normalize_success_statString_diffSizeAlso("feature")

    def back_normalize_success_statString_diffSizeAlso(self, axis):
        obj1 = self.constructor([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
        obj2 = self.constructor([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
        alsoLess = self.constructor([[1, 2], [1, 2], [1, 2]])
        alsoMore = self.constructor([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 1, 2]])

        expObj = self.constructor([[0, 0, 0], [.5, .5, .5], [2, 2, 2]])
        expAlsoL = self.constructor([[0, 1], [0, .5], [0, -1]])
        expAlsoM = self.constructor([[0, 1, 0, 1], [0, .5, 0, .5], [0, -1, 0, -1]])

        ret1 = self.normalizeHelper(obj1, axis, subtract="unique count", divide="median", also=alsoLess)
        ret2 = self.normalizeHelper(obj2, axis, subtract="unique count", divide="median", also=alsoMore)

        assert ret1 is None
        assert ret2 is None
        assert expObj == obj1
        assert expObj == obj2
        assert expAlsoL == alsoLess
        assert expAlsoM == alsoMore

    #################
    # features.fill #
    #################
    @logCountAssertionFactory(4)
    def test_features_fill_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        ret = obj1.features.fill(match.missing, fill.mean) #RET CHECK
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'], useLog=False)
        assert obj1 == exp1
        assert ret is None

        obj2 = obj0.copy()
        obj2.features.fill([3, 7], None)
        obj2.features.fill(match.missing, fill.mean)
        exp2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        exp2.features.setNames(['a', 'b', 'c'], useLog=False)
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.features.fill(match.missing, fill.mean, returnModified=True)
        exp3 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp3.features.setNames(['a', 'b', 'c'], useLog=False)
        expRet = self.constructor([[False, False, False], [True, False, True], [False, False, True], [False, False, False]])
        expRet.features.setNames(['a_modified', 'b_modified', 'c_modified'], useLog=False)
        assert obj3 == exp3
        assert ret == expRet

    def test_features_fill_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.features.fill(match.nonNumeric, fill.mean)
        exp1 = self.constructor([[1, 2, 3], [5, 11, 6], [7, 11, 6], [7, 8, 9]])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

        obj2 = obj0.copy()
        obj2.features.fill([3, 7], 'na')
        obj2.features.fill(match.nonNumeric, fill.mean)
        exp2 = self.constructor([[1, 2, 9], [1, 11, 9], [1, 11, 9], [1, 8, 9]])
        exp2.features.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.features.fill(match.nonNumeric, fill.mean, features=['b',2], returnModified=True)
        exp3 = self.constructor([[1, 2, 3], ['na', 11, 6], [7, 11, 6], [7, 8, 9]])
        exp3.features.setNames(['a', 'b', 'c'])
        expRet = self.constructor([[False, False], [False, True], [False, True], [False, False]])
        expRet.features.setNames(['b_modified', 'c_modified'])
        assert obj3 == exp3
        assert ret == expRet

    @raises(InvalidArgumentValue)
    def test_features_fill_mean_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fill(match.missing, fill.mean)

    def test_features_fill_median_missing(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, None)
        obj.features.fill(match.missing, fill.median)
        exp = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3], ['na', 11, 'na'], [7, 11, 'na'], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, 'na')
        obj.features.fill(match.nonNumeric, fill.median)
        exp = self.constructor([[1, 2, 3], [7, 5, 6], [7, 5, 6], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_features_fill_median_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fill(match.missing, fill.median)

    def test_features_fill_mode(self):
        obj0 = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj0.features.fill(9, None)
        obj0.features.fill(match.missing, fill.mode)
        exp0 = self.constructor([[1, 2, 3], [7, 11, 3], [7, 11, 3], [7, 8, 3]])
        exp0.features.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([['a','b','c'], [None, 'd', None], ['e','d','c'], ['e','f','g']], featureNames=['a', 'b', 'c'])
        obj1.features.fill('c', None)
        obj1.features.fill(match.missing, fill.mode)
        exp1 = self.constructor([['a','b','g'], ['e','d', 'g'], ['e','d', 'g'], ['e','f', 'g']])
        exp1.features.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

    @raises(InvalidArgumentValue)
    def test_features_fill_mode_allMatches(self):
        obj = self.constructor([[1, None, 3], [4, None, 6], [7, None, 9]])
        obj.features.fill(match.missing, fill.mode)

    def test_features_fill_zero(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, None)
        obj.features.fill(match.missing, 0, features=['b', 'c'])
        exp = self.constructor([[1, 2, 3], [None, 0, 0], [7, 0, 0], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(0, 100)
        exp = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_forwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.forwardFill)
        exp = self.constructor([[1, 2, 3], [1, 11, 3], [7, 11, 3], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_features_fill_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.forwardFill)

    def test_features_fill_backwardFill(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(11, None)
        obj.features.fill(match.missing, fill.backwardFill)
        exp = self.constructor([[1, 2, 3], [7, 8, 9], [7, 8, 9], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_features_fill_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, None, 3], [None, 11, None], [7, 11, None], [7, None, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.backwardFill)

    def test_features_fill_interpolate(self):
        obj = self.constructor([[1, 2, 3], [None, 11, None], [7, 11, None], [7, 8, 9]], featureNames=['a', 'b', 'c'])
        obj.features.fill(match.missing, fill.interpolate)
        exp = self.constructor([[1, 2, 3], [4, 11, 5], [7, 11, 7], [7, 8, 9]])
        exp.features.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_features_fill_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.features.fill(negative, 0)
        assert toTest == exp

    def test_features_fill_custom_fill(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, -3, 4], [5, 2, -3, 8], [9, 10, 11, 4]]
        exp = self.constructor(expData)

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.features.fill(match.negative, firstValue)
        assert toTest == exp

    def test_features_fill_custom_fillAndMatch(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, -3, 4], [5, 2, -3, 8], [9, 10, 11, 4]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.features.fill(negative, firstValue)
        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_features_fill_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.features.fill(999, float('nan'))
        obj2.features.fill(999, None)
        obj3.features.fill(999, numpy.nan)
        obj1.features.fill(numpy.nan, 0)
        obj2.features.fill(numpy.nan, 0)
        obj3.features.fill(float('nan'), 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_features_fill_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fill(999, None)
        obj.features.fill([1, numpy.nan], 0)

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_features_fill_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fill(999, None)
        obj.features.fill(match.missing, 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_features_fill_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.features.fill(999, 'na')

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_features_fill_NamePath_preservation(self):
        data = [['a'], ['b'], [1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.features.fill(match.nonNumeric, 0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'


    ###################
    # points.fill #
    ###################
    @logCountAssertionFactory(4)
    def test_points_fill_mean_missing(self):
        obj0 = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        ret = obj1.points.fill(match.missing, fill.mean) # RET CHECK
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'], useLog=False)
        assert obj1 == exp1
        assert ret is None

        obj2 = obj0.copy()
        obj2.points.fill([4, 8], None)
        obj2.points.fill(match.missing, fill.mean)
        exp2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        exp2.points.setNames(['a', 'b', 'c'], useLog=False)
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.points.fill(match.missing, fill.mean, returnModified=True)
        exp3 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        expRet = self.constructor([[False, False, False, False], [True, False, True, False], [False, False, False, True]])
        exp3.points.setNames(['a', 'b', 'c'], useLog=False)
        expRet.points.setNames(['a_modified', 'b_modified', 'c_modified'], useLog=False)
        assert obj3 == exp3
        assert ret == expRet

    def test_points_fill_mean_nonNumeric(self):
        obj0 = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj1 = obj0.copy()
        obj1.points.fill(match.nonNumeric, fill.mean)
        exp1 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

        obj2 = obj0.copy()
        obj2.points.fill([4, 8], 'na')
        obj2.points.fill(match.nonNumeric, fill.mean)
        exp2 = self.constructor([[1, 2, 3, 2], [6, 6, 6, 6], [9, 1, 11, 7]])
        exp2.points.setNames(['a', 'b', 'c'])
        assert obj2 == exp2

        obj3 = obj0.copy()
        ret = obj3.points.fill(match.nonNumeric, fill.mean, points=['b', 2], returnModified=True)
        exp3 = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 7]])
        expRet = self.constructor([[True, False, True, False], [False, False, False, True]])
        exp3.points.setNames(['a', 'b', 'c'])
        expRet.points.setNames(['b_modified', 'c_modified'])
        assert obj3 == exp3
        assert ret == expRet

    @raises(InvalidArgumentValue)
    def test_points_fill_mean_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fill(match.missing, fill.mean)

    def test_points_fill_median_missing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.median)
        exp = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 11, 9]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_points_fill_median_nonNumeric(self):
        obj = self.constructor([[1, 2, 3, 4], ['na', 6, 'na', 8], [9, 1, 11, 'na']], pointNames=['a', 'b', 'c'])
        obj.points.fill(11, 'na')
        obj.points.fill(match.nonNumeric, fill.median)
        exp = self.constructor([[1, 2, 3, 4], [7, 6, 7, 8], [9, 1, 5, 5]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_points_fill_median_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fill(match.missing, fill.median)

    def test_points_fill_mode(self):
        obj0 = self.constructor([[1, 2, 3, 3], [None, 6, 8, 8], [9, 9, 11, None]], pointNames=['a', 'b', 'c'])
        obj0.points.fill(9, None)
        obj0.points.fill(match.missing, fill.mode)
        exp0 = self.constructor([[1, 2, 3, 3], [8, 6, 8, 8], [11, 11, 11, 11]], pointNames=['a', 'b', 'c'])
        exp0.points.setNames(['a', 'b', 'c'])
        assert obj0 == exp0

        obj1 = self.constructor([['a', 'b', 'c', 'c'], [None, 'f', 'h', 'h'], ['i', 'i', 'k', None]], pointNames=['a', 'b', 'c'])
        obj1.points.fill('b', None)
        obj1.points.fill(match.missing, fill.mode)
        exp1 = self.constructor([['a', 'c', 'c', 'c'], ['h', 'f', 'h', 'h'], ['i', 'i', 'k', 'i']], pointNames=['a', 'b', 'c'])
        exp1.points.setNames(['a', 'b', 'c'])
        assert obj1 == exp1

    @raises(InvalidArgumentValue)
    def test_points_fill_mode_allMatches(self):
        obj = self.constructor([[1, 2, 3], [None, None, None], [7, 8, 9]])
        obj.points.fill(match.missing, fill.mode)

    def test_points_fill_zero(self):
        obj = self.constructor([[1, 2, None], [None, 11, 6], [7, 11, None], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.points.fill(11, None)
        obj.points.fill(match.missing, 0, points=['b', 'c'])
        exp = self.constructor([[1, 2, None], [0, 0, 6], [7, 0, 0], [7, 8, 9]])
        exp.points.setNames(['a', 'b', 'c', 'd'])
        assert obj == exp

    def test_points_fill_constant(self):
        obj = self.constructor([[1, 2, 3], [0, 0, 0], [7, 0, 0], [7, 8, 9]], pointNames=['a', 'b', 'c', 'd'])
        obj.points.fill(0, 100)
        exp = self.constructor([[1, 2, 3], [100, 100, 100], [7, 100, 100], [7, 8, 9]])
        exp.points.setNames(['a', 'b', 'c', 'd'])
        assert obj == exp

    def test_points_fill_forwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.forwardFill)
        exp = self.constructor([[1, 2, 3, 4], [5, 5, 5, 8], [9, 1, 11, 11]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_points_fill_forwardFill_firstFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [None, 6, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.forwardFill)

    def test_points_fill_backwardFill(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, 11, 2]], pointNames=['a', 'b', 'c'])
        obj.points.fill(11, None)
        obj.points.fill(match.missing, fill.backwardFill)
        exp = self.constructor([[1, 2, 3, 4], [5, 8, 8, 8], [1, 1, 2, 2]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    @raises(InvalidArgumentValue)
    def test_points_fill_backwardFill_lastFeatureValueMissing(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [9, 1, 11, None]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.backwardFill)

    def test_points_fill_interpolate(self):
        obj = self.constructor([[1, 2, 3, 4], [5, None, None, 8], [None, 1, None, 5]], pointNames=['a', 'b', 'c'])
        obj.points.fill(match.missing, fill.interpolate)
        exp = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 3, 5]])
        exp.points.setNames(['a', 'b', 'c'])
        assert obj == exp

    def test_points_fill_custom_match(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        toTest.points.fill(negative, 0)
        assert toTest == exp
        assertNoNamesGenerated(toTest)

    def test_points_fill_custom_fill(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 1, 4], [5, 5, 5, 8], [9, 10, 11, 9]]
        exp = self.constructor(expData)

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.points.fill(match.negative, firstValue)
        assert toTest == exp

    def test_points_fill_custom_fillAndMatch(self):
        data = [[1, 2, -3, 4], [5, -6, -7, 8], [9, 10, 11, -12]]
        toTest = self.constructor(data)

        expData = [[1, 2, 1, 4], [5, 5, 5, 8], [9, 10, 11, 9]]
        exp = self.constructor(expData)

        def negative(value):
            return value < 0

        def firstValue(feat, match):
            first = feat[0]
            ret = []
            for i, val in enumerate(feat):
                if match(val):
                    ret.append(first)
                else:
                    ret.append(val)
            return ret

        toTest.points.fill(negative, firstValue)
        assert toTest == exp

    def test_points_fill_fillValuesWithNaN_constant(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj1 = self.constructor(data)
        obj2 = self.constructor(data)
        obj3 = self.constructor(data)
        obj1.points.fill(999, float('nan'))
        obj2.points.fill(999, None)
        obj3.points.fill(999, numpy.nan)
        obj1.points.fill(numpy.nan, 0)
        obj2.points.fill(numpy.nan, 0)
        obj3.points.fill(float('nan'), 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj1 == exp
        assert obj2 == obj1
        assert obj3 == obj1

    def test_points_fill_fillValuesWithNaN_list(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fill(999, None)
        obj.points.fill([1, numpy.nan], 0)

        exp = self.constructor([[0, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_points_fill_fillValuesWithNaN_function(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fill(999, None)
        obj.points.fill(match.missing, 0)

        exp = self.constructor([[1, 2, 0, 4], [5, 0, 0, 8], [9, 10, 11, 0]])
        assert obj == exp

    def test_points_fill_fillNumericWithNonNumeric(self):
        data = [[1, 2, 999, 4], [5, 999, 999, 8], [9, 10, 11, 999]]
        obj = self.constructor(data)
        obj.points.fill(999, 'na')

        exp = self.constructor([[1, 2, 'na', 4], [5, 'na', 'na', 8], [9, 10, 11, 'na']])
        assert obj == exp

    def test_points_fill_NamePath_preservation(self):
        data = [['a', 'b', 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.points.fill(match.nonNumeric, 0)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    #######################
    # fillUsingAllData #
    #######################
    @oneLogEntryExpected
    def test_fillUsingAllData_kNeighborsRegressor_missing(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        ret = toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, **kwarguments) # RET CHECK
        assert toTest == expTest
        assert ret is None

    @oneLogEntryExpected
    def test_fillUsingAllData_kNeighborsRegressor_returnModified_all(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        ret = toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, returnModified=True, **kwarguments)
        expRet = self.constructor([[False, True, True], [False, False, False], [False, False, False], [False, False, False], [True, False, True]])
        expRet.features.setNames([name + "_modified" for name in expRet.features.getNames()], useLog=False)
        assert toTest == expTest
        assert ret == expRet

    def test_fillUsingAllData_kNeighborsRegressor_nonNumeric(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, 'na', 'x'], [1, 3, 9], [2, 1, 6], [3, 2, 3], ['na', 3, 'x']]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.nonNumeric, fill.kNeighborsRegressor, **kwarguments)
        assert toTest == expTest

    @oneLogEntryExpected
    def test_fillUsingAllData_kNeighborsRegressor_pointsLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, points=[2, 3, 4], **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsRegressor_featuresLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 2, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [2, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, features=[1,0], **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsRegressor_pointsFeaturesLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = data = [[1, None, None], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, 6], [1, 3, 9], [2, 1, 6], [3, 2, 3], [None, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor, points=0, features=2, **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_missing(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_nonNumeric(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.nonNumeric, fill.kNeighborsClassifier, **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_pointsLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, points=[2, 3, 4], **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_featuresLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, 3, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, features=[1,0], **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_pointsFeaturesLimited(self):
        fNames = ['a', 'b', 'c']
        pNames = ['p0', 'p1', 'p2', 'p3', 'p4']
        data = data = [[1, None, None], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)
        expData = [[1, None, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [None, 3, None]]
        expTest = self.constructor(expData, pointNames=pNames, featureNames=fNames)
        toTest.fillUsingAllData(match.missing, fill.kNeighborsClassifier, points=0, features=2, **kwarguments)
        assert toTest == expTest

    def test_fillUsingAllData_kNeighborsClassifier_lazyNameGeneration(self):
        data = [[1, 'na', 'x'], [1, 3, 6], [2, 1, 6], [1, 3, 7], ['na', 3, 'x']]
        kwarguments = {'n_neighbors': 3}
        toTest = self.constructor(data)
        expData = [[1, 3, 6], [1, 3, 6], [2, 1, 6], [1, 3, 7], [1, 3, 6]]
        expTest = self.constructor(expData)
        toTest.fillUsingAllData(match.nonNumeric, fill.kNeighborsClassifier, **kwarguments)
        assert toTest == expTest
        assertNoNamesGenerated(toTest)

    def test_fillUsingAllData_NamePath_preservation(self):
        data = [[None, None, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        toTest = self.constructor(data)

        toTest._name = "TestName"
        toTest._absPath = "TestAbsPath"
        toTest._relPath = "testRelPath"

        toTest.fillUsingAllData(match.missing, fill.kNeighborsRegressor)

        assert toTest.name == "TestName"
        assert toTest.absolutePath == "TestAbsPath"
        assert toTest.relativePath == 'testRelPath'

    ####################################
    # points.splitByCollapsingFeatures #
    ####################################
    @logCountAssertionFactory(3)
    def test_points_splitByCollapsingFeatures_sequentialFeatures(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]
        ptNames = ["0", "1", "2"]
        ftNames = ["ret0", "ret1", "coll0", "coll1", "coll2", "coll3"]

        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0,0,"coll0",1], [0,0,"coll1",2], [0,0,"coll2",3], [0,0,"coll3",4],
                   [1,1,"coll0",5], [1,1,"coll1",6], [1,1,"coll2",7], [1,1,"coll3",8],
                   [2,2,"coll0",-1], [2,2,"coll1",-2], [2,2,"coll2",-3], [2,2,"coll3",-4]]
        expPnames = ["0_0", "0_1", "0_2", "0_3",
                     "1_0", "1_1", "1_2", "1_3",
                     "2_0", "2_1", "2_2", "2_3"]
        expFNames = ["ret0", "ret1", "ftNames", "ftValues"]

        exp = self.constructor(expData, pointNames=expPnames, featureNames=expFNames)

        nameFeatureNames = "ftNames"
        nameFeatureValues = "ftValues"
        test0 = toTest.copy()
        toCollapse = ["coll0", "coll1", "coll2", "coll3"]
        test0.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test0 == exp

        test1 = toTest.copy()
        toCollapse = [2, 3, 4, 5]
        test1.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test1 == exp

        test2 = toTest.copy()
        toCollapse = [2, "coll1", 4, "coll3"]
        test2.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test2 == exp

    @logCountAssertionFactory(3)
    def test_points_splitByCollapsingFeatures_nonSequentialFeatures(self):
        data = [[1,0,2,0,3,4], [5,1,6,1,7,8], [-1,2,-2,2,-3,-4]]
        ptNames = ["0", "1", "2"]
        ftNames = ["coll0", "ret0", "coll1", "ret1", "coll2", "coll3"]

        toTest = self.constructor(data, pointNames=ptNames, featureNames=ftNames)

        expData = [[0,0,"coll3",4], [0,0,"coll1",2], [0,0,"coll2",3], [0,0,"coll0",1],
                   [1,1,"coll3",8], [1,1,"coll1",6], [1,1,"coll2",7], [1,1,"coll0",5],
                   [2,2,"coll3",-4], [2,2,"coll1",-2], [2,2,"coll2",-3], [2,2,"coll0",-1]]
        expPnames = ["0_0", "0_1", "0_2", "0_3",
                     "1_0", "1_1", "1_2", "1_3",
                     "2_0", "2_1", "2_2", "2_3"]
        expFNames = ["ret0", "ret1", "ftNames", "ftValues"]

        exp = self.constructor(expData, pointNames=expPnames, featureNames=expFNames)

        nameFeatureNames = "ftNames"
        nameFeatureValues = "ftValues"
        test0 = toTest.copy()
        toCollapse = ["coll3", "coll1", "coll2", "coll0"]
        test0.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test0 == exp

        test1 = toTest.copy()
        toCollapse = [5, 2, 4, 0]
        test1.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test1 == exp

        test2 = toTest.copy()
        toCollapse = [5, "coll1", "coll2", 0]
        test2.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert test2 == exp

    def test_points_splitByCollapsingFeatures_noPointNames(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]
        ftNames = ["ret0", "ret1", "coll0", "coll1", "coll2", "coll3"]

        toTest = self.constructor(data, featureNames=ftNames)

        expData = [[0,0,"coll0",1], [0,0,"coll1",2], [0,0,"coll2",3], [0,0,"coll3",4],
                   [1,1,"coll0",5], [1,1,"coll1",6], [1,1,"coll2",7], [1,1,"coll3",8],
                   [2,2,"coll0",-1], [2,2,"coll1",-2], [2,2,"coll2",-3], [2,2,"coll3",-4]]
        expFNames = ["ret0", "ret1", "ftNames", "ftValues"]

        exp = self.constructor(expData, featureNames=expFNames)

        toCollapse = ["coll0", "coll1", "coll2", "coll3"]
        toTest.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert toTest == exp

    def test_points_splitByCollapsingFeatures_noNames(self):
        data = [[0,0,1,2,3,4], [1,1,5,6,7,8], [2,2,-1,-2,-3,-4]]

        toTest = self.constructor(data)
        coll0 = DEFAULT_PREFIX + str(2)
        coll1 = DEFAULT_PREFIX + str(3)
        coll2 = DEFAULT_PREFIX + str(4)
        coll3 = DEFAULT_PREFIX + str(5)

        expData = [[0,0,coll0,1], [0,0,coll1,2], [0,0,coll2,3], [0,0,coll3,4],
                   [1,1,coll0,5], [1,1,coll1,6], [1,1,coll2,7], [1,1,coll3,8],
                   [2,2,coll0,-1], [2,2,coll1,-2], [2,2,coll2,-3], [2,2,coll3,-4]]

        exp = self.constructor(expData)
        exp.features.setName(2, "ftNames")
        exp.features.setName(3, "ftValues")

        toCollapse = [2, 3, 4, 5]
        toTest.points.splitByCollapsingFeatures(toCollapse, "ftNames", "ftValues")
        assert toTest == exp

    #####################################
    # points.combineByExpandingFeatures #
    #####################################
    @logCountAssertionFactory(3)
    def test_points_combineByExpandingFeatures(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r2', 9.9], ["p1", 100, 'r3', 9.8],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0], ["p2", 100, 'r3', 5.9],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2], ["p3", 100, 'r3', 11.0],
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1], ["p1", 200, 'r3', 19.8]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.9, 9.8],
                   ["p2", 100, 6.5, 6.0, 5.9],
                   ["p3", 100, 11, 11.2, 11.0],
                   ["p1", 200, 18.1, 20.1, 19.8]]
        expFNames = ['type', 'dist', 'r1', 'r2', 'r3']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        test0 = toTest.copy()
        test0.points.combineByExpandingFeatures('run', 'time')
        assert test0 == exp

        test1 = toTest.copy()
        test1.points.combineByExpandingFeatures(2, 3)
        assert test1 == exp

        test2 = toTest.copy()
        test2.points.combineByExpandingFeatures('run', 3)
        assert test2 == exp

    def test_points_combineByExpandingFeatures_withMissing(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r3', 9.8],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0], ["p2", 100, 'r3', 5.9],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2],
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1], ["p1", 200, 'r3', 19.8]]
        pNames = [str(i) for i in range(10)]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.8, None],
                   ["p2", 100, 6.5, 5.9, 6.0],
                   ["p3", 100, 11, None, 11.2],
                   ["p1", 200, 18.1, 19.8, 20.1]]
        expFNames = ['type', 'dist', 'r1', 'r3', 'r2']
        expPNames = ["0", "2", "5", "7"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        toTest.points.combineByExpandingFeatures('run', 'time')
        assert toTest == exp

    def test_points_combineByExpandingFeatures_nonConcurrentNamesAndValues(self):
        data = [[100, "r1", 'p1', 9.5], [100, "r2", 'p1', 9.9], [100, "r3", 'p1', 9.8],
                [100, "r1", 'p2', 6.5], [100, "r2", 'p2', 6.0], [100, "r3", 'p2', 5.9],
                [100, "r1", 'p3', 11], [100, "r2", 'p3', 11.2], [100, "r3", 'p3', 11.0],
                [200, "r1", 'p1', 18.1], [200, "r2", 'p1', 20.1], [200, "r3", 'p1', 19.8]]
        pNames = [str(i) for i in range(12)]
        fNames = ['dist', 'run', 'type', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[100, 9.5, 9.9, 9.8, "p1"],
                   [100, 6.5, 6.0, 5.9, "p2"],
                   [100, 11, 11.2, 11.0, "p3"],
                   [200, 18.1, 20.1, 19.8, "p1"]]
        expFNames = ['dist', 'r1', 'r2', 'r3', 'type']
        expPNames = ["0", "3", "6", "9"]
        exp = self.constructor(expData, pointNames=expPNames, featureNames=expFNames)

        toTest.points.combineByExpandingFeatures('run', 'time')
        assert toTest == exp

    def test_points_combineByExpandingFeatures_noPointNames(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r2', 9.9], ["p1", 100, 'r3', 9.8],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0], ["p2", 100, 'r3', 5.9],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2], ["p3", 100, 'r3', 11.0],
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1], ["p1", 200, 'r3', 19.8]]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, featureNames=fNames)

        expData = [["p1", 100, 9.5, 9.9, 9.8],
                   ["p2", 100, 6.5, 6.0, 5.9],
                   ["p3", 100, 11, 11.2, 11.0],
                   ["p1", 200, 18.1, 20.1, 19.8]]
        expFNames = ['type', 'dist', 'r1', 'r2', 'r3']
        exp = self.constructor(expData, featureNames=expFNames)

        toTest.points.combineByExpandingFeatures('run', 'time')
        assert toTest == exp

    def test_points_combineByExpandingFeatures_noNames(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r2', 9.9], ["p1", 100, 'r3', 9.8],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0], ["p2", 100, 'r3', 5.9],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2], ["p3", 100, 'r3', 11.0],
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1], ["p1", 200, 'r3', 19.8]]
        toTest = self.constructor(data)

        expData = [["p1", 100, 9.5, 9.9, 9.8],
                   ["p2", 100, 6.5, 6.0, 5.9],
                   ["p3", 100, 11, 11.2, 11.0],
                   ["p1", 200, 18.1, 20.1, 19.8]]
        exp = self.constructor(expData)
        exp.features.setName(2, 'r1')
        exp.features.setName(3, 'r2')
        exp.features.setName(4, 'r3')

        toTest.points.combineByExpandingFeatures(2, 3)
        assert toTest == exp

    @raises(ImproperObjectAction)
    def test_points_combineByExpandingFeatures_2valuesSameFeature(self):
        data = [["p1", 100, 'r1', 9.5], ["p1", 100, 'r2', 9.9], ["p1", 100, 'r3', 9.8],
                ["p2", 100, 'r1', 6.5], ["p2", 100, 'r2', 6.0], ["p2", 100, 'r3', 5.9],
                ["p3", 100, 'r1', 11], ["p3", 100, 'r2', 11.2], ["p3", 100, 'r1', 11.0], # r1 in p3 twice
                ["p1", 200, 'r1', 18.1], ["p1", 200, 'r2', 20.1], ["p1", 200, 'r3', 19.8]]
        pNames = [str(i) for i in range(12)]
        fNames = ['type', 'dist', 'run', 'time']
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.points.combineByExpandingFeatures('run', 'time')

    ###########################
    # features.splitByParsing #
    ###########################
    @oneLogEntryExpected
    def test_features_splitByParsing_integer(self):
        data = [[0, "a1", 0], [1, "b2", 1], [2, "c3", 2]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged", "f1"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a", "1", 0], [1, "b", "2", 1], [2, "c", "3", 2]]
        expFNames = ["f0", "split0", "split1", "f1"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(1, 1, ["split0", "split1"])
        assert toTest == exp

    @oneLogEntryExpected
    def test_features_splitByParsing_string(self):
        data = [["a-1", 0], ["b-2", 1], ["c-3", 2]]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["a", "1", 0], ["b", "2", 1], ["c", "3", 2]]
        expFNames = ["split0", "split1", "f0"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(0, '-', ["split0", "split1"])
        assert toTest == exp

    def test_features_splitByParsing_listIntegers(self):
        data = [[0, "a1z9000AAA"], [1, "b2y8000BBB"], [2, "c3x7000CCC"]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a1", "z9", "000", "AAA"], [1, "b2", "y8", "000", "BBB"],
                   [2, "c3", "x7", "000", "CCC"]]
        expFNames = ["f0", "split0", "split1", "split2", "split3"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing("merged", [2,4,7], ["split0", "split1", "split2", "split3"])
        assert toTest == exp

    def test_features_splitByParsing_listStrings(self):
        data = [[0, "a1/9000-AAA"], [1, "b2/8000-BBB"], [2, "c3/7000-CCC"]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a1", "9000", "AAA"], [1, "b2", "8000", "BBB"],
                   [2, "c3", "7000", "CCC"]]
        expFNames = ["f0", "split0", "split1", "split2"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(1, ['/','-'], ["split0", "split1", "split2"])
        assert toTest == exp

    @oneLogEntryExpected
    def test_features_splitByParsing_listMixed(self):
        data = [[0, "a1/9z000-AAA"], [1, "b2/8y000-BBB"], [2, "c3/7x000-CCC"]]
        pNames = ["0", "1", "2"]
        fNames = ["f0", "merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [[0, "a1", "9z", "000", "AAA"], [1, "b2", "8y", "000", "BBB"],
                   [2, "c3", "7x", "000", "CCC"]]
        expFNames = ["f0", "split0", "split1", "split2", "split3"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        toTest.features.splitByParsing(1, ['/', 5, '-'], ["split0", "split1", "split2", "split3"])
        assert toTest == exp

    def test_features_splitByParsing_function(self):
        data = [["a1z9000AAA"], ["b2y8000BBB"], ["c3x7000CCC"]]
        pNames = ["a", "b", "c"]
        fNames = ["merged"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["a1z9", "AAA"], ["b2y8", "BBB"], ["c3x7", "CCC"]]
        expFNames = ["split0", "split1"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        def splitter(value):
            return value.split('000')

        toTest.features.splitByParsing("merged", splitter, ["split0", "split1"])
        assert toTest == exp

    def test_features_splitByParsing_regex(self):
        data = [["a1z9000AAA", '001'], ["b2y8000BBB", '001'], ["c3x7000CCC", '002']]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        expData = [["a1", "9000AAA", '001'], ["b2", "8000BBB", '001'], ["c3", "7000CCC", '002']]
        expFNames = ["split0", "split1", "f0"]
        exp = self.constructor(expData, pointNames=pNames, featureNames=expFNames)

        def splitter(value):
            import re
            return re.split('[xyz]', value)

        toTest.features.splitByParsing("merged", splitter, ["split0", "split1"])
        assert toTest == exp

    def test_features_splitByParsing_noNames(self):
        data = [[0, "a1", 0], [1, "b2", 1], [2, "c3", 2]]
        toTest = self.constructor(data)

        expData = [[0, "a", "1", 0], [1, "b", "2", 1], [2, "c", "3", 2]]
        exp = self.constructor(expData)
        exp.features.setName(1,"split0")
        exp.features.setName(2,"split1")

        toTest.features.splitByParsing(1, 1, ["split0", "split1"])
        assert toTest == exp

    @raises(InvalidArgumentValueCombination)
    def test_features_splitByParsing_shortSplitList(self):
        data = [["a-1", 0], ["b-2", 1], ["c3", 2]]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.features.splitByParsing("merged", '-', ["split0", "split1"])

    @raises(InvalidArgumentValueCombination)
    def test_features_splitByParsing_longSplitList(self):
        data = [["a-1", 0], ["b-2-2", 1], ["c-3", 2]]
        pNames = ["a", "b", "c"]
        fNames = ["merged", "f0"]
        toTest = self.constructor(data, pointNames=pNames, featureNames=fNames)

        toTest.features.splitByParsing("merged", '-', ["split0", "split1"])

class HighLevelAll(HighLevelDataSafe, HighLevelModifying):
    pass
