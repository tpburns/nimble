"""
Unit tests for mlpy_interface.py

"""

from __future__ import absolute_import
import UML

from nose.tools import *
import numpy.testing

from UML.exceptions import InvalidArgumentValue
from UML.interfaces.mlpy_interface import Mlpy

from .test_helpers import checkLabelOrderingAndScoreAssociations
from .skipTestDecorator import SkipMissing
from ..assertionHelpers import logCountAssertionFactory
from ..assertionHelpers import noLogEntryExpected, oneLogEntryExpected

mlpy = UML.importExternalLibraries.importModule("mlpy")

mlpySkipDec = SkipMissing('mlpy')

@mlpySkipDec
@noLogEntryExpected
def test_Mlpy_version():
    interface = Mlpy()
    assert interface.version() == mlpy.__version__

@mlpySkipDec
@oneLogEntryExpected
def testMlpyHandmadeSVMClassification():
    """ Test mlpy() by calling on SVM classification with handmade output """

    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={})

    assert ret is not None

    expected = [[1.]]
    expectedObj = UML.createData('Matrix', expected, useLog=False)

    numpy.testing.assert_approx_equal(ret.data[0, 0], 1.)

@mlpySkipDec
@oneLogEntryExpected
def testMlpyHandmadeLogisticRegression():
    """ Test mlpy() by calling on logistic regression on handmade output """

    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.LibLinear", trainingObj, trainY="Y", testX=testObj,
                            output=None, arguments={"solver_type": "l2r_lr"})

    assert ret is not None

    expected = [[1.]]
    expectedObj = UML.createData('Matrix', expected, useLog=False)

    numpy.testing.assert_approx_equal(ret.data[0, 0], 1.)

@mlpySkipDec
@oneLogEntryExpected
def testMlpyHandmadeKNN():
    """ Test mlpy() by calling on knn classification on handmade output """

    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [0, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.KNN", trainingObj, trainY="Y", testX=testObj,
                            output=None, arguments={"k": 1})

    assert ret is not None

    numpy.testing.assert_approx_equal(ret.data[0, 0], 1.)
    numpy.testing.assert_approx_equal(ret.data[1, 0], 0.)

@mlpySkipDec
@oneLogEntryExpected
def testMlpyHandmadePCA():
    """ Test mlpy() by calling PCA and checking the output has the correct dimension """
    data = [[1, 1, 1], [2, 2, 2], [4, 4, 4]]
    trainingObj = UML.createData('Matrix', data, useLog=False)

    data2 = [[4, 4, 4]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.PCA", trainingObj, testX=testObj, output=None, arguments={'k': 1})

    assert ret is not None
    # check return has the right dimension
    assert len(ret.data[0]) == 1

@mlpySkipDec
@oneLogEntryExpected
def testMlpyHandmadeKernelPCA():
    """ Test mlpy() by calling PCA with a kernel transformation, checking the output has the correct dimension """
    data = [[1, 1], [2, 2], [3, 3]]
    trainObj = UML.createData('Matrix', data, useLog=False)

    data2 = [[4, 4]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.KPCA", trainObj, testX=testObj, output=None,
                            arguments={"kernel": "KernelGaussian", 'k': 1})

    assert ret is not None
    # check return has the right dimension
    assert len(ret.data[0]) == 1

@mlpySkipDec
@raises(InvalidArgumentValue)
@noLogEntryExpected
def testMlpyHandmadeInnerProductTrainingPCAException():
    """ Test mlpy by calling a kernel based leaner with no kernel or transformed data """
    data = [[1, 1], [2, 2], [3, 3], [7, 7]]
    trainObj = UML.createData('Matrix', data, useLog=False)

    data2 = [[4, 4]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.KPCA", trainObj, testX=testObj, output=None, arguments={'k': 1})

    assert ret is not None

@mlpySkipDec
@oneLogEntryExpected
def testMlpyHandmadeInnerProductTrainingPCA():
    """ Test mlpy by calling PCA with data that has already been run through a kernel """
    import mlpy

    data = [[1, 1], [2, 2], [3, 3], [7, 7]]
    kernData = mlpy.kernel_linear(data, data)
    trainObj = UML.createData('Matrix', kernData, useLog=False)

    data2 = [[4, 4]]
    kernData2 = mlpy.kernel_linear(data2, data2)
    testObj = UML.createData('Matrix', kernData2, useLog=False)

    ret = UML.trainAndApply("mlpy.KPCA", trainObj, testX=testObj, output=None, arguments={'k': 1})

    assert ret is not None

@mlpySkipDec
@logCountAssertionFactory(3)
def testMlpyScoreMode():
    """ Test mlpy() scoreMode flags"""
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    # default scoreMode is 'label'
    ret = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    bestScores = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert len(bestScores.points) == 2
    assert len(bestScores.features) == 2

    allScores = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert len(allScores.points) == 2
    assert len(allScores.features) == 3

    checkLabelOrderingAndScoreAssociations([0, 1, 2], bestScores, allScores)

@mlpySkipDec
@logCountAssertionFactory(3)
def testMlpyScoreModeBinary():
    """ Test mlpy() scoreMode flags, binary case"""
    variables = ["Y", "x1", "x2"]
    data = [[1, 1, 1], [1, 0, 1], [1, -1, -1], [-1, 30, 2], [-1, 30, 3], [-1, 34, 4]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 1], [25, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    # default scoreMode is 'label'
    ret = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={})
    assert len(ret.points) == 2
    assert len(ret.features) == 1

    bestScores = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={},
                                   scoreMode='bestScore')
    assert len(bestScores.points) == 2
    assert len(bestScores.features) == 2

    allScores = UML.trainAndApply("mlpy.LibSvm", trainingObj, trainY="Y", testX=testObj, arguments={},
                                  scoreMode='allScores')
    assert len(allScores.points) == 2
    assert len(allScores.features) == 2

    checkLabelOrderingAndScoreAssociations([1, -1], bestScores, allScores)

@mlpySkipDec
@logCountAssertionFactory(3)
def testMlpyRegression():
    """ Test mlpy() regressors problematic in previous implementations """

    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.Ridge", trainingObj, trainY="Y", testX=testObj, output=None, arguments={})
    assert ret is not None

    ret = UML.trainAndApply("mlpy.PLS", trainingObj, trainY="Y", testX=testObj, output=None, arguments={'iters': 10})
    assert ret is not None

    ret = UML.trainAndApply("mlpy.ElasticNet", trainingObj, trainY="Y", testX=testObj, output=None,
                            arguments={'lmb': .1, 'eps': .1})
    assert ret is not None

@mlpySkipDec
@raises(InvalidArgumentValue)
def testMlpyOLSDisallowed():
    """ Test mlpy's broken OLS Learner is disallowed """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2)

    UML.trainAndApply("mlpy.OLS", trainingObj, trainY="Y", testX=testObj, output=None, arguments={})

@mlpySkipDec
@raises(InvalidArgumentValue)
def testMlpyLARSDisallowed():
    """ Test mlpy's broken LARS Learner is disallowed """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2)

    UML.trainAndApply("mlpy.LARS", trainingObj, trainY="Y", testX=testObj, output=None, arguments={})

@mlpySkipDec
@logCountAssertionFactory(4)
def testMlpyRequiredInitArgs():
    """ Test mlpy() learners with required parameters to their __init__ methods"""
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, -300, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.PLS", trainingObj, trainY="Y", testX=testObj, output=None, arguments={'iters': 10})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.DLDA", trainingObj, trainY="Y", testX=testObj, output=None, arguments={'delta': 1})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.ElasticNet", trainingObj, trainY="Y", testX=testObj, output=None,
                            arguments={'lmb': 1, 'eps': 1})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.ElasticNetC", trainingObj, trainY="Y", testX=testObj, output=None,
                            arguments={'lmb': 1, 'eps': 1})
    assert ret is not None

@mlpySkipDec
@logCountAssertionFactory(2)
def testMlpyUnknownCrashes():
    """ Test mlpy on learners failing for undiagnosed reasons in previous implementation """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, -300, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.Golub", trainingObj, trainY="Y", testX=testObj, output=None, arguments={})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.Perceptron", trainingObj, trainY="Y", testX=testObj, output=None, arguments={})
    assert ret is not None

@mlpySkipDec
@logCountAssertionFactory(5)
def testMlpyKernelLearners():
    """ Test mlpy on a handful of learners that rely on kernels """
    variables = ["Y", "x1", "x2"]
    data4d = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj4d = UML.createData('Matrix', data4d, featureNames=variables, useLog=False)

    data2d = [[0, 1, 1], [0, 0, 1], [0, 3, 2], [1, -300, 2], [1, 1, 500]]
    trainingObj2d = UML.createData('Matrix', data2d, featureNames=variables, useLog=False)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2, useLog=False)

    ret = UML.trainAndApply("mlpy.KFDA", trainingObj2d, trainY="Y", testX=testObj, output=None,
                            arguments={'kernel': 'KernelLinear'})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.KFDAC", trainingObj2d, trainY="Y", testX=testObj, output=None,
                            arguments={'kernel': 'KernelPolynomial'})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.KPCA", trainingObj4d, trainY="Y", testX=testObj, output=None,
                            arguments={'kernel': 'KernelGaussian'})
    assert ret is not None
    ret = UML.trainAndApply("mlpy.KernelRidge", trainingObj4d, trainY="Y", testX=testObj, output=None,
                            arguments={'kernel': 'KernelSigmoid'})
    assert ret is not None
    # 2 classes
    ret = UML.trainAndApply("mlpy.Parzen", trainingObj2d, trainY="Y", testX=testObj, output=None,
                            arguments={'kernel': 'KernelGaussian'})
    assert ret is not None

@mlpySkipDec
@raises(InvalidArgumentValue)
def testMlpyKernelExponentialDisallowed():
    """ Test mlpy that trying to use KernelExponential throws an exception """
    variables = ["Y", "x1", "x2"]

    data2d = [[0, 1, 1], [0, 0, 1], [0, 3, 2], [1, -300, 2], [1, 1, 500]]
    trainingObj2d = UML.createData('Matrix', data2d, featureNames=variables)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2)

    ret = UML.trainAndApply("mlpy.KFDA", trainingObj2d, trainY="Y", testX=testObj, output=None,
                            arguments={'kernel': 'KernelExponential'})
    assert ret is not None

@mlpySkipDec
@logCountAssertionFactory(2)
def testMlpyClusteringLearners():
    """ Test mlpy exposes appropriately wrapped clustering learners """
    variables = ["Y", "x1", "x2"]
    data = [[0, 1, 1], [0, 0, 1], [1, -300, 2]]
    trainingObj = UML.createData('Matrix', data, featureNames=variables, useLog=False)

    UML.trainAndApply("mlpy.MFastHCluster", trainingObj, output=None, arguments={'t': 1})
    UML.trainAndApply("mlpy.kmeans", trainingObj, output=None, arguments={'k': 2})

@mlpySkipDec
@noLogEntryExpected
def testMlpyListLearners():
    """ Test mlpy's listMlpyLearners() by checking the output for those learners we unit test """

    ret = UML.listLearners('mlpy')

    assert 'KPCA' in ret
    assert 'PCA' in ret
    assert 'KNN' in ret
    assert "LibLinear" in ret
    assert "LibSvm" in ret

    toExclude = []

    for name in ret:
        if name not in toExclude:
            params = UML.learnerParameters('mlpy.' + name)
            assert params is not None
            defaults = UML.learnerDefaultValues('mlpy.' + name)
            for pSet in params:
                for dSet in defaults:
                    for key in dSet.keys():
                        assert key in pSet

@mlpySkipDec
@logCountAssertionFactory(8)
def test_applier_acceptsNewArguments():
    """ Test an mlpy function that accept arguments for pred and transform """
    data = [[0, 1, 1], [0, 0, 1], [1, -300, 2]]
    dataObj = UML.createData('Matrix', data, useLog=False)

    # MFastHCluster.pred takes a 't' argument.
    expected = UML.trainAndApply("mlpy.MFastHCluster", dataObj, arguments={'t': 1})
    tl = UML.train('mlpy.MFastHCluster', dataObj, arguments={'t': 0})
    origArgs = tl.apply(dataObj)
    newArgs = tl.apply(dataObj, arguments={'t': 1})
    assert origArgs != newArgs
    assert newArgs == expected

    # PCA.transform takes a 'k' argument, by default it is None
    expected = UML.trainAndApply("mlpy.PCA", dataObj, k=1)
    tl = UML.train('mlpy.PCA', dataObj)
    origArgs = tl.apply(dataObj)
    newArgs = tl.apply(dataObj, k=1)
    assert origArgs != newArgs
    assert newArgs == expected

@mlpySkipDec
def test_applier_exception():
    """ Test an mlpy function with invalid arguments for pred and transform """
    data = [[0, 1, 1], [0, 0, 1], [1, -300, 2]]
    dataObj = UML.createData('Matrix', data)
    # MFastHCluster.pred does not take a 'foo' argument
    tl = UML.train('mlpy.MFastHCluster', dataObj, t=1)
    try:
        newArgs = tl.apply(dataObj, arguments={'foo': 1})
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

    # PCA.transform does not take a 'foo' argument
    tl = UML.train('mlpy.PCA', dataObj)
    try:
        newArgs = tl.apply(dataObj, foo=1)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

@mlpySkipDec
def test_getScores_exception():
    """ Test an mlpy function with invalid arguments for pred_values"""
    data = [[0, 1, 1], [0, 0, 1], [1, 3, 2], [2, -300, 2], [3, 1, 500]]
    trainingObj = UML.createData('Matrix', data)

    data2 = [[2, 3], [-200, 0]]
    testObj = UML.createData('Matrix', data2)

    # LibLinear.pred_values does not take a 'foo' argument.
    tl = UML.train('mlpy.LibLinear', trainingObj, 0)
    # in arguments parameter
    try:
        newArgs = tl.getScores(testObj, arguments={'foo': 1})
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass
    # as keyword argument
    try:
        newArgs = tl.getScores(testObj, foo=1)
        assert False # expected InvalidArgumentValue
    except InvalidArgumentValue:
        pass

