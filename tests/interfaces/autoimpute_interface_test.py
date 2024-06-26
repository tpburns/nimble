
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Tests for autoimpute interface.
"""
import functools

import numpy as np

import nimble
from nimble import match
from nimble import fill
from nimble.exceptions import InvalidArgumentValue
from nimble.calculate import rootMeanSquareError, fractionCorrect
from tests.helpers import raises
from tests.helpers import getDataConstructors
from tests.helpers import skipMissingPackage

autoimputeSkipDec = skipMissingPackage('autoimpute')

# fillMatching is inplace so no views
constructors = getDataConstructors(includeViews=False)

def getDataWithMissing(constructor, assignNames=True, yBinary=False):
    if isinstance(constructor, str):
        constructor = functools.partial(nimble.data, constructor)
    mu = np.array([5.0, 0.0])
    r = np.array([
            [  3.40, -2.75],
            [ -2.75,  5.50],])

    # Generate the random samples.
    num = 100
    d = np.random.multivariate_normal(mu, r, size=num)
    # insert missing values
    rm1 = np.random.random_sample(num) > 0.85
    d[rm1, 1] = np.nan
    if yBinary:
        d[:, 0] = np.random.choice(2, 100)

    if assignNames:
        data = constructor(d, featureNames=['y', 'x'])
    else:
        data = constructor(d)

    return data

def backend_imputation(learnerName, **kwargs):
    for constructor in constructors:
        data = getDataWithMissing(constructor)
        orig = data.copy()
        matches = data.matchingElements(match.missing)
        nimble.fillMatching(learnerName, matches, data, **kwargs)

        for dFt, mFt, oFt in zip(data.features, matches.features, orig.features):
            for i in range(len(dFt)):
                if mFt[i]:
                    assert dFt[i] == dFt[i] # not nan
                    assert dFt[i] != oFt[i]
                else:
                    assert dFt[i] == oFt[i]

@autoimputeSkipDec
def test_autoimpute_SingleImputer():
    backend_imputation('autoimpute.SingleImputer', strategy='least squares')
    # check also that the object itself is a valid input
    from autoimpute.imputations import SingleImputer
    backend_imputation(SingleImputer, strategy='least squares')

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_SingleImputer_exception_noStrategy():
    # this would work by default, testing that we override to require strategy argument
    backend_imputation('autoimpute.SingleImputer')

@autoimputeSkipDec
def test_autoimpute_MultipleImputer():
    backend_imputation('autoimpute.MultipleImputer', strategy={'x': 'mean', 'y': 'random'})

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MultipleImputer_exception_noStrategy():
    # this would work by default, testing that we override to require strategy argument
    backend_imputation('autoimpute.MultipleImputer')

@autoimputeSkipDec
def test_autoimpute_MiLinearRegression():
    for constructor in constructors:
        data = getDataWithMissing(constructor)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
        # test data cannot have missing values
        testX.features.fillMatching(fill.mean, match.missing)

        rmse = nimble.trainAndTest('autoimpute.MiLinearRegression', rootMeanSquareError,
                                   trainX, trainY, testX, testY,
                                   mi_kwgs={'n': 1, 'strategy': {'x': 'mean'}})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='mean')

        exp = nimble.trainAndTest('skl.LinearRegression', rootMeanSquareError,
                                  trainX, trainY, testX, testY)

        np.testing.assert_almost_equal(rmse, exp)

@autoimputeSkipDec
def test_autoimpute_MiLinearRegression_noNames():
    for constructor in constructors:
        data = getDataWithMissing(constructor, False)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels=0)
        testX.features.fillMatching(fill.mean, match.missing)

        rmse = nimble.trainAndTest('autoimpute.MiLinearRegression', rootMeanSquareError,
                                   trainX, trainY, testX, testY,
                                   mi_kwgs={'n': 1, 'strategy':'mode'})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='mode')

        exp = nimble.trainAndTest('skl.LinearRegression', rootMeanSquareError,
                                  trainX, trainY, testX, testY)

        np.testing.assert_almost_equal(rmse, exp)

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MiLinearRegression_exception_noStrategy():
    data = getDataWithMissing('Matrix')
    trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
    testX.features.fillMatching(fill.mean, match.missing)

    nimble.trainAndTest('autoimpute.MiLinearRegression', rootMeanSquareError,
                        trainX, trainY, testX, testY, mi_kwgs={'n': 1})

@autoimputeSkipDec
def test_autoimpute_MiLogisticRegression():
    for constructor in constructors:
        data = getDataWithMissing(constructor, yBinary=True)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
        # test data cannot have missing values
        testX.features.fillMatching(fill.mean, match.missing)
        fc = nimble.trainAndTest('autoimpute.MiLogisticRegression', fractionCorrect,
                                 trainX, trainY, testX, testY, model_lib='sklearn',
                                 mi_kwgs={'n': 1, 'strategy': {'x': 'mean'},
                                          'seed': 0})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='mean')

        exp = nimble.trainAndTest('skl.LogisticRegression', fractionCorrect,
                                  trainX, trainY, testX, testY, randomSeed=0)

        np.testing.assert_almost_equal(fc, exp)

@autoimputeSkipDec
def test_autoimpute_MiLogisticRegression_directMultipleImputer():
    for constructor in constructors:
        data = getDataWithMissing(constructor, yBinary=True)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
        # test data cannot have missing values
        testX.features.fillMatching(fill.mean, match.missing)
        trainArgs = ['autoimpute.MiLogisticRegression', fractionCorrect,
                     trainX, trainY, testX, testY]
        try:
            fc = nimble.trainAndTest(*trainArgs, model_lib='sklearn',
                                     mi=nimble.Init('MultipleImputer', n=1,
                                                strategy='interpolate'))
            imputer = 'autoimpute.MultipleImputer'
        # learners in version >=0.12 require MiceImputer
        except ValueError:
            fc = nimble.trainAndTest(*trainArgs, model_lib='sklearn',
                                     mi=nimble.Init('MiceImputer', n=1,
                                                strategy='interpolate'))
            imputer = 'autoimpute.MiceImputer'

        nimble.fillMatching(imputer, match.missing, trainX,
                            n=1, strategy='interpolate')

        exp = nimble.trainAndTest('skl.LogisticRegression', fractionCorrect,
                                  trainX, trainY, testX, testY)

        np.testing.assert_almost_equal(fc, exp)

@autoimputeSkipDec
def test_autoimpute_MiLogisticRegression_noNames():
    for constructor in constructors:
        data = getDataWithMissing(constructor, assignNames=False, yBinary=True)
        trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels=0)

        testX.features.fillMatching(fill.mean, match.missing)
        fc = nimble.trainAndTest('autoimpute.MiLogisticRegression', fractionCorrect,
                                 trainX, trainY, testX, testY, model_lib='sklearn',
                                 mi_kwgs={'n': 1, 'strategy': 'median'})

        nimble.fillMatching('autoimpute.SingleImputer', match.missing, trainX,
                            strategy='median')

        exp = nimble.trainAndTest('skl.LogisticRegression', fractionCorrect,
                                  trainX, trainY, testX, testY)

        np.testing.assert_almost_equal(fc, exp)

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MiLogisticRegression_exception_noStrategy():
    data = getDataWithMissing('Matrix', yBinary=True)
    trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')
    testX.features.fillMatching(fill.mean, match.missing)
    nimble.trainAndTest('autoimpute.MiLogisticRegression', fractionCorrect,
                        trainX, trainY, testX, testY)

@autoimputeSkipDec
@raises(InvalidArgumentValue)
def test_autoimpute_MiLogisticRegression_exception_directMultipleImputerNoStrategy():
    data = getDataWithMissing('Matrix', yBinary=True)
    trainX, trainY, testX, testY = data.trainAndTestSets(0.25, labels='y')

    testX.features.fillMatching(fill.mean, match.missing)
    nimble.trainAndTest('autoimpute.MiLogisticRegression', fractionCorrect,
                        trainX, trainY, testX, testY,
                        mi=nimble.Init('MultipleImputer', n=1))

@autoimputeSkipDec
def testLearnerTypes():
    learners = ['autoimpute.' + l for l in nimble.learnerNames('autoimpute')]
    allowed = ['classification', 'regression', 'transformation', 'UNKNOWN']
    # TODO MissingnessClassifier is UNKNOWN because getScores does not align
    # with expectation for classification
    assert all(lt in allowed for lt in nimble.learnerType(learners))
