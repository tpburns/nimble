"""
Contains the MultiOutputLinearRegression custom learner class.
"""

from __future__ import absolute_import

import numpy
from six.moves import range
try:
    from sklearn.linear_model import LinearRegression
    imported = True
except ImportError:
    imported = False

import nimble
from nimble.customLearners import CustomLearner


class MultiOutputLinearRegression(CustomLearner):
    """
    Learner which trains a separate linear regerssion model on each of
    the features of the prediction data. The backend learner is provided
    by scikit-learn.
    """

    learnerType = 'regression'

    def train(self, trainX, trainY):
        self._models = []
        rawTrainX = trainX.copy(to='numpymatrix')

        for i in range(len(trainY.features)):
            currY = trainY.features.copy(i, useLog=False)
            rawCurrY = currY.copy(to='numpyarray', outputAs1D=True)

            currModel = LinearRegression()
            currModel.fit(rawTrainX, rawCurrY)
            self._models.append(currModel)

    def apply(self, testX):
        results = []
        rawTestX = testX.copy(to='numpymatrix')

        for i in range(len(self._models)):
            curr = self._models[i].predict(rawTestX)
            results.append(curr)

        results = numpy.matrix(results)
        results = results.transpose()

        return nimble.createData("Matrix", results, useLog=False)
