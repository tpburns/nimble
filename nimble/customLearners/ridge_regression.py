"""
Contains the RidgeRegression custom learner class.
"""

from __future__ import absolute_import

import numpy

import nimble
from nimble.customLearners import CustomLearner


class RidgeRegression(CustomLearner):
    learnerType = 'regression'

    def train(self, trainX, trainY, lamb=0):
        self.lamb = lamb

        # setup for intercept term
        #		ones = nimble.createData("Matrix", numpy.ones(len(trainX.points)))
        #		ones.transpose()
        #		trainX = trainX.copy()
        #		trainX.features.add(ones)

        # trainX and trainY are input as points in rows, features in columns
        # in other words: Points x Features.
        # for X data, we want both Points x Features and Features x Points
        # for Y data, we only want Points x Features
        rawXPxF = trainX.copy(to="numpymatrix")
        rawXFxP = rawXPxF.transpose()
        rawYPxF = trainY.copy(to="numpymatrix")

        featureSpace = rawXFxP * rawXPxF
        lambdaMatrix = lamb * numpy.identity(len(trainX.features))
        #		lambdaMatrix[len(trainX.features)-1][len(trainX.features)-1] = 0

        inv = numpy.linalg.inv(featureSpace + lambdaMatrix)
        self.w = inv * rawXFxP * rawYPxF

    def apply(self, testX):
    # setup intercept
    #		ones = nimble.createData("Matrix", numpy.ones(len(testX.points)))
    #		ones.transpose()
    #		testX = testX.copy()
    #		testX.features.add(ones)

        # testX input as points in rows, features in columns
        rawXPxF = testX.copy(to="numpyarray")
        rawXFxP = rawXPxF.transpose()

        pred = numpy.dot(self.w.transpose(), rawXFxP)

        return nimble.createData("Matrix", pred.transpose(), useLog=False)
