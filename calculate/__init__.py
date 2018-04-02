"""
This loosely groups together functions which perform calculations on data
objects and other UML defined objects, including functions which can be
used as performance functions in the UML testing and cross validation API.
Some of these are also availale as methods off of data objects; the
versions here are functions, and take any inputs as arguments.

"""

from __future__ import absolute_import
from .confidence import confidenceIntervalHelper

from .loss import meanSquareLoss
from .loss import rootMeanSquareLoss
from .loss import sumSquareLoss
from .loss import crossEntropyLoss
from .loss import exponentialLoss
from .loss import quadraticLoss
from .loss import hingeLoss
from .loss import fractionIncorrectLoss
from .loss import l2Loss
from .loss import l1Loss
from .loss import meanAbsoluteError
from .loss import meanFeaturewiseRootMeanSquareError
from .loss import varianceFractionRemaining
from .matrix import elementwiseMultiply
from .matrix import elementwisePower
from .similarity import correlation
from .similarity import cosineSimilarity
from .similarity import covariance
from .similarity import fractionCorrect
from .similarity import rSquared
from .statistic import maximum
from .statistic import mean
from .statistic import median
from .statistic import mode
from .statistic import minimum
from .statistic import uniqueCount
from .statistic import proportionMissing
from .statistic import proportionZero
from .statistic import quartiles
from .statistic import residuals
from .statistic import standardDeviation
from .utility import detectBestResult

__all__ = ['confidenceIntervalHelper', 'meanSquareLoss', 'rootMeanSquareLoss',
           'sumSquareLoss', 'crossEntropyLoss', 'exponentialLoss',
           'quadraticLoss', 'hingeLoss', 'fractionIncorrectLoss', 'l2Loss',
           'l1Loss', 'correlation', 'cosineSimilarity',
           'covariance', 'detectBestResult', 'elementwiseMultiply',
           'elementwisePower', 'fractionCorrect',
           'maximum', 'mean', 'meanAbsoluteError',
           'meanFeaturewiseRootMeanSquareError', 'median', 'mode', 'minimum',
           'proportionMissing', 'proportionZero', 'quartiles',
           'residuals', 'rSquared', 'standardDeviation',
           'uniqueCount', 'varianceFractionRemaining']
