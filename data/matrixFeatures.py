"""

"""
from __future__ import absolute_import

import numpy

from .axis import Axis
from .matrixAxis import MatrixAxis
from .features import Features

class MatrixFeatures(MatrixAxis, Axis, Features):
    """

    """
    def __init__(self, source, **kwds):
        self.source = source
        self.axis = 'feature'
        kwds['axis'] = self.axis
        kwds['source'] = self.source
        super(MatrixFeatures, self).__init__(**kwds)

    def _add_implementation(self, toAdd, insertBefore):
        """
        Insert the features from the toAdd object to the right of the
        provided index in this object, the remaining points from this
        object will continue to the right of the inserted points.
        """
        startData = self.source.data[:, :insertBefore]
        endData = self.source.data[:, insertBefore:]
        self.source.data = numpy.concatenate((startData, toAdd.data, endData),
                                             1)
