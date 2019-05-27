"""
Tests for the UML.randUML submodule

"""

from __future__ import absolute_import
import UML
import random
import numpy
import nose

import UML.randomness
from UML.randomness import pythonRandom
from UML.randomness import numpyRandom
from six.moves import range
from .assertionHelpers import logCountAssertionFactory


@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def testSetRandomSeedExplicit():
    """ Test UML.setRandomSeed yields uml accessible random objects with the correct random behavior """
    expPy = random.Random(1333)
    expNp = numpy.random.RandomState(1333)
    UML.setRandomSeed(1333)

    for i in range(50):
        assert pythonRandom.random() == expPy.random()
        assert numpyRandom.rand() == expNp.rand()


@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
def testSetRandomSeedNone():
    """ Test UML.setRandomSeed operates as expected when passed None (-- use system time as seed) """
    UML.setRandomSeed(None)
    pyState = pythonRandom.getstate()
    npState = numpyRandom.get_state()

    origPy = random.Random()
    origPy.setstate(pyState)
    origNp = numpy.random.RandomState()
    origNp.set_state(npState)

    UML.setRandomSeed(None)

    assert origPy.random() != pythonRandom.random()
    assert origNp.rand() != numpyRandom.rand()


@nose.with_setup(UML.randomness.startAlternateControl, UML.randomness.endAlternateControl)
@logCountAssertionFactory(3)
def testSetRandomSeedPropagate():
    """ Test that UML.setRandomSeed will correctly control how randomized methods in UML perform """
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [17, 18, 19], [2, 2, 2], [3, 3, 3], [4, 4, 4],
            [5, 5, 5]]
    featureNames = ['1', '2', '3']
    toTest1 = UML.createData("List", data, featureNames=featureNames, useLog=False)
    toTest2 = toTest1.copy()
    toTest3 = toTest1.copy()

    UML.setRandomSeed(1337)
    toTest1.points.shuffle(useLog=False)

    UML.setRandomSeed(1336)
    toTest2.points.shuffle(useLog=False)

    UML.setRandomSeed(1337)
    toTest3.points.shuffle(useLog=False)

    assert toTest1 == toTest3
    assert toTest1 != toTest2
