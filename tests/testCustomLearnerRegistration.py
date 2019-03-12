from __future__ import absolute_import
import six.moves.configparser
import tempfile
import os
import numpy
import copy
from nose.tools import raises

import UML
from UML.exceptions import InvalidArgumentType, InvalidArgumentValue
from UML.customLearners import CustomLearner
from UML.customLearners.ridge_regression import RidgeRegression
from UML.configuration import configSafetyWrapper
from .logger.testLoggingCount import noLogEntryExpected


class LoveAtFirstSightClassifier(CustomLearner):
    """ Always predicts the value of the first class it sees in the most recently trained data """
    learnerType = 'classification'

    def incrementalTrain(self, trainX, trainY):
        if hasattr(self, 'scope'):
            self.scope = numpy.union1d(self.scope, trainY.copyAs('numpyarray').flatten())
        else:
            self.scope = numpy.unique(trainY.copyAs('numpyarray'))
        self.prediction = trainY[0, 0]

    def apply(self, testX):
        ret = []
        for point in testX.points:
            ret.append([self.prediction])
        return UML.createData("Matrix", ret)

    def getScores(self, testX):
        ret = []
        for point in testX.points:
            currScores = []
            for value in self.scope:
                if value == self.prediction:
                    currScores.append(1)
                else:
                    currScores.append(0)
            ret.append(currScores)
        return UML.createData("Matrix", ret)

    @classmethod
    def options(cls):
        return ['option']


class UncallableLearner(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, foo):
        return None

    def apply(self, testX):
        return None

    @classmethod
    def options(cls):
        return ['option']


@raises(InvalidArgumentValue)
@configSafetyWrapper
def testCustomPackageNameCollision():
    """ Test registerCustomLearner raises an exception when the given name collides with a real package """
    avail = UML.interfaces.available
    nonCustom = None
    for inter in avail:
        if not isinstance(inter, UML.interfaces.CustomLearnerInterface):
            nonCustom = inter
            break
    if nonCustom is not None:
        UML.registerCustomLearner(nonCustom.getCanonicalName(), LoveAtFirstSightClassifier)
    else:
        raise InvalidArgumentValue("Can't test, just pass")


@raises(InvalidArgumentValue)
@configSafetyWrapper
def testDeregisterBogusPackageName():
    """ Test deregisterCustomLearner raises an exception when passed a bogus customPackageName """
    UML.deregisterCustomLearner("BozoPackage", 'ClownClassifier')


@raises(InvalidArgumentType)
@configSafetyWrapper
def testDeregisterFromNonCustom():
    """ Test deregisterCustomLearner raises an exception when trying to deregister from a non-custom interface """
    UML.deregisterCustomLearner("Mlpy", 'KNN')


@raises(InvalidArgumentValue)
@configSafetyWrapper
def testDeregisterBogusLearnerName():
    """ Test deregisterCustomLearner raises an exception when passed a bogus learnerName """
    #save state before registration, to check against final state
    saved = copy.copy(UML.interfaces.available)
    try:
        UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
        UML.deregisterCustomLearner("Foo", 'BogusNameLearner')
        # should never get to here
        assert False
    finally:
        UML.deregisterCustomLearner("Foo", "LoveAtFirstSightClassifier")
        assert saved == UML.interfaces.available


@configSafetyWrapper
def testMultipleCustomPackages():
    """ Test registerCustomLearner correctly instantiates multiple custom packages """

    #save state before registration, to check against final state
    saved = copy.copy(UML.interfaces.available)

    UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
    UML.registerCustomLearner("Bar", RidgeRegression)
    UML.registerCustomLearner("Baz", UncallableLearner)

    assert UML.listLearners("Foo") == ["LoveAtFirstSightClassifier"]
    assert UML.listLearners("Bar") == ["RidgeRegression"]
    assert UML.listLearners("Baz") == ["UncallableLearner"]

    assert UML.learnerParameters("Foo.LoveAtFirstSightClassifier") == [[]]
    assert UML.learnerParameters("Bar.RidgeRegression") == [['lamb']]
    assert UML.learnerParameters("Baz.UncallableLearner") == [['foo']]

    UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
    UML.deregisterCustomLearner("Bar", 'RidgeRegression')
    UML.deregisterCustomLearner("Baz", 'UncallableLearner')

    assert saved == UML.interfaces.available


@configSafetyWrapper
def testMultipleLearnersSinglePackage():
    """ Test multiple learners grouped in the same custom package """
    #TODO test is not isolated from above.

    #save state before registration, to check against final state
    saved = copy.copy(UML.interfaces.available)

    UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
    UML.registerCustomLearner("Foo", RidgeRegression)
    UML.registerCustomLearner("Foo", UncallableLearner)

    learners = UML.listLearners("Foo")
    assert len(learners) == 3
    assert "LoveAtFirstSightClassifier" in learners
    assert "RidgeRegression" in learners
    assert "UncallableLearner" in learners

    assert UML.learnerParameters("Foo.LoveAtFirstSightClassifier") == [[]]
    assert UML.learnerParameters("Foo.RidgeRegression") == [['lamb']]
    assert UML.learnerParameters("Foo.UncallableLearner") == [['foo']]

    UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')

    learners = UML.listLearners("Foo")
    assert len(learners) == 2
    assert "RidgeRegression" in learners
    assert "UncallableLearner" in learners

    UML.deregisterCustomLearner("Foo", 'RidgeRegression')
    UML.deregisterCustomLearner("Foo", 'UncallableLearner')

    assert saved == UML.interfaces.available


# test that register and deregister learner records the
# registartion to UML.settings / the config file
@configSafetyWrapper
def testEffectToSettings():
    """ Test that (de)registering is correctly recorded by UML.settings """
    UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
    regSec = 'RegisteredLearners'
    opName = 'Foo.LoveAtFirstSightClassifier'
    opValue = 'UML.tests.testCustomLearnerRegistration.'
    opValue += 'LoveAtFirstSightClassifier'

    # check that it is listed immediately
    assert UML.settings.get(regSec, opName) == opValue
    # check that it is recorded to file
    tempSettings = UML.configuration.loadSettings()
    assert tempSettings.get(regSec, opName) == opValue

    UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')

    # check that deregistration removed the entry in the
    # RegisteredLearners section
    try:
        UML.settings.get(regSec, opName) == opValue
        assert False
    except six.moves.configparser.NoOptionError:
        pass

# test that registering a sample custom learner with option names
# will affect UML.settings and the config file
@configSafetyWrapper
def testRegisterLearnerWithOptionNames():
    """ Test the availability of a custom learner's options """
    UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
    learnerName = 'LoveAtFirstSightClassifier'

    assert UML.settings.get('Foo', learnerName + '.option') == ""

    UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')


# test what happens when you deregister a custom learner that has
# in memory changes to its options in UML.settings
@configSafetyWrapper
def testDeregisterWithSettingsChanges():
    """ Safety check: unsaved options discarded after deregistration """
    UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
    UML.registerCustomLearner("Foo", UncallableLearner)

    loveName = 'LoveAtFirstSightClassifier'
    uncallName = 'UncallableLearner'
    UML.settings.set('Foo', loveName + '.option', "hello")
    UML.settings.set('Foo', uncallName + '.option', "goodbye")
    assert UML.settings.get('Foo', loveName + '.option') == "hello"

    UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')

    # should no longer have options for that learner in foo
    try:
        UML.settings.get('Foo', loveName + '.option')
        assert False
    except six.moves.configparser.NoOptionError:
        pass

    UML.deregisterCustomLearner("Foo", 'UncallableLearner')

    # should no longer have a section for foo
    try:
        UML.settings.get('Foo', loveName + '.option')
        assert False
    except six.moves.configparser.NoSectionError:
        pass


@configSafetyWrapper
def test_settings_autoRegister():
    """ Test that the auto registration helper correctly registers learners """
    # establish that this custom package does not currently exist
    try:
        fooInt = UML.helpers.findBestInterface('Foo')
        assert False
    except InvalidArgumentValue:
        pass

    # make changes to UML.settings
    name = 'Foo.LoveAtFirstSightClassifier'
    value = LoveAtFirstSightClassifier.__module__ + '.' + 'LoveAtFirstSightClassifier'
    UML.settings.set("RegisteredLearners", name, value)

    # call helper
    UML.helpers.autoRegisterFromSettings()

    # check that correct learners are available through UML.interfaces.available
    assert 'LoveAtFirstSightClassifier' in UML.listLearners('Foo')


@configSafetyWrapper
def test_settings_autoRegister_safety():
    """ Test that the auto registration will not crash on strange inputs """

    name = "Foo.Bar"
    value = "Bogus.Attribute"
    UML.settings.set("RegisteredLearners", name, value)

    # should throw warning, not exception
    UML.helpers.autoRegisterFromSettings()

@configSafetyWrapper
@noLogEntryExpected
def test_logCount():
    UML.registerCustomLearner("Foo", LoveAtFirstSightClassifier)
    lst = UML.listLearners("Foo")
    params = UML.learnerParameters("Foo.LoveAtFirstSightClassifier")
    defaults = UML.learnerDefaultValues("Foo.LoveAtFirstSightClassifier")
    lType = UML.learnerType("Foo.LoveAtFirstSightClassifier")
    UML.deregisterCustomLearner("Foo", 'LoveAtFirstSightClassifier')
