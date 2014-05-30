"""
Unit tests for regressors_interface.py

"""
import UML

from UML.interfaces.regressors_interface_old import setRegressorLocation
from UML.interfaces.regressors_interface_old import getRegressorLocation
from UML.data import Matrix
import tempfile


def _testRegressorsHandmade():
	""" Test regressor() by calling a regressor with known output using files """
	setRegressorLocation("/home/tpburns/Dropbox/Regressors")

	trainingIn = tempfile.NamedTemporaryFile(suffix=".csv")
	trialIn = tempfile.NamedTemporaryFile(suffix=".csv")
	actualOut = tempfile.NamedTemporaryFile(suffix=".csv")

	trainingIn.write("3,107,4.0\n")
	trainingIn.write("5,101,4.0\n")
	trainingIn.write("5,104,4.0\n")
	trainingIn.write("5,106,4.0\n")
	trainingIn.flush()

	trialIn.write("5,107,4.0\n")
	trialIn.flush()

	UML.trainAndApply("regressor.ConstantRegressor",trainingIn.name, trainY=2, testX=trialIn.name, output=actualOut.name, arguments={'randomObject':None})

	actualOut.seek(0)
	line = actualOut.readline()
	print line
	assert line.strip() == "4.0"


def _testRegressorsHandmadeTrainMatrix():
	""" Test regressor() against handmade output with a Matrix object as training input """

	setRegressorLocation("/home/tpburns/Dropbox/Regressors")
	learnerName = 'LinearRegressor'

	trialIn = tempfile.NamedTemporaryFile()
	actualOut = tempfile.NamedTemporaryFile()

	trialIn.write("3,3\n")
	trialIn.flush()

	data = [[1,1,2],[2,2,4]]
	training = UML.createData('Matrix', data)

	UML.trainAndApply("regressor." + learnerName, training, trainY=2, testX=trialIn.name, output=actualOut.name,  arguments={})

	actualOut.seek(0)
	line = actualOut.readline()
	line = line.strip()
	print line
	assert line != ''
	assert line == "6.0"


def _testRegressorsLocation():
	""" Test setRegressorLocation() """

	path = '/test/path'
	setRegressorLocation(path)

	assert getRegressorLocation() == path


def _testRegressorsPresent():
	""" Test regressorsPresent() will return false for obviously wrong path values """

	# default is none - should be false
	setRegressorLocation(None)
	assert not regressorsPresent()

	# pathes which are not directories - should be false
	setRegressorLocation('')
	assert not regressorsPresent()
	setRegressorLocation('/bin/bash')
	assert not regressorsPresent()


def _testFindRegressorClassName():
	""" Test fineRegressorClassName() against simple inputs """

	emptyFile = tempfile.NamedTemporaryFile()
	assert findRegressorClassName(emptyFile.name) is None

	rubbishFile = tempfile.NamedTemporaryFile()
	rubbishFile.write("# default is none - should be false")
	rubbishFile.write("setRegressorLocation(None)")
	rubbishFile.write("assert not regressorsPresent()")
	rubbishFile.write("\t# pathes which are not directories - should be false")
	rubbishFile.write("\tsetRegressorLocation('')")
	rubbishFile.write("\tassert not regressorsPresent()")
	rubbishFile.flush()
	assert findRegressorClassName(rubbishFile.name) is None

	beginsWithFile = tempfile.NamedTemporaryFile()
	beginsWithFile.write("import numpy, copy, bisect, random, math")
	beginsWithFile.write("import inspect")
	beginsWithFile.write("\n")
	beginsWithFile.write("class Regressor():")
	beginsWithFile.flush()
	assert findRegressorClassName(beginsWithFile.name) is None

	endsWithFile = tempfile.NamedTemporaryFile()
	endsWithFile.write("def regressorFunction(Regressor):")
	endsWithFile.flush()
	assert findRegressorClassName(endsWithFile.name) is None

	validFile = tempfile.NamedTemporaryFile()
	validFile.write("#Copyright Gimbel Technologies, LLC, 2008 All Rights Reserved.")
	validFile.write("\n")
	validFile.write("from Regressor import Regressor")
	validFile.write("import numpy")
	validFile.write("\n")
	validFile.write("class ConstantRegressor(asdfRegressor):")
	validFile.write("\n\"A regressor that always returns the same value for its predictions\"")
	validFile.flush()
	result = findRegressorClassName(validFile.name)
	assert result == "ConstantRegressor"


def _testRegressorsListLearners():
	""" Test Regressors's listRegressorLearners() by checking the output for those learners we unit test """
	
	setRegressorLocation('/home/tpburns/Dropbox/Regressors')
	
	ret = UML.listLearners('regressors')
	assert 'ConstantRegressor' in ret
	assert 'LinearRegressor' in ret



