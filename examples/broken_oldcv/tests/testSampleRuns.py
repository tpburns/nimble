from UML.examples.allowImports import boilerplate
boilerplate()
from UML import UMLPath
from UML import trainAndApply
from UML import normalizeData
from UML import createData
from UML import crossValidateReturnBest
from UML import crossValidate
from UML import splitData
from UML import functionCombinations
from UML import trainAndTest
from UML.metrics import fractionIncorrect

import os
exampleDirPath = UMLPath + "/datasets/"

def testEverythingVolumeOne():
	"""
	Try to test some full use cases: load data, split data, normalize data, run crossValidate or
	crossValidateReturnBest, and get results.  Use the classic iris data set for classification.
	"""
	pathOrig = os.path.join(os.path.dirname(__file__), "../../datasets/iris.csv")

	# we specify that we want a Matrix object returned, and with just the path it will
	# decide automatically the format of the file that is being loaded
	processed = createData("Matrix", pathOrig)

	assert processed.data is not None

	partOne = processed.extractPointsByCoinToss(0.5)
	partOneTest = partOne.extractPointsByCoinToss(0.1)
	partTwoX = processed
	partTwoY = processed.extractFeatures('Type')

	assert partOne.pointCount > 55
	assert partOne.pointCount < 80
	assert partTwoX.pointCount > 65
	assert partTwoX.pointCount < 85
	assert partTwoY.pointCount == partTwoX.pointCount
	assert partOne.pointCount + partTwoX.pointCount + partOneTest.pointCount == 150

	trainX = partOne
	trainY = partOne.extractFeatures('Type')
	testX = partOneTest
	testY = partOneTest.extractFeatures('Type')
	

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRunOne = 'trainAndTest("mlpy.LibSvm", trainX, trainY, testX, testY, {"C":<.01|.1|.1|10|100>,"gamma":<.01|.1|.1|10|100>,"kernel_type":"<rbf|sigmoid>"}, [fractionIncorrect])'
	runsOne = functionCombinations(toRunOne)
	extraParams = {'trainAndTest':trainAndTest, 'fractionIncorrect':fractionIncorrect}
	fullCrossValidateResults = crossValidate(trainX, trainY, runsOne, numFolds=10, extraParams=extraParams, sendToLog=False)
	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runsOne, mode='min', numFolds=10, extraParams=extraParams, sendToLog=False)

	#Check that the error rate for each function is between 0 and 1
	for result in fullCrossValidateResults.items():
		assert result[1] >= 0.0
		assert result[1] <= 1.0
	assert bestFunction is not None
	assert performance >= 0.0
	assert performance <= 1.0

	trainObj = trainX
	testObj = partTwoX

	# use normalizeData to modify our data; we call a dimentionality reduction algorithm to
	# simply our mostly redundant points. k is the desired number of dimensions in the output
	normalizeData('mlpy.PCA', trainObj, testX=testObj, arguments={'k':1})

	# assert that we actually do have fewer dimensions
	assert trainObj.data[0].size == 1
	assert testObj.data[0].size == 1

def testDataPrepExample():
	"""
		Functional test for data preparation
	"""

	# string manipulation to get and make paths
	pathOrig = os.path.join(os.path.dirname(__file__), "../../datasets/adult_income_classification_tiny.csv")
	pathOut = os.path.join(os.path.dirname(__file__), "../../datasets/adult_income_classification_tiny_numerical.csv")

	# we specify that we want a Matrix object returned, and with just the path it will
	# decide automatically the format of the file that is being loaded
	processed = createData("List", pathOrig)

	# this feature is a precalculated similarity rating. Lets not make it too easy....
	processed.extractFeatures('fnlwgt')

	#convert assorted features from strings to binary category columns
	processed.replaceFeatureWithBinaryFeatures('sex')
	processed.replaceFeatureWithBinaryFeatures('marital-status')
	processed.replaceFeatureWithBinaryFeatures('occupation')
	processed.replaceFeatureWithBinaryFeatures('relationship')
	processed.replaceFeatureWithBinaryFeatures('race')
	processed.replaceFeatureWithBinaryFeatures('native-country')

	# convert 'income' column (the classification label) to a single numerical column
	processed.transformFeatureToIntegerFeature('income')

	#scrub the rest of the string valued data -- the ones we converted are the non-redundent ones
	processed.dropFeaturesContainingType(basestring)

	# output the split and normalized sets for later usage
	processed.writeFile(pathOut, includeFeatureNames=True)

def testCrossValidateExample():
	"""
		Functional test for load-data-to-classification-results example of crossvalidation
	"""
	# path to input specified by command line argument
	pathIn = os.path.join(os.path.dirname(__file__), "../../datasets/adult_income_classification_tiny_numerical.csv")
	allData = createData("Matrix", pathIn, fileType="csv")
	trainX, trainY, testX, testY = splitData(allData, labelID='income', fractionForTestSet=.15)

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'trainAndTest("mlpy.LibSvm", trainX, trainY, testX, testY, {"C":<.01|.1|.1|10|100>,"gamma":<.01|.1|.1|10|100>,"kernel_type":"<rbf|sigmoid>"}, [fractionIncorrect])'
	runs = functionCombinations(toRun)
	extraParams = {'trainAndTest':trainAndTest, 'fractionIncorrect':fractionIncorrect}

	bestFunction, performance = crossValidateReturnBest(trainX, trainY, runs, mode='min', numFolds=10, extraParams=extraParams)
	assert bestFunction is not None
	assert performance > 0.0

def testNormalizing():
	"""
		Functional test of data normalization
	"""
	# we separate into classes accoring to whether x1 is positive or negative
	variables = ["y","x1","x2","x3"]
	data1 = [[1,6,0,0], [1,3,0,0], [0,-5,0,0],[0,-3,0,0]]
	trainObj = createData('Matrix', data1, variables)
	trainObjY = trainObj.extractFeatures('y')

	# data we're going to classify
	variables2 = ["x1","x2","x3"]
	data2 = [[1,0,0],[4,0,0],[-1,0,0], [-2,0,0]]
	testObj = createData('Matrix', data2, variables2)

	# baseline check
	assert trainObj.data[0].size == 3
	assert testObj.data[0].size == 3

	# use normalizeData to modify our data; we call a dimentionality reduction algorithm to
	# simply our mostly redundant points. k is the desired number of dimensions in the output
	normalizeData('mlpy.PCA', trainObj, testX=testObj, arguments={'k':1})

	# assert that we actually do have fewer dimensions
	assert trainObj.data[0].size == 1
	assert testObj.data[0].size == 1

	ret = trainAndApply('mlpy.KNN', trainObj, trainObjY, testObj, arguments={'k':1})

	# assert we get the correct classes
	assert ret.data[0,0] == 1
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 0
	assert ret.data[3,0] == 0

