

from allowImports import boilerplate
boilerplate()


if __name__ == "__main__":
	import os.path
	import UML
	from UML import crossValidateReturnBest
	from UML import functionCombinations
	from UML import createData
	from UML import trainAndTest
	from UML.metrics import fractionIncorrect

	# path to input specified by command line argument
	pathIn = os.path.join(UML.UMLPath, "datasets/sparseSample.mtx")
	allData = createData('sparse', pathIn, fileType="mtx")

	print "data loaded"

	yData = allData.extractFeatures([5])
	xData = allData

	yData = yData.copyAs(format="Matrix")

	print "data formatted"

	# setup parameters we want to cross validate over, and the functions and metrics to evaluate
	toRun = 'trainAndTest("shogun.MulticlassOCAS", trainX, trainY, testX, testY, {"C":<1.0>}, [fractionIncorrect])'
	runs = functionCombinations(toRun)
	extraParams = {'trainAndTest':trainAndTest, 'fractionIncorrect':fractionIncorrect}

	print "runs prepared"

	bestFunction, performance = crossValidateReturnBest(xData, yData, runs, mode='min', numFolds=5, extraParams=extraParams, sendToLog=True)
	#print bestFunction
	#print performance
