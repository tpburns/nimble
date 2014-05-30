"""
Unit tests for shogun_interface.py

"""

import numpy
import scipy.sparse
from numpy.random import rand, randint
from nose.tools import *

import UML

from UML.exceptions import ArgumentException

from UML.interfaces.shogun_interface_old import setShogunLocation
from UML.interfaces.shogun_interface_old import getShogunLocation


def testShogunLocation():
	""" Test setShogunLocation() """
	path = '/test/path/shogun'
	setShogunLocation(path)

	assert getShogunLocation() == path

@raises(ArgumentException)
def testShogun_shapemismatchException():
	""" Test shogun() raises exception when the shape of the train and test data don't match """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [1,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[3]]
	testObj = UML.createData('Matrix', data2)

	args = {}
	ret = UML.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


@raises(ArgumentException)
def testShogun_singleClassException():
	""" Test shogun() raises exception when the training data only has a single label """
	variables = ["Y","x1","x2"]
	data = [[-1,1,0], [-1,0,1], [-1,0,0]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[3,3]]
	testObj = UML.createData('Matrix', data2)

	args = {}
	ret = UML.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

@raises(ArgumentException)
def testShogun_multiClassDataToBinaryAlg():
	""" Test shogun() raises ArgumentException when passing multiclass data to a binary classifier """
	variables = ["Y","x1","x2"]
	data = [[5,-11,-5], [1,0,1], [2,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[5,3], [-1,0]]
	testObj = UML.createData('Matrix', data2)

	args = {'kernel':'GaussianKernel', 'width':2, 'size':10}
	ret = UML.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)


def testShogunHandmadeBinaryClassification():
	""" Test shogun() by calling a binary linear classifier """
	variables = ["Y","x1","x2"]
	data = [[0,1,0], [-0,0,1], [1,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[3,3], [-1,0]]
	testObj = UML.createData('Matrix', data2)

	args = {}
	ret = UML.trainAndApply("shogun.LibLinear", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunHandmadeBinaryClassificationWithKernel():
	""" Test shogun() by calling a binary linear classifier with a kernel """
	variables = ["Y","x1","x2"]
	data = [[5,-11,-5], [1,0,1], [1,3,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[5,3], [-1,0]]
	testObj = UML.createData('Matrix', data2)

	args = {'kernel':'GaussianKernel', 'width':2, 'size':10}
#	args = {}
	ret = UML.trainAndApply("shogun.LibSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	# shogun binary classifiers seem to return confidence values, not class ID
	assert ret.data[0,0] > 0

def testShogunKMeans():
	""" Test shogun() by calling the Kmeans classifier, a distance based machine """
	variables = ["Y","x1","x2"]
	data = [[0,0,0], [0,0,1], [1,8,1], [1,7,1], [2,1,9], [2,1,8]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[0,-10], [10,1], [1,10]]
	testObj = UML.createData('Matrix', data2)

	args = {'distance':'ManhattanMetric'}
	ret = UML.trainAndApply("shogun.KNN", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None
	print ret.data

	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunMulticlassSVM():
	""" Test shogun() by calling a multilass classifier with a kernel """
	variables = ["Y","x1","x2"]
	data = [[0,0,0], [0,0,1], [1,-118,1], [1,-117,1], [2,1,191], [2,1,118], [3,-1000,-500]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[0,0], [-101,1], [1,101], [1,1]]
	testObj = UML.createData('Matrix', data2)

	args = {'C':.5, 'kernel':'LinearKernel'}
#	args = {'C':1}
#	args = {}
	ret = UML.trainAndApply("shogun.GMNPSVM", trainingObj, trainY="Y", testX=testObj, output=None, arguments=args)

	assert ret is not None

	assert ret.data[0,0] == 0
	assert ret.data[1,0] == 1
	assert ret.data[2,0] == 2


def testShogunSparseRegression():
	""" Test shogun() sparse data instantiation by calling on a sparse regression learner with a large, but highly sparse, matrix """

	x = 100
	c = 10
	points = randint(0,x,c)
	cols = randint(0,x,c)
	data = rand(c)
	A = scipy.sparse.coo_matrix( (data, (points,cols)), shape=(x,x))
	obj = UML.createData('Sparse', A)

	labelsData = numpy.random.rand(x)
	labels = UML.createData('Matrix', labelsData.reshape((x,1)))

	ret = UML.trainAndApply('shogun.MulticlassOCAS', trainX=obj, trainY=labels, testX=obj)

	assert ret is not None


def testShogunRossData():
	""" Test shogun() by calling classifers using the problematic data from Ross """
	
	p0 = [1,  0,    0,    0,    0.21,  0.12]
	p1 = [2,  0,    0.56, 0.77, 0,     0]
	p2 = [1,  0.24, 0,    0,    0.12,  0]
	p3 = [1,  0,    0,    0,    0,     0.33]
	p4 = [2,  0.55, 0,    0.67, 0.98,  0]
	p5 = [1,  0,    0,    0,    0.21,  0.12]
	p6 = [2,  0,    0.56, 0.77, 0,     0]
	p7 = [1,  0.24, 0,    0,    0.12,  0]

	data = [p0,p1,p2,p3,p4,p5,p6,p7]

	trainingObj = UML.createData('Matrix', data)

	data2 = [[0, 0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98, 0]]
	testObj = UML.createData('Matrix', data2)

	args = {'C':1.0}
	argsk = {'C':1.0, 'kernel':"LinearKernel"}

	ret = UML.trainAndApply("shogun.MulticlassLibSVM", trainingObj, trainY=0, testX=testObj, output=None, arguments=argsk)
	assert ret is not None

	ret = UML.trainAndApply("shogun.MulticlassLibLinear", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
	assert ret is not None

	ret = UML.trainAndApply("shogun.LaRank", trainingObj, trainY=0, testX=testObj, output=None, arguments=argsk)
	assert ret is not None

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
	assert ret is not None


def testShogunEmbeddedRossData():
	""" Test shogun() by MulticlassOCAS with the ross data embedded in random data """
	
	p0 = [3,  0,    0,    0,    0.21,  0.12]
	p1 = [2,  0,    0.56, 0.77, 0,     0]
	p2 = [3,  0.24, 0,    0,    0.12,  0]
	p3 = [3,  0,    0,    0,    0,     0.33]
	p4 = [2,  0.55, 0,    0.67, 0.98,  0]
	p5 = [3,  0,    0,    0,    0.21,  0.12]
	p6 = [2,  0,    0.56, 0.77, 0,     0]
	p7 = [3,  0.24, 0,    0,    0.12,  0]

	data = [p0,p1,p2,p3,p4,p5,p6,p7]

	numpyData = numpy.zeros((50,10))

	for i in xrange(50):
		for j in xrange(10):
			if i < 8 and j < 6:
				numpyData[i,j] = data[i][j]
			else:
				if j == 0:
					numpyData[i,j] = numpy.random.randint(2,3)
				else:
					numpyData[i,j] = numpy.random.rand()

	trainingObj = UML.createData('Matrix', numpyData)

	data2 = [[0, 0, 0, 0, 0.33,0, 0, 0, 0.33], [0.55, 0, 0.67, 0.98,0.55, 0, 0.67, 0.98, 0]]
	testObj = UML.createData('Matrix', data2)

	args = {'C':1.0}

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY=0, testX=testObj, output=None, arguments=args)
	assert ret is not None

	for value in ret.data:
		assert value == 2 or value == 3


def testShogunScoreModeMulti():
	""" Test shogun() returns the right dimensions when given different scoreMode flags, multi case"""
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,3],[-200,0]]
	testObj = UML.createData('Matrix', data2)

	# default scoreMode is 'label'
	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert ret.pointCount == 2
	assert ret.featureCount == 2

	ret = UML.trainAndApply("shogun.MulticlassOCAS", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert ret.pointCount == 2
	assert ret.featureCount == 3


def testShogunScoreModeBinary():
	""" Test shogun() returns the right dimensions when given different scoreMode flags, binary case"""
	variables = ["Y","x1","x2"]
	data = [[-1,1,1], [-1,0,1], [1,30,2], [1,30,3]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,1],[25,0]]
	testObj = UML.createData('Matrix', data2)

	# default scoreMode is 'label'
	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={})
	assert ret.pointCount == 2
	assert ret.featureCount == 1

	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='bestScore')
	assert ret.pointCount == 2
	assert ret.featureCount == 2

	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={}, scoreMode='allScores')
	assert ret.pointCount == 2
	assert ret.featureCount == 2


# TODO def testShogunMultiClassStrategyMultiDataBinaryAlg():
def notRunnable():
	""" Test shogun() will correctly apply the provided strategies when given multiclass data and a binary learner """
	variables = ["Y","x1","x2"]
	data = [[0,1,1], [0,0,1], [1,3,2], [2,-300,2]]
	trainingObj = UML.createData('Matrix', data, featureNames=variables)

	data2 = [[2,3],[-200,0]]
	testObj = UML.createData('Matrix', data2)

	ret = UML.trainAndApply("shogun.SVMOcas", trainingObj, trainY="Y", testX=testObj, arguments={}, multiClassStrategy="OneVsOne")
	




def testShogunListLearners():
	""" Test shogun's listShogunLearners() by checking the output for those learners we unit test """

	ret = UML.listLearners('shogun')

	assert 'LibSVM' in ret
	assert 'LibLinear' in ret
	assert 'MulticlassLibSVM' in ret
	assert 'MulticlassOCAS' in ret


	from shogun.Features import RealFeatures
	from shogun.Features import BinaryLabels
	from shogun.Features import MulticlassLabels
	from shogun.Features import RegressionLabels
	import shogun.Kernel
	import shogun.Classifier

	for i in xrange(len(ret)):
		funcName = ret[i]

		toCall = getattr(shogun.Classifier, funcName)
		trialObj = toCall()
		probType = trialObj.get_machine_problem_type()
		if probType == 0:
			data = [[1,0], [0,1], [3,2]]
			labels = [[-1], [-1], [1]]
		else:
			data = [[0,0], [0,1], [-118,1], [-117,1], [1,191], [1,118], [-1000,-500]]
			labels = [[0],[0],[1],[1],[2],[2],[3]]

		data = numpy.array(data, dtype=numpy.float)
		data = data.transpose()
		labels = numpy.array(labels)
		labels = labels.transpose()
		labels = labels.flatten()
		labels = labels.astype(float)

		trainFeat = RealFeatures()
		trainFeat.set_feature_matrix(numpy.array(data, dtype=numpy.float))	
		if probType == 0:
			trainLabels = BinaryLabels(labels)
		elif probType == 1:
			trainLabels = RegressionLabels(labels)
		else:
			trainLabels = MulticlassLabels(labels)
		trialObj.set_labels(trainLabels)

		if hasattr(trialObj, 'set_kernel'):
			kern = shogun.Kernel.LinearKernel(trainFeat,trainFeat)
			trialObj.set_kernel(kern)
		if hasattr(trialObj, 'set_distance'):
			dist = shogun.Kernel.ManhattanMetric(trainFeat, trainFeat)
			trialObj.set_distance(dist)
		if hasattr(trialObj, 'set_C'):
			try:
				trialObj.set_C(1, 1)
			except TypeError:
				trialObj.set_C(1)

		trialObj.train(trainFeat)

