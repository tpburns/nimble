"""
An example script defining a custom learner and running it on the
data given via a path on the command line.

"""

from allowImports import boilerplate

boilerplate()

import sys
import numpy
import itertools
import random

import UML
from UML.customLearners import CustomLearner
from UML.helpers import generateClassificationData
from UML.calculate import fractionIncorrect
from UML.exceptions import ArgumentException

class LogisticRegressionSelectByRegularization(CustomLearner):
    learnerType = "classification"

    def train(
            self, trainX, trainY, desiredNonZero, verboseStandardOut=False,
            allowSubLogging=False):
        if desiredNonZero > trainX.featureCount:
            desiredNonZero = trainX.featureCount

        def countNonZero(tl):
            coefs = tl.getAttributes()['coef_']
            return numpy.count_nonzero(coefs)

        # Define those arguments that will be fixed over all calls
        arguments = {}
        arguments['penalty'] = "l1"

        inC = 1
        sklLogReg = "scikitlearn.LogisticRegression"
        trained = UML.train(sklLogReg, trainX, trainY, C=inC, useLog=allowSubLogging, **arguments)

        # the first entry will be our downward multiplicative adjustment,
        # the second entry will be our upward adjustment
        adjustment = [3./4, 5./4]
        iteration = 1
        numNZ = countNonZero(trained)
        
        # we stop when the number of non-zero coefficients is precisely
        # what we want it to be
        while numNZ != desiredNonZero:
            # determine which adjustment is appropriate
            if numNZ < desiredNonZero:
                multiplier = adjustment[1]
            else:  # numNZ > desiredNonZero
                multiplier = adjustment[0]
            inC = inC * multiplier
            
            trained = UML.train(sklLogReg, trainX, trainY, C=inC,  useLog=allowSubLogging, **arguments)
            numNZ = countNonZero(trained)
            iteration += 1

            if verboseStandardOut:
                print '\n# iteration #: ' + str(iteration)
                print 'multiplier: ' + str(multiplier)
                print 'inC: ' + str(inC)
                print 'nz: '+str(numpy.count_nonzero(trained.getAttributes()['coef_']))

        self._trained = trained
        # expose everything from the underlying learner
        toExpose = trained.getAttributes()
        for key in toExpose:
            setattr(self, key, toExpose[key])

        coefs = trained.getAttributes()['coef_']
        self.wantedIndices = list(numpy.nonzero(coefs)[1])

    def apply(self, testX):
        return self._trained.apply(testX, useLog=False)

class LogisticRegressionSelectByRegularization(CustomLearner):
    learnerType = "classification"

    def train(
            self, trainX, trainY, desiredNonZero, verboseStandardOut=False,
            allowSubLogging=False):
        if desiredNonZero > trainX.featureCount:
            desiredNonZero = trainX.featureCount

        def countNonZero(tl):
            coefs = tl.getAttributes()['coef_']
            return numpy.count_nonzero(coefs)

        # Define those arguments that will be fixed over all calls
        arguments = {}
        arguments['penalty'] = "l1"

        inC = 1
        sklLogReg = "scikitlearn.LogisticRegression"
        trained = UML.train(sklLogReg, trainX, trainY, C=inC, useLog=allowSubLogging, **arguments)

        # the first entry will be our downward multiplicative adjustment,
        # the second entry will be our upward adjustment
        adjustment = [3. / 4, 5. / 4]
        iteration = 1
        numNZ = countNonZero(trained)

        # we stop when the number of non-zero coefficients is precisely
        # what we want it to be
        while numNZ != desiredNonZero:
            # determine which adjustment is appropriate
            if numNZ < desiredNonZero:
                multiplier = adjustment[1]
            else:  # numNZ > desiredNonZero
                multiplier = adjustment[0]
            inC = inC * multiplier

            trained = UML.train(sklLogReg, trainX, trainY, C=inC, useLog=allowSubLogging, **arguments)
            numNZ = countNonZero(trained)
            iteration += 1

            if verboseStandardOut:
                print '\n# iteration #: ' + str(iteration)
                print 'multiplier: ' + str(multiplier)
                print 'inC: ' + str(inC)
                print 'nz: ' + str(numpy.count_nonzero(trained.getAttributes()['coef_']))

        self._trained = trained
        # expose everything from the underlying learner
        toExpose = trained.getAttributes()
        for key in toExpose:
            setattr(self, key, toExpose[key])

    def apply(self, testX):
        return self._trained.apply(testX)


class LogisticRegressionSelectByOmission(CustomLearner):
    learnerType = "classification"

    def train(self, trainX, trainY, numberToOmit, method, C, **kwargs):
        accepted = ["least magnitude", "highest magnitude", "least value",
                "highest value"]

        kwargs['C'] = C
        kwargs['penalty'] = "l1"

        side = method.split(" ")[0]
        ordering = method.split(" ")[1]

        if method not in accepted:
            pretty = UML.exceptions.prettyListString(accepted)
            raise ArgumentException("method must be in: " + pretty)

        # do NOT log the sub calls
        kwargs['useLog'] = False

        sklLogReg = "scikitlearn.LogisticRegression"
        trained = UML.train(sklLogReg, trainX, trainY, **kwargs)

        self.origCoefs = trained.getAttributes()['coef_'].flatten()
        coefs = self.origCoefs

        if ordering == 'magnitude':
            coefs = map(abs, coefs)

        withIndices = zip(coefs, range(len(coefs)))
        withIndices.sort(key=lambda x:x[0])

        # retrain without those features????
        removalIndices = [withIndices[n][1] for n in range(numberToOmit)]
        self.wantedIndices = list(set(xrange(trainX.featureCount)) - set(removalIndices))

        inTrainX = trainX.copyFeatures(self.wantedIndices)
        self._trained = UML.train(sklLogReg, inTrainX, trainY, **kwargs)


    def apply(self, testX):
        inTestX = testX.copyFeatures(self.wantedIndices)
        return self._trained.apply(inTestX, useLog=False)



class ReducedRidge(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, alpha, wantedIndices):
        name = 'scikitlearn.RidgeClassifier'
        self.wantedIndices = wantedIndices
        redTrainX = trainX.copyFeatures(wantedIndices)
        self.tl = UML.train(name, redTrainX, trainY, alpha, useLog=False)

    def apply(self, testX):
        redTestX = testX.copyFeatures(self.wantedIndices)
        return self.tl.apply(redTestX, useLog=False)


class ReducedLogisticRegression(CustomLearner):
    learnerType = 'classification'

    def train(self, trainX, trainY, C, wantedIndices):
        name = 'scikitlearn.LogisticRegression'
        self.wantedIndices = wantedIndices
        redTrainX = trainX.copyFeatures(wantedIndices)
        self.tl = UML.train(name, redTrainX, trainY, C, useLog=False)

    def apply(self, testX):
        redTestX = testX.copyFeatures(self.wantedIndices)
        return self.tl.apply(redTestX, useLog=False)


def sanityCheck(trainX, totalScores):
    assert trainX.featureCount == 84

    for name in trainX.getFeatureNames():
        assert name[-3:] == "(M)" or name[-3:] == "(F)"

    # gotta fix data's __getitem__ so we can just pass python's sum function
    def summer(point):
        total = 0
        for value in point:
            total += value
        return total

    summed = trainX.calculateForEachPoint(summer)
    summed.setFeatureName(0, "totalScorePosOrNeg")
    assert summed == totalScores

def seperateData(dataAll, omitList):
    # grab the features we want to be in the training data; including raw
    # data that we may later choose to omit
    nameOfFirst = "I do not enjoy watching dance performances. (M)"
    indexOfFirst = dataAll.getFeatureIndex(nameOfFirst)

    usedData = dataAll.extractFeatures(start=indexOfFirst, end=None)
    usedData.appendFeatures(dataAll.copyFeatures("isMale"))
    usedData.appendFeatures(dataAll.copyFeatures("InTestSet"))

    usedData.extractFeatures(omitList)

    def selectInTestSet(point):
        if point["InTestSet"] > 0:
            return True
        return False

    testingSet = usedData.extractPoints(selectInTestSet)
    testY = testingSet.extractFeatures("isMale")
    testingSet.extractFeatures("InTestSet")
    testX = testingSet

    trainY = usedData.extractFeatures("isMale")
    usedData.extractFeatures("InTestSet")
    trainX = usedData

    return trainX, trainY, testX, testY

def printCoefficientsHR(trainedLearner):
    coefs = trainedLearner.getAttributes()["coef_"]
    coefs = coefs.flatten()
    chosen = []
    chosenCoefs = []
    for i in xrange(len(coefs)):
        value = coefs[i]
        if value != 0:
            chosen.append(trainX.getFeatureName(i))
            chosenCoefs.append(coefs[i])

    # display those questions which were the most useful
    print "\n"
    i = 1
    for question, coef in zip(chosen, chosenCoefs):
#       print str(i).ljust(3) + "\t" + str(round(coef,3)).ljust(8) + question.strip()
        print str(i).ljust(3) + "\t" + str(coef).ljust(20) + "\t" + question.strip()
        i = i + 1

def printCoefficientsPythonLists(trainedLearner, trainX, randomize=False):
    coefsFromLearner = trainedLearner.getAttributes()["coef_"]
    coefsFromLearner = coefsFromLearner.flatten()
    chosen = []
    coefs = []
    paired = []
    for i in xrange(len(coefsFromLearner)):
        name = trainX.getFeatureName(i).strip()
        paired.append((name, coefsFromLearner[i]))

    if randomize:
        random.shuffle(paired)

    for (n, c) in paired:
        chosen.append(n)
        coefs.append(c)

    # display those questions which were the most useful
    print ""
    print chosen
    print len(chosen)
    print ""
    print coefs
    print len(coefs)

def saveCoefficients(trainedLearner, names, outPath):
    coefs = trainedLearner.getAttributes()["coef_"]
    coefsObj = UML.createData("List", coefs, featureNames=names, pointNames=['coefs'])
    coefsObj.transpose()

    coefsObj.writeFile(outPath, 'csv', includeNames=True)



def standardizeScoreScale(obj):
    allNames = obj.getFeatureNames()
    negScored = []
    for i, name in enumerate(allNames):
        assert name[-1] == ")" and name[-3] == "("
        assert name[-2] == 'F' or name[-2] == 'M'
        if name[-2] == 'F':
            negScored.append(i)

    # confirm scoring range assumptions
    for f in xrange(obj.featureCount):
        for p in xrange(obj.pointCount):
            fname = obj.getFeatureName(f)
            if fname[-2] == 'M':
                assert obj[p,f] >= 0 and obj[p, f] <= 4
            else:
                assert obj[p,f] <= 0 and obj[p, f] >= -4

    def reverseScorePolarity(feature):
        ret = []
        for elem in feature:
            if elem == 0:
                ret.append(elem)
            else:
                ret.append(-elem)
        return ret

#   reduced = obj.copyPoints(end=20)
#   reduced = reduced.copyFeatures([0,1,2,58,59,60])
#   print reduced.getFeatureNames()
#   reduced.show('Before')

    obj.transformEachFeature(reverseScorePolarity, features=negScored)

#   reduced = obj.copyPoints(end=20)
#   reduced = reduced.copyFeatures([0,1,2,58,59,60])
#   reduced.show('After')

#   exit(0)
    return


def printAccuracy(trainedLearner, trainX, trainY, testX, testY):
#   print "\n\n"
    errorOutSample = trainedLearner.test(testX, testY, performanceFunction=fractionIncorrect)
    print "Out of sample error rate: " + str(round(errorOutSample*100,1)) + "%"
    errorInSample = trainedLearner.test(trainX, trainY, performanceFunction=fractionIncorrect, useLog=False)
    print "In sample error rate: " + str(round(errorInSample*100,1)) + "%"
    print "\n"



##############################
### FULL TRIAL CHOICE CODE ###
##############################



def SVC_full_data_outSample_only(trainX, trainY, testX, testY):
#   Cs = tuple([4**k for k in xrange(-8,8)])
#   bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, C=0.3) #19.7% out of sample error
#   bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, kernel="poly", degree=2, coef0=1, C=0.01) #19.2%
    bestError = UML.trainAndTest("scikitlearn.SVC", trainX, trainY, testX, testY, performanceFunction=fractionIncorrect, kernel="poly", degree=3, coef0=1, C=0.1) 
    print "bestError out of sample: ", str(round(bestError*100,1)) + "%"
    sys.exit(0)



def trial_Coefficient_removal_by_least_magnitude(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])
#       cVals = (100., 55., 10., 5.5, 1., 0.55, 0.1, 0.055, 0.01, 0.0055, 0.001, 0.00055, 0.0001)
#       cVals = 1

    print "Cross validated over C with values of: " + str(cVals)

    results = []
    omit = [0,10,15,20,25,30,35,40,45,50,55,60,65,70]
    for num in omit:
        trainedLearner = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least magnitude",
                C=cVals, performanceFunction=fractionIncorrect)
        result = trainedLearner.test(testX, testY,
            performanceFunction=fractionIncorrect)
        results.append(result)

    pnames = ['number Omitted', 'out sample error: fractionIncorrect']
    objName = predictionMode
    raw = UML.createData("List", [omit, results], pointNames=pnames, name=objName)
    figurePath = './results-least_magnitude.png'
    raw.plotPointAgainstPoint(0,1, outPath=figurePath)
    exit(0)


def trial_Coefficient_removal_by_least_value(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])
#       cVals = (100., 55., 10., 5.5, 1., 0.55, 0.1, 0.055, 0.01, 0.0055, 0.001, 0.00055, 0.0001)
#       cVals = 1

    results = []
    omit = [0,10,15,20,25,30,35,40,45,50,55,60,65,70]
    for num in omit:
        trainedLearner = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least value",
                C=cVals, performanceFunction=fractionIncorrect)
        result = trainedLearner.test(testX, testY, performanceFunction=fractionIncorrect)
        print result
        results.append(result)

    pnames = ['number Omitted', 'out sample error: fractionIncorrect']
    objName = predictionMode
    raw = UML.createData("List", [omit, results], pointNames=pnames, name=objName)
    figurePath = './results-least_value.png'
    raw.plotPointAgainstPoint(0,1, outPath=figurePath)
    exit(0)


def analysis_removal_comparison(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])
#   cVals = 0.001

    num = 35

    allQsList = trainX.getFeatureNames()
#   allQs = set(allQsList)
    LVQs = []
    LMQs = []
    tlLV = []
    tlLM = []

    omit = [10,15,20,25,30,35,40,45,50,55,60,65,70]
#   omit = [35]
    for i, num in enumerate(omit):
        trainedLearnerLV = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least value",
                C=cVals, performanceFunction=fractionIncorrect)

        tlLV.append(trainedLearnerLV)

        trainedLearnerLM = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least magnitude",
                C=cVals, performanceFunction=fractionIncorrect)

        tlLM.append(trainedLearnerLM)

        print "\nnum " + str(num) + "\n"

        LVWanted = trainedLearnerLV.getAttributes()['wantedIndices']
        LMWanted = trainedLearnerLM.getAttributes()['wantedIndices']

        LVQs.append(set(numpy.array(allQsList)[LVWanted]))
        LMQs.append(set(numpy.array(allQsList)[LMWanted]))

        # In Both
#       print "In Both:" + str(LVQs[i] & LMQs[i])
#       print ""

        # difference
#       print "Least Value unique:" + str(LVQs[i] - LMQs[i])
#       print ""
#       print "Least Magnitude unique" + str(LMQs[i] - LVQs[i])
#       print ""

        # Excluded from both
#       print "removed from both: " + str(allQs[i] - (LMQs[i] | LVQs[i]))

        # assertions
        #if i > 0:
            #print numpy.equal(tlLV[i].getAttributes()['origCoefs'], tlLV[i-1].getAttributes()['origCoefs'])
            #assert tlLV[i].getAttributes()['origCoefs'] == tlLV[i-1].getAttributes()['origCoefs']
            #print LVQs[i] - LVQs[i-1]
            # inaccurate: without the same original coefs, the results won't
            # be exact
            # assert LVQs[i] < LVQs[i-1]
            # assert LMQs[i] < LMQs[i-1]

    exit(0)


def analysis_randomness_effects(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])

    tlLV = []
    resultsLV = []
    resultsLM = []
    tlLM = []

    omit = [0,10,15,20,25,30,35,40,45,50,55,60,65,70]
    for i, num in enumerate(omit):
        trainedLearnerLV1 = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least value",
                C=cVals, performanceFunction=fractionIncorrect)
        tlLV.append(trainedLearnerLV1)
        trainedLearnerLV2 = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least value",
                C=cVals, performanceFunction=fractionIncorrect)
        tlLV.append(trainedLearnerLV2)
#       trainedLearnerLV3 = UML.train(
#               name, trainX, trainY, numberToOmit=num, method="least value",
#               C=cVals, performanceFunction=fractionIncorrect)
#       tlLV.append(trainedLearnerLV3)

        result1 = trainedLearnerLV1.test(testX, testY, performanceFunction=fractionIncorrect)
        resultsLV.append(result1)
        result2 = trainedLearnerLV2.test(testX, testY, performanceFunction=fractionIncorrect)
        resultsLV.append(result2)
#       result3 = trainedLearnerLV3.test(testX, testY, performanceFunction=fractionIncorrect)
#       resultsLV.append(result3)
    
        trainedLearnerLM1 = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least magnitude",
                C=cVals, performanceFunction=fractionIncorrect)
        tlLM.append(trainedLearnerLM1)
        trainedLearnerLM2 = UML.train(
                name, trainX, trainY, numberToOmit=num, method="least magnitude",
                C=cVals, performanceFunction=fractionIncorrect)
        tlLM.append(trainedLearnerLM2)
#       trainedLearnerLM3 = UML.train(
#               name, trainX, trainY, numberToOmit=num, method="least magnitude",
#               C=cVals, performanceFunction=fractionIncorrect)
#       tlLM.append(trainedLearnerLM3)

        result1 = trainedLearnerLM1.test(testX, testY, performanceFunction=fractionIncorrect)
        resultsLM.append(result1)
        result2 = trainedLearnerLM2.test(testX, testY, performanceFunction=fractionIncorrect)
        resultsLM.append(result2)
#       result3 = trainedLearnerLM3.test(testX, testY, performanceFunction=fractionIncorrect)
#       resultsLM.append(result3)

    # COMPARE!
    # out of sample error
#   pnames = ['number Omitted', 'out sample error: fractionIncorrect']
#   objName = "Least Value randomness analysis"
#   corrOmit = [val for pair in zip(omit, omit) for val in pair]
#   corrOmit = [val for pair in zip(omit, omit, omit) for val in pair]
#   raw = UML.createData("List", [corrOmit, resultsLV], pointNames=pnames, name=objName)
#   figurePath = './results-least_value_triple_trials.png'
#   raw.plotPointAgainstPoint(0,1, outPath=figurePath)

#   objName = "Least Magnitude randomness analysis"
#   raw = UML.createData("List", [corrOmit, resultsLM], pointNames=pnames, name=objName)
#   figurePath = './results-least_magnitude_triple_trials.png'
#   raw.plotPointAgainstPoint(0,1, outPath=figurePath)

    # coefficients
    allTL = tlLV + tlLM
    currTL = allTL[0]
    currCoefs = currTL.getAttributes()['origCoefs'].flatten().reshape(1,75)
    coefsObj = UML.createData("Matrix", currCoefs)

    for i in xrange(1,len(allTL)):
        currTL = allTL[i]
        currCoefs = currTL.getAttributes()['origCoefs'].flatten().reshape(1,75)
        currCoefsObj = UML.createData("Matrix", currCoefs)
        coefsObj.appendPoints(currCoefsObj)

#   print coefsObj.pointCount
#   print coefsObj.featureCount

    coefCorr = coefsObj.featureSimilarities("correlation")
    # BUT THIS IS WIERD since the questions are 'scored' on different
    # scales depending on whether it ends with an (M) or (F)
    coefCorr.setPointNames([str(val) for val in xrange(75)])
    coefCorr.setFeatureNames([str(val) for val in xrange(75)])
    coefCorr.show("coef correlation", maxWidth=None, maxHeight=80,
        includeObjectName=False)

    exit(0)


def analysis_finalModel_decisionScore_proba(trainX, trainY, testX, testY):
    normalizer = noNormalization
    selector = featSelect_handmade_orig
    trainer = train_LogReg_with_L2

    normalizer(trainX, testX)
    (trainX, testX) = selector(trainX, trainY, testX, desiredNonZeroCoefficients)
    trainedLearner = trainer(trainX, trainY, testX, testY)

    printAccuracy(trainedLearner, trainX, trainY, testX, testY)

    toCheckMe = [1,0,2,2,3,0,0,2,3,1,1,1,1,1,1,4,3,3,1,3,0,3,2]
    toCheckRando = [2,3,0,0,4,1,0,1,2,2,3,0,3,4,1,1,2,0,4,4,3,1,1]
    toCheckAllF = [0,0,4,4,4,0,0,0,0,0,0,4,4,0,0,0,0,4,0,0,4,4,4]
    toCheckAllM = [4,4,0,0,0,4,4,4,4,4,4,0,0,4,4,4,4,0,4,4,0,0,0]
    toCheck = [toCheckMe, toCheckRando, toCheckAllF, toCheckAllM]
    #print dir(trainedLearner.backend)
#       print trainedLearner.backend.coef_
    print trainedLearner.backend.predict(toCheck)
    print trainedLearner.backend.decision_function(toCheck)
    print trainedLearner.backend.predict_proba(toCheck)



def analysis_finalModel_perGenderAvgScores(trainX, trainY, testX, testY):
    normalizer = noNormalization
    selector = featSelect_handmade_random
    trainer = train_LogReg_with_L2

    normalizer(trainX, testX)
    (trainX, testX) = selector(trainX, trainY, testX, desiredNonZeroCoefficients)
    trainedLearner = trainer(trainX, trainY, testX, testY)
    coefs = trainedLearner.backend.coef_.flatten()
    print coefs.shape

    trainX.appendPoints(testX)
    trainY.appendPoints(testY)

    nzIDs = trainY.copyAs("numpyarray",outputAs1D=True).nonzero()[0]

    malePoints = trainX.extractPoints(list(nzIDs))
    femalePoints = trainX

    print malePoints.pointCount
    print femalePoints.pointCount

    mMeans = malePoints.featureStatistics("mean").copyAs("pythonlist", outputAs1D=True)
    fMeans = femalePoints.featureStatistics("mean").copyAs("pythonlist", outputAs1D=True)

    print mMeans
    print ""
    print fMeans

    names = trainX.getFeatureNames()
    zipped = zip(mMeans, fMeans, coefs, names)

    def r((m,f,c,n)):
        return (round(m,3),round(f,3),round(c,3),n)

    zipped = map(r, zipped)

#   for i, (mMean, fMean) in enumerate()
    for (mMean, fMean, coef, Q) in zipped:
        msg = str(mMean).ljust(6) + " "
        msg += str(fMean).ljust(6) + " "
        if coef < 0:
            msg += "-"
        else:
            msg += " "
        msg += str(abs(coef)).ljust(6) + " "
        msg += Q
        print msg


#################################
### NORMALIZATION CHOICE CODE ###
#################################

def noNormalization(trainX, testX):
    trainX.name = "trainX noNorm"
    testX.name = "testX noNorm"

    return

def normalize_Feature_subtract_mean(trainX, testX):
    vals = []

    def fn(feat):
        workspace = numpy.array(feat)
        mn = numpy.mean(workspace)
        vals.append(mn)

        return workspace - mn

    trainX.applyToFeatures(fn)

    def fnLookup(feat):
        workspace = numpy.array(feat)
        mn = vals[feat.index()]

        return workspace - mn

    testX.applyToFeatures(fnLookup)

    trainX.name = "trainX norm:sub_mean"
    testX.name = "testX norm:sub_mean"

def normalize_Feature_subtract_mean_div_std(trainX, testX):
    vals = []

    def fn(feat):
        workspace = numpy.array(feat)
        mn = numpy.mean(workspace)
        std = numpy.std(workspace)
        vals.append((mn,std))

        return (workspace - mn) / std

    trainX.applyToFeatures(fn)

    def fnLookup(feat):
        workspace = numpy.array(feat)
        (mn,std) = vals[feat.index()]

        return (workspace - mn) / std

    testX.applyToFeatures(fnLookup)

    trainX.name = "trainX norm:stdScore"
    testX.name = "testX norm:stdScore"


#####################################
### FEATURE SELECTION CHOICE CODE ###
#####################################

def wantedIndiceGrabber(tl, data):
    wi = tl.getAttributes()['wantedIndices']
    return data.copyFeatures(wi)

def featSelect_All(trainX, trainY, testX, numWanted):
    return (trainX, testX)

def featSelect_LogRegRegularization(trainX, trainY, testX, numWanted):
    name = "Custom.LogisticRegressionSelectByRegularization"
    tl = UML.train(name, trainX, trainY, desiredNonZero=numWanted, useLog=False)

    redTrain = wantedIndiceGrabber(tl, trainX)
    redTest = wantedIndiceGrabber(tl, testX)

    redTrain.name = trainX.name + " sel:Reg"
    redTest.name = testX.name + " sel:Reg"

    return (redTrain, redTest)

def featSelect_LogRegOmit_LeastValue(trainX, trainY, testX, numWanted):
    name = "Custom.LogisticRegressionSelectByOmission"
    nto = trainX.featureCount - numWanted
    cVals = tuple([100. / (10**n) for n in range(7)])
    tl = UML.train(
        name, trainX, trainY, method="least value", numberToOmit=nto,
        C=cVals, performanceFunction=fractionIncorrect, useLog=False)

    redTrain = wantedIndiceGrabber(tl, trainX)
    redTest = wantedIndiceGrabber(tl, testX)

    redTrain.name = trainX.name + " sel:omitLV"
    redTest.name = testX.name + " sel:omitLV"

    return (redTrain, redTest)

def featSelect_LogRegOmit_LeastMagnitude(trainX, trainY, testX, numWanted):
    name = "Custom.LogisticRegressionSelectByOmission"
    nto = trainX.featureCount - numWanted
    cVals = tuple([100. / (10**n) for n in range(7)])
    tl = UML.train(
        name, trainX, trainY, method="least magnitude", numberToOmit=nto,
        C=cVals, performanceFunction=fractionIncorrect, useLog=False)

    redTrain = wantedIndiceGrabber(tl, trainX)
    redTest = wantedIndiceGrabber(tl, testX)

    redTrain.name = trainX.name + " sel:omitLM"
    redTest.name = testX.name + " sel:omitLM"

    return (redTrain, redTest)

def featSelect_handmade_orig(trainX, trainY, testX, numWanted):
    wanted = []
    wanted.append(" I contradict others. (M)")
    wanted.append(" I would like to have more power than other people. (M)")
    wanted.append(" I give money to charity. (F)")
    wanted.append(" I am deeply moved by others' misfortunes. (F)")
    wanted.append(" I would feel very sorry for an animal caught in a trap. (F)")
    wanted.append(" I am good at making impromptu speeches. (M)")
    wanted.append(" I rarely worry. (M)")
    wanted.append(" I am relaxed most of the time. (M)")
    wanted.append(" I am not easily annoyed. (M)")
    wanted.append(" I rarely notice my emotional reactions. (M)")
    wanted.append(" I recover quickly from stress and illness. (M)")
    wanted.append(" I am afraid of many things. (F)")
    wanted.append(" I find that my feelings are easily hurt. (F)")
    wanted.append(" I put on a show to impress people. (M)")
    wanted.append(" I tell people what they want to hear so they do what I want. (M)")
    wanted.append(" I have a dark outlook on the future. (M)")
    wanted.append(" I like to solve complex problems. (M)")
    wanted.append(" I would never make a high-risk investment. (F)")
    wanted.append(" I am out for my own personal gain. (M)")
    wanted.append(" I behave in unusual and strange ways. (M)")
    wanted.append(" I talk a lot. (F)")
    wanted.append(" I laugh aloud. (F)")
    wanted.append(" I show my feelings when I'm happy. (F)")

    redTrain = trainX.copyFeatures(wanted)
    redTest = testX.copyFeatures(wanted)
    return (redTrain, redTest)

def featSelect_handmade_random(trainX, trainY, testX, numWanted):
    wanted = []
    wanted.append(" I contradict others. (M)")
    wanted.append(" I tell people what they want to hear so they do what I want. (M)")
    wanted.append(" I am good at making impromptu speeches. (M)")
    wanted.append(" I behave in unusual and strange ways. (M)")
    wanted.append(" I am not easily annoyed. (M)")
    wanted.append(" I would feel very sorry for an animal caught in a trap. (F)")
    wanted.append(" I recover quickly from stress and illness. (M)")
    wanted.append(" I rarely worry. (M)")
    wanted.append(" I am out for my own personal gain. (M)")
    wanted.append(" I give money to charity. (F)")
    wanted.append(" I like to solve complex problems. (M)")
    wanted.append(" I am deeply moved by others' misfortunes. (F)")
    wanted.append(" I put on a show to impress people. (M)")
    wanted.append(" I am relaxed most of the time. (M)")
    wanted.append(" I talk a lot. (F)")
    wanted.append(" I show my feelings when I'm happy. (F)")
    wanted.append(" I find that my feelings are easily hurt. (F)")
    wanted.append(" I rarely notice my emotional reactions. (M)")
    wanted.append(" I am afraid of many things. (F)")
    wanted.append(" I laugh aloud. (F)")
    wanted.append(" I would like to have more power than other people. (M)")
    wanted.append(" I would never make a high-risk investment. (F)")
    wanted.append(" I have a dark outlook on the future. (M)")

    redTrain = trainX.copyFeatures(wanted)
    redTest = testX.copyFeatures(wanted)
    return (redTrain, redTest)

############################
### TRAINING CHOICE CODE ###
############################


def train_LogReg_with_L1(trainX, trainY, testX, testY):
    name = "scikitlearn.LogisticRegression"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print "Cross validated over C with values of: " + str(cVals)
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, penalty='l1',
        performanceFunction=fractionIncorrect)

    return trainedLearner


def train_LogReg_with_L2(trainX, trainY, testX, testY):
    name = "scikitlearn.LogisticRegression"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print "Cross validated over C with values of: " + str(cVals)
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, penalty='l2',
        performanceFunction=fractionIncorrect)

    return trainedLearner

def train_ridgeClassifier(trainX, trainY, testX, testY):
    aVals = tuple([1. / (10**n) for n in range(9)])
    print "Cross validated over alpha with values of: " + str(aVals)
    trainedLearner = UML.train('skl.RidgeClassifier', trainX, trainY,
        alpha=aVals, performanceFunction=fractionIncorrect)

    return trainedLearner

def train_SVM_Linear_kernel(trainX, trainY, testX, testY):
    name = "scikitlearn.SVC"
    cVals = tuple([100. / (10**n) for n in range(7)])

    print "Cross validated over C with values of: " + str(cVals)
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, kernel='linear',
        performanceFunction=fractionIncorrect, max_iter=2000)

    return trainedLearner


def train_SVM_rbf_kernel(trainX, trainY, testX, testY):
    name = "scikitlearn.SVC"
    cVals = tuple([100. / (10**n) for n in range(7)])
    gamVals = tuple([100. / (10**n) for n in range(7)])

    print "Cross validated over C with values of: " + str(cVals)
    print "Cross validated over gamma with values of: " + str(gamVals)
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, gamma=gamVals, kernel='rbf',
        performanceFunction=fractionIncorrect, max_iter=2000)

    return trainedLearner


def train_SVM_with_poly_kernel_deg_2(trainX, trainY, testX, testY):
    return train_SVM_with_poly_kernel(trainX, trainY, testX, testY, 2)

def train_SVM_with_poly_kernel_deg_3(trainX, trainY, testX, testY):
    return train_SVM_with_poly_kernel(trainX, trainY, testX, testY, 3)

def train_SVM_with_poly_kernel_deg_4(trainX, trainY, testX, testY):
    return train_SVM_with_poly_kernel(trainX, trainY, testX, testY, 4)

def train_SVM_with_poly_kernel(trainX, trainY, testX, testY, degree):
    name = "scikitlearn.SVC"
    cVals = tuple([100. / (10**n) for n in range(7)])
    coef0Vals = (0,10,100)

    print "Cross validated over C with values of: " + str(cVals)
    print "Cross validated over coef0 with values of: " + str(coef0Vals)
    trainedLearner = UML.train(
        name, trainX, trainY, C=cVals, coef0=coef0Vals, kernel='poly',
        degree=degree, performanceFunction=fractionIncorrect, max_iter=2000)

    return trainedLearner



####################################
### SELECT AND TRAIN CHOICE CODE ###
####################################


def selAndTrain_by_regularization_pick35(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByRegularization"
    print "Finding exactly " + str(35) + " coefficients..."
    trainedLearner = UML.train(name, trainX, trainY, desiredNonZero=35)

    return trainedLearner


def selAndTrain_by_least_magnitude_pick35(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print "Cross validated over C with values of: " + str(cVals)
    print "Finding exactly " + str(35) + " coefficients..."
    trainedLearner = UML.train(name, trainX, trainY, numberToOmit=40,
        method='least magnitude', C=cVals, performanceFunction=fractionIncorrect)

    return trainedLearner


def selAndTrain_by_least_value_pick35(trainX, trainY, testX, testY):
    name = "custom.LogisticRegressionSelectByOmission"
    cVals = tuple([100. / (10**n) for n in range(7)])
    print "Cross validated over C with values of: " + str(cVals)
    print "Finding exactly " + str(35) + " coefficients..."
    trainedLearner = UML.train(name, trainX, trainY, numberToOmit=40,
        method='least value', C=cVals, performanceFunction=fractionIncorrect)

    return trainedLearner


def omittedCultureQuestions():
    toOmit = []
    toOmit.append("I do not enjoy watching dance performances. (M)")
    toOmit.append(" I would be good at rescuing someone from a burning building. (M)")
    toOmit.append(" I am interested in science. (M)")
    toOmit.append(" I find political discussions interesting. (M)")
    toOmit.append(" I face danger confidently. (M)")
    toOmit.append(" I do not like concerts. (M)")
    toOmit.append(" I do not enjoy going to art museums. (M)")
    toOmit.append(" I would fear walking in a high-crime part of a city. (F)")
    toOmit.append(" I begin to panic when there is danger. (F)")

    return toOmit

def ommittedHandchosenQuestions():
    toOmit = []
    toOmit.append(" I would feel very badly for a long time if stole. (F)")
    toOmit.append(" I feel sympathy for those who are worse off than myself. (F)")
    toOmit.append(" I believe that I am better than others. (M)")
    toOmit.append(" I am not easily disturbed by events. (M)")
    toOmit.append(" I worry about things. (F)")
    toOmit.append(" I am a worrier. (F)")
    toOmit.append(" I panic easily. (F)")
    toOmit.append(" I get overwhelmed by emotions. (F)")
    toOmit.append(" I take risks that could cause trouble for me. (M)")
    toOmit.append(" I don't talk a lot. (M)")
    toOmit.append(" I get even with others. (M)")
    toOmit.append(" I smile a lot. (F)")

    return toOmit

if __name__ == "__main__":

    # Some variables to control the flow of the program
    defaultInFile = "/Users/spencer2/Dropbox/Spencer/Work/ClearerThinking.org/Programs and modules/Gender continuum test/gender continuum train and test ready to predict.csv"
    defaultCoefOutFile = ""
    omitCulture = True
    desiredNonZeroCoefficients = 35
    performSanityCheck = False

    UML.registerCustomLearner("custom", LogisticRegressionSelectByRegularization)
    UML.registerCustomLearner("custom", LogisticRegressionSelectByOmission)

    if len(sys.argv) <= 1:
        origFileName = defaultInFile
        coefOutFile = defaultCoefOutFile
    elif len(sys.argv) <= 2:
        origFileName = sys.argv[1]
        coefOutFile = defaultCoefOutFile
    else:
        origFileName = sys.argv[1]
        coefOutFile = sys.argv[2]
    
    dataAll = UML.createData("Matrix", origFileName, featureNames=True, fileType='csv',
        ignoreNonNumericalFeatures=True)

    # call helper to remove extraneous features, omit undesired
    # ones, and to seperate into training and testing sets according
    # to the 'InTestSet' feature.
    toOmit = []
    if omitCulture:
        toOmit += omittedCultureQuestions()
    trainX, trainY, testX, testY = seperateData(dataAll, toOmit)

    # bring all questions on the 0-5 scale
    standardizeScoreScale(trainX)
    standardizeScoreScale(testX)

    trainX.name = "Training Data"
    trainY.name = "Training Labels"
    testX.name = "Testing Data"
    testY.name = "Testing Labels"

    # Check some of the expectations we had on the data to make sure
    # we've extracted the right stuff
    if performSanityCheck:
        totalScores = dataAll.extractFeatures("totalScorePosOrNeg")
        sanityCheck(trainX, totalScores)  # defined above __main__

    print ""
    print "Train points: " + str(trainX.pointCount) 
    print "Test points: " + str(testX.pointCount) 
    print "Starting number of features: " + str(trainX.featureCount)
    print ""


    fullTrialChoices = []
    fullTrialChoices.append(SVC_full_data_outSample_only)  # 0
    fullTrialChoices.append(trial_Coefficient_removal_by_least_magnitude)  # 1
    fullTrialChoices.append(trial_Coefficient_removal_by_least_value)  # 2
    fullTrialChoices.append(analysis_removal_comparison)  # 3
    fullTrialChoices.append(analysis_randomness_effects)  # 4
    fullTrialChoices.append(analysis_finalModel_decisionScore_proba)  # 5
    fullTrialChoices.append(analysis_finalModel_perGenderAvgScores)  # 6
    fullTrialMode = fullTrialChoices[6]
    print fullTrialMode.__name__
    fullTrialMode(trainX, trainY, testX, testY) 
    exit(0)

    normFeatureChoices = []
    normFeatureChoices.append(noNormalization)  # 0
    normFeatureChoices.append(normalize_Feature_subtract_mean)  # 1
    normFeatureChoices.append(normalize_Feature_subtract_mean_div_std)  # 2
    normMode = normFeatureChoices[0]

    featSelectChoices = []
    featSelectChoices.append(featSelect_All)  # 0
    featSelectChoices.append(featSelect_LogRegRegularization)  # 1
    featSelectChoices.append(featSelect_LogRegOmit_LeastValue)  # 2
    featSelectChoices.append(featSelect_LogRegOmit_LeastMagnitude)  # 3
    featSelectChoices.append(featSelect_handmade_orig)  # 4
    featSelectChoices.append(featSelect_handmade_random)  # 5
    selectMode = featSelectChoices[4]

    trainChoices = []
    trainChoices.append(train_LogReg_with_L1)  # 0
    trainChoices.append(train_LogReg_with_L2)  # 1
    trainChoices.append(train_ridgeClassifier)  # 2
    trainChoices.append(train_SVM_Linear_kernel)  # 3
    trainChoices.append(train_SVM_rbf_kernel)  # 4
    trainChoices.append(train_SVM_with_poly_kernel_deg_2)  # 5
    trainChoices.append(train_SVM_with_poly_kernel_deg_3)  # 6
    trainChoices.append(train_SVM_with_poly_kernel_deg_4)  # 7
    trainMode = trainChoices[1]

    selAndTrainChoices = []
    selAndTrainChoices.append(selAndTrain_by_regularization_pick35)  # 0
    selAndTrainChoices.append(selAndTrain_by_least_value_pick35)  # 1
    selAndTrainChoices.append(selAndTrain_by_least_magnitude_pick35)  # 2
    
    # for safety
    origTrainX = trainX.copy()
    origTestX = testX.copy()

#   choices = itertools.product(normFeatureChoices, featSelectChoices[2:], trainChoices[:2])
    choices = [(normMode, selectMode, trainMode)]

    for (normalizer, selector, trainer) in choices:
        trainX = origTrainX.copy()
        testX = origTestX.copy()
        print normalizer.__name__
        print selector.__name__
        print trainer.__name__

        normalizer(trainX, testX)
        (trainX, testX) = selector(trainX, trainY, testX, desiredNonZeroCoefficients)
        trainedLearner = trainer(trainX, trainY, testX, testY)

        printAccuracy(trainedLearner, trainX, trainY, testX, testY)

        # grab the feature names associated with the non-zero coefficients
#       printCoefficientsPythonLists(trainedLearner, trainX, randomize=True)
#       saveCoefficients(trainedLearner, trainX.getFeatureNames(), coefOutFile)


    exit(0)

    choices = itertools.product(normFeatureChoices, selAndTrainChoices)

    for (normalizer, trainer) in choices:
        trainX = origTrainX.copy()
        testX = origTestX.copy()
        print normalizer.__name__
        print trainer.__name__

        normalizer(trainX, testX)
        trainedLearner = trainer(trainX, trainY, testX, testY)

#       print trainedLearner.getAttributes()

        # grab the feature names associated with the non-zero coefficients
    #   printCoefficientsHR(trainedLearner)

        #Now measure the accuracy of the model
        printAccuracy(trainedLearner, trainX, trainY, testX, testY)

    exit(0)