
##########################################################################
# Copyright 2024 Sparkwave LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

"""
Module demonstrating the nimble-forced One vs One and One vs All
multi-class classification strategies. Also demonstrates the possible
output formats allowed when calling learners.
"""

import nimble
from nimble import trainAndTest
from nimble import trainAndApply
from nimble.calculate import fractionIncorrect

if __name__ == "__main__":
    variables = ["x1", "x2", "x3", "label"]
    data1 = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2],
             [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1],
             [0, 1, 0, 2], [0, 0, 1, 3], [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3],
             [1, 0, 0, 3], [0, 1, 0, 1], [0, 0, 1, 2]]
    trainObj = nimble.data(source=data1, featureNames=variables)

    # formated vertically for easier comparison with printed results
    data2 = [[1, 0, 0, 1],
             [0, 1, 0, 2],
             [0, 0, 1, 3]]
    testObj = nimble.data(source=data2, featureNames=variables)
    tesObjNoY = testObj.features.copy([0,1,2])

    results = trainAndTest('sciKitLearn.SVC', fractionIncorrect, 
                           trainX=trainObj, trainY=3, testX=testObj, testY=3)
    print('Standard trainAndTest call, fractionIncorrect: ' + str(results))
    print("")

    resultsLabelsOvO = trainAndApply('sciKitLearn.SVC', trainX=trainObj, trainY=3,
                             testX=tesObjNoY, multiClassStrategy="OneVsOne")
    print('One vs One predictions (aka labels format):')
    print(resultsLabelsOvO)

    resultsBestScoreOvO = trainAndApply('sciKitLearn.SVC', trainX=trainObj, trainY=3,
                                     testX=tesObjNoY, scoreMode='bestScore',
                                     multiClassStrategy="OneVsOne")
    print('One vs One, best score format:')
    print(resultsBestScoreOvO)

    resultsAllScoresOvO = trainAndApply('sciKitLearn.SVC', trainX=trainObj, trainY=3,
                                     testX=tesObjNoY, scoreMode='allScores',
                                     multiClassStrategy="OneVsOne")
    print('One vs One, all scores format:')
    print(resultsAllScoresOvO)

    resultsLabelsOvA = trainAndApply('sciKitLearn.SVC', trainX=trainObj, trainY=3,
                             testX=tesObjNoY, multiClassStrategy="OneVsAll")
    print('One vs All predictions (aka labels format):')
    print(resultsLabelsOvA)

    resultsBestScoreOvA = trainAndApply('sciKitLearn.SVC', trainX=trainObj, trainY=3,
                                     testX=tesObjNoY, scoreMode='bestScore',
                                     multiClassStrategy="OneVsAll")
    print('One vs All, best score format:')
    print(resultsBestScoreOvA)

    resultsAllScoresOvA = trainAndApply('sciKitLearn.SVC', trainX=trainObj, trainY=3,
                                     testX=tesObjNoY, scoreMode='allScores',
                                     multiClassStrategy="OneVsAll")
    print('One vs All, all scores format:')
    print(resultsAllScoresOvA)
