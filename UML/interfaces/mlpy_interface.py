"""
Relies on being scikit-learn 0.9 or above

OLS and LARS learners are not allowed as learners. KernelExponential is
not allowed as a Kernel.

TODO: multinomialHMM requires special input processing for obs param
"""

from __future__ import absolute_import
import importlib
import copy
import sys

import numpy
from six.moves import range

import UML
from UML.exceptions import InvalidArgumentValue
from UML.interfaces.universal_interface import UniversalInterface
from UML.interfaces.interface_helpers import PythonSearcher
from UML.interfaces.interface_helpers import removeFromTailMatchedLists
from UML.helpers import inspectArguments
from UML.docHelpers import inheritDocstringsFactory

# Contains path to mlpy root directory
mlpyDir = None

# a dictionary mapping names to learners, or modules
# containing learners. To be used by findInPackage
locationCache = {}


@inheritDocstringsFactory(UniversalInterface)
class Mlpy(UniversalInterface):
    """
    This class is an interface to mlpy.
    """

    _XDataAliases = ['X', 'x', 'T', 't', 'K', 'Kt']
    _YDataAliases = ['Y', 'y']
    _DataAliases = _XDataAliases + _YDataAliases

    def __init__(self):
        if mlpyDir is not None:
            sys.path.insert(0, mlpyDir)

        self.mlpy = importlib.import_module('mlpy')

        def isLearner(obj):
            hasLearn = hasattr(obj, 'learn')
            hasPred = hasattr(obj, 'pred')
            hasTrans = hasattr(obj, 'transform')

            if hasLearn and (hasPred or hasTrans):
                return True
            return False

        self._searcher = PythonSearcher(self.mlpy, dir(self.mlpy), {},
                                        isLearner, 1)

        super(Mlpy, self).__init__()

    #######################################
    ### ABSTRACT METHOD IMPLEMENTATIONS ###
    #######################################

    def accessible(self):
        try:
            import mlpy
        except ImportError:
            return False
        return True


    def _listLearnersBackend(self):
        possibilities = self._searcher.allLearners()

        exclude = ['OLS', 'LARS']
        ret = []
        for name in possibilities:
            if not name in exclude:
                ret.append(name)

        ret.append('MFastHCluster')
        ret.append('kmeans')

        return ret


    def learnerType(self, name):
        obj = self.findCallable(name)
        if name.lower() == 'liblinear' or name.lower() == 'libsvm':
            return "UNKNOWN"
        if hasattr(obj, 'labels'):
            return 'classification'

        return 'UNKNOWN'


    def _findCallableBackend(self, name):
        if name == 'kmeans':
            return _Kmeans
        if name == 'MFastHCluster':
            return _MFastHCluster

        return self._searcher.findInPackage(None, name)


    def _getParameterNamesBackend(self, name):
        ret = self._paramQuery(name, None)
        if ret is None:
            return ret
        (objArgs, _, _, _) = ret
        if objArgs[0] == 'self':
            objArgs = objArgs[1:]
        return [objArgs]


    def _getLearnerParameterNamesBackend(self, learnerName):
        ignore = self._DataAliases + ['self']
        if learnerName == 'MFastHCluster':
            ignore.remove('t')
        init = self._paramQuery('__init__', learnerName, ignore)
        learn = self._paramQuery('learn', learnerName, ignore)
        pred = self._paramQuery('pred', learnerName, ignore)
        transform = self._paramQuery('transform', learnerName, ignore)

        if pred is not None:
            ret = init[0] + learn[0] + pred[0]
        elif transform is not None:
            ret = init[0] + learn[0] + transform[0]
        else:
            msg = "Cannot get parameter names for leaner " + learnerName
            raise InvalidArgumentValue(msg)

        return [ret]


    def _getDefaultValuesBackend(self, name):
        ret = self._paramQuery(name, None)
        if ret is None:
            return ret
        (objArgs, _, _, d) = ret
        ret = {}
        if d is not None:
            for i in range(len(d)):
                ret[objArgs[-(i + 1)]] = d[-(i + 1)]

        return [ret]


    def _getLearnerDefaultValuesBackend(self, learnerName):
        ignore = self._DataAliases + ['self']
        init = self._paramQuery('__init__', learnerName, ignore)
        learn = self._paramQuery('learn', learnerName, ignore)
        pred = self._paramQuery('pred', learnerName, ignore)
        transform = self._paramQuery('transform', learnerName, ignore)

        if pred is not None:
            toProcess = [init, learn, pred]
        elif transform is not None:
            toProcess = [init, learn, transform]

        ret = {}
        for stage in toProcess:
            currNames = stage[0]
            currDefaults = stage[3]
            if stage[3] is not None:
                for i in range(len(currDefaults)):
                    key = currNames[-(i + 1)]
                    value = currDefaults[-(i + 1)]
                    ret[key] = value

        return [ret]


    def _getScores(self, learnerName, learner, testX, newArguments,
                   storedArguments, customDict):
        if hasattr(learner, 'pred_values'):
            method = 'pred_values'
            ignore = self._XDataAliases + ['self']
            if 'useT' in customDict and customDict['useT']:
                ignore.remove('t')
            backendArgs = self._paramQuery(method, learnerName, ignore)[0]
            scoreArgs = self._getMethodArguments(backendArgs, newArguments,
                                                 storedArguments)
            return learner.pred_values(testX, **scoreArgs)
        else:
            raise NotImplementedError('Cannot get scores for this learner')


    def _getScoresOrder(self, learner):
        return learner.labels()


    def isAlias(self, name):
        return name.lower() == self.getCanonicalName().lower()


    def getCanonicalName(self):
        return 'mlpy'


    def _inputTransformation(self, learnerName, trainX, trainY, testX,
                             arguments, customDict):
        if trainX is not None:
            customDict['match'] = trainX.getTypeString()
            if trainX.getTypeString() == 'Matrix':
                transTrainX = trainX.data
            else:
                transTrainX = trainX.copy(to='numpy matrix')
        else:
            transTrainX = None

        if trainY is not None:
            transTrainY = trainY.copy(to='numpy array', outputAs1D=True)
        else:
            transTrainY = None

        if testX is not None:
            if testX.getTypeString() == 'Matrix':
                transTestX = testX.data
            else:
                transTestX = testX.copy(to='numpy matrix')
        else:
            transTestX = None

        if 'kernel' in arguments:
            if (arguments['kernel'] is None
                    and trainX is not None
                    and len(trainX.points) != len(trainX.features)):
                msg = "For this learner, in the absence of specifying a "
                msg += "kernel, the trainX parameter must be square "
                msg += "(representing the inner product space of the features)"
                raise InvalidArgumentValue(msg)

            if isinstance(arguments['kernel'], self.mlpy.KernelExponential):
                msg = "This interface disallows the use of KernelExponential; "
                msg = "it is bugged in some versions of mlpy"
                raise InvalidArgumentValue(msg)

        customDict['useT'] = False
        if learnerName == 'MFastHCluster':
            customDict['useT'] = True

        return (transTrainX, transTrainY, transTestX, copy.deepcopy(arguments))


    def _outputTransformation(self, learnerName, outputValue,
                              transformedInputs, outputType, outputFormat,
                              customDict):
        if outputFormat == "label" and len(outputValue.shape) == 1:
            # we are sometimes given a matrix, this will take care of that
            outputValue = numpy.array(outputValue).flatten()
            # In the case of prediction we are given a row vector,
            # yet we want a column vector
            outputValue = outputValue.reshape(len(outputValue), 1)

        # TODO correct
        outputType = 'Matrix'
        if outputType == 'match':
            outputType = customDict['match']
        return UML.createData(outputType, outputValue, useLog=False)


    def _trainer(self, learnerName, trainX, trainY, arguments, customDict):
        # get parameter names
        initNames = self._paramQuery('__init__', learnerName, ['self'])[0]
        learnNames = self._paramQuery('learn', learnerName, ['self'])[0]
        predNames = self._paramQuery('pred', learnerName, ['self'])
        if predNames is not None:
            customDict['predNames'] = predNames[0]
        transNames = self._paramQuery('transform', learnerName, ['self'])
        if transNames is not None:
            customDict['transNames'] = transNames[0]

        # pack parameter sets
        initParams = {}
        for name in initNames:
            initParams[name] = arguments[name]
        learnParams = {}
        for name in learnNames:
            if name in self._XDataAliases:
                value = trainX
            elif name in self._YDataAliases:
                value = trainY
            else:
                value = arguments[name]
            learnParams[name] = value

        # use patch if necessary
        patchedLearners = ["DLDA", "Parzen", "ElasticNet", "ElasticNetC"]
        if learnerName in patchedLearners:
            patchModule = importlib.import_module("UML.interfaces.mlpy_patches")
            initLearner = getattr(patchModule, learnerName)
        else:
            initLearner = self.findCallable(learnerName)
        learner = initLearner(**initParams)
        learner.learn(**learnParams)

        return learner


    def _incrementalTrainer(self, learner, trainX, trainY, arguments,
                            customDict):
        raise RuntimeError()


    def _applier(self, learnerName, learner, testX, newArguments,
                 storedArguments, customDict):
        if hasattr(learner, 'pred'):
            method = 'pred'
            toCall = self._pred
        elif hasattr(learner, 'transform'):
            method = 'transform'
            toCall = self._transform
        else:
            msg = "Cannot apply this learner to data, no predict or "
            msg = "transform function"
            raise TypeError(msg)
        ignore = self._XDataAliases + ['self']
        if 'useT' in customDict and customDict['useT']:
            ignore.remove('t')

        backendArgs = self._paramQuery(method, learnerName, ignore)[0]
        applyArgs = self._getMethodArguments(backendArgs, newArguments,
                                             storedArguments)
        return toCall(learner, testX, applyArgs, customDict)


    def _getAttributes(self, learnerBackend):
        raise RuntimeError()


    def _optionDefaults(self, option):
        return None


    def _configurableOptionNames(self):
        return ['location']


    def _exposedFunctions(self):
        return [self._pred, self._transform]


    def version(self):
        return self.mlpy.__version__


    def _pred(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying predict function of a mlpy
        learner object.
        """
        params = customDict['predNames']
        if len(params) > 0:
            if customDict['useT']:
                testX = arguments['t']
                del arguments['t']
            return learner.pred(testX, **arguments)
        else:
            return learner.pred(**arguments)

    def _transform(self, learner, testX, arguments, customDict):
        """
        Wrapper for the underlying transform function of a mlpy learner
        object.
        """
        return learner.transform(testX, **arguments)


    ###############
    ### HELPERS ###
    ###############

    def _paramQuery(self, name, parent, ignore=None):
        """
        Takes the name of some mlpy object or function, returns
        a list of parameters used to instantiate that object or run that
        function, or None if the desired thing cannot be found.
        """
        if ignore is None:
            ignore = []
        namedModule = self._searcher.findInPackage(parent, name)

        # for python 3
        # in python 3, inspectArguments(mlpy.KNN.__init__) works,
        # but returns back wrong arguments. we need to purposely run
        # self._paramQueryHardCoded(name, parent, ignore) for KNN, PCA...
        excludeList = ['libsvm', 'knn', 'liblinear', 'maximumlikelihoodc',
                       'KernelAdatron'.lower(), 'ClassTree'.lower(),
                       'MFastHCluster'.lower(), 'kmeans']
        if sys.version_info.major > 2 and 'kernel' not in name.lower():
            if parent is None or parent.lower() in excludeList:
                return self._paramQueryHardCoded(name, parent, ignore)

        if not namedModule is None:
            try:
                (args, v, k, d) = inspectArguments(namedModule)
                (args, d) = removeFromTailMatchedLists(args, d, ignore)
                if 'seed' in args:
                    index = args.index('seed')
                    negdex = index - len(args)
                    d[negdex] = UML.randomness.generateSubsidiarySeed()
                return (args, v, k, d)
            except TypeError:
                try:
                    (args, v, k, d) = inspectArguments(namedModule.__init__)
                    (args, d) = removeFromTailMatchedLists(args, d, ignore)
                    if 'seed' in args:
                        index = args.index('seed')
                        negdex = index - len(args)
                        d[negdex] = UML.randomness.generateSubsidiarySeed()
                    return (args, v, k, d)
                except TypeError:
                    pass

        return self._paramQueryHardCoded(name, parent, ignore)


    def _paramQueryHardCoded(self, name, parent, ignore):
        """
        Returns a list of parameters for in package entities that we
        have hard coded, under the assumption that it is difficult or
        impossible to find that data automatically.
        """
        pnames = []
        pvarargs = None
        pkeywords = None
        pdefaults = []

        if parent is None:
            return None

        # TODO for python 3
        # in python 3, mlpy's KNN, PCA ... may have different arguments
        # than those in python 2 mlpy.
        if parent.lower() == 'LibSvm'.lower():
            if name == '__init__':
                pnames = ['svm_type', 'kernel_type', 'degree', 'gamma',
                          'coef0', 'C', 'nu', 'eps', 'p', 'cache_size',
                          'shrinking', 'probability', 'weight']
                pdefaults = ['c_svc', 'linear', 3, 0.001, 0, 1, 0.5, 0.001,
                             0.1, 100, True, False, {}]
            elif name == 'learn':
                pnames = ['x', 'y']
            elif name == 'pred' or name == 'pred_values':
                pnames = ['t']
            else:
                return None
        elif parent.lower() == 'KNN'.lower():
            if name == '__init__':
                pnames = ['k']
            elif name == 'learn':
                pnames = ['x', 'y']
            elif name == 'pred':
                pnames = ['t']
            else:
                return None
        elif parent.lower() == 'LibLinear'.lower():
            if name == '__init__':
                pnames = ['solver_type', 'C', 'eps', 'weight']
                pdefaults = ['l2r_lr', 1, 0.01, {}]
            elif name == 'learn':
                pnames = ['x', 'y']
            elif name == 'pred' or name == 'pred_values':
                pnames = ['t']
            else:
                return None
        elif parent.lower() == 'MaximumLikelihoodC'.lower():
            if name == '__init__':
                pass
            elif name == 'learn':
                pnames = ['x', 'y']
            elif name == 'pred':
                pnames = ['t']
            else:
                return None
        elif parent.lower() == 'KernelAdatron'.lower():
            if name == '__init__':
                pnames = ['C', 'maxsteps', 'eps']
                pdefaults = [1000, 1000, 0.01]
            elif name == 'learn':
                pnames = ['K', 'y']
            elif name == 'pred':
                pnames = ['Kt']
            else:
                return None
        elif parent.lower() == 'ClassTree'.lower():
            if name == '__init__':
                pnames = ['stumps', 'minsize']
                pdefaults = [0, 1]
            elif name == 'learn':
                pnames = ['x', 'y']
            elif name == 'pred':
                pnames = ['t']
            else:
                return None
        elif parent.lower() == 'MFastHCluster'.lower():
            if name == '__init__':
                pnames = ['method']
                pdefaults = ['single']
            elif name == 'learn':
                pnames = ['x']
            elif name == 'pred':
                pnames = ['t']
            else:
                return None
        elif parent.lower() == 'kmeans'.lower():
            if name == '__init__':
                pnames = ['k', 'plus', 'seed']
                pdefaults = [False, UML.randomness.generateSubsidiarySeed()]
            elif name == 'learn':
                pnames = ['x']
            elif name == 'pred':
                pnames = []
            else:
                return None

        else:
            return None

        ret = (pnames, pvarargs, pkeywords, pdefaults)
        (newArgs, newDefaults) = removeFromTailMatchedLists(ret[0], ret[3],
                                                            ignore)
        return (newArgs, ret[1], ret[2], newDefaults)


class _Kmeans(object):
    def __init__(self, k, plus=False, seed=0):
        self.k = k
        self.plus = plus
        self.seed = seed

    def learn(self, x):
        self.x = x

    def labels(self):
        return None

    def pred(self):
        import mlpy

        (self.clusters, _, _) = mlpy.kmeans(self.x, self.k, self.plus,
                                            self.seed)
        return self.clusters


class _MFastHCluster(object):
    def __init__(self, method='single'):
        import mlpy

        self.obj = mlpy.MFastHCluster(method)

    def learn(self, x):
        self.obj.linkage(x)

    def pred(self, t):
        return self.obj.cut(t)
