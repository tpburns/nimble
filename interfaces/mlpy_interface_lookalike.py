"""




"""

import mlpy_interface_old as mlpy
from universal_interface_lookalike import UniversalInterfaceLookalike

class Mlpy(UniversalInterfaceLookalike):
	"""

	"""

	def __init__(self):
		"""

		"""
		super(Mlpy, self).__init__()

	def trainAndApply(self, learnerName, trainX, trainY=None, testX=None, arguments={}, output=None, scoreMode='label', multiClassStrategy='default', sendToLog=True):
		return mlpy.mlpy(learnerName, trainX, trainY, testX, arguments, output, scoreMode, multiClassStrategy, sendToLog)


	def listLearners(self):
		"""
		Return a list of all learners callable through this interface.

		"""
		return mlpy.listMlpyLearners()

	def getLearnerParameterNames(self, name):
		return mlpy.getParameters(name)

	def getLearnerDefaultValues(self, name):
		return mlpy.getDefaultValues(name)

	def _getParameterNames(self, name):
		"""
		Find params for instantiation and function calls 
		TAKES string name, 
		RETURNS list of list of param names to make the chosen call
		"""
		return mlpy.getParameters(name)


	def _getDefaultValues(self, name):
		"""
		Find default values
		TAKES string name, 
		RETURNS list of dict of param names to default values
		"""
		return mlpy.getDefaultValues(name)

	def isAlias(self, name):
		"""
		Returns true if the name is an accepted alias for this interface

		"""
		if name.lower() in ['mlpyold']:
			return True
		else:
			return False


	def getCanonicalName(self):
		"""
		Returns the string name that will uniquely identify this interface

		"""
		return "mlpyOLD"
