"""
	A class that manages logging from a high level.  Creates two low-level logger objects -
	a human readable logger and a machine-readable logger - and passes run information to each
	of them.  The logs are put in the default location unless 
"""

import os

from human_readable_test import HumanReadableRunLog
from machine_readable_test import MachineReadableRunLog


class LogManager(object):

	def __init__(self, logLocation=None, logName=None):
		if logLocation is None:
			logLocation = os.environ['HOME']+'/'

		if logName is None:
			logName = "uMLLog"

		self.humanReadableLog = HumanReadableRunLog(logLocation + logName + ".txt")
		self.machineReadableLog = MachineReadableRunLog(logLocation + logName + ".mr")

	def logRun(self, trainData, testData, function, metrics, runTime, extraInfo=None):
		"""
			Pass the information about this run to both logs:  human and machine
			readable, which will write it out to the log files.
		"""
		self.humanReadableLog.logRun(trainData, testData, function, metrics, runTime, extraInfo)
		self.machineReadableLog.logRun(trainData, testData, function, metrics, runTime, extraInfo)