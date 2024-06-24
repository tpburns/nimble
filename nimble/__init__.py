
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
Nimble offers interfaces into other machine learning packages and
tools for data representation and processing. Available at
the top level in this package are the functions necessary to
create data objects, call machine learning algorithms on that
data, and do package level configuration and information querying.
"""
# pylint: disable=cyclic-import

# Import those functions that we want to be accessible in the
# top level
from nimble.core.configuration import nimblePath
from nimble.core.configuration import showAvailablePackages
from nimble.core.create import data
from nimble.core.create import ones
from nimble.core.create import zeros
from nimble.core.create import identity
from nimble.core.create import fetchFiles
from nimble.core.learn import learnerType
from nimble.core.learn import learnerNames
from nimble.core.learn import showLearnerNames
from nimble.core.learn import learnerParameters
from nimble.core.learn import showLearnerParameters
from nimble.core.learn import learnerParameterDefaults
from nimble.core.learn import showLearnerParameterDefaults
from nimble.core.learn import loadTrainedLearner
from nimble.core.learn import train
from nimble.core.learn import trainAndApply
from nimble.core.learn import trainAndTest
from nimble.core.learn import trainAndTestOnTrainingData
from nimble.core.learn import normalizeData
from nimble.core.learn import fillMatching
from nimble.core.learn import Init
from nimble.core.tune import Tune
from nimble.core.tune import Tuning
from nimble.core.logger import log
from nimble.core.logger import showLog
from nimble.core.interfaces import CustomLearner

# imports we don't want in __all__
from nimble import core
from nimble._utility import _setAll, _customMlGetattrHelper

# import submodules accessible to the user (in __all__)
from nimble import learners
from nimble import calculate
from nimble import random
from nimble import match
from nimble import fill
from nimble import exceptions

# load settings from configuration file (comments below for Sphinx docstring)
#: User control over configurable options.
#:
#: Use nimble.settings.get() to see all sections and options.
#:
#: See Also
#: --------
#: nimble.core.configuration.SessionConfiguration
#:
#: Keywords
#: --------
#: configure, configuration, options
settings = core.configuration.loadSettings()

# initialize the interfaces
core.interfaces.initInterfaceSetup()

# initialize the logging file
core.logger.initLoggerAndLogConfig()

__all__ = _setAll(vars(), includeModules=True, ignore=[
        'core',
    ])

__version__ = "0.5.3"

def __getattr__(name):
    # The standard AttributeError messsage generated by python
    base = f"module 'nimble' has no attribute '{name}'. "
    extend = _customMlGetattrHelper(name)

    if extend is not None:
        raise AttributeError(base + extend)

    # Ideally this would call something to reproduce the default behavior
    # (for example, self.__getattribute__() in the case of classes) but it
    # isn't clear how to do that in this case. Thus, we raise the error
    # ourselves using the standard format.
    raise AttributeError(base)
