"""
Collect the interfaces that will be accessible to the user.
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import importlib
import abc
from .universal_interface import UniversalInterface

displayErrors = False


def collectVisiblePythonModules(modulePath):
    """
    Find files in this directory which could be python importable.
    """
    possibleFiles = os.listdir(modulePath)
    pythonModules = []
    for fileName in possibleFiles:
        if '.' not in fileName:
            continue
        (name, extension) = fileName.rsplit('.', 1)
        if extension == 'py' and not name.startswith('_'):
            pythonModules.append(name)
    return pythonModules


def collectUnexpectedInterfaces(pythonModules):
    """
    Import possible modules and check for possible interfaces.
    """
    possibleInterfaces = []
    # setup seen with the interfaces we know we don't want to load/try to load
    seen = set(["UniversalInterface", "CustomLearnerInterface"])
    for toImport in pythonModules:
        importedModule = importlib.import_module('.' + toImport, __package__)
        contents = dir(importedModule)

        # for each attribute of the module, we will check to see if it is a
        # subclass of the UniversalInterface
        for valueName in contents:
            value = getattr(importedModule, valueName)
            if (isinstance(value, abc.ABCMeta)
                    and issubclass(value, UniversalInterface)):
                if not valueName in seen:
                    seen.add(valueName)
                    possibleInterfaces.append(value)
    return possibleInterfaces


def collect(modulePath):
    """
    Collect the interfaces which import properly.
    """
    pythonModules = collectVisiblePythonModules(modulePath)
    possibleInterfaces = collectUnexpectedInterfaces(pythonModules)

    # now have a list of possible interfaces, which we will try to instantiate
    instantiated = []
    for toInstantiate in possibleInterfaces:
        tempObj = None
        try:
            tempObj = toInstantiate()
        # if ANYTHING goes wrong, just go on without that interface
        except Exception as e:
            if displayErrors:
                print(str(e))
            continue
        if tempObj is not None:
            instantiated.append(tempObj)
    # key: canonical names
    # value: interface using that name, or None if that name has a collision
    nameToInterface = {}
    for namedInterface in instantiated:
        canonicalName = namedInterface.getCanonicalName()
        if canonicalName in list(nameToInterface.keys()):
            # TODO register error with subsystem
            nameToInterface[canonicalName] = None
        else:
            nameToInterface[canonicalName] = namedInterface

    # interfaces should not accept as aliases
    # the canonical names of other interfaces
    for currName in nameToInterface:
        if nameToInterface[currName] is None:
            continue
        for checkName in nameToInterface:
            # we only care about valid interfaces other than currName's
            if (currName != checkName
                    and nameToInterface[checkName] is not None):
                # if currName's interface accepts another's canonical name,
                # we erase it
                if nameToInterface[currName].isAlias(checkName):
                    # TODO register error with subsystem
                    nameToInterface[currName] = None
                    break

    validatedByName = []
    for name in nameToInterface:
        if nameToInterface[name] is not None:
            validatedByName.append(nameToInterface[name])

    return validatedByName
