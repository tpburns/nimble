
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

from nimble import fill
from nimble.exceptions import InvalidArgumentValue
from tests.helpers import noLogEntryExpected
from tests.helpers import getDataConstructors
from tests.helpers import raises


@noLogEntryExpected
def test_constant_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    constant = 100
    expected = [1, 2, 2, 9]
    for constructor in getDataConstructors():
        toTest = constructor(data, useLog=False)
        exp = constructor(expected, useLog=False)
        assert fill.constant(toTest, match, constant) == exp

@noLogEntryExpected
def test_constant_number_ignoreMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    constant = 100
    expected = [1, 100, 100, 9]
    for constructor in getDataConstructors():
        toTest = constructor(data, useLog=False)
        exp = constructor(expected, useLog=False)
        assert fill.constant(toTest, match, constant) == exp

@noLogEntryExpected
def test_constant_string_ignoreMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    constant = ""
    expected = [1, "", "", 9]
    for constructor in getDataConstructors():
        toTest = constructor(data, useLog=False)
        exp = constructor(expected, useLog=False)
        assert fill.constant(toTest, match, constant) == exp

@noLogEntryExpected
def test_constant_allMatches():
    data = [1, 2, 2, 9]
    match = lambda x: x in [1, 2, 9]
    constant = 100
    expected = [100, 100, 100, 100]
    for constructor in getDataConstructors():
        toTest = constructor(data, useLog=False)
        exp = constructor(expected, useLog=False)
        assert fill.constant(toTest, match, constant) == exp

@noLogEntryExpected
def backend_fill(func, data, match, expected=None):
    "backend for fill functions that do not require additional arguments"
    for constructor in getDataConstructors():
        toTest = constructor(data, useLog=False)
        exp = constructor(expected, useLog=False)
        assert func(toTest, match) == exp

@noLogEntryExpected
def backend_fill_exception(func, data, match, exceptionType):
    "backend for fill functions when testing exception raising"
    for constructor in getDataConstructors(includeSparse=False):
        with raises(exceptionType):
            toTest = constructor(data, useLog=False)
            func(toTest, match)

def test_mean_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    expected = [1, 2, 2, 9]
    backend_fill(fill.mean, data, match, expected)

def test_mean_ignoreMatch():
    data = [1, 2, 2, 9]
    match = lambda x: x == 2
    expected = [1, 5, 5, 9]
    backend_fill(fill.mean, data, match, expected)

def test_mean_allMatches_exception():
    data = [1, 2, 2, 9]
    match = lambda x: x in [1, 2, 2, 9]
    backend_fill_exception(fill.mean, data, match, InvalidArgumentValue)

def test_mean_cannotCalculate_exception():
    data = ['a', 'b', 3, 4]
    match = lambda x: x == 'b'
    backend_fill_exception(fill.mean, data, match, InvalidArgumentValue)

def test_median_noMatches():
    data = [1, 2, 9, 2]
    match = lambda x: False
    expected = [1, 2, 9, 2]
    backend_fill(fill.median, data, match, expected)

def test_median_ignoreMatch():
    data = [1, 2, 9, 2]
    match = lambda x: x == 2
    expected = [1, 5, 9, 5]
    backend_fill(fill.median, data, match, expected)

def test_median_allMatches_exception():
    data = [1, 2, 9, 2]
    match = lambda x: x in [1, 2, 9]
    backend_fill_exception(fill.median, data, match, InvalidArgumentValue)

def test_median_cannotCalculate_exception():
    data = ['a', 'b', 3, 4]
    match = lambda x: x == 'b'
    backend_fill_exception(fill.median, data, match, InvalidArgumentValue)

def test_mode_noMatches():
    data = [1, 2, 2, 9]
    match = lambda x: False
    expected = [1, 2, 2, 9]
    backend_fill(fill.mode, data, match, expected)

def test_mode_ignoreMatch():
    data = [1, 2, 2, 2, 9, 9]
    match = lambda x: x == 2
    expected = [1, 9, 9, 9, 9, 9]
    backend_fill(fill.mode, data, match, expected)

def test_mode_allMatches_exception():
    data = [1, 2, 2, 9, 9]
    match = lambda x: x in [1, 2, 9]
    backend_fill_exception(fill.mode, data, match, InvalidArgumentValue)

def test_forwardFill_noMatches():
    data = [1, 2, 3, 4]
    match = lambda x: False
    expected = data
    backend_fill(fill.forwardFill, data, match, expected)

def test_forwardFill_withMatch():
    data = [1, 2, 3, 4]
    match = lambda x: x == 2
    expected = [1, 1, 3, 4]
    backend_fill(fill.forwardFill, data, match, expected)

def test_forwardFill_consecutiveMatches():
    data = [1, 2, 2, 2, 3, 4, 5]
    match = lambda x: x == 2
    expected = [1, 1, 1, 1, 3, 4, 5]
    backend_fill(fill.forwardFill, data, match, expected)

def test_forwardFill_InitialContainsMatch_exception():
    data = [1, 2, 3, 4]
    match = lambda x: x == 1
    backend_fill_exception(fill.forwardFill, data, match, InvalidArgumentValue)

def test_backwardFill_noMatches():
    data = [1, 2, 3, 4]
    match = lambda x: False
    expected = data
    backend_fill(fill.backwardFill, data, match, expected)

def test_backwardFill_withMatch():
    data = [1, 2, 3, 4]
    match = lambda x: x == 2
    expected = [1, 3, 3, 4]
    backend_fill(fill.backwardFill, data, match, expected)

def test_backwardFill_consecutiveMatches():
    data = [1, 2, 2, 2, 3, 4, 5]
    match = lambda x: x == 2
    expected = [1, 3, 3, 3, 3, 4, 5]
    backend_fill(fill.backwardFill, data, match, expected)

def test_backwardFill_FinalContainsMatch_exception():
    data = [1, 2, 3, 4]
    match = lambda x: x == 4
    backend_fill_exception(fill.backwardFill, data, match, InvalidArgumentValue)

def test_interpolate_noMatches():
    data = [1, 2, 2, 10]
    match = lambda x: False
    expected = data
    backend_fill(fill.interpolate, data, match, expected)

def test_interpolate_withMatch():
    data = [1, 2, 2, 10]
    match = lambda x: x == 2
    # linear function y = x + 3
    expected = [1, 4, 7, 10]
    backend_fill(fill.interpolate, data, match, expected)

@noLogEntryExpected
def test_interpolate_withArguments():
    data = [1, "na", "na", 5]
    arguments = {}
    # linear function y = 2x + 5
    arguments['xp'] = [0, 4, 8]
    arguments['fp'] = [5, 13, 21]
    match = lambda x: x == "na"
    expected = [1, 7, 9, 5]
    for constructor in getDataConstructors(includeSparse=False):
        toTest = constructor(data, useLog=False)
        exp = constructor(expected, useLog=False)
        assert fill.interpolate(toTest, match, **arguments) == exp

@noLogEntryExpected
def test_interpolate_xKwargIncluded_exception():
    data = [1, "na", "na", 5]
    arguments = {}
    # linear function y = 2x + 5
    arguments['xp'] = [0, 4, 8]
    arguments['fp'] = [5, 13, 21]
    arguments['x'] = [1]  # disallowed argument
    match = lambda x: x == "na"
    for constructor in getDataConstructors(includeSparse=False):
        with raises(TypeError):
            toTest = constructor(data, useLog=False)
            ret = fill.interpolate(toTest, match, **arguments)
