"""
Support for query strings.
"""
import operator
import re

from nimble.exceptions import InvalidArgumentValue
from nimble._utility import _setAll
from . import match

operatorDict = {'!=': operator.ne, '==': operator.eq, '<=': operator.le,
                '>=': operator.ge, '<': operator.lt, '>': operator.gt}

class QueryString:
    """
    A callable object for filtering in Nimble data objects.

    Query strings are used when an object method iterates through the
    elements, points, or features of the object as a quick method to
    define a boolean function. Since queries can be elementwise or
    operate over points and features, the strings will differ sligtly,
    but all follow the same rules. All queries require an operator.
    Valid operators are::

      ==  equal to
      !=  not equal to
      >=  greater than or equal to
      >   greater than
      <=  less than or equal to
      <   less than
      is  special value

    For the "is" operator, the following value can be a function name
    from the nimble.match module (i.e. missing, negative, zero, etc.) or
    "True", "False", or "None".

    For an elementwise query string, only an operator and value are
    required. The "is" operator must be separated from the value by a
    single space, but the space is optional for the other operators.
    For example::

      "==3"         elements equal to 3
      "> 10"        elements greater than 10
      "is missing"  elements that are missing values

    For a points or features query, the point or feature name must be
    included prior to the operator. The operator must be separated from
    the name and the value by single space characters.
    For example::

      "pt3 <= 1"     elements in pt3 that are less than or equal to 1
      "ft1 == True"  elements in ft1 that are equal to the string "True"
      "ft2 is False" elements in ft2 that are the python False object

    The Examples below show how query strings interact with Nimble data
    object methods.

    Note: Query strings do not work for every situation or sometimes
    cannot be parsed because another part of the string also contains an
    operator. All methods that accept query strings also accept a custom
    function that can handle more complex cases.

    Parameters
    ----------
    string : str
        The string to parse. See above for string requirements.
    elementQuery : bool, None
        When set to True, identifies that the query is intended to be
        elementwise. When set to False, identifies that the query is
        specific to a point or feature. This can help eliminate
        ambiguity if the string contains multiple operators.

    Examples
    --------
    >>> lst = [[0, True, -1.0],
    ...        [-1, True, 2.0],
    ...        [2, True, -1.0],
    ...        [-1, False, 3.0]]
    >>> fnames = ['ft1', 'ft2', 'ft3']
    >>> toQuery = nimble.data('DataFrame', lst, featureNames=fnames)
    >>> missing = toQuery.matchingElements("is nonZero")
    >>> missing
    DataFrame(
        [[False  True True]
         [ True  True True]
         [ True  True True]
         [ True False True]]
        featureNames={'ft1':0, 'ft2':1, 'ft3':2}
        )
    >>> toQuery.points.delete('ft3 == -1')
    >>> toQuery
    DataFrame(
        [[-1  True 2.000]
         [-1 False 3.000]]
        featureNames={'ft1':0, 'ft2':1, 'ft3':2}
        )
    """
    _accepted = {n: getattr(match, n) for n in _setAll(vars(match))}
    _accepted['True'] = lambda e: e is True
    _accepted['False'] = lambda e: e is False
    _accepted['None'] = lambda e: e is None

    def __init__(self, string, elementQuery=None):
        if not isinstance(string, str):
            msg = 'string for QueryString is not a string'
            raise InvalidArgumentValue(msg)
        self.string = string
        # elementQuery can be True or False when QueryStrings are constructed
        # to indicate whether the operation part of the string is expected at
        # the beginning (elementwise) or the middle (axiswise) to eliminate
        # ambiguity in some cases.
        self.function = None
        self.filter = None
        self.identifier = None

        # search for "is", if present determine if it is the operator or part
        # of an axis name or string value
        startsIs = re.match(r'(is( not)? )(.*)', string)
        startsOp = re.match(r'(==|!=|>=|>|<=|<) ?(.*)', string)
        # use positive lookaheads to find overlapping results then need
        # group(1) match because overall match must have 0 length
        matchesIs = re.finditer(r'(?=( is( not)? ))', string)
        containsIs = [m.group(1) for m in matchesIs]
        matchesOp = re.finditer('(?=( (==|!=|>=|>|<=|<) ))', string)
        containsOp = [m.group(1) for m in matchesOp]
        isAsAxisName = ((startsIs and containsOp and not containsIs)
                        or (len(containsIs) == 1 and len(containsOp) == 1
                        and not re.search(r'is( not)? ',
                                          string.split(containsOp[0])[1])))

        wise = None
        if not (containsIs or startsIs) or isAsAxisName:
            if startsOp and (not containsOp or elementQuery is True):
                self.function = operatorDict[startsOp.group(1)]
                try: # convert from a string, if possible
                    self.filter = float(startsOp.group(2))
                except ValueError:
                    self.filter = startsOp.group(2)
                wise = 'element'
            elif (len(containsOp) == 1
                    and (not startsOp or elementQuery is False)):
                try: # must have single operator otherwise is ambiguous
                    name, optr, value = re.split(r' (==|!=|>=|>|<=|<) ',
                                                 string)
                    self.identifier = name
                    self.function = operatorDict[optr]
                    try: # convert from a string, if possible
                        self.filter = float(value)
                    except ValueError:
                        self.filter = value
                    wise = 'axis'
                except ValueError:
                    pass
        # last "is" needs to be valid for remaining cases
        elif startsIs and not (containsIs or containsOp):
            funcName = startsIs.group(3)
            if funcName in QueryString._accepted:
                func = QueryString._accepted[funcName]
                if startsIs.group(2): # is not
                    self.function = lambda e: not func(e)
                else:
                    self.function = func
                wise = 'element'
        else:
            identifier, funcName = string.rsplit(containsIs[-1], 1)
            if funcName in QueryString._accepted:
                self.identifier = identifier
                func = QueryString._accepted[funcName]
                if 'not' in containsIs[-1]:
                    self.function = lambda v: not func(v)
                else:
                    self.function = func
                wise = 'axis'

        queryHelp = 'See help(nimble.match.QueryString) for query string '
        queryHelp += 'requirements'
        if elementQuery is True and wise == 'axis':
            msg = 'The query string is designated as elementwise but does not '
            msg += 'begin with an operator. {}'
            raise InvalidArgumentValue(msg.format(queryHelp))
        if elementQuery is False and wise == 'element':
            msg = 'The query string is designated for points/features but '
            msg += 'appears to be elementwise. {}'
            raise InvalidArgumentValue(msg.format(queryHelp))
        if self.function is None:
            msg = 'QueryString was not able to parse the string. {}'
            raise InvalidArgumentValue(msg.format(queryHelp))

    def __call__(self, value):
        args = []
        if self.identifier is not None:
            args.append(value[self.identifier])
        else:
            args.append(value)
        if self.filter is not None:
            args.append(self.filter)

        return self.function(*args)

    def __repr__(self):
        return "QueryString({})".format(self.string)

    def __str__(self):
        return self.string
