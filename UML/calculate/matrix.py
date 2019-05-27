from __future__ import absolute_import

from UML.data import Base
from UML.exceptions import InvalidArgumentType

def elementwiseMultiply(left, right):
    """
    Perform element wise multiplication of two provided UML Base objects
    with the result being returned in a separate UML Base object. Both
    objects must contain only numeric data. The pointCount and featureCount
    of both objects must be equal. The types of the two objects may be
    different. None is always returned.

    """
    # check left is UML
    if not isinstance(left, Base):
        msg = "'left' must be an instance of a UML data object"
        raise InvalidArgumentType(msg)

    left = left.copy()
    left.elements.multiply(right, useLog=False)
    return left


def elementwisePower(left, right):
    """
    Perform an element-wise power operation, with the values in the left object
    as the bases and the values in the right object as exponents. A new object
    will be created, and the input obects will be un-modified.
    """
    # check left is UML

    if not isinstance(left, Base):
        msg = "'left' must be an instance of a UML data object"
        raise InvalidArgumentType(msg)

    left = left.copy()
    left.elements.power(right, useLog=False)
    return left
