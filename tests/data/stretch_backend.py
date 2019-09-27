"""
Tests for using stretch attribute to trigger broadcasting operations
"""
import operator

from nose.tools import raises

from nimble.exceptions import ImproperObjectAction
from nimble.randomness import pythonRandom
from .baseObject import DataTestObject
from ..assertionHelpers import assertNoNamesGenerated

# TODO
# point and feature names?
  # If base has? if stretch has?
 # Chained operations

class StretchBackend(DataTestObject):

    ##############
    # Exceptions #
    ##############

    @raises(ImproperObjectAction)
    def test_2D_stretch_exception(self):
        toStretch = self.constructor([[1, 2], [3, 4]])
        toStretch.stretch

    ########################
    # Base / Stretch Point #
    ########################

    # zero safe

    def test_handmade_Base_Stretch_add_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[2, 4, 3, 3], [6, 8, 7, 7], [1, 1, -2, -4]])
        ret1 = base + toStretch.stretch
        ret2 = toStretch.stretch + base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    def test_handmade_Base_Stretch_sub_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 0, 3, 5], [4, 4, 7, 9], [-1, -3, -2, -2]])
        exp_r = self.constructor([[0, 0, -3, -5], [-4, -4, -7, -9], [1, 3, 2, 2]])
        ret1 = base - toStretch.stretch
        ret2 = toStretch.stretch - base

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    def test_handmade_Base_Stretch_mul_point(self):
        base = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        toStretch = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[1, 4, 0, -4], [5, 12, 0, -8], [0, -2, 0, 3]])
        ret1 = base * toStretch.stretch
        ret2 = toStretch.stretch * base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    # zero exception

    def test_handmade_Base_Stretch_truediv_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        exp_r = self.constructor([[1, 1, (-1/3), (-1/2)], [(1/5), (1/3), (-1/7), (-1/4)],
                                  [-1, -1, (1/3), (1/2)]])
        ret1 = base1 / stretch1.stretch
        ret2 = stretch1.stretch / base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 / stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch / base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_floordiv_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 1, -3, -2], [5, 3, -7, -4], [-1, -1, 3, 2]])
        exp_r = self.constructor([[1, 1, -1, -1], [0, 0, -1, -1], [-1, -1, 0, 0]])
        ret1 = base1 // stretch1.stretch
        ret2 = stretch1.stretch // base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 // stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch // base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_mod_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [0, -1, -2, -3]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        exp_r = self.constructor([[0, 0, 2, 2], [1, 2, 6, 6], [0, 0, -1, -2]])
        ret1 = base1 % stretch1.stretch
        ret2 = stretch1.stretch % base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 % stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch % base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_pow_point(self):
        base1 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        stretch1 = self.constructor([[1, 2, -1, -2]])
        base2 = self.constructor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, 0]])
        stretch2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 4, (1/3), (1/16)], [5, 36, (1/7), (1/64)], [-1, 4, (-1/3), (1/16)]])
        exp_r = self.constructor([[1, 4, -1, 16], [1, 64, -1, 256], [1, (1/4), -1, (1/16)]])
        ret1 = base1 ** stretch1.stretch
        ret2 = stretch1.stretch ** base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 ** stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch ** base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    ##########################
    # Base / Stretch Feature #
    ##########################

    # zero safe

    def test_handmade_Base_Stretch_add_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp = self.constructor([[2, 3, 4], [6, 7, 8], [7, 8, 9], [-1, -2, -3]])
        ret1 = base + toStretch.stretch
        ret2 = toStretch.stretch + base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    def test_handmade_Base_Stretch_sub_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[0, 1, 2], [2, 3, 4], [7, 8, 9], [1, 0, -1]])
        exp_r = self.constructor([[0, -1, -2], [-2, -3, -4], [-7, -8, -9], [-1, 0, 1]])
        ret1 = base - toStretch.stretch
        ret2 = toStretch.stretch - base

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    def test_handmade_Base_Stretch_mul_feature(self):
        base = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        toStretch = self.constructor([[1], [2], [0], [-1]])
        exp = self.constructor([[1, 2, 3], [8, 10, 12], [0, 0, 0], [0, 1, 2]])
        ret1 = base * toStretch.stretch
        ret2 = toStretch.stretch * base

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    # zero exception

    def test_handmade_Base_Stretch_truediv_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[1, 2, 3], [2, (5/2), 3], [-7, -8, -9], [(1/2), 1, (3/2)]])
        exp_r = self.constructor([[1, (1/2), (1/3)], [(1/2), (2/5), (1/3)],
                                  [(-1/7), (-1/8), (-1/9)], [2, 1, (2/3)]])
        ret1 = base1 / stretch1.stretch
        ret2 = stretch1.stretch / base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 / stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch / base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_floordiv_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[1, 2, 3], [2, 2, 3], [-7, -8, -9], [0, 1, 1]])
        exp_r = self.constructor([[1, 0, 0], [0, 0, 0], [-1, -1, -1], [2, 1, 0]])
        ret1 = base1 // stretch1.stretch
        ret2 = stretch1.stretch // base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 // stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch // base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_mod_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[0, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, -1]])
        exp_r = self.constructor([[0, 1, 1], [2, 2, 2], [6, 7, 8], [0, 0, -2]])
        ret1 = base1 % stretch1.stretch
        ret2 = stretch1.stretch % base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 % stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch % base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Base_Stretch_pow_feature(self):
        base1 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -2, -3]])
        stretch1 = self.constructor([[1], [2], [-1], [-2]])
        base2 = self.constructor([[1, 2, 3], [4, 5, 6], [7, 8, -1], [0, -1, -2]])
        stretch2 = self.constructor([[1], [2], [0], [-1]])
        exp_l = self.constructor([[1, 2, 3], [16, 25, 36], [(1/7), (1/8), (1/9)], [1, (1/4), (1/9)]])
        exp_r = self.constructor([[1, 1, 1], [16, 32, 64], [-1, 1, -1], [(-1/2), (1/4), (-1/8)]])
        ret1 = base1 ** stretch1.stretch
        ret2 = stretch1.stretch ** base1

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            base2 ** stretch2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            stretch2.stretch ** base2
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    #####################
    # Stretch / Stretch #
    #####################
    @raises(ImproperObjectAction)
    def test_stretch_stretch_bothPoints_exception(self):
        pt1 = self.constructor([[1, 2]])
        pt2 = self.constructor([[3, 4]])

        pt1.stretch + pt2.stretch

    @raises(ImproperObjectAction)
    def test_stretch_stretch_bothFeatures_exception(self):
        ft1 = self.constructor([[1], [2]])
        ft2 = self.constructor([[3], [4]])

        ft1.stretch + ft2.stretch

    # zero safe

    def test_handmade_Stretch_Stretch_add(self):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[1, 2, 0, -1], [2, 3, 1, 0], [3, 4, 2, 1], [0, 1, -1, -2]])
        ret1 = ft.stretch + pt.stretch
        ret2 = pt.stretch + ft.stretch
        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)

    def test_handmade_Stretch_Stretch_sub(self):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[-1, -2, 0, 1], [0, -1, 1, 2], [1, 0, 2, 3], [-2, -3, -1, 0]])
        exp_r = self.constructor([[1, 2, 0, -1], [0, 1, -1, -2], [-1, 0, -2, -3], [2, 3, 1, 0]])
        ret1 = ft.stretch - pt.stretch
        ret2 = pt.stretch - ft.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    def test_handmade_Stretch_Stretch_mul(self):
        ft = self.constructor([[0], [1], [2], [-1]])
        pt = self.constructor([[1, 2, 0, -1]])
        exp = self.constructor([[0, 0, 0, 0], [1, 2, 0, -1], [2, 4, 0, -2], [-1, -2, 0, 1]])
        ret1 = ft.stretch * pt.stretch
        ret2 = pt.stretch * ft.stretch

        assert ret1.isIdentical(exp)
        assert ret2.isIdentical(exp)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

    # zero exception

    def test_handmade_Stretch_Stretch_truediv(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, (1/2), (-1/2), -1], [2, 1, -1, -2],
                                  [3, (3/2), (-3/2), -3], [-1, (-1/2), (1/2), 1]])
        exp_r = self.constructor([[1, 2, -2, -1], [(1/2), 1, -1, (-1/2)],
                                  [(1/3), (2/3), (-2/3), (-1/3)], [-1, -2, 2, 1]])
        ret1 = ft1.stretch / pt1.stretch
        ret2 = pt1.stretch / ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            ft2.stretch / pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch / ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Stretch_Stretch_floordiv(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 0, -1, -1], [2, 1, -1, -2], [3, 1, -2, -3], [-1, -1, 0, 1]])
        exp_r = self.constructor([[1, 2, -2, -1], [0, 1, -1, -1], [0, 0, -1, -1], [-1, -2, 2, 1]])
        ret1 = ft1.stretch // pt1.stretch
        ret2 = pt1.stretch // ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            ft2.stretch // pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch // ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Stretch_Stretch_mod(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [2], [-1]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[0, 1, -1, 0], [0, 0, 0, 0], [0, 1, -1, 0], [0, 1, -1, 0]])
        exp_r = self.constructor([[0, 0, 0, 0], [1, 0, 0, 1], [1, 2, 1, 2], [0, 0, 0, 0]])
        ret1 = ft1.stretch % pt1.stretch
        ret2 = pt1.stretch % ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            ft2.stretch % pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch % ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def test_handmade_Stretch_Stretch_pow(self):
        ft1 = self.constructor([[1], [2], [3], [-1]])
        pt1 = self.constructor([[1, 2, -2, -1]])
        ft2 = self.constructor([[0], [1], [-1], [0]])
        pt2 = self.constructor([[1, 2, 0, -1]])
        exp_l = self.constructor([[1, 1, 1, 1], [2, 4, (1/4), (1/2)], [3, 9, (1/9), (1/3)], [-1, 1, 1, -1]])
        exp_r = self.constructor([[1, 2, -2, -1], [1, 4, 4, 1], [1, 8, -8, -1], [1, (1/2), (-1/2), -1]])
        ret1 = ft1.stretch ** pt1.stretch
        ret2 = pt1.stretch ** ft1.stretch

        assert ret1.isIdentical(exp_l)
        assert ret2.isIdentical(exp_r)
        assertNoNamesGenerated(ret1)
        assertNoNamesGenerated(ret2)

        try:
            ft2.stretch ** pt2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

        try:
            pt2.stretch ** ft2.stretch
            assert False # expected ZeroDivisionError
        except ZeroDivisionError:
            pass

    def back_stretchSetNames(self, obj1, obj2, expPts, expFts):
        # random operation for each, the output is not important only the names
        possibleOps = [operator.add, operator.sub, operator.mul, operator.truediv,
                       operator.floordiv, operator.mod, operator.pow]
        op1 = possibleOps[pythonRandom.randint(0, 6)]
        op2 = possibleOps[pythonRandom.randint(0, 6)]
        ret1 = op1(obj1, obj2)
        ret2 = op2(obj2, obj1)

        if expPts is None:
            assert not ret1.points._namesCreated()
            assert not ret2.points._namesCreated()
        else:
            assert ret1.points.getNames() == expPts
            assert ret2.points.getNames() == expPts
        if expFts is None:
            assert not ret1.features._namesCreated()
            assert not ret2.features._namesCreated()
        else:
            assert ret1.features.getNames() == expFts
            assert ret2.features.getNames() == expFts

    def test_stretchSetNames(self):
        pNames = ['p0', 'p1', 'p2']
        fNames = ['f0', 'f1', 'f2', 'f3']
        offFNames = ['fA', 'fB', 'fC', 'fD']
        offPNames = ['pA', 'pB', 'pC']
        single = ['s']
        single3 = ['s_1', 's_2', 's_3']
        single4 = ['s_1', 's_2', 's_3', 's_4']

        baseRaw = [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]]
        baseNoNames = self.constructor(baseRaw)
        basePtNames = self.constructor(baseRaw, pointNames=pNames)
        baseFtNames = self.constructor(baseRaw, featureNames=fNames)
        baseAllNames = self.constructor(baseRaw, pointNames=pNames, featureNames=fNames)

        stretchPt_Raw = [1, 2, -1, -2]
        stretchPt_NoNames = self.constructor(stretchPt_Raw).stretch
        stretchPt_MatchFtNames = self.constructor(stretchPt_Raw, featureNames=fNames).stretch
        stretchPt_NoMatchFtNames = self.constructor(stretchPt_Raw, featureNames=offFNames).stretch
        stretchPt_WithPtName = self.constructor(stretchPt_Raw, pointNames=single).stretch
        stretchPt_AllNamesFtMatch = self.constructor(stretchPt_Raw, pointNames=single,
                                                  featureNames=fNames).stretch
        stretchPt_AllNamesNoFtMatch = self.constructor(stretchPt_Raw, pointNames=single,
                                                    featureNames=offFNames).stretch

        stretchFt_Raw = [[1], [2], [-1]]
        stretchFt_NoNames = self.constructor(stretchFt_Raw).stretch
        stretchFt_MatchPtNames = self.constructor(stretchFt_Raw, pointNames=pNames).stretch
        stretchFt_NoMatchPtNames = self.constructor(stretchFt_Raw, pointNames=offPNames).stretch
        stretchFt_WithFtName = self.constructor(stretchFt_Raw, featureNames=single).stretch
        stretchFt_AllNamesPtMatch = self.constructor(stretchFt_Raw, pointNames=pNames,
                                                  featureNames=single).stretch
        stretchFt_AllNamesNoPtMatch = self.constructor(stretchFt_Raw, pointNames=offPNames,
                                                    featureNames=single).stretch

        ### Base no names ###
        self.back_stretchSetNames(baseNoNames, stretchPt_NoNames, None, None)
        self.back_stretchSetNames(baseNoNames, stretchFt_NoNames, None, None)
        self.back_stretchSetNames(baseNoNames, stretchPt_MatchFtNames, None, fNames)
        self.back_stretchSetNames(baseNoNames, stretchFt_MatchPtNames, pNames, None)
        self.back_stretchSetNames(baseNoNames, stretchPt_WithPtName, single3, None)
        self.back_stretchSetNames(baseNoNames, stretchFt_WithFtName, None, single4)
        self.back_stretchSetNames(baseNoNames, stretchPt_AllNamesFtMatch, single3, fNames)
        self.back_stretchSetNames(baseNoNames, stretchFt_AllNamesPtMatch, pNames, single4)

        ### Base pt names ###
        self.back_stretchSetNames(basePtNames, stretchPt_NoNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchFt_NoNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchPt_MatchFtNames, pNames, fNames)
        self.back_stretchSetNames(basePtNames, stretchFt_MatchPtNames, pNames, None)
        self.back_stretchSetNames(basePtNames, stretchFt_NoMatchPtNames, None, None)
        self.back_stretchSetNames(basePtNames, stretchPt_WithPtName, None, None)
        self.back_stretchSetNames(basePtNames, stretchFt_WithFtName, pNames, single4)
        self.back_stretchSetNames(basePtNames, stretchPt_AllNamesFtMatch, None, fNames)
        self.back_stretchSetNames(basePtNames, stretchFt_AllNamesPtMatch, pNames, single4)
        self.back_stretchSetNames(basePtNames, stretchFt_AllNamesNoPtMatch, None, single4)

        ### Base ft names ###
        self.back_stretchSetNames(baseFtNames, stretchPt_NoNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_NoNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchPt_MatchFtNames, None, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_MatchPtNames, pNames, fNames)
        self.back_stretchSetNames(baseFtNames, stretchPt_NoMatchFtNames, None, None)
        self.back_stretchSetNames(baseFtNames, stretchPt_WithPtName, single3, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_WithFtName, None, None)
        self.back_stretchSetNames(baseFtNames, stretchPt_AllNamesFtMatch, single3, fNames)
        self.back_stretchSetNames(baseFtNames, stretchFt_AllNamesPtMatch, pNames, None)
        self.back_stretchSetNames(baseFtNames, stretchPt_AllNamesNoFtMatch, single3, None)

        ### Base all names ###
        self.back_stretchSetNames(baseAllNames, stretchPt_NoNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_NoNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchPt_MatchFtNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_MatchPtNames, pNames, fNames)
        self.back_stretchSetNames(baseAllNames, stretchPt_NoMatchFtNames, pNames, None)
        self.back_stretchSetNames(baseAllNames, stretchFt_NoMatchPtNames, None, fNames)
        self.back_stretchSetNames(baseAllNames, stretchPt_WithPtName, None, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_WithFtName, pNames, None)
        self.back_stretchSetNames(baseAllNames, stretchPt_AllNamesFtMatch, None, fNames)
        self.back_stretchSetNames(baseAllNames, stretchFt_AllNamesPtMatch, pNames, None)
        self.back_stretchSetNames(baseAllNames, stretchPt_AllNamesNoFtMatch, None, None)
        self.back_stretchSetNames(baseAllNames, stretchFt_AllNamesNoPtMatch, None, None)

        ### Both Stretch ###
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_NoNames, None, None)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_MatchPtNames, pNames, None)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_WithFtName, None, single4)
        self.back_stretchSetNames(stretchPt_NoNames, stretchFt_AllNamesPtMatch, pNames, single4)

        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_NoNames, None, fNames)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_MatchPtNames, pNames, fNames)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_WithFtName, None, None)
        self.back_stretchSetNames(stretchPt_MatchFtNames, stretchFt_AllNamesPtMatch, pNames, None)

        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_NoNames, single3, None)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_MatchPtNames, None, None)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_WithFtName, single3, single4)
        self.back_stretchSetNames(stretchPt_WithPtName, stretchFt_AllNamesPtMatch, None, single4)

        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_NoNames, single3, fNames)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_MatchPtNames, None, fNames)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_WithFtName, single3, None)
        self.back_stretchSetNames(stretchPt_AllNamesFtMatch, stretchFt_AllNamesPtMatch, None, None)
